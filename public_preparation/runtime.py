"""Public task reconstruction and real Harbor fixture controls; no inference probes."""

import argparse
import ast
import copy
import hashlib
import json
import os
import secrets
import shutil
import subprocess
import time

from experiments.configuration_paths import (
    configuration_path,
    resolve_configuration_path,
)
from public_preparation.catalog import ROOT, catalog
from public_preparation.materialize import (
    current_source_path,
    materialize_acceptance,
    resolve,
)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def policy_fingerprints(names):
    """Bind relocated inputs and the path registry used to resolve them."""
    paths = [
        resolve_configuration_path(ROOT / current_source_path(name)) for name in names
    ]
    paths += [
        ROOT / "experiments/configuration_paths.py",
        ROOT / "configs/configuration-paths.json",
    ]
    return {
        str(path.relative_to(ROOT)): digest(path.read_bytes())
        for path in paths
        if path.is_file()
    }


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")


def run(args, *, log=None, cwd=ROOT, timeout=1800):
    env = {
        k: v
        for k, v in os.environ.items()
        if not any(
            t in k.upper() for t in ["KEY", "TOKEN", "SECRET", "CREDENTIAL", "PASSWORD"]
        )
    }
    env.update(
        PUBLIC_NO_INFERENCE="1",
        MSWEA_SILENT_STARTUP="1",
        MSWEA_GLOBAL_CONFIG_DIR=str(ROOT / ".local/mini"),
        GIT_TERMINAL_PROMPT="0",
    )
    if log:
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("w") as out:
            p = subprocess.run(
                list(map(str, args)),
                cwd=cwd,
                env=env,
                stdout=out,
                stderr=subprocess.STDOUT,
                timeout=timeout,
            )
        if p.returncode:
            raise RuntimeError(
                f"Command exit {p.returncode}; inspect {log.relative_to(ROOT)}"
            )
        return ""
    p = subprocess.run(
        list(map(str, args)),
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if p.returncode:
        raise RuntimeError(p.stderr[-3000:])
    return p.stdout


def export(cache, commit, path):
    from public_replay.prepare import export_commit

    if path.exists():
        return
    return export_commit(cache, commit, path)


def constants(path):
    result = {}
    for node in ast.parse(path.read_text()).body:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
        ):
            try:
                result[node.targets[0].id] = ast.literal_eval(node.value)
            except ValueError, TypeError:
                pass
    return result


def task_config(tid):
    spec = catalog()[tid]
    spec.update(
        json.loads(
            (ROOT / "public_preparation/configs/reference-pins.json").read_text()
        )[tid]
    )
    if not spec["reproducer"]:
        data = json.loads((ROOT / "tasks" / tid / "task.json").read_text())
        spec["reproducer"] = (
            data["reproduction_command"].split("<<'PY'\n", 1)[1].rsplit("\nPY", 1)[0]
            + "\n"
        )
    return spec


def source(spec, state, network):
    cache = ROOT / ".local/upstream.git"
    if not cache.exists():
        run(["git", "init", "--bare", cache])
    for commit in [spec["starting_commit"], spec["reference_commit"]]:
        present = (
            subprocess.run(
                [
                    "git",
                    "--git-dir",
                    str(cache),
                    "cat-file",
                    "-e",
                    commit + "^{commit}",
                ],
                capture_output=True,
            ).returncode
            == 0
        )
        if not present:
            if not network:
                raise RuntimeError(
                    "Missing pinned commit; rerun with --allow-setup-network"
                )
            run(
                [
                    "git",
                    "--git-dir",
                    cache,
                    "fetch",
                    "--depth=1",
                    spec["repository"],
                    commit,
                ],
                log=state / (commit + ".fetch.log"),
            )
    baseline = state / "baseline"
    if not baseline.exists():
        identity = export(cache, spec["starting_commit"], baseline)
        if identity["baseline_checksum"] != spec["baseline_checksum"]:
            raise ValueError("Baseline differs from pinned historical file identity")
        write(state / "source.json", identity)
    # Retrieve only changed production files into a host-only reference fixture.
    names = run(
        [
            "git",
            "--git-dir",
            cache,
            "diff",
            "--name-only",
            spec["starting_commit"],
            spec["reference_commit"],
        ]
    ).splitlines()
    production = [
        name
        for name in names
        if any(name.startswith(prefix + "/") for prefix in spec["source_paths"])
        and name.endswith(".py")
    ]
    fixture = {}
    for name in production:
        fixture[name] = run(
            ["git", "--git-dir", cache, "show", spec["reference_commit"] + ":" + name]
        )
    if not fixture:
        raise RuntimeError("Reference contains no permitted production changes")
    # A new author-written regression exercises the public reproducer, while the
    # independently retained acceptance suite checks the complete task contract.
    regression = spec["test_paths"][0] + "/test_public_regression.py"
    fixture[regression] = (
        '"""Public reproducer regression, independent of historical outcomes."""\n\nimport runpy\n\n\ndef test_public_reproduction() -> None:\n    """Require the reported behavior to be corrected."""\n    runpy.run_path("/task/reproduce.py", run_name="__main__")\n'
    )
    write(state / "reference-fixture.json", fixture)
    spec["reference_files"] = production
    return spec


def build_image(context, tag, *, network=False):
    run(
        [
            "docker",
            "build",
            "--platform",
            "linux/arm64",
            "--network=" + ("default" if network else "none"),
            "--pull=false",
            "-t",
            tag,
            context,
        ],
        log=context / "build.log",
    )
    return run(["docker", "image", "inspect", tag, "--format", "{{.Id}}"]).strip()


def prepare(tid, network=False):
    import toml

    spec = task_config(tid)
    state = ROOT / ".local/public-v1" / tid
    state.mkdir(parents=True, exist_ok=True)
    spec = source(spec, state, network)
    acceptance = state / "evaluator/test_acceptance.py"
    extracted = materialize_acceptance(tid, ROOT / ".local/upstream.git", acceptance)
    baseline = state / "baseline"
    bundle = ROOT / spec["task_bundle"]
    if not bundle.is_dir():
        raise RuntimeError("Retained task bundle missing: " + spec["task_bundle"])
    if not (state / "dependency.json").exists():
        if not network:
            raise RuntimeError("Dependency build requires --allow-setup-network")
        context = state / "build/dependencies"
        context.mkdir(parents=True, exist_ok=True)
        if not (context / "source").exists():
            shutil.copytree(baseline, context / "source")
        import tomllib

        manifest = tomllib.loads(
            (baseline / spec["package"] / "pyproject.toml").read_text()
        )
        groups = [
            g
            for g in ["test", "lint", "typing", "test_integration"]
            if g in manifest.get("dependency-groups", {})
        ]
        if tid in {"LC-TIMEOUT-001", "LC-RESPONSES-USAGE-001"}:
            groups = [g for g in groups if g != "test_integration"]
        if not groups:
            raise RuntimeError(
                "No supported locked dependency groups in pinned manifest"
            )
        lock = baseline / spec["package"] / "uv.lock"
        if not lock.is_file():
            raise RuntimeError("Pinned package uv.lock missing")
        python_image = "python@sha256:4c92ffcde4dd6f1ff72a24518f49fd4990b27134987dfa31a733badde66df9f8"
        env_file = ROOT / "tasks" / tid / "environment.json"
        if env_file.is_file():
            python_image = json.loads(env_file.read_text()).get(
                "python_image", python_image
            )
        if tid in {"LC-TIMEOUT-001", "LC-RESPONSES-USAGE-001", "LC-CHROMA-NONE-001"}:
            python_image = "python@sha256:73aa5faee130c60da030e105ec971ccf57473fbd22ce73bd6fc9ba99a40b838d"
        dockerfile = f"""FROM ghcr.io/astral-sh/uv@sha256:eb2843a1e56fd9e30c7276ce1a52cba86e64c7b385f5e3279a0e08e02dd058fc AS uv
FROM {python_image}
COPY --from=uv /uv /usr/local/bin/uv
COPY source /workspace
WORKDIR /workspace/{spec["package"]}
ENV UV_PROJECT_ENVIRONMENT=/opt/venv UV_PYTHON_DOWNLOADS=never
RUN uv sync --frozen {" ".join("--group " + g for g in groups)}
ENV TIKTOKEN_CACHE_DIR=/opt/tiktoken HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
RUN mkdir -p /opt/tiktoken && /opt/venv/bin/python -c 'import importlib.util; exec("import tiktoken; tiktoken.get_encoding(\\\"o200k_base\\\"); tiktoken.get_encoding(\\\"cl100k_base\\\")") if importlib.util.find_spec("tiktoken") else None'
RUN uv pip freeze --python /opt/venv/bin/python > /opt/requirements.lock
ENV PATH=/opt/venv/bin:/usr/local/bin:/usr/bin:/bin PYTHONDONTWRITEBYTECODE=1 HOME=/tmp
"""
        if tid == "LC-HF-ACCELERATOR-001":
            dockerfile = dockerfile.replace(
                "COPY source /workspace",
                "RUN apt-get update && apt-get install -y --no-install-recommends build-essential=12.9 && rm -rf /var/lib/apt/lists/*\nCOPY source /workspace",
            )
        if tid == "LC-ASYNC-TOOLS-001":
            core_lock = tomllib.loads((baseline / "libs/core/uv.lock").read_text())
            lines = []
            for name in ["blockbuster", "forbiddenfruit"]:
                package = next(p for p in core_lock["package"] if p["name"] == name)
                hashes = [p["hash"] for p in package.get("wheels", [])]
                if package.get("sdist"):
                    hashes.append(package["sdist"]["hash"])
                lines.append(
                    name
                    + "=="
                    + package["version"]
                    + " "
                    + " ".join("--hash=" + h for h in hashes)
                )
            (context / "supplemental.lock").write_text("\n".join(lines) + "\n")
            dockerfile = dockerfile.replace(
                "RUN uv pip freeze",
                "COPY supplemental.lock /tmp/supplemental.lock\nRUN uv --no-config pip install --python /opt/venv/bin/python --no-deps --require-hashes -r /tmp/supplemental.lock\nRUN uv pip freeze",
            )
        (context / "Dockerfile").write_text(dockerfile)
        image = build_image(
            context, "maintenance-public-deps:" + tid.lower(), network=True
        )
        write(
            state / "dependency.json",
            {
                "image": image,
                "source_lock_sha256": digest(lock.read_bytes()),
                "groups": groups,
                "python_image": python_image,
                "model_requests": 0,
            },
        )
    dep = json.loads((state / "dependency.json").read_text())["image"]
    key = state / "receipt.key"
    if not key.exists():
        key.write_bytes(secrets.token_bytes(32))
        key.chmod(0o600)
    lockroot = ROOT / "public_preparation/locks"
    lockroot.mkdir(parents=True, exist_ok=True)
    observed = run(
        [
            "docker",
            "run",
            "--rm",
            "--network=none",
            dep,
            "cat",
            "/opt/requirements.lock",
        ]
    )
    resolved_lock = lockroot / (tid + ".requirements.lock")
    if resolved_lock.exists() and resolved_lock.read_text() != observed:
        raise RuntimeError(
            "Resolved environment differs from the published dependency lock: " + tid
        )
    if not resolved_lock.exists():
        resolved_lock.write_text(observed)
    pytest_opts = "-p no:cacheprovider"
    cov = run(
        [
            "docker",
            "run",
            "--rm",
            "--network=none",
            dep,
            "/opt/venv/bin/python",
            "-c",
            "import importlib.util; print(int(importlib.util.find_spec('pytest_cov') is not None))",
        ]
    ).strip()
    if cov == "1":
        pytest_opts += " --no-cov"
    image_ids = {}
    verifier_constants = constants(bundle / "tests/verify.py")
    pythonpath = verifier_constants.get(
        "PYTHONPATH",
        "/workspace/"
        + spec["package"]
        + ":/workspace/libs/core:/workspace/libs/standard-tests",
    )
    for role in ["agent", "verifier"]:
        context = state / "build" / role
        context.mkdir(parents=True, exist_ok=True)
        if not (context / "source").exists():
            shutil.copytree(baseline, context / "source")
        (context / "reproduce.py").write_text(spec["reproducer"])
        lines = [
            f"FROM maintenance-public-deps:{tid.lower()}",
            "USER root",
            "RUN rm -rf /workspace",
            "COPY source /workspace",
            "COPY reproduce.py /task/reproduce.py",
        ]
        if role == "agent":
            # Use task-specific original isolation tests when available; these
            # never receive reference fixtures, evaluator source or receipt keys.
            iso = ROOT / "tasks" / tid / "isolation.py"
            shutil.copyfile(
                iso if iso.is_file() else ROOT / "harbor_integration/isolation.py",
                context / "isolation.py",
            )
            (context / "public-isolation.py").write_text(public_isolation(spec))
            lines += [
                "COPY isolation.py /task/historical-isolation.py",
                "COPY public-isolation.py /task/isolation.py",
            ]
        else:
            if not (context / "baseline").exists():
                shutil.copytree(baseline, context / "baseline")
            if (context / "tests").exists():
                shutil.rmtree(context / "tests")
            shutil.copytree(bundle / "tests", context / "tests")
            if extracted:
                shutil.copyfile(acceptance, context / "tests/test_acceptance.py")
                shutil.copyfile(
                    acceptance.with_name("UPSTREAM-LICENSE"),
                    context / "tests/UPSTREAM-LICENSE",
                )
            shutil.copyfile(key, context / "tests/receipt.key")
            verify = context / "tests/verify.py"
            text = (
                verify.read_text()
                .replace('"-p no:cacheprovider --no-cov"', repr(pytest_opts))
                .replace('"-p no:cacheprovider"', repr(pytest_opts))
            )
            verify.write_text(text)
            lines += ["COPY baseline /baseline", "COPY tests /tests"]
        lines += [
            "RUN chmod -R a-w /workspace /task && mkdir -p /logs/agent /logs/verifier /logs/artifacts && chmod 777 /logs/agent /logs/verifier /logs/artifacts"
        ]
        if role == "agent":
            for src in spec["source_paths"]:
                lines += [
                    f"RUN find /workspace/{src} -type d -exec chmod 777 {{}} + && find /workspace/{src} -type f -exec chmod 666 {{}} +"
                ]
            for tests in spec["test_paths"]:
                lines += [f"RUN chmod 1777 /workspace/{tests}"]
        else:
            lines += [
                "RUN chmod -R a-w /baseline /tests && chmod 400 /tests/receipt.key"
            ]
        lines += [
            f"ENV PYTHONPATH={pythonpath}",
            "ENV PYTEST_ADDOPTS=" + json.dumps(pytest_opts),
            "ENV OPENAI_API_KEY=sk-offline-placeholder",
            f"WORKDIR /workspace/{spec['package']}",
            "USER " + ("65532:65532" if role == "agent" else "root"),
        ]
        (context / "Dockerfile").write_text("\n".join(lines) + "\n")
        image_ids[role] = build_image(
            context, "maintenance-public-" + role + ":" + tid.lower()
        )
    generated = state / "harbor-task"
    if generated.exists():
        shutil.rmtree(generated)
    shutil.copytree(bundle, generated)
    if extracted:
        shutil.copyfile(acceptance, generated / "tests/test_acceptance.py")
        shutil.copyfile(
            acceptance.with_name("UPSTREAM-LICENSE"),
            generated / "tests/UPSTREAM-LICENSE",
        )
    generated_verify = generated / "tests/verify.py"
    generated_verify.write_text(
        generated_verify.read_text()
        .replace('"-p no:cacheprovider --no-cov"', repr(pytest_opts))
        .replace('"-p no:cacheprovider"', repr(pytest_opts))
    )
    task = toml.loads((generated / "task.toml").read_text())
    task["metadata"].update(
        public_protocol=spec["protocol_id"], historical_equivalence="not claimed"
    )
    task["environment"]["docker_image"] = image_ids["agent"]
    task["verifier"]["environment"]["docker_image"] = image_ids["verifier"]
    (generated / "task.toml").write_text(toml.dumps(task))
    # Keep all original isolation and resource constraints from the bundle.
    templates = list((ROOT / "configs").glob("*" + tid + "*model.json"))
    preferred = [
        p for p in templates if p.name.startswith("held-out-") and "minimax" in p.name
    ] or [
        p for p in templates if p.name.startswith("comparison-") and "minimax" in p.name
    ]
    if not preferred:
        preferred = [
            configuration_path(
                ROOT,
                "configs/context-counter-repaired-calibration-v1-minimax-model.json",
            )
        ]
    job = resolve(json.loads(preferred[0].read_text()))
    agent = job["agents"][0]
    policy = agent["kwargs"]["policy"]
    policy["receipt_key"] = str(key)
    policy["reference_fixture"] = str(state / "reference-fixture.json")
    policy["source_provenance"] = {
        "starting_commit": spec["starting_commit"],
        "baseline_checksum": spec["baseline_checksum"],
    }
    policy["reproduction"] = {"expected_failure": spec["expected_failure"]}
    # Capture and finalization semantics, budgets and provider settings are inherited.
    fingerprints = policy_fingerprints(policy.get("code_fingerprints", {}))
    policy["code_fingerprints"] = fingerprints
    job["tasks"][0]["path"] = str(generated)
    job["jobs_dir"] = str(ROOT / "runs/public-v1")
    for fixture in [
        None,
        "baseline",
        "reference",
        "reference_budget",
        "reference_error",
    ]:
        config = copy.deepcopy(job)
        if fixture is not None:
            config["agents"][0]["kwargs"]["fixture"] = fixture
        else:
            config["agents"][0]["kwargs"].pop("fixture", None)
        write(state / ((fixture or "model") + ".json"), config)
    public = {
        k: v
        for k, v in spec.items()
        if k not in ["retained_evaluator", "retained_acceptance"]
    }
    public.update(
        acceptance_materialization=extracted,
        resolved_requirements_sha256=digest(observed.encode()),
        images=image_ids,
        dependency_image=dep,
        source_lock_sha256=json.loads((state / "dependency.json").read_text())[
            "source_lock_sha256"
        ],
        model_requests=0,
        status="built_controls_required",
    )
    public["task_bundle_sha256"] = digest(
        json.dumps(
            {
                str(f.relative_to(generated)): digest(f.read_bytes())
                for f in generated.rglob("*")
                if f.is_file()
            },
            sort_keys=True,
        ).encode()
    )
    public["setting_id"] = digest(json.dumps(public, sort_keys=True).encode())
    write(state / "prepared.json", public)
    return public


def public_isolation(spec):
    return (
        """import importlib,json,os,socket,subprocess
from pathlib import Path
checks={}
checks['no_history']=not list(Path('/workspace').rglob('.git'))
checks['no_evaluator']=not any(Path(p).exists() for p in ['/tests','/baseline','/reference','/evaluator','/run/secrets','/host'])
checks['no_keys']=all(k=='OPENAI_API_KEY' and v=='sk-offline-placeholder' for k,v in os.environ.items() if 'API_KEY' in k or k.endswith('_TOKEN') or 'SECRET' in k)
checks['unprivileged']=os.getuid()==65532
checks['no_docker_socket']=not Path('/var/run/docker.sock').exists()
for relative in TESTROOTS:
    root=Path('/workspace')/relative
    probe=root/'test_public_write_probe.py'
    probe.write_text('')
    probe.unlink()
    existing=next(root.rglob('test_*.py'))
    checks['existing_readonly_'+relative]=not os.access(existing,os.W_OK)
    renamed=existing.with_name('public_rename_probe')
    try:
        existing.rename(renamed)
        renamed.rename(existing)
        checks['existing_rename_denied_'+relative]=False
    except OSError:checks['existing_rename_denied_'+relative]=True
for relative in SOURCEPATHS:
    checks['production_writable_'+relative]=os.access(Path('/workspace')/relative,os.W_OK)
    probe=Path('/workspace')/relative/'public_source_probe.py'
    probe.write_text('')
    probe.unlink()
    link=Path('/workspace')/relative/'public_link_probe'
    link.symlink_to('/host-private/resources')
    checks['host_symlink_denied_'+relative]=not link.exists()
    link.unlink()
for name in ['/opt/venv/public_probe','/workspace/public_probe']:
    try:
        Path(name).write_text('')
        checks['readonly_'+name]=False
    except OSError:checks['readonly_'+name]=True
if ORIGINAL_PROBE:
    legacy=subprocess.run(['python','/task/historical-isolation.py'],capture_output=True,text=True)
    checks['retained_isolation_suite']=legacy.returncode==0
with socket.socket() as sock:
    sock.settimeout(1)
    checks['offline']=sock.connect_ex(('1.1.1.1',443)) != 0
for name in MODULES:
    module=importlib.import_module(name)
    checks['workspace_import_'+name]=str(Path(module.__file__).resolve()).startswith('/workspace/')
print(json.dumps(checks))
assert all(checks.values()),checks
""".replace(
            "ORIGINAL_PROBE",
            repr(
                (ROOT / "tasks" / spec["task_id"] / "isolation.py").is_file()
                or spec["package"] == "libs/partners/openai"
            ),
        )
        .replace("TESTROOTS", repr(spec["test_paths"]))
        .replace("SOURCEPATHS", repr(spec["source_paths"]))
        .replace(
            "MODULES",
            repr(
                [
                    {
                        "libs/core": "langchain_core",
                        "libs/langchain_v1": "langchain",
                        "libs/partners/openai": "langchain_openai",
                        "libs/partners/chroma": "langchain_chroma",
                        "libs/partners/fireworks": "langchain_fireworks",
                        "libs/partners/openrouter": "langchain_openrouter",
                        "libs/partners/huggingface": "langchain_huggingface",
                        "libs/text-splitters": "langchain_text_splitters",
                    }[spec["package"]],
                    "langchain_core",
                ]
            ),
        )
    )


def validate(tid):
    state = ROOT / ".local/public-v1" / tid
    public = json.loads((state / "prepared.json").read_text())
    from public_preparation.wrapper_controls import validate_wrapper

    wrapper = validate_wrapper(tid)
    controls = []
    binding_path = state / "control-bindings.json"
    bindings = json.loads(binding_path.read_text()) if binding_path.exists() else {}
    for fixture in ["baseline", "reference", "reference_budget", "reference_error"]:
        job = "public-v1-" + tid.lower() + "-" + fixture
        expected_config = digest(
            (state / (fixture + ".json")).read_bytes()
            + json.dumps(
                {
                    str(f.relative_to(state / "harbor-task")): digest(f.read_bytes())
                    for f in (state / "harbor-task").rglob("*")
                    if f.is_file()
                },
                sort_keys=True,
            ).encode()
        )
        bound = bindings.get(fixture, {})
        if bound.get("configuration_sha256") == expected_config:
            job = bound["job"]
        existing = (
            list((ROOT / "runs/public-v1" / job).glob("*/verifier/checks.json"))
            if bound.get("configuration_sha256") == expected_config
            and bound.get("passed", True)
            else []
        )
        if not existing:
            if (ROOT / "runs/public-v1" / job).exists():
                job += "-" + str(int(time.time()))
            run(
                [
                    ROOT / ".venv/bin/harbor",
                    "run",
                    "-c",
                    state / (fixture + ".json"),
                    "--job-name",
                    job,
                ],
                log=state / (fixture + "-launch.log"),
            )
        bindings[fixture] = {"job": job, "configuration_sha256": expected_config}
        write(binding_path, bindings)
        trials = list((ROOT / "runs/public-v1" / job).glob("*/verifier/checks.json"))
        if len(trials) != 1:
            raise RuntimeError("Harbor control produced no verifier checks: " + fixture)
        checks = json.loads(trials[0].read_text())
        trial = trials[0].parents[1]
        reward = json.loads((trial / "verifier/reward.json").read_text())
        receipt = json.loads((trial / "agent/receipt.json").read_text())["receipt"]
        isolation = json.loads(
            next(
                line
                for line in reversed(
                    (trial / "agent/isolation.json").read_text().splitlines()
                )
                if line.startswith("{")
            )
        )
        accepted = bool(
            reward.get(
                "patch_accepted", reward.get("patch_acceptance", reward.get("reward"))
            )
        )
        expected = fixture != "baseline"
        correct = (
            accepted == expected
            and receipt["model_requests"] == 0
            and all(isolation.values())
        )
        expected_stop = {
            "baseline": "submitted",
            "reference": "submitted",
            "reference_budget": "budget_exhausted",
            "reference_error": "error",
        }[fixture]
        correct = (
            correct
            and receipt.get("termination_reason") == expected_stop
            and receipt.get("workspace_quiesced") is True
        )
        raw_trace = (
            (trial / "agent/requests.jsonl").read_text()
            if (trial / "agent/requests.jsonl").exists()
            else ""
        )
        correct = correct and not any(
            json.loads(line).get("event") == "model_request"
            for line in raw_trace.splitlines()
            if line.strip()
        )
        if expected:
            correct = (
                correct
                and set(
                    [
                        "file_policy",
                        "regression_on_baseline",
                        "reproducer",
                        "regression",
                        "existing_tests",
                        "lint",
                        "format",
                        "types",
                        "hidden_acceptance",
                    ]
                ).issubset(checks["checks"])
                and all(
                    c.get("exit_code") == (1 if name == "regression_on_baseline" else 0)
                    for name, c in checks["checks"].items()
                    if isinstance(c, dict)
                )
            )
        result = {
            "fixture": fixture,
            "passed": correct,
            "accepted": accepted,
            "model_requests": receipt["model_requests"],
            "termination_reason": receipt.get("termination_reason"),
            "checks": {
                k: {"exit_code": v.get("exit_code")}
                for k, v in checks.get("checks", {}).items()
            },
            "isolation": isolation,
            "error": checks.get("error"),
            "artifact_dir": str(trial.relative_to(ROOT)),
        }
        bindings[fixture]["passed"] = correct
        write(binding_path, bindings)
        controls.append(result)
        write(state / "controls-progress.json", controls)
        if not correct:
            raise RuntimeError(
                f"{tid} {fixture} failed public checks; inspect {state.relative_to(ROOT)}/controls-progress.json"
            )
    required_inputs = [
        ROOT / "experiments/configuration_paths.py",
        ROOT / "configs/configuration-paths.json",
        state / "prepared.json",
        state / "model.json",
        state / "reference-fixture.json",
        state / "receipt.key",
        *list((state / "harbor-task").rglob("*")),
        *list((ROOT / "public_preparation").glob("*.py")),
        *list((ROOT / "public_preparation/configs").glob("*.json")),
        *list((ROOT / "configs").glob("mini-swe*.yaml")),
    ]
    files = {
        str(f.relative_to(ROOT)): digest(f.read_bytes())
        for f in required_inputs
        if f.is_file()
    }
    receipt = {
        "task_id": tid,
        "protocol_id": public["protocol_id"],
        "setting_id": public["setting_id"],
        "historical_equivalence": "not claimed",
        "model_requests": 0,
        "baseline_rejected": True,
        "reference_accepted": True,
        "public_checks_preserved": True,
        "wrapper_controls": wrapper,
        "controls": controls,
        "files": files,
        "validated_at": time.time(),
    }
    write(state / "validated.json", receipt)
    # Tracked receipts contain numeric outcomes only, never source or patches/keys.
    write(
        ROOT / "reports/public-readiness" / (tid + ".json"),
        {k: v for k, v in receipt.items() if k != "files"},
    )
    return receipt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["prepare", "validate"])
    p.add_argument("task", choices=sorted(catalog()))
    p.add_argument("--allow-setup-network", action="store_true")
    args = p.parse_args()
    result = (
        prepare(args.task, args.allow_setup_network)
        if args.action == "prepare"
        else validate(args.task)
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
