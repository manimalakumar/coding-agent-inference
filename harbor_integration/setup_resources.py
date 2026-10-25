"""Trusted setup for resource-configured historical tasks; never mounted to agents."""

import argparse
import hashlib
import json
import shlex
import shutil
import subprocess
import time
import tomllib
import urllib.request
from pathlib import Path

from agent.schemas import Task
from experiments.configuration_paths import configuration_path
from harbor_integration.prepare_task import build
from sandbox.docker import digest, manifest
from sandbox.prepare import export_snapshot

ROOT = Path(__file__).resolve().parents[1]


def setup(reference, task_file, reuse, state, reference_commit, fixture_spec):
    started = time.monotonic()
    task = Task.model_validate_json(task_file.read_text())
    environment = json.loads((task_file.parent / "environment.json").read_text())
    previous = json.loads((reuse / "setup.json").read_text())
    assert previous["setup_status"] == "validated"
    key = digest(
        {
            "task": task.model_dump(),
            "environment": environment,
            "reproducer": (task_file.parent / task.reproduction_script).read_text(),
            "isolation": (task_file.parent / "isolation.py").read_text(),
            "dependency_image": previous["image_id"],
            "recipe": Path(__file__).read_text(),
        }
    )[:16]
    target = state / task.task_id / key
    target.mkdir(parents=True, exist_ok=True)
    record_file = target / "setup.json"
    if record_file.exists():
        saved = json.loads(record_file.read_text())
        if saved["setup_status"] == "validated":
            assert digest(manifest(target / "baseline")) == saved["baseline_checksum"]
            return target
    provenance_file = target / "provenance.json"
    if not provenance_file.exists():
        provenance_file.write_text(
            json.dumps(export_snapshot(reference, task, target / "baseline"), indent=2)
        )
    provenance = json.loads(provenance_file.read_text())
    assert digest(manifest(target / "baseline")) == provenance["baseline_checksum"]
    for asset in [task.reproduction_script, "isolation.py"]:
        shutil.copyfile(task_file.parent / asset, target / asset)
        (target / asset).chmod(0o444)
    context = target / "dependency-build"
    context.mkdir(exist_ok=True)
    if not (context / "source").exists():
        shutil.copytree(target / "baseline", context / "source")
    wheels = context / "wheels"
    wheels.mkdir(exist_ok=True)
    lock = tomllib.loads((target / "baseline" / environment["lockfile"]).read_text())
    artifacts = {}
    for name in environment["packages"]:
        package = next(p for p in lock["package"] if p["name"] == name)
        asset = next(
            (
                w
                for w in package.get("wheels", [])
                if w["url"].endswith(("py3-none-any.whl", "py2.py3-none-any.whl"))
                or (
                    environment.get("python_version", "").startswith("3.13")
                    and "cp313-cp313-" in w["url"]
                    and "manylinux" in w["url"]
                    and "aarch64" in w["url"]
                )
            ),
            package.get("sdist"),
        )
        assert asset, "No platform-independent artifact for " + name
        destination = wheels / asset["url"].rsplit("/", 1)[1]
        if not destination.exists():
            destination.write_bytes(
                urllib.request.urlopen(asset["url"], timeout=60).read()
            )
        assert (
            hashlib.sha256(destination.read_bytes()).hexdigest()
            == asset["hash"].split(":")[1]
        )
        record = dict(asset) | {"version": package["version"]}
        if not destination.name.endswith(".whl"):
            # Historical pure-Python sdist; keep build tooling isolated from runtime.
            constraints = target / "build-constraints.txt"
            constraints.write_text("setuptools==80.9.0\nwheel==0.45.1\n")
            command = [
                "uv",
                "build",
                "--wheel",
                "--python",
                str(ROOT / ".venv/bin/python"),
                "--build-constraint",
                str(constraints),
                "--out-dir",
                str(wheels),
                str(destination),
            ]
            built = subprocess.run(command, capture_output=True, text=True, timeout=150)
            (target / (name + "-build.log")).write_text(built.stdout + built.stderr)
            if built.returncode:
                raise RuntimeError("Historical build failed: " + name)
            built_wheel = next(wheels.glob(name.replace("-", "_") + "-*.whl"))
            record["built_wheel_sha256"] = hashlib.sha256(
                built_wheel.read_bytes()
            ).hexdigest()
            record["build_constraints"] = constraints.read_text()
            destination.unlink()  # Docker installs only the verified/built wheels offline.
        artifacts[name] = record
    # Preserve every unaffected runtime pin and record each changed/added artifact hash.
    lines = (reuse / "requirements.lock").read_text().splitlines(True)
    for name, artifact in artifacts.items():
        variants = {name, name.replace("-", "_")}
        first = next(
            (
                i
                for i, line in enumerate(lines)
                if any(line.lower().startswith(v + "==") for v in variants)
            ),
            None,
        )
        entry = (
            name
            + "=="
            + artifact["version"]
            + " --hash=sha256:"
            + artifact.get("built_wheel_sha256", artifact["hash"].split(":")[1])
            + "\n"
        )
        if first is None:
            lines.append(entry)
        else:
            last = next(
                (
                    i
                    for i in range(first + 1, len(lines))
                    if lines[i].strip()
                    and not lines[i][0].isspace()
                    and not lines[i].startswith("#")
                ),
                len(lines),
            )
            lines[first:last] = [entry]
    (target / "requirements.lock").write_text("".join(lines))
    base = "maintenance-task-dependencies:" + previous["image_id"].split(":")[1][:16]
    subprocess.run(["docker", "tag", previous["image_id"], base], check=True)
    pythonpath = ":".join(
        "/workspace/" + p
        for p in dict.fromkeys(
            [task.package, "libs/core", "libs/standard-tests", "libs/partners/openai"]
        )
    )
    env = f"""ENV PATH=/opt/venv/bin:/usr/local/bin:/usr/bin:/bin HOME=/tmp PYTHONDONTWRITEBYTECODE=1 OPENAI_API_KEY=sk-offline-placeholder TIKTOKEN_CACHE_DIR=/opt/tiktoken
ENV PYTHONPATH={pythonpath}
ENV PYTEST_ADDOPTS="-p no:cacheprovider --no-cov"
"""
    dockerfile = f"""FROM {previous["uv_image"]} AS uv
FROM {base} AS metadata
USER root
COPY --from=uv /uv /usr/local/bin/uv
COPY wheels /wheels
RUN UV_CACHE_DIR=/tmp/uv-cache uv pip install --offline --no-deps --python /opt/venv/bin/python /wheels/*.whl
COPY source /workspace
RUN UV_CACHE_DIR=/tmp/uv-cache uv pip install --offline --no-deps --no-build-isolation --python /opt/venv/bin/python --reinstall-package langchain-core --reinstall-package langchain-openai --reinstall-package langchain-tests -e /workspace/libs/core -e /workspace/libs/partners/openai -e /workspace/libs/standard-tests
RUN /opt/venv/bin/python -c 'import importlib.metadata as m,json,sys; print(json.dumps({{"python":sys.version,"packages":sorted((d.metadata["Name"],d.version) for d in m.distributions())}},indent=2))' > /opt/environment.json
FROM {base}
USER root
RUN rm -rf /opt/venv
COPY --from=metadata /opt/venv /opt/venv
COPY --from=metadata /opt/environment.json /opt/environment.json
{env}
USER 65532:65532
WORKDIR /workspace/{task.package}
"""
    (context / "Dockerfile").write_text(dockerfile)
    if environment.get("python_image"):
        # A separate historical task runtime, not a change to host/controller Python.
        dockerfile = f"""FROM {previous["uv_image"]} AS uv
FROM {environment["python_image"]} AS metadata
COPY --from=uv /uv /usr/local/bin/uv
COPY wheels /wheels
COPY requirements.lock /opt/requirements.lock
COPY build-constraints.txt /opt/build-constraints.txt
COPY source /workspace
RUN uv venv --python /usr/local/bin/python /opt/venv && UV_LINK_MODE=copy uv pip sync --python /opt/venv/bin/python --find-links /wheels --build-constraint /opt/build-constraints.txt /opt/requirements.lock
RUN /opt/venv/bin/python -c 'import importlib.metadata as m,json,sys; print(json.dumps({{"python":sys.version,"packages":sorted((d.metadata["Name"],d.version) for d in m.distributions())}},indent=2))' > /opt/environment.json
FROM {environment["python_image"]}
COPY --from=metadata /opt/venv /opt/venv
COPY --from=metadata /opt/environment.json /opt/environment.json
COPY --from={base} /opt/tiktoken /opt/tiktoken
{env}
USER 65532:65532
WORKDIR /workspace/{task.package}
"""
        (context / "Dockerfile").write_text(dockerfile)
        shutil.copyfile(target / "requirements.lock", context / "requirements.lock")
        (context / "build-constraints.txt").write_text(
            "pdm-backend==2.4.4\nsetuptools==80.9.0\nwheel==0.45.1\n"
        )
        image_tag = "maintenance-task-dependencies:" + task.task_id.lower()
        built = subprocess.run(
            [
                "docker",
                "build",
                "--network=default",
                "--pull=false",
                "-t",
                image_tag,
                str(context),
            ],
            capture_output=True,
            text=True,
            timeout=300,
        )
        (context / "build.log").write_text(built.stdout + built.stderr)
        if built.returncode:
            raise RuntimeError(
                "Task runtime build failed; see " + str(context / "build.log")
            )
        image = subprocess.check_output(
            ["docker", "image", "inspect", image_tag, "--format", "{{.Id}}"], text=True
        ).strip()
    else:
        image = build(context, "maintenance-task-dependencies:" + task.task_id.lower())
    report = {
        "task_id": task.task_id,
        "task": task.model_dump(),
        "commit": task.commit,
        "baseline_checksum": provenance["baseline_checksum"],
        "archive_sha256": provenance["archive_sha256"],
        "image_id": image,
        "dependency_image_reused_from": previous["image_id"],
        "dependency_policy": "Retain unaffected pins; align required Core test/runtime packages with baseline lock; immutable checked artifacts and isolated build tooling.",
        "dependency_artifacts": artifacts,
        "sandbox_config": json.loads(
            (configuration_path(ROOT, "configs/sandbox-mini-v5.json")).read_text()
        ),
        "setup_status": "in_progress",
        "setup_model_calls": 0,
        "setup_tokens": 0,
        "setup_directory": str(target),
        "checks": {},
    }
    report["sandbox_config"]["writable"] = list(task.source_paths + task.test_paths)
    try:

        def run(name, command):
            args = [
                "docker",
                "run",
                "--rm",
                "--network=none",
                "--read-only",
                "--cap-drop=ALL",
                "--security-opt=no-new-privileges",
                "--user=65532:65532",
                "--tmpfs=/tmp:rw,nosuid,nodev,noexec,size=512m,mode=1777",
                "-v",
                str(target / "baseline") + ":/workspace:ro",
                "-v",
                str(target / "reproduce.py") + ":/task/reproduce.py:ro",
                image,
                "bash",
                "-o",
                "pipefail",
                "-c",
                command,
            ]
            value = subprocess.run(args, capture_output=True, text=True, timeout=150)
            report["checks"][name] = {
                "exit_code": value.returncode,
                "output": value.stdout + value.stderr,
                "command": command,
            }
            (target / (name + ".log")).write_text(value.stdout + value.stderr)
            return value

        imports = (
            "import importlib,importlib.metadata as m,json,sys; assert sys.version_info[:2]=="
            + repr(tuple(environment.get("python_series", [3, 14])))
            + "; "
        )
        imports += "; ".join(
            f"assert m.version({n!r})=={v!r}, (m.version({n!r}),{v!r})"
            for n, v in task.runtime_versions.items()
        )
        imports += (
            "; sources={n:importlib.import_module(n).__file__ for n in "
            + repr(task.workspace_imports)
            + "}; assert all(p.startswith('/workspace/') for p in sources.values()); print(json.dumps(sources))"
        )
        assert (
            run("workspace_imports", "python -c " + shlex.quote(imports)).returncode
            == 0
        ), "Workspace/environment validation failed"
        reproduced = run("reproduction", task.reproduction_command)
        assert (
            reproduced.returncode == 1
            and task.expected_failure in reproduced.stdout + reproduced.stderr
        ), "Expected issue not reproduced"
        for check in task.development_checks:
            assert run(check.name, check.command).returncode == 0, (
                "Existing check failed: " + check.name
            )
        snapshot = subprocess.check_output(
            [
                "docker",
                "run",
                "--rm",
                "--network=none",
                image,
                "cat",
                "/opt/environment.json",
            ]
        )
        (target / "environment.json").write_bytes(snapshot)
        report["environment_checksum"] = hashlib.sha256(snapshot).hexdigest()
        report["dependency_manifest_checksum"] = hashlib.sha256(
            (target / "requirements.lock").read_bytes()
        ).hexdigest()
        spec = json.loads(fixture_spec.read_text())
        files = {
            p: subprocess.check_output(
                ["git", "-C", str(reference), "show", reference_commit + ":" + p],
                text=True,
            )
            for p in spec["production_paths"]
        }
        files.update(spec["new_tests"])
        (target / "reference-fixture.json").write_text(json.dumps(files))
        (target / "trusted-reference-provenance.json").write_text(
            json.dumps(
                {
                    "commit": reference_commit,
                    "fixture_sha256": hashlib.sha256(
                        (target / "reference-fixture.json").read_bytes()
                    ).hexdigest(),
                },
                indent=2,
            )
        )
        report["setup_status"] = "validated"
    except Exception as exc:
        report["setup_status"] = "blocked_by_environment"
        report["blocker"] = type(exc).__name__ + ": " + str(exc)
        raise
    finally:
        report["setup_elapsed_seconds"] = time.monotonic() - started
        record_file.write_text(json.dumps(report, indent=2))
    return target


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for name in ["reference", "task", "reuse", "state", "fixture-spec"]:
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--reference-commit", required=True)
    args = p.parse_args()
    print(
        setup(
            args.reference.resolve(),
            args.task.resolve(),
            args.reuse.resolve(),
            args.state.resolve(),
            args.reference_commit,
            args.fixture_spec.resolve(),
        )
    )
