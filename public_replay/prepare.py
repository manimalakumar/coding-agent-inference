"""Trusted, zero-inference preparation; not imported by Replay or measured agents."""

import asyncio
import hashlib
import hmac
import io
import json
import os
import secrets
import shutil
import stat
import subprocess
import tarfile
import time
from pathlib import Path
from types import SimpleNamespace

from public_replay.bundle import Bundle, canonical, digest, safe_path


def command(args, *, cwd=None, timeout=1800, log=None):
    """Never use a shell for host commands; no provider credential propagation."""
    env = {
        k: v
        for k, v in os.environ.items()
        if not any(
            word in k.upper()
            for word in ["KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL"]
        )
    }
    result = subprocess.run(
        args, cwd=cwd, env=env, capture_output=True, timeout=timeout
    )
    if log:
        Path(log).write_bytes(result.stdout + result.stderr)
    if result.returncode:
        raise RuntimeError(
            f"Setup command failed ({result.returncode}): {args[0]}; inspect trusted setup log"
        )
    return result.stdout


def tree_identity(root):
    files = {}
    for p in sorted(root.rglob("*")):
        if p.is_symlink() or ".git" in p.relative_to(root).parts:
            raise ValueError("Snapshot links/history prohibited")
        if p.is_file():
            files[p.relative_to(root).as_posix()] = digest(p.read_bytes())
    return digest(json.dumps(files, sort_keys=True).encode())


def export_commit(cache, commit, destination):
    destination.mkdir(parents=True, exist_ok=False)
    archive = command(["git", "--git-dir", str(cache), "archive", commit])
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        for member in tar.getmembers():
            safe_path(destination, member.name)
            if (
                not (member.isfile() or member.isdir())
                or ".git" in Path(member.name).parts
            ):
                raise ValueError("Unsupported snapshot member")
        tar.extractall(destination, filter="data")
    return {
        "archive_sha256": digest(archive),
        "baseline_checksum": tree_identity(destination),
    }


def doctor(bundle: Bundle, resource_root: Path | None = None, check_docker=False):
    report = {
        "public_replay": "ready",
        "model_requests": 0,
        "task_readiness": bundle.read("tasks.json"),
        "git": bool(shutil.which("git")),
        "docker": "not checked",
        "private_verifier": "unavailable: private fixtures intentionally not distributed",
        "paid_execution": "not exposed by this public CLI",
    }
    if check_docker:
        docker = shutil.which("docker")
        try:
            report["docker"] = (
                command(
                    [
                        docker or "docker",
                        "info",
                        "--format",
                        "{{.OSType}}/{{.Architecture}}",
                    ],
                    timeout=30,
                )
                .decode()
                .strip()
            )
        except RuntimeError, OSError, subprocess.TimeoutExpired:
            report["docker"] = (
                "unavailable; install/start Docker and put its credential helper on PATH"
            )
    if resource_root:
        report["resolved_preparation"] = (
            json.loads((resource_root / "resolved.json").read_text())
            if (resource_root / "resolved.json").is_file()
            else "not prepared; run trusted prepare"
        )
    if resource_root and isinstance(report.get("resolved_preparation"), dict):
        resolved = report["resolved_preparation"]
        recipe = (
            bundle.read("tasks.json")
            .get(resolved.get("task"), {})
            .get("portable_recipe", {})
        )
        required = [
            "upstream.git",
            "baseline",
            "reference",
            "keys/receipt.key",
            "build/expected-requirements.lock",
        ]
        report["missing_resources"] = [
            name for name in required if not (resource_root / name).exists()
        ]
        key = resource_root / "keys/receipt.key"
        report["key_permissions_private"] = (
            key.is_file() and stat.S_IMODE(key.stat().st_mode) == 0o600
        )
        lock = resource_root / "build/expected-requirements.lock"
        report["dependency_manifest_matches"] = (
            lock.is_file() and lock.read_text() == recipe.get("requirements_lock")
        )
        report["exact_commits_present"] = {}
        for field in ["starting_commit", "reference_commit"]:
            commit = recipe.get(field)
            if commit:
                try:
                    command(
                        [
                            "git",
                            "--git-dir",
                            str(resource_root / "upstream.git"),
                            "cat-file",
                            "-e",
                            commit + "^{commit}",
                        ],
                        timeout=15,
                    )
                    report["exact_commits_present"][field] = True
                except OSError, RuntimeError, subprocess.TimeoutExpired:
                    report["exact_commits_present"][field] = False
        if check_docker:
            try:
                actual = json.loads(
                    command(
                        [
                            shutil.which("docker") or "docker",
                            "image",
                            "inspect",
                            resolved["image"],
                        ],
                        timeout=30,
                    )
                )[0]["Id"]
                report["image_identity_matches"] = actual == resolved["image_id"]
            except OSError, RuntimeError, subprocess.TimeoutExpired:
                report["image_identity_matches"] = False
        report["historical_execution_readiness"] = (
            "unavailable: original private verifier not reconstructed; public controls only"
        )
    return report


class ControlBudget:
    """No model calls; same tool timeout, output and file-size limits as frozen runs."""

    submission_only = False

    def __init__(self, seconds=1200):
        self.deadline = time.monotonic() + seconds

    def remaining_seconds(self):
        return max(0, self.deadline - time.monotonic())

    def check_time(self):
        if not self.remaining_seconds():
            raise TimeoutError("Control budget exhausted")


class DockerControlEnvironment:
    def __init__(self, docker, container):
        self.docker, self.container = docker, container

    async def exec(self, *, command: str, timeout_sec):
        proc = await asyncio.create_subprocess_exec(
            self.docker,
            "exec",
            self.container,
            "bash",
            "-c",
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            out, err = await asyncio.wait_for(proc.communicate(), timeout_sec)
        except BaseException:
            proc.kill()
            await proc.wait()
            raise
        return SimpleNamespace(
            return_code=proc.returncode,
            stdout=out.decode(errors="replace"),
            stderr=err.decode(errors="replace"),
        )


class ControlTrace:
    def __init__(self):
        self.records = []

    def emit(self, event, **kwargs):
        self.records.append({"event": event, **kwargs})


async def wrapper_checks(docker, container, commands, expected_failure):
    from public_replay._historical_tools import HarborTools

    trace = ControlTrace()
    tools = HarborTools(
        DockerControlEnvironment(docker, container),
        asyncio.get_running_loop(),
        ControlBudget(),
        trace,
        expected_failure,
    )
    results = []
    for cmd, expected in commands:
        # Public reference checks bypass ONLY the baseline-first state once the
        # independent baseline control has already established the failure.
        result = await asyncio.to_thread(tools.execute, {"command": cmd})
        results.append(
            {
                "command": cmd,
                "expected_exit": expected,
                "observed_exit": result["returncode"],
                "passed": result["returncode"] == expected,
                "output": result["output"],
            }
        )
    return results


def file_policy(baseline, candidate, recipe):
    changes = []
    existing = {
        p.relative_to(baseline).as_posix(): digest(p.read_bytes())
        for p in baseline.rglob("*")
        if p.is_file()
    }
    current = {
        p.relative_to(candidate).as_posix(): digest(p.read_bytes())
        for p in candidate.rglob("*")
        if p.is_file()
    }
    for name in sorted(set(existing) | set(current)):
        if existing.get(name) == current.get(name):
            continue
        allowed = name in recipe["source_files"] and name in current
        path = Path(name)
        allowed = allowed or (
            name not in existing
            and path.parent.as_posix() == "libs/core/tests/unit_tests"
            and path.name.startswith("test_")
            and path.suffix == ".py"
        )
        changes.append({"path": name, "allowed": allowed})
    return {"passed": all(c["allowed"] for c in changes), "changes": changes}


def controls(docker, image, root, baseline, reference, recipe):
    """Public reconstruction controls are NOT the historical hidden evaluator."""
    os.environ["MSWEA_GLOBAL_CONFIG_DIR"] = str(root / "mini-config")
    os.environ["MSWEA_SILENT_STARTUP"] = "1"
    from public_replay._historical_tools import HarborTools

    receipts = []
    source_rel = recipe["source_files"]
    for scenario in ["baseline", "reference", "reference_budget", "reference_error"]:
        workspace = root / "controls" / scenario / "workspace"
        shutil.copytree(baseline, workspace)
        for p in workspace.rglob("*"):
            p.chmod(0o555 if p.is_dir() else 0o444)
        workspace.chmod(0o555)
        for name in source_rel:
            (workspace / name).chmod(0o666)
        test_root = workspace / "libs/core/tests/unit_tests"
        test_root.chmod(0o1777)
        task = root / "controls" / scenario / "task"
        task.mkdir()
        (task / "reproduce.py").write_text(recipe["reproducer"])
        (task / "reproduce.py").chmod(0o444)

        def start_container(workspace):
            return (
                command(
                    [
                        docker,
                        "run",
                        "-d",
                        "--network=none",
                        "--read-only",
                        "--user",
                        "65532:65532",
                        "--cap-drop=ALL",
                        "--security-opt=no-new-privileges",
                        "--cpus=2",
                        "--memory=2g",
                        "--pids-limit=128",
                        "--tmpfs",
                        "/tmp:rw,nosuid,nodev,size=256m",
                        "--mount",
                        f"type=bind,src={workspace},dst=/workspace",
                        "--mount",
                        f"type=bind,src={task},dst=/task,readonly",
                        "-w",
                        "/workspace/libs/core",
                        image,
                        "sleep",
                        "3600",
                    ]
                )
                .decode()
                .strip()
            )

        container = start_container(workspace)
        try:
            first = asyncio.run(
                wrapper_checks(
                    docker,
                    container,
                    [("python /task/reproduce.py", 1)],
                    recipe["expected_failure"],
                )
            )
            if (
                not first[0]["passed"]
                or recipe["expected_failure"] not in first[0]["output"]
            ):
                raise RuntimeError(
                    "Prepared baseline does not reproduce; infrastructure control failed"
                )
            if scenario != "baseline":
                # Trusted host writes reference; no history/reference mount enters container.
                for name in source_rel:
                    (workspace / name).write_bytes((reference / name).read_bytes())
            isolated = """python - <<'PY'
import os,socket,subprocess
from pathlib import Path
assert not Path('/workspace/.git').exists()
assert not Path('/host').exists() and not Path('/reference').exists()
assert not Path('/run/secrets').exists() and not Path('/evaluator').exists()
assert not any('KEY' in k or 'TOKEN' in k or 'SECRET' in k for k in os.environ)
p=Path('/workspace/libs/core/tests/unit_tests/test_public_location.py')
p.write_text('def test_public_location() -> None:\\n    assert True\\n')
try: Path('/workspace/libs/core/tests/unit_tests/language_models/llms/test_base.py').write_text('bad')
except PermissionError: pass
else: raise AssertionError('Existing test was writable')
link=Path('/tmp/host-link');link.symlink_to('/Users')
assert not link.exists()
with socket.socket() as s:
    s.settimeout(1)
    assert s.connect_ex(('1.1.1.1',443)) != 0
print('Read/write/test-root/history/credential/symlink/network controls passed')
PY"""
            checks = [
                (isolated, 0),
                ("python /task/reproduce.py", 1 if scenario == "baseline" else 0),
                *[(c, 0) for c in recipe["required_commands"]],
            ]

            # Baseline-first already checked above; avoid repeating first-tool gate.
            async def more():
                trace = ControlTrace()
                tools = HarborTools(
                    DockerControlEnvironment(docker, container),
                    asyncio.get_running_loop(),
                    ControlBudget(),
                    trace,
                    recipe["expected_failure"],
                )
                tools.reproduced = True
                result = []
                for cmd, expected in checks:
                    r = await asyncio.to_thread(tools.execute, {"command": cmd})
                    result.append(
                        {
                            "command": cmd,
                            "expected_exit": expected,
                            "observed_exit": r["returncode"],
                            "passed": r["returncode"] == expected,
                            "output": r["output"],
                        }
                    )
                return result

            all_checks = checks
            checks = all_checks[:1]
            isolation_result = asyncio.run(more())
            command([docker, "rm", "-f", container])
            container = None
            # Capture before trusted checks in a separate fresh environment.
            candidate = tree_identity(workspace)
            frozen = {name: (workspace / name).read_text() for name in source_rel}
            frozen["libs/core/tests/unit_tests/test_public_location.py"] = (
                workspace / "libs/core/tests/unit_tests/test_public_location.py"
            ).read_text()
            capture_path = workspace.parent / "frozen-candidate.json"
            capture_path.write_bytes(canonical(frozen))
            capture_path.chmod(0o444)
            evaluation = workspace.parent / "trusted-evaluation"
            shutil.copytree(workspace, evaluation)
            container = start_container(evaluation)
            checks = all_checks[1:]
            checks_result = isolation_result + asyncio.run(more())
            assert tree_identity(workspace) == candidate
            # Terminal events are synthetic controls, never measured attempts.

            termination = {
                "baseline": "submitted",
                "reference": "submitted",
                "reference_budget": "budget_exhausted",
                "reference_error": "error",
            }[scenario]
            policy = file_policy(baseline, workspace, recipe)
            receipt = {
                "file_policy": policy,
                "scenario": scenario,
                "termination_reason": termination,
                "termination_control": "simulated stop label; separate adapter tests cover controller finalization",
                "explicit_submission": termination == "submitted",
                "candidate_checksum": candidate,
                "captured_before_evaluation": True,
                "separate_clean_evaluation": True,
                "candidate_capture_sha256": digest(capture_path.read_bytes()),
                "public_candidate_accepted": scenario != "baseline"
                and policy["passed"]
                and all(c["passed"] for c in first + checks_result),
                "checks": first + checks_result,
                "control_only": True,
                "model_requests": 0,
                "historical_hidden_acceptance": "not reconstructed",
                "public_check_passed": policy["passed"]
                and all(c["passed"] for c in first + checks_result),
            }
            # Independently sign a local control receipt; key is outside all mounts.
            receipt["signature"] = hmac.new(
                (root / "keys/receipt.key").read_bytes(),
                canonical(receipt),
                hashlib.sha256,
            ).hexdigest()
            receipts.append(receipt)
        finally:
            if container:
                command([docker, "rm", "-f", container])
    return receipts


def prepare(
    bundle: Bundle,
    project_root: Path,
    resource_root: Path,
    upstream_url: str,
    task: str,
    run_controls=False,
):
    """One supported recipe; every other task remains explicitly unvalidated."""
    started = time.monotonic()
    task_data = bundle.read("tasks.json").get(task, {})
    recipe = task_data.get("portable_recipe")
    if not recipe:
        raise ValueError(
            "No independently reconstructable public recipe for this task; private verifier is not bundled"
        )
    project_root, root = project_root.resolve(), resource_root.resolve()
    if root.is_relative_to(project_root) or project_root.is_relative_to(root):
        raise ValueError("Host-only resources must be outside the public project tree")
    root.mkdir(parents=True, exist_ok=False)
    (root / "keys").mkdir(mode=0o700)
    (root / "keys/receipt.key").write_bytes(secrets.token_bytes(32))
    (root / "keys/receipt.key").chmod(0o600)
    cache = root / "upstream.git"
    command(["git", "init", "--bare", str(cache)], log=root / "git-init.log")
    for label in ["starting_commit", "reference_commit"]:
        commit = recipe[label]
        if len(commit) != 40 or any(c not in "0123456789abcdef" for c in commit):
            raise ValueError("Task commit must be an exact SHA")
        command(
            [
                "git",
                "--git-dir",
                str(cache),
                "fetch",
                "--depth=1",
                upstream_url,
                commit,
            ],
            log=root / (label + ".fetch.log"),
        )
    baseline = root / "baseline"
    identity = export_commit(cache, recipe["starting_commit"], baseline)
    if identity["baseline_checksum"] != recipe["baseline_checksum"]:
        raise ValueError("Baseline file identity differs from original provenance")
    reference = root / "reference"
    export_commit(cache, recipe["reference_commit"], reference)
    context = root / "build"
    shutil.copytree(baseline, context / "source")
    (context / "Dockerfile").write_text(recipe["dockerfile"])
    (context / "expected-requirements.lock").write_text(recipe["requirements_lock"])
    docker = shutil.which("docker")
    if not docker:
        raise RuntimeError("Docker not found; source prepared, execution unavailable")
    builder = "maintenance-public-" + secrets.token_hex(4)
    image = "maintenance-public-" + task.lower() + ":" + digest(canonical(recipe))[:12]
    command(
        [docker, "buildx", "create", "--name", builder, "--driver", "docker-container"],
        log=root / "builder.log",
    )
    try:
        command(
            [
                docker,
                "buildx",
                "build",
                "--builder",
                builder,
                "--platform",
                recipe["platform"],
                "--pull",
                "--no-cache",
                "--load",
                "-t",
                image,
                str(context),
            ],
            log=root / "build.log",
        )
    finally:
        command([docker, "buildx", "rm", builder])
    image_info = json.loads(command([docker, "image", "inspect", image]))[0]
    config = {
        "prepare_code_sha256": digest(Path(__file__).read_bytes()),
        "tool_wrapper_sha256": digest(
            Path(__file__).with_name("_historical_tools.py").read_bytes()
        ),
        "recipe_sha256": digest(canonical(recipe)),
        "schema_version": 1,
        "kind": "new portable setup control, not measured execution",
        "historical_semantic_protocol": recipe["historical_semantic_protocol"],
        "task": task,
        "source": identity,
        "resource_root": str(root),
        "project_root": str(project_root),
        "image": image,
        "image_id": image_info["Id"],
        "platform": recipe["platform"],
        "base_images": recipe["base_images"],
        "network": "none in task containers",
        "private_verifier": "unavailable",
        "receipt_key": str(root / "keys/receipt.key"),
        "model_requests": 0,
    }
    config["setting_id"] = digest(canonical(config))
    (root / "resolved.json").write_bytes(canonical(config))
    results = (
        controls(docker, image, root, baseline, reference, recipe)
        if run_controls
        else []
    )
    result = {
        "setting_id": config["setting_id"],
        "task": task,
        "build": "passed",
        "model_requests": 0,
        "elapsed_seconds": time.monotonic() - started,
        "controls": results,
        "historical_hidden_verifier": "unavailable; no full execution reproducibility claim",
        "other_tasks": "not independently reconstructed",
    }
    (root / "control-receipt.json").write_bytes(canonical(result))
    return result
