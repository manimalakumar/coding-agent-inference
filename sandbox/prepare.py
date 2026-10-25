"""Trusted setup only: Git export, frozen dependencies, baseline and Docker gates."""

import argparse
import hashlib
import io
import json
import shutil
import subprocess
import tarfile
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from agent.schemas import SandboxConfig, Task
from experiments.configuration_paths import configuration_path
from sandbox.docker import InfrastructureError, Sandbox, digest, docker_ready, manifest

PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_STATE = PROJECT / ".local/resources"


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def export_snapshot(reference: Path, task: Task, destination: Path) -> dict:
    """Git history is used only here; the export never includes .git."""
    archive = subprocess.run(
        ["git", "-C", str(reference), "archive", "--format=tar", task.commit],
        check=True,
        capture_output=True,
    ).stdout
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        for member in tar.getmembers():
            if not (member.isfile() or member.isdir()):
                raise InfrastructureError("Source archive contains links/special files")
            path = Path(member.name)
            if path.is_absolute() or ".." in path.parts or ".git" in path.parts:
                raise InfrastructureError("Unsafe archive path")
        tar.extractall(destination, filter="data")
    files = manifest(destination)
    for path in destination.rglob("*"):
        path.chmod(0o555 if path.is_dir() else 0o444)
    destination.chmod(0o555)
    return {
        "commit": task.commit,
        "archive_sha256": hashlib.sha256(archive).hexdigest(),
        "baseline_checksum": digest(files),
        "files": files,
    }


def logged(command: list[str], cwd: Path, log: Path, timeout: int = 1800) -> None:
    with log.open("w") as output:
        result = subprocess.run(
            command, cwd=cwd, stdout=output, stderr=subprocess.STDOUT, timeout=timeout
        )
    if result.returncode:
        raise InfrastructureError(f"Setup command failed; see {log}")


def image_digest(image: str) -> str:
    subprocess.run(
        ["docker", "pull", image], check=True, capture_output=True, timeout=300
    )
    value = subprocess.run(
        ["docker", "image", "inspect", image, "--format", "{{index .RepoDigests 0}}"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if "@sha256:" not in value:
        raise InfrastructureError("Base image digest could not be resolved")
    return value


def prepare(
    reference: Path,
    task_file: Path,
    config_file: Path,
    state: Path,
    reuse_environment_from: Path | None = None,
) -> dict:
    started = time.monotonic()
    task = Task.model_validate_json(task_file.read_text())
    config = SandboxConfig.model_validate_json(config_file.read_text())
    task_assets = (
        {
            task.reproduction_script: digest(
                (task_file.parent / task.reproduction_script).read_text()
            )
        }
        if task.reproduction_script
        else {}
    )
    reused = None
    if reuse_environment_from is not None:
        reused = json.loads((reuse_environment_from / "setup.json").read_text())
        if reused.get("setup_status") != "validated":
            raise InfrastructureError(
                "Only a validated frozen environment can be reused"
            )
    fingerprint = digest(
        {
            "task": task.model_dump(),
            "sandbox": config.model_dump(),
            "task_assets": task_assets,
            "reused_image_id": reused["image_id"] if reused else None,
            "recipe": (PROJECT / "sandbox/Dockerfile").read_text(),
            "requirements": (task_file.parent / "requirements.in").read_text(),
        }
    )
    root = state / task.task_id / fingerprint[:16]
    root.mkdir(parents=True, exist_ok=True)
    existing = root / "setup.json"
    if existing.exists():
        saved = json.loads(existing.read_text())
        if saved.get("setup_status") == "validated":
            if digest(manifest(root / "baseline")) != saved["baseline_checksum"]:
                raise InfrastructureError("Immutable baseline checksum mismatch")
            for name, expected in saved.get("task_assets", {}).items():
                if digest((root / name).read_text()) != expected:
                    raise InfrastructureError("Read-only task asset checksum mismatch")
            return {"setup_directory": str(root), **saved}
    report = {
        "task_id": task.task_id,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "setup_fingerprint": fingerprint,
        "setup_status": "in_progress",
        "sandbox_config": config.model_dump(),
        "task": task.model_dump(),
        "setup_model_calls": 0,
        "setup_tokens": 0,
        "task_assets": task_assets,
    }
    try:
        provenance = root / "provenance.json"
        if not provenance.exists():
            if (root / "baseline").exists():
                raise InfrastructureError(
                    "Incomplete export; choose a fresh state directory"
                )
            write_json(provenance, export_snapshot(reference, task, root / "baseline"))
        provenance_data = json.loads(provenance.read_text())
        if digest(manifest(root / "baseline")) != provenance_data["baseline_checksum"]:
            raise InfrastructureError("Immutable baseline checksum mismatch")
        report.update({k: v for k, v in provenance_data.items() if k != "files"})
        shutil.copy2(task_file, root / "task.json")
        for name in task_assets:
            target = root / name
            if target.exists():
                if (
                    target.is_symlink()
                    or digest(target.read_text()) != task_assets[name]
                ):
                    raise InfrastructureError("Existing task asset was modified")
            else:
                shutil.copy2(task_file.parent / name, target)
                target.chmod(0o444)
        report["docker_version"] = docker_ready()
        if reused is not None:
            for field in ["image_id", "base_image", "uv_image"]:
                report[field] = reused[field]
            for name in ["environment.json", "requirements.lock"]:
                shutil.copy2(reuse_environment_from / name, root / name)
            report["environment_reused_from"] = str(reuse_environment_from.resolve())
        else:
            context = root / "build"
            context.mkdir(exist_ok=True)
            # No controller, evaluator, credentials, run logs, or reference clone in context.
            if not (context / "source").exists():
                shutil.copytree(root / "baseline", context / "source")
            shutil.copy2(PROJECT / "sandbox/Dockerfile", context / "Dockerfile")
            shutil.copy2(
                task_file.parent / "requirements.in", context / "requirements.in"
            )
            report["base_image"] = image_digest(config.base_image)
            report["uv_image"] = image_digest("ghcr.io/astral-sh/uv:0.11.29")
            tag = "maintenance-economics:" + fingerprint[:16]
            logged(
                [
                    "docker",
                    "build",
                    "--build-arg",
                    "BASE=" + report["base_image"],
                    "--build-arg",
                    "UV_IMAGE=" + report["uv_image"],
                    "-t",
                    tag,
                    ".",
                ],
                context,
                root / "build.log",
            )
            report["image_id"] = subprocess.run(
                ["docker", "image", "inspect", tag, "--format", "{{.Id}}"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            for name in ["environment.json", "requirements.lock"]:
                content = subprocess.run(
                    [
                        "docker",
                        "run",
                        "--rm",
                        "--network=none",
                        "--read-only",
                        "--cap-drop=ALL",
                        "--security-opt=no-new-privileges",
                        "--entrypoint=/bin/cat",
                        report["image_id"],
                        "/opt/" + name,
                    ],
                    check=True,
                    capture_output=True,
                    timeout=30,
                ).stdout
                (root / name).write_bytes(content)
        report["environment_checksum"] = digest((root / "environment.json").read_text())
        report["dependency_manifest_checksum"] = digest(
            (root / "requirements.lock").read_text()
        )
        box = Sandbox(
            config,
            task,
            root / "baseline",
            root / ("preflight-workspace-" + uuid.uuid4().hex[:8]),
            root / "task.json",
            report["image_id"],
            state / "resources",
        )
        if box.workspace.exists():
            shutil.rmtree(box.workspace)
        box.prepare()
        report["isolation"] = box.isolation_checks()
        report["environment"] = box.execute(
            "cat /opt/environment.json; cat /opt/requirements.lock", 30
        )
        import_script = (
            "import importlib, importlib.metadata as m, json, sys\n"
            f"assert '.'.join(map(str,sys.version_info[:3])) == {config.python!r}\n"
            f"for name,version in {task.runtime_versions!r}.items():\n"
            " assert m.version(name)==version, (name,m.version(name))\n"
            "sources={}\n"
            f"for name in {task.workspace_imports!r}:\n"
            " module=importlib.import_module(name)\n"
            " assert module.__file__.startswith('/workspace/'), module.__file__\n"
            " sources[name]=module.__file__\n"
            "print(json.dumps(sources))\n"
        )
        imports = box.execute("python - <<'PY'\n" + import_script + "PY", 30)
        report["workspace_import_check"] = imports
        if imports["exit_code"]:
            raise InfrastructureError("Workspace import/environment check failed")
        if not task.reproduction_command or not task.expected_failure:
            raise InfrastructureError(
                "Task needs an explicit reproduction and expected failure"
            )
        reproduction = box.execute(task.reproduction_command, 30)
        report["reproduction"] = reproduction
        if (
            reproduction["exit_code"] == 0
            or task.expected_failure not in reproduction["output"]
        ):
            raise InfrastructureError("Reported timeout failure was not confirmed")
        report["baseline_development_checks"] = {}
        for check in task.development_checks:
            baseline_tests = box.execute(check.command, 120)
            report["baseline_development_checks"][check.name] = baseline_tests
            if baseline_tests["exit_code"]:
                raise InfrastructureError(
                    "Existing development check fails before agent: " + check.name
                )
        report["resources"] = box.resource_record()
        report["setup_status"] = "validated"
    except (InfrastructureError, subprocess.SubprocessError, OSError) as exc:
        report["setup_status"] = "blocked_by_environment"
        report["blocker"] = str(exc)
    finally:
        report["setup_elapsed_seconds"] = time.monotonic() - started
        write_json(root / "setup-attempts" / (uuid.uuid4().hex + ".json"), report)
        write_json(root / "setup.json", report)
    return {"setup_directory": str(root), **report}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument(
        "--task", type=Path, default=PROJECT / "tasks/LC-TIMEOUT-001/task.json"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=configuration_path(PROJECT, "configs/sandbox-v1.json"),
    )
    parser.add_argument("--state", type=Path, default=DEFAULT_STATE)
    parser.add_argument("--reuse-environment-from", type=Path)
    args = parser.parse_args()
    result = prepare(
        args.reference, args.task, args.config, args.state, args.reuse_environment_from
    )
    print(
        json.dumps(
            {k: result[k] for k in ["setup_directory", "setup_status"]}
            | {"blocker": result.get("blocker")},
            indent=2,
        )
    )
    raise SystemExit(0 if result["setup_status"] == "validated" else 2)


if __name__ == "__main__":
    main()
