"""Trusted task preparation: immutable Git export, offline editable metadata and gates.

Never mounted into evaluated containers. Reference code is exported only to a private
fixture used for local positive controls, never to the agent image or instructions.
"""

import argparse
import hashlib
import json
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


def setup(reference, task_file, reuse, state, reference_commit):
    started = time.monotonic()
    task = Task.model_validate_json(task_file.read_text())
    previous = json.loads((reuse / "setup.json").read_text())
    assert previous["setup_status"] == "validated"
    key = digest(
        {
            "task": task.model_dump(),
            "reproducer": (task_file.parent / task.reproduction_script).read_text(),
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
        provenance = export_snapshot(reference, task, target / "baseline")
        provenance_file.write_text(json.dumps(provenance, indent=2))
    provenance = json.loads(provenance_file.read_text())
    assert digest(manifest(target / "baseline")) == provenance["baseline_checksum"]
    assert not list((target / "baseline").rglob(".git"))
    shutil.copyfile(
        task_file.parent / task.reproduction_script, target / "reproduce.py"
    )
    (target / "reproduce.py").chmod(0o444)
    shutil.copyfile(reuse / "requirements.lock", target / "requirements.lock")
    context = target / "dependency-build"
    context.mkdir(exist_ok=True)
    if not (context / "source").exists():
        shutil.copytree(target / "baseline", context / "source")
    # Download exactly the SDK wheel pinned in this source snapshot during trusted setup.
    lock = tomllib.loads((target / "baseline/libs/partners/openai/uv.lock").read_text())
    sdk = next(p for p in lock["package"] if p["name"] == "openai")
    assert sdk["version"] == task.runtime_versions["openai"]
    wheel = next(w for w in sdk["wheels"] if w["url"].endswith("py3-none-any.whl"))
    wheels = context / "wheels"
    wheels.mkdir(exist_ok=True)
    wheel_path = wheels / wheel["url"].rsplit("/", 1)[1]
    if not wheel_path.exists():
        wheel_path.write_bytes(urllib.request.urlopen(wheel["url"], timeout=60).read())
    assert (
        hashlib.sha256(wheel_path.read_bytes()).hexdigest()
        == wheel["hash"].split(":")[1]
    )
    lines = (target / "requirements.lock").read_text().splitlines(True)
    first = next(i for i, line in enumerate(lines) if line.startswith("openai=="))
    last = next(
        i
        for i in range(first + 1, len(lines))
        if lines[i] and not lines[i][0].isspace() and not lines[i].startswith("#")
    )
    lines[first:last] = [
        "openai==" + sdk["version"] + " --hash=" + wheel["hash"] + "\n"
    ]
    (target / "requirements.lock").write_text("".join(lines))
    base = "maintenance-task-dependencies:" + previous["image_id"].split(":")[1][:16]
    subprocess.run(["docker", "tag", previous["image_id"], base], check=True)
    env = """ENV PATH=/opt/venv/bin:/usr/local/bin:/usr/bin:/bin HOME=/tmp PYTHONDONTWRITEBYTECODE=1 OPENAI_API_KEY=sk-offline-placeholder TIKTOKEN_CACHE_DIR=/opt/tiktoken
ENV PYTHONPATH=/workspace/libs/partners/openai:/workspace/libs/core:/workspace/libs/standard-tests
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
WORKDIR /workspace/libs/partners/openai
"""
    (context / "Dockerfile").write_text(dockerfile)
    image = build(context, "maintenance-task-dependencies:" + task.task_id.lower())
    report = {
        "task_id": task.task_id,
        "task": task.model_dump(),
        "commit": task.commit,
        "baseline_checksum": provenance["baseline_checksum"],
        "archive_sha256": provenance["archive_sha256"],
        "image_id": image,
        "dependency_image_reused_from": previous["image_id"],
        "dependency_policy": "Existing runtime/check pins retained except OpenAI SDK aligned to this baseline uv.lock (1.92.0); wheel hash verified. Workspace editable metadata rebuilt offline.",
        "sandbox_config": json.loads(
            (configuration_path(ROOT, "configs/sandbox-mini-v5.json")).read_text()
        ),
        "setup_status": "in_progress",
        "setup_model_calls": 0,
        "setup_tokens": 0,
        "setup_directory": str(target),
        "checks": {},
        "sdk_wheel_provenance": wheel,
    }
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

        imports = "import importlib,importlib.metadata as m,json,sys; assert sys.version_info[:3]==(3,14,6); "
        imports += "; ".join(
            f"assert m.version({name!r})=={version!r}"
            for name, version in task.runtime_versions.items()
        )
        imports += (
            "; sources={name:importlib.import_module(name).__file__ for name in "
            + repr(task.workspace_imports)
            + "}; assert all(p.startswith('/workspace/') for p in sources.values()); print(json.dumps(sources))"
        )
        checked = run(
            "workspace_imports", "python -c " + __import__("shlex").quote(imports)
        )
        assert checked.returncode == 0, "Workspace import/environment validation failed"
        reproduced = run("reproduction", task.reproduction_command)
        assert (
            reproduced.returncode == 1
            and task.expected_failure in reproduced.stdout + reproduced.stderr
        ), "Expected issue was not reproduced"
        for check in task.development_checks:
            checked = run(check.name, check.command)
            assert checked.returncode == 0, "Existing check failed: " + check.name
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
        fixture = {
            "libs/partners/openai/langchain_openai/chat_models/base.py": subprocess.check_output(
                [
                    "git",
                    "-C",
                    str(reference),
                    "show",
                    reference_commit
                    + ":libs/partners/openai/langchain_openai/chat_models/base.py",
                ],
                text=True,
            ),
            "libs/partners/openai/tests/unit_tests/test_usage_fixture.py": """from langchain_openai.chat_models.base import _create_usage_metadata_responses


def test_usage_details_fixture() -> None:
    usage = {"input_tokens": 91, "input_tokens_details": {"cached_tokens": 14}, "output_tokens": 37, "output_tokens_details": {"reasoning_tokens": 6}, "total_tokens": 128}
    result = _create_usage_metadata_responses(usage)
    assert result["input_token_details"]["cache_read"] == 14
    assert result["output_token_details"]["reasoning"] == 6
""",
        }
        # Format the trusted new regression file with the task's exact formatter version.
        fixture_test = target / "fixture-test.py"
        fixture_test.write_text(
            fixture["libs/partners/openai/tests/unit_tests/test_usage_fixture.py"]
        )
        subprocess.run(
            [
                "docker",
                "run",
                "--rm",
                "--network=none",
                "--user=0:0",
                "-v",
                str(fixture_test) + ":/tmp/test_usage_fixture.py",
                image,
                "ruff",
                "format",
                "--no-cache",
                "--isolated",
                "--target-version",
                "py39",
                "/tmp/test_usage_fixture.py",
            ],
            check=True,
            capture_output=True,
        )
        fixture["libs/partners/openai/tests/unit_tests/test_usage_fixture.py"] = (
            fixture_test.read_text()
        )
        (target / "reference-fixture.json").write_text(json.dumps(fixture))
        (target / "trusted-reference-provenance.json").write_text(
            json.dumps(
                {
                    "commit": reference_commit,
                    "production_path": "libs/partners/openai/langchain_openai/chat_models/base.py",
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
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--task", type=Path, required=True)
    p.add_argument("--reuse", type=Path, required=True)
    p.add_argument("--state", type=Path, required=True)
    p.add_argument("--reference-commit", required=True)
    args = p.parse_args()
    print(
        setup(
            args.reference.resolve(),
            args.task.resolve(),
            args.reuse.resolve(),
            args.state.resolve(),
            args.reference_commit,
        )
    )
