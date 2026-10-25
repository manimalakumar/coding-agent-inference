"""Trusted export of the validated snapshot into standard Harbor task images."""

import argparse
import hashlib
import json
import secrets
import shutil
import subprocess
from pathlib import Path

import toml
import yaml
from harbor.models.job.config import JobConfig
from harbor.models.task.config import TaskConfig

from experiments.configuration_paths import configuration_path

ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / "harbor_tasks/LC-TIMEOUT-001"
SOURCE = "libs/partners/openai/langchain_openai/chat_models"
TESTS = "libs/partners/openai/tests/unit_tests"


def build(context, tag):
    result = subprocess.run(
        ["docker", "build", "--network=none", "--pull=false", "-t", tag, str(context)],
        capture_output=True,
        text=True,
        timeout=300,
    )
    (context / "build.log").write_text(result.stdout + result.stderr)
    if result.returncode:
        raise RuntimeError(result.stderr[-2000:])
    return subprocess.check_output(
        ["docker", "image", "inspect", tag, "--format", "{{.Id}}"], text=True
    ).strip()


def prepare(setup_dir):
    saved = json.loads((setup_dir / "setup.json").read_text())
    assert saved["setup_status"] == "validated"
    from sandbox.docker import digest, manifest

    assert digest(manifest(setup_dir / "baseline")) == saved["baseline_checksum"]
    state = setup_dir / "harbor-v1"
    state.mkdir(exist_ok=True)
    key = state / "receipt.key"
    if not key.exists():
        key.write_bytes(secrets.token_bytes(32))
        key.chmod(0o600)
    image_id = saved["image_id"]
    image = "maintenance-harbor-base:" + image_id.split(":")[1][:16]
    subprocess.run(["docker", "tag", image_id, image], check=True)
    assert (
        subprocess.check_output(
            ["docker", "image", "inspect", image, "--format", "{{.Id}}"], text=True
        ).strip()
        == image_id
    )
    env_lines = """ENV PATH=/opt/venv/bin:/usr/local/bin:/usr/bin:/bin HOME=/tmp PYTHONDONTWRITEBYTECODE=1 OPENAI_API_KEY=sk-offline-placeholder TIKTOKEN_CACHE_DIR=/opt/tiktoken
ENV PYTHONPATH=/workspace/libs/partners/openai:/workspace/libs/core:/workspace/libs/standard-tests
ENV PYTEST_ADDOPTS="-p no:cacheprovider --no-cov"
WORKDIR /workspace/libs/partners/openai
"""
    for role in ["agent", "verifier"]:
        context = state / role
        context.mkdir(exist_ok=True)
        if not (context / "source").exists():
            shutil.copytree(setup_dir / "baseline", context / "source")
        shutil.copyfile(setup_dir / "reproduce.py", context / "reproduce.py")
        lines = [
            f"FROM {image}",
            "USER root",
            "COPY source /workspace",
            "COPY reproduce.py /task/reproduce.py",
        ]
        if role == "agent":
            shutil.copyfile(
                ROOT / "harbor_integration/isolation.py", context / "isolation.py"
            )
            lines += ["COPY isolation.py /task/isolation.py"]
        else:
            if not (context / "baseline").exists():
                shutil.copytree(setup_dir / "baseline", context / "baseline")
            if (context / "tests").exists():
                shutil.rmtree(context / "tests")
            shutil.copytree(TASK / "tests", context / "tests")
            shutil.copyfile(key, context / "tests/receipt.key")
            lines += ["COPY baseline /baseline", "COPY tests /tests"]
        lines += [
            "RUN chmod -R a-w /workspace /task && mkdir -p /logs/agent /logs/verifier /logs/artifacts && chmod 777 /logs/agent /logs/verifier /logs/artifacts"
        ]
        if role == "agent":
            lines += [
                f"RUN find /workspace/{SOURCE} -type d -exec chmod 777 {{}} + && find /workspace/{SOURCE} -type f -exec chmod 666 {{}} + && chmod 1777 /workspace/{TESTS}",
                env_lines,
                "USER 65532:65532",
            ]
        else:
            lines += [
                "RUN chmod -R a-w /baseline /tests && chmod 400 /tests/receipt.key",
                env_lines,
                "USER root",
            ]
        (context / "Dockerfile").write_text("\n".join(lines) + "\n")
    agent_id = build(state / "agent", "maintenance-harbor-agent:v1")
    verifier_id = build(state / "verifier", "maintenance-harbor-verifier:v1")
    compose = {
        "services": {
            "main": {
                "pull_policy": "never",
                "read_only": True,
                "user": "65532:65532",
                "cap_drop": ["ALL"],
                "security_opt": ["no-new-privileges:true"],
                "network_mode": "none",
                "pids_limit": 128,
                "tmpfs": ["/tmp:rw,nosuid,nodev,noexec,size=512m,mode=1777"],
                "volumes": [
                    f"production:/workspace/{SOURCE}",
                    f"development-tests:/workspace/{TESTS}",
                ],
            }
        },
        "volumes": {"production": {}, "development-tests": {}},
    }
    (TASK / "environment/docker-compose.yaml").write_text(
        yaml.safe_dump(compose, sort_keys=False)
    )
    (TASK / "tests/docker-compose.yaml").write_text(
        yaml.safe_dump(
            {
                "services": {
                    "main": {
                        "network_mode": "none",
                        "user": "root",
                        "security_opt": ["no-new-privileges:true"],
                        "pids_limit": 128,
                        "tmpfs": ["/tmp:rw,nosuid,nodev,noexec,size=768m,mode=1777"],
                    }
                }
            },
            sort_keys=False,
        )
    )
    config = {
        "schema_version": "1.4",
        "metadata": {
            "task_id": "LC-TIMEOUT-001",
            "starting_commit": saved["commit"],
            "baseline_checksum": saved["baseline_checksum"],
            "experimental_setting": "harbor-v1",
            "approval_status": "checks_passed_pending_review",
            "shared_workflow_seconds": 1200,
        },
        "environment": {
            "docker_image": agent_id,
            "network_mode": "no-network",
            "cpus": 2,
            "memory_mb": 2048,
            "workdir": "/workspace/libs/partners/openai",
        },
        "agent": {"timeout_sec": 1200, "user": "65532:65532"},
        "verifier": {
            "timeout_sec": 1200,
            "environment_mode": "separate",
            "user": "root",
            "environment": {
                "docker_image": verifier_id,
                "network_mode": "no-network",
                "cpus": 2,
                "memory_mb": 2048,
            },
        },
        "artifacts": [],
    }
    config["verifier"]["collect"] = [
        {
            "command": "rm -f /logs/artifacts/candidate.tar && tar --mode='u+rwX,go+rX' -cf /logs/artifacts/candidate.tar -C /workspace "
            + SOURCE
            + " "
            + TESTS,
            "user": "root",
            "timeout_sec": 90,
        }
    ]
    TaskConfig.model_validate(config)
    (TASK / "task.toml").write_text(toml.dumps(config))
    # Only issue and approved generic requirements become the task instruction.
    text = saved["task"]["instructions"] + "\n\n"

    from public_preparation.materialize import load_mini

    prompt = load_mini("mini-swe-explicit-test-policy.yaml")["agent"][
        "instance_template"
    ]
    text += prompt[prompt.index("## Required completion checklist") :]
    text += "\nHarbor executes hidden verification in a separate environment after you finish. The shared agent-plus-verifier deadline is 20 minutes; submit promptly to leave verification time.\n"
    (TASK / "instruction.md").write_text(text)
    # Trusted reference fixture is outside both the agent image and instructions.
    fixture_dir = next(
        p
        for p in sorted(setup_dir.glob("evaluator-selftest-*"))
        if (p / "result.json").exists()
        and json.loads((p / "result.json").read_text()).get("status")
        == "checks_passed_pending_review"
    )
    files = {
        SOURCE + "/_client_utils.py": (
            fixture_dir / "workspace" / SOURCE / "_client_utils.py"
        ).read_text(),
        TESTS + "/test_timeout_selftest.py": (
            fixture_dir / "workspace" / TESTS / "test_timeout_selftest.py"
        ).read_text(),
    }
    fixture = state / "reference-fixture.json"
    fixture.write_text(json.dumps(files))
    policy = {
        "budgets": saved["sandbox_config"]["budgets"],
        "model": json.loads(
            (configuration_path(ROOT, "configs/minimax-m2.5-mini-v5.json")).read_text()
        ),
        "prompt_config": "mini-swe-harbor-initial-instructions.yaml",
        "receipt_key": str(key),
        "reference_fixture": str(fixture),
    }
    from public_preparation.materialize import load_mini

    policy["mini_config"] = load_mini(policy["prompt_config"])
    policy["code_fingerprints"] = {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in [
            *(ROOT / "harbor_integration").glob("*.py"),
            ROOT / "agent/budget.py",
            ROOT / "agent/mini_runtime.py",
            ROOT / "agent/provider.py",
            ROOT / "agent/schemas.py",
            configuration_path(ROOT, "configs/mini-swe.yaml"),
        ]
    }
    policy_file = state / "controller-policy.json"
    policy_file.write_text(json.dumps(policy, indent=2) + "\n")
    for mode in ["baseline", "reference", "model"]:
        agent = {
            "import_path": "harbor_integration.host_mini:HostMiniSweAgent",
            "model_name": "openrouter/minimax/minimax-m2.5",
            "kwargs": {"policy": policy},
        }
        if mode != "model":
            agent["kwargs"]["fixture"] = mode
        job = {
            "job_name": "harbor-" + mode,
            "jobs_dir": str(
                ROOT
                / ("reports/harbor-validation" if mode != "model" else "runs/harbor")
            ),
            "n_attempts": 1,
            "n_concurrent_trials": 1,
            "retry": {"max_retries": 0},
            "environment": {"type": "docker", "delete": True},
            "agents": [agent],
            "tasks": [{"path": str(TASK)}],
        }
        JobConfig.model_validate(job)
        (configuration_path(ROOT, f"configs/harbor-{mode}.json")).write_text(
            json.dumps(job, indent=2) + "\n"
        )
    record = {
        "harbor_version": "0.24.0",
        "mini_swe_agent_version": "2.4.6",
        "agent_image": agent_id,
        "verifier_image": verifier_id,
        "starting_commit": saved["commit"],
        "baseline_checksum": saved["baseline_checksum"],
        "model_calls": 0,
        "task_directory": str(TASK),
        "trusted_setup_directory": str(state),
    }
    (ROOT / "reports/harbor-preparation.json").write_text(
        json.dumps(record, indent=2) + "\n"
    )
    return record


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--setup", type=Path, required=True)
    print(json.dumps(prepare(p.parse_args().setup.resolve()), indent=2))
