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


def prepare(setup_dir, task_dir, prompt_file, reference_fixture, label, config_prefix):
    TASK = task_dir
    task_id = task_dir.name
    saved = json.loads((setup_dir / "setup.json").read_text())
    assert saved["setup_status"] == "validated"
    SOURCE = saved["task"]["source_paths"][0]
    TESTS = saved["task"]["test_paths"][0]
    package = saved["task"]["package"]
    pythonpath = ":".join(
        "/workspace/" + p
        for p in dict.fromkeys(
            [package, "libs/core", "libs/standard-tests", "libs/partners/openai"]
        )
    )
    from sandbox.docker import digest, manifest

    assert digest(manifest(setup_dir / "baseline")) == saved["baseline_checksum"]
    state = setup_dir / label
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
    env_lines = f"""ENV PATH=/opt/venv/bin:/usr/local/bin:/usr/bin:/bin HOME=/tmp PYTHONDONTWRITEBYTECODE=1 OPENAI_API_KEY=sk-offline-placeholder TIKTOKEN_CACHE_DIR=/opt/tiktoken
ENV PYTHONPATH={pythonpath}
ENV PYTEST_ADDOPTS="-p no:cacheprovider --no-cov"
WORKDIR /workspace/{package}
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
            shutil.copyfile(setup_dir / "isolation.py", context / "isolation.py")
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
    agent_id = build(state / "agent", "maintenance-harbor-agent:" + task_id.lower())
    verifier_id = build(
        state / "verifier", "maintenance-harbor-verifier:" + task_id.lower()
    )
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
            "task_id": task_id,
            "starting_commit": saved["commit"],
            "baseline_checksum": saved["baseline_checksum"],
            "experimental_setting": label,
            "approval_status": "checks_passed_pending_review",
            "shared_workflow_seconds": 1200,
        },
        "environment": {
            "docker_image": agent_id,
            "network_mode": "no-network",
            "cpus": 2,
            "memory_mb": 2048,
            "workdir": "/workspace/" + package,
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
    # Instructions were reviewed independently; never append another task's checklist.
    assert (TASK / "instruction.md").read_text() == saved["task"]["instructions"]
    fixture = state / "reference-fixture.json"
    files = json.loads(reference_fixture.read_text())
    normalization = state / "reference-normalization"
    normalization.mkdir(exist_ok=True)
    normalized = {}
    for relative, content in files.items():
        path = normalization / Path(relative).name
        path.write_text(content)
        command = [
            "docker",
            "run",
            "--rm",
            "--network=none",
            "--user=0:0",
            "-v",
            str(setup_dir / "baseline") + ":/workspace:ro",
            "-v",
            str(normalization) + ":/reference",
            saved["image_id"],
            "ruff",
            "format",
            "--no-cache",
            "--config",
            "/workspace/" + package + "/pyproject.toml",
            "/reference/" + path.name,
        ]
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        files[relative] = path.read_text()
        normalized[relative] = {
            "before_sha256": hashlib.sha256(content.encode()).hexdigest(),
            "after_sha256": hashlib.sha256(files[relative].encode()).hexdigest(),
            "command": command,
            "output": result.stdout + result.stderr,
        }
    (state / "reference-normalization.json").write_text(
        json.dumps(normalized, indent=2)
    )
    fixture.write_text(json.dumps(files))
    policy = {
        "budgets": saved["sandbox_config"]["budgets"],
        "source_provenance": {
            "starting_commit": saved["commit"],
            "baseline_checksum": saved["baseline_checksum"],
        },
        "model": json.loads(
            (configuration_path(ROOT, "configs/minimax-m2.5-mini-v5.json")).read_text()
        ),
        "prompt_config": prompt_file.name,
        "reproduction": {"expected_failure": saved["task"]["expected_failure"]},
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
            "import_path": "harbor_integration.host_task:HostMiniSweAgent",
            "model_name": "openrouter/minimax/minimax-m2.5",
            "kwargs": {"policy": policy},
        }
        if mode != "model":
            agent["kwargs"]["fixture"] = mode
        job = {
            "job_name": config_prefix + "-" + mode,
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
        (configuration_path(ROOT, f"configs/{config_prefix}-{mode}.json")).write_text(
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
    (ROOT / f"reports/{config_prefix}-preparation.json").write_text(
        json.dumps(record, indent=2) + "\n"
    )
    return record


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--setup", type=Path, required=True)
    p.add_argument("--task", type=Path, required=True)
    p.add_argument("--prompt", type=Path, required=True)
    p.add_argument("--reference-fixture", type=Path, required=True)
    p.add_argument("--label", required=True)
    p.add_argument("--prefix", required=True)
    args = p.parse_args()
    print(
        json.dumps(
            prepare(
                args.setup.resolve(),
                args.task.resolve(),
                args.prompt.resolve(),
                args.reference_fixture.resolve(),
                args.label,
                args.prefix,
            ),
            indent=2,
        )
    )
