"""Audit each frozen comparison task through the actual HarborTools wrapper; no inference."""

import asyncio
import base64
import hashlib
import json
import shlex
import subprocess
from pathlib import Path
from types import SimpleNamespace

import toml
import yaml

from agent.budget import Budget
from agent.schemas import Budgets
from experiments.configuration_paths import configuration_path
from experiments.launch import verify_freeze
from harbor_integration.host_task import HarborTools
from harbor_integration.trace import Trace

LABEL = "comparison-4x2x3-v1-exact-tool-controls-v1"


class LocalDockerExec:
    def __init__(self, container, workdir):
        self.container, self.workdir = container, workdir

    async def exec(self, command, timeout_sec):
        result = await asyncio.to_thread(
            subprocess.run,
            [
                "docker",
                "exec",
                "--user=65532:65532",
                "--workdir=" + self.workdir,
                self.container,
                "bash",
                "-c",
                command,
            ],
            capture_output=True,
            text=True,
            timeout=timeout_sec,
        )
        return SimpleNamespace(
            return_code=result.returncode, stdout=result.stdout, stderr=result.stderr
        )


def write_files(files):
    data = base64.b64encode(json.dumps(files).encode()).decode()
    script = (
        "import base64,json;from pathlib import Path;files=json.loads(base64.b64decode("
        + repr(data)
        + '));[(Path("/workspace")/p).write_text(c) for p,c in files.items()]'
    )
    return "python -c " + shlex.quote(script)


async def task_control(project, bundle, output):
    task = bundle["task_id"]
    job = json.loads(
        (
            configuration_path(
                project, f"configs/comparison-4x2x3-v1-{task}-minimax-reference.json"
            )
        ).read_text()
    )
    policy = job["agents"][0]["kwargs"]["policy"]
    fixture = json.loads(Path(policy["reference_fixture"]).read_text())
    task_path = Path(bundle["task_directory"])
    cfg = toml.loads((task_path / "task.toml").read_text())
    compose = yaml.safe_load(
        (task_path / "environment/docker-compose.yaml").read_text()
    )["services"]["main"]
    reused = json.loads(
        (project / f"reports/comparison-4x2x3-v1-{task}-controls.json").read_text()
    )
    assert reused["baseline"]["status"] == reused["reference"]["status"] == "passed"
    reference_path = project / reused["reference"]["trial"]
    reference_config = json.loads((reference_path / "config.json").read_text())
    assert reference_config["agent"]["kwargs"]["policy"] == policy
    reference_checks = json.loads((reference_path / "verifier/checks.json").read_text())
    baseline_path = project / reused["baseline"]["trial"]
    assert (
        json.loads((baseline_path / "verifier/checks.json").read_text())["checks"][
            "reproducer"
        ]["exit_code"]
        == 1
    )
    package = cfg["environment"]["workdir"]
    tests = {k: v for k, v in fixture.items() if "/tests/unit_tests/" in k}
    sources = {k: v for k, v in fixture.items() if k not in tests}
    assert len(tests) == 1
    test = next(iter(tests))
    assert Path(test).parent.name == "unit_tests" and Path(test).name.startswith(
        "test_"
    )
    assert test in reference_checks["files_changed"]
    source_paths = ["/workspace/" + p for p in sources]
    test_paths = ["/workspace/" + p for p in tests]
    args = [
        "docker",
        "run",
        "--detach",
        "--rm",
        "--network=none",
        "--read-only",
        "--user=" + compose["user"],
        "--cap-drop=ALL",
        "--security-opt=no-new-privileges:true",
        "--pids-limit=" + str(compose["pids_limit"]),
        "--cpus=" + str(cfg["environment"]["cpus"]),
        "--memory=" + str(cfg["environment"]["memory_mb"]) + "m",
    ]
    for value in compose["tmpfs"]:
        args += ["--tmpfs", value]
    for value in compose["volumes"]:
        args += ["--volume", value.split(":", 1)[1]]
    args += [bundle["agent_image"], "sleep", "infinity"]
    container = subprocess.check_output(args, text=True).strip()
    trace_path = output / (task + "-tool-results.jsonl")
    trace = Trace(trace_path, "")
    results = []
    budget = Budget(Budgets.model_validate(policy["budgets"]))
    tools = HarborTools(
        LocalDockerExec(container, package),
        asyncio.get_running_loop(),
        budget,
        trace,
        policy["reproduction"]["expected_failure"],
    )
    commands = [
        ("baseline_reproducer", "python /task/reproduce.py", 1),
        (
            "baseline_types",
            "mypy --cache-dir=/tmp/mypy " + " ".join(map(shlex.quote, source_paths)),
            0,
        ),
        ("write_root_regression", write_files(tests), 0),
        (
            "regression_on_baseline",
            "python -m pytest -q --disable-socket --allow-unix-socket "
            + " ".join(map(shlex.quote, test_paths)),
            1,
        ),
        ("apply_trusted_reference", write_files(sources), 0),
        ("reference_reproducer", "python /task/reproduce.py", 0),
        (
            "reference_regression",
            "python -m pytest -q --disable-socket --allow-unix-socket "
            + " ".join(map(shlex.quote, test_paths)),
            0,
        ),
        (
            "reference_lint",
            "ruff check --no-cache "
            + " ".join(map(shlex.quote, source_paths + test_paths)),
            0,
        ),
        (
            "reference_format",
            "ruff format --check --no-cache "
            + " ".join(map(shlex.quote, source_paths + test_paths)),
            0,
        ),
        (
            "reference_types",
            "mypy --cache-dir=/tmp/mypy " + " ".join(map(shlex.quote, source_paths)),
            0,
        ),
    ]
    try:
        for name, command, expected in commands:
            try:
                result = await asyncio.to_thread(tools.execute, {"command": command})
                results.append(
                    {
                        "check": name,
                        "exit_code": result["returncode"],
                        "expected_exit_code": expected,
                        "passed": result["returncode"] == expected,
                        "output": result["output"],
                        "command": command
                        if "write_" not in name and name != "apply_trusted_reference"
                        else "[trusted fixture write; see reference_fixture SHA]",
                    }
                )
            except Exception as exc:
                results.append(
                    {
                        "check": name,
                        "passed": False,
                        "error": type(exc).__name__ + ": " + str(exc),
                    }
                )
                break
    finally:
        subprocess.run(
            ["docker", "rm", "-f", container],
            capture_output=True,
            timeout=20,
            check=True,
        )
    report = {
        "task_id": task,
        "frozen_task_environment": "comparison-4x2x3-v1",
        "model_requests": 0,
        "inference_cost_usd": 0,
        "status": "passed"
        if all(r["passed"] for r in results) and len(results) == len(commands)
        else "blocked_by_environment",
        "actual_wrapper": "harbor_integration.host_task.HarborTools.execute",
        "wrapper_sha256": hashlib.sha256(
            (project / "harbor_integration/host_task.py").read_bytes()
        ).hexdigest(),
        "agent_image": bundle["agent_image"],
        "verifier_image": bundle["verifier_image"],
        "baseline_checksum": bundle["baseline_checksum"],
        "reference_fixture_sha256": hashlib.sha256(
            Path(policy["reference_fixture"]).read_bytes()
        ).hexdigest(),
        "reused_native_baseline_control": str(baseline_path.relative_to(project)),
        "reused_native_reference_control": str(reference_path.relative_to(project)),
        "reused_reference_check_exits": {
            k: v["exit_code"] for k, v in reference_checks["checks"].items()
        },
        "validated_regression_file": test,
        "checks": results,
        "comparison_launch_authorized": False,
        "scope_note": "Applies only to this exact frozen task environment. Feature-task repair, different dependencies/images/instructions/checks require affected controls again.",
    }
    (output / (task + ".json")).write_text(json.dumps(report, indent=2) + "\n")
    return {k: v for k, v in report.items() if k != "checks"}


async def run(project):
    verify_freeze(
        project,
        "configs/comparison-four-tasks.sha256",
        "configs/comparison-four-tasks.frozen-setting.json",
    )
    output = project / "reports" / LABEL
    output.mkdir(exist_ok=False)
    plan = json.loads(
        (
            configuration_path(project, "configs/comparison-4x2x3-v1.plan.json")
        ).read_text()
    )
    results = []
    for bundle in plan["task_bundles"]:
        results.append(await task_control(project, bundle, output))
    summary = {
        "setting": LABEL,
        "model_requests": 0,
        "inference_cost_usd": 0,
        "comparison_round_not_launched": True,
        "tasks": results,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    asyncio.run(run(Path(__file__).resolve().parents[1]))
