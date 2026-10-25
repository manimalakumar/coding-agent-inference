"""Zero-inference task readiness through actual HarborTools and native Harbor."""

import asyncio
import json
import os
import shlex
import subprocess
from pathlib import Path

import toml
import yaml

from agent.budget import Budget
from agent.schemas import Budgets
from experiments.comparison_wrapper_controls import LocalDockerExec, write_files
from experiments.configuration_paths import configuration_path
from experiments.prepare_heldout_tasks import LABEL, ROOT, TASKS, write
from harbor_integration.host_task import HarborTools
from harbor_integration.trace import Trace


async def wrapper(task_id, revision):
    spec = TASKS[task_id]
    job = json.loads(
        (
            configuration_path(ROOT, f"configs/{LABEL}-{task_id}-minimax-model.json")
        ).read_text()
    )
    policy = job["agents"][0]["kwargs"]["policy"]
    h = Path(job["tasks"][0]["path"])
    cfg = toml.loads((h / "task.toml").read_text())
    c = yaml.safe_load((h / "environment/docker-compose.yaml").read_text())["services"][
        "main"
    ]
    out = ROOT / "reports" / LABEL / revision
    out.mkdir(parents=True, exist_ok=True)
    destination = out / (task_id + "-wrapper.json")
    if destination.exists():
        raise FileExistsError("Preserve earlier control results")
    args = [
        "docker",
        "run",
        "--rm",
        "--detach",
        "--network=none",
        "--read-only",
        "--user=65532:65532",
        "--cap-drop=ALL",
        "--security-opt=no-new-privileges:true",
        "--pids-limit=128",
        "--cpus=2",
        "--memory=2048m",
    ]
    for value in c["tmpfs"]:
        args += ["--tmpfs", value]
    for value in c["volumes"]:
        args += ["--volume", value.split(":", 1)[1]]
    args += [cfg["environment"]["docker_image"], "sleep", "infinity"]
    container = subprocess.check_output(args, text=True).strip()
    tools = HarborTools(
        LocalDockerExec(container, cfg["environment"]["workdir"]),
        asyncio.get_running_loop(),
        Budget(Budgets.model_validate(policy["budgets"])),
        Trace(out / (task_id + "-tools.jsonl"), ""),
        spec["expected"],
    )
    fixture = json.loads(Path(policy["reference_fixture"]).read_text())
    tests = {p: t for p, t in fixture.items() if "/tests/" in p}
    sources = {p: t for p, t in fixture.items() if p not in tests}
    cmds = [
        ("baseline_reproduction", "python /task/reproduce.py", 1),
        ("isolation", "python /task/isolation.py", 0),
    ]
    packages = sorted(
        {spec["package"]}
        | ({"libs/core"} if task_id == "LC-ASYNC-TOOLS-001" else set())
    )
    for package in packages:
        files = ["/workspace/" + p for p in sources if p.startswith(package + "/")]
        if files:
            cmds.append(
                (
                    "baseline_types_" + package,
                    "cd /workspace/"
                    + package
                    + " && mypy --cache-dir=/dev/null "
                    + " ".join(files),
                    0,
                )
            )
    cmds += [
        ("write_regression", write_files(tests), 0),
        (
            "regression_on_baseline",
            "python -m pytest -q --disable-socket --allow-unix-socket "
            + " ".join("/workspace/" + p for p in tests),
            1,
        ),
        ("apply_reference", write_files(sources), 0),
        ("reference_reproduction", "python /task/reproduce.py", 0),
        (
            "regression",
            "python -m pytest -q --disable-socket --allow-unix-socket "
            + " ".join("/workspace/" + p for p in tests),
            0,
        ),
    ]
    for package in packages:
        files = ["/workspace/" + p for p in fixture if p.startswith(package + "/")]
        source = ["/workspace/" + p for p in sources if p.startswith(package + "/")]
        for name, command, paths in [
            ("lint", "ruff check --no-cache", files),
            ("format", "ruff format --check --no-cache", files),
            ("types", "mypy --cache-dir=/dev/null", source),
        ]:
            if paths:
                cmds.append(
                    (
                        "reference_" + name + "_" + package,
                        "cd /workspace/"
                        + package
                        + " && "
                        + command
                        + " "
                        + " ".join(paths),
                        0,
                    )
                )
    for package, module in spec["existing"]:
        cmds.append(
            (
                "existing_" + package,
                "cd /workspace/"
                + package
                + " && python -m pytest "
                + shlex.quote(module)
                + " -q --disable-socket --allow-unix-socket",
                0,
            )
        )
    results = []
    try:
        for name, command, expected in cmds:
            try:
                r = await asyncio.to_thread(tools.execute, {"command": command})
                results.append(
                    {
                        "check": name,
                        "expected_exit_code": expected,
                        "exit_code": r["returncode"],
                        "passed": r["returncode"] == expected,
                        "output": r["output"],
                    }
                )
            except Exception as exc:
                results.append({"check": name, "passed": False, "error": str(exc)})
                break
            print(task_id, name, results[-1]["passed"], flush=True)
    finally:
        subprocess.run(
            ["docker", "rm", "-f", container], capture_output=True, check=True
        )
    report = {
        "task_id": task_id,
        "model_requests": 0,
        "status": "passed"
        if len(results) == len(cmds) and all(x["passed"] for x in results)
        else "blocked_by_environment",
        "actual_wrapper": "harbor_integration.host_task.HarborTools.execute",
        "checks": results,
    }
    write(destination, json.dumps(report, indent=2))
    return report


def native(task_id, mode, revision):
    out = ROOT / "reports" / LABEL / revision
    out.mkdir(parents=True, exist_ok=True)
    job = f"{LABEL}-{task_id}-{mode}-{revision}"
    dest = out / (task_id + "-" + mode + ".json")
    if dest.exists():
        raise FileExistsError("Preserve previous native controls")
    config = configuration_path(ROOT, f"configs/{LABEL}-{task_id}-minimax-{mode}.json")
    env = os.environ | {
        "MSWEA_SILENT_STARTUP": "1",
        "MSWEA_GLOBAL_CONFIG_DIR": str(ROOT / ".state/mini"),
    }
    p = subprocess.run(
        [str(ROOT / ".venv/bin/harbor"), "run", "-c", str(config), "--job-name", job],
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    write(out / (task_id + "-" + mode + ".log"), p.stdout + p.stderr)
    folders = list((ROOT / "reports/harbor-validation" / job).glob(task_id + "__*"))
    result = {
        "task_id": task_id,
        "mode": mode,
        "model_requests": 0,
        "status": "setup_failure",
        "command_exit": p.returncode,
    }
    if folders:
        trial = folders[0]
        result["trial"] = str(trial.relative_to(ROOT))
        if (trial / "verifier/checks.json").exists():
            checks = json.loads((trial / "verifier/checks.json").read_text())
            reward = json.loads((trial / "verifier/reward.json").read_text())
            receipt = checks.get("controller_receipt", {})
            ok = (
                reward.get("task_success") == int(mode != "baseline")
                and receipt.get("model_requests") == 0
            )
            if mode == "baseline":
                ok = (
                    ok
                    and checks["checks"].get("reproducer", {}).get("exit_code") == 1
                    and checks["checks"].get("hidden_acceptance", {}).get("exit_code")
                    == 1
                    and checks["checks"].get("existing_tests", {}).get("exit_code") == 0
                )
            if mode != "baseline":
                ok = ok and reward.get("explicit_submission") == int(
                    mode == "reference"
                )
            result.update(
                status="passed" if ok else "failed",
                rewards=reward,
                checks={k: v["exit_code"] for k, v in checks["checks"].items()},
                error=checks.get("error"),
                receipt=receipt,
            )
    write(dest, json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)
    return result


if __name__ == "__main__":
    import sys

    task, kind, revision = sys.argv[1:]
    if kind == "wrapper":
        asyncio.run(wrapper(task, revision))
    else:
        native(task, kind, revision)
