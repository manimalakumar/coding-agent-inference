"""Exercise public source/checks through the production HarborTools wrapper."""

import asyncio
import base64
import json
import os
import shlex
import shutil
import zlib

from public_preparation.catalog import ROOT
from public_preparation.runtime import constants, run, task_config, write
from public_replay.prepare import ControlBudget, ControlTrace, DockerControlEnvironment


def injection(files):
    data = base64.b64encode(zlib.compress(json.dumps(files).encode())).decode()
    return (
        "python - <<'PY'\nimport base64,json,zlib\nfrom pathlib import Path\nfiles=json.loads(zlib.decompress(base64.b64decode("
        + repr(data)
        + ")))\nfor name,content in files.items():\n    p=Path('/workspace')/name\n    p.parent.mkdir(parents=True,exist_ok=True)\n    p.write_text(content)\nPY"
    )


def existing_commands(spec):
    c = constants(ROOT / spec["task_bundle"] / "tests/verify.py")
    if "EXISTING" in c:
        return [
            "cd /workspace/"
            + package
            + " && python -m pytest "
            + module
            + " -q --disable-socket --allow-unix-socket"
            for package, module in c["EXISTING"]
        ]
    envfile = ROOT / "tasks" / spec["task_id"] / "environment.json"
    if envfile.is_file():
        env = json.loads(envfile.read_text())
        if env.get("existing_tests"):
            return [
                "python -m pytest "
                + " ".join(env["existing_tests"])
                + " -q --disable-socket --allow-unix-socket"
            ]
    taskfile = ROOT / "tasks" / spec["task_id"] / "task.json"
    if taskfile.is_file():
        task = json.loads(taskfile.read_text())
        if task.get("development_checks"):
            return [c["command"] for c in task["development_checks"]]
    raise RuntimeError("Explicit focused-test command unavailable: " + spec["task_id"])


def validate_wrapper(tid):
    os.environ.update(
        MSWEA_SILENT_STARTUP="1",
        MSWEA_GLOBAL_CONFIG_DIR=str(ROOT / ".local/mini"),
        PUBLIC_NO_INFERENCE="1",
    )
    from harbor_integration.host_task import HarborTools

    spec = task_config(tid)
    state = ROOT / ".local/public-v1" / tid
    prepared = json.loads((state / "prepared.json").read_text())
    fixture = json.loads((state / "reference-fixture.json").read_text())
    docker = shutil.which("docker") or "docker"
    args = [
        docker,
        "run",
        "-d",
        "--network=none",
        "--read-only",
        "--user=65532:65532",
        "--cap-drop=ALL",
        "--security-opt=no-new-privileges",
        "--cpus=2",
        "--memory=2g",
        "--pids-limit=128",
        "--tmpfs",
        "/tmp:rw,nosuid,nodev,size=512m",
    ]
    for rel in spec["source_paths"] + spec["test_paths"]:
        args += ["--mount", "type=volume,dst=/workspace/" + rel]
    args += [prepared["images"]["agent"], "sleep", "1200"]
    container = run(args).strip()

    async def check():
        trace = ControlTrace()
        tools = HarborTools(
            DockerControlEnvironment(docker, container),
            asyncio.get_running_loop(),
            ControlBudget(),
            trace,
            spec["expected_failure"],
        )
        results = []

        async def command(name, cmd, expected=0):
            result = await asyncio.to_thread(tools.execute, {"command": cmd})
            numeric = {
                "check": name,
                "expected_exit": expected,
                "observed_exit": result["returncode"],
                "passed": result["returncode"] == expected,
            }
            results.append(numeric)
            write(state / "wrapper-progress.json", results)
            # Source-bearing diagnostics stay in ignored setup storage.
            (state / (name.replace("/", "_") + ".log")).write_text(result["output"])
            if not numeric["passed"]:
                raise RuntimeError(
                    "Actual command wrapper failed "
                    + name
                    + "; inspect ignored setup log"
                )

        await command("baseline_reproduces", "python /task/reproduce.py", 1)
        await command("isolation_workspace_imports", "python /task/isolation.py")
        tests = {
            name: value
            for name, value in fixture.items()
            if any(name.startswith(p + "/") for p in spec["test_paths"])
        }
        await command("create_regression", injection(tests))
        reg = "python -m pytest -q --disable-socket --allow-unix-socket " + " ".join(
            "/workspace/" + p for p in tests
        )
        await command("regression_baseline", reg, 1)
        await command(
            "apply_reference",
            injection({p: v for p, v in fixture.items() if p not in tests}),
        )
        await command("reference_reproduces", "python /task/reproduce.py")
        await command("regression_reference", reg)
        for i, cmd in enumerate(existing_commands(spec)):
            await command("focused_tests_" + str(i), cmd)
        packages = {}
        for name in fixture:
            package = next(
                (
                    prefix
                    for prefix in [
                        spec["package"],
                        "libs/core",
                        "libs/langchain_v1",
                        "libs/partners/openai",
                    ]
                    if name.startswith(prefix + "/")
                ),
                None,
            )
            if package:
                packages.setdefault(package, []).append(name)
        for package, files in packages.items():
            paths = " ".join(shlex.quote("/workspace/" + p) for p in files)
            prefix = "cd /workspace/" + package + " && "
            await command("lint_" + package, prefix + "ruff check --no-cache " + paths)
            await command(
                "format_" + package, prefix + "ruff format --check --no-cache " + paths
            )
            production = [p for p in files if p not in tests]
            if production:
                types = (
                    "ty check"
                    if tid == "LC-JSON-INPUT-001"
                    else "mypy --cache-dir=/dev/null"
                )
                await command(
                    "types_" + package,
                    prefix
                    + types
                    + " "
                    + " ".join("/workspace/" + p for p in production),
                )
        assert all(r["event"] == "tool_result" for r in trace.records)
        return results

    try:
        result = asyncio.run(check())
        write(
            state / "wrapper-controls.json",
            {"passed": True, "model_requests": 0, "checks": result},
        )
        return result
    finally:
        run([docker, "rm", "-fv", container])
