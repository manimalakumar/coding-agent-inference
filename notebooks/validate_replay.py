"""Validate every tutorial code cell from an isolated public copy; zero inference."""

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.configuration_paths import configuration_path  # noqa: E402


def worker():
    from IPython.core.interactiveshell import InteractiveShell
    from IPython.utils.capture import capture_output

    # Initialize IPython before denying OS processes; tutorial execution is guarded.
    shell = InteractiveShell.instance()
    private = "/Experimental-" + "Agents/"
    denied = []
    reads = set()

    def audit(event, args):
        forbidden = event in {
            "subprocess.Popen",
            "os.system",
            "os.exec",
            "os.posix_spawn",
            "socket.connect",
            "socket.getaddrinfo",
            "socket.sendto",
        }
        if event == "open" and args and isinstance(args[0], (str, bytes)):
            name = os.fsdecode(args[0])
            forbidden = (
                forbidden
                or private in name
                or (name.endswith(".zip") and not name.startswith(str(ROOT)))
                or name.endswith("credentials.env")
                or "/.config/openrouter/" in name
            )
            if not forbidden and name.startswith(str(ROOT)):
                reads.add(name)
        if forbidden:
            denied.append(event)
            raise RuntimeError("Replay blocked: " + event)

    sys.addaudithook(audit)
    # Negative controls prove the guard fires before any external side effect.
    import socket

    probes = {}
    for label, probe in [
        ("network", lambda: socket.create_connection(("203.0.113.1", 443))),
        ("process", lambda: subprocess.run(["docker", "info"])),
        ("credentials", lambda: open(".local/credentials.env")),
        ("private_archive", lambda: open("/unavailable" + private + "main.ipynb")),
        ("private_zip", lambda: open("/unavailable/B-Experimental-Agents-Public.zip")),
    ]:
        try:
            probe()
        except RuntimeError as exc:
            assert str(exc).startswith("Replay blocked:")
            probes[label] = "blocked before side effect"
        else:
            raise AssertionError("Negative control failed: " + label)
    denied.clear()
    nb = json.loads((ROOT / "main.ipynb").read_text())
    passed = []
    with capture_output():
        for index, cell in enumerate(nb["cells"]):
            if cell["cell_type"] != "code":
                continue
            result = shell.run_cell("".join(cell["source"]))
            if result.error_in_exec or result.error_before_exec:
                raise RuntimeError(
                    "Notebook cell "
                    + str(index)
                    + ": "
                    + str(result.error_in_exec or result.error_before_exec)
                )
            passed.append(index)
    from public_replay.bundle import Bundle

    assert (
        Bundle().manifest["content_id"]
        == shell.user_ns["bundle"].manifest["content_id"]
    )
    ns = shell.user_ns
    assert ns["MODE"] == "replay" and ns["CONFIRM_PAID_EXECUTION"] is False
    assert ns["heldout_counts"] == {
        "held_out_two": 4,
        "held_out_six": 36,
        "held_out_total": 40,
    }, ns["heldout_counts"]
    assert ns["spending"]["known_operational_spend_usd"] == "2.76420370"
    assert len(ns["development_roster"]) == 5 and len(ns["heldout_roster"]) == 8
    assert len(ns["by_id"]) == 60
    assert not denied, denied
    from experiments.portable_prepare import (
        doctor,
        execution_readiness,
        obtain_source,
        validate_spec,
    )

    readiness = doctor(ROOT)
    assert len(readiness["tasks"]) == 13
    assert all(len(s["starting_commit"]) == 40 for s in readiness["tasks"].values())
    for task in readiness["tasks"]:
        try:
            execution_readiness(ROOT, [task])
        except RuntimeError:
            pass
        else:
            raise AssertionError("Unprepared task passed Execute gate")
        try:
            obtain_source(task, root=ROOT)
        except PermissionError:
            pass
        else:
            raise AssertionError("Source preparation lacked explicit opt-in")
    from experiments.launch import launch

    study_catalog = json.loads(
        (configuration_path(ROOT, "configs/study.json")).read_text()
    )["studies"]
    for study_id in study_catalog:
        try:
            launch(
                ROOT,
                mode="execute",
                confirmed=True,
                study_id=study_id,
                job_name="validation-never-launch",
            )
        except (RuntimeError, ValueError) as exc:
            assert "preparation controls have not passed" in str(
                exc
            ) or "explicit public:TASK_ID" in str(exc), str(exc)
        else:
            raise AssertionError("Unprepared original launcher reached inference")
    try:
        validate_spec(
            {
                "repository": "https://github.com/langchain-ai/langchain.git",
                "starting_commit": "main",
            }
        )
    except ValueError:
        pass
    else:
        raise AssertionError("Unpinned source accepted")
    print(
        json.dumps(
            {
                "status": "passed",
                "notebook_cells": len(nb["cells"]),
                "executed_code_cells": passed,
                "saved_records": 60,
                "development_tasks": 5,
                "heldout_tasks": 8,
                "heldout_launches": 40,
                "known_operational_spend_usd": ns["spending"][
                    "known_operational_spend_usd"
                ],
                "source_specs_validated": 13,
                "original_launcher_study_gates_checked": len(study_catalog),
                "execute_gates_checked": 13,
                "source_network_opt_in_gates_checked": 13,
                "negative_controls": probes,
                "forbidden_io_attempts": denied,
                "model_requests": 0,
                "private_project_reads": 0,
                "relocated": True,
                "read_input_files": len(reads),
            }
        )
    )


def main():
    if "--worker" in sys.argv:
        worker()
        return
    before = hashlib.sha256((ROOT / "main.ipynb").read_bytes()).hexdigest()
    with tempfile.TemporaryDirectory(prefix="agents-public-replay-") as temp:
        public = Path(temp) / "repository"
        shutil.copytree(
            ROOT,
            public,
            ignore=shutil.ignore_patterns(
                ".venv",
                ".notebook-venv",
                ".local",
                "__pycache__",
                ".pytest_cache",
                ".ruff_cache",
                "*.pyc",
            ),
        )
        env = {
            k: v
            for k, v in os.environ.items()
            if not any(
                s in k.upper()
                for s in ["KEY", "TOKEN", "SECRET", "CREDENTIAL", "PASSWORD"]
            )
        }
        env.update(
            PYTHONPATH=str(public),
            PYTHONDONTWRITEBYTECODE="1",
            IPYTHONDIR=str(Path(temp) / "ipython"),
            MPLCONFIGDIR=str(Path(temp) / "matplotlib"),
        )
        completed = subprocess.run(
            [sys.executable, str(public / "notebooks/validate_replay.py"), "--worker"],
            cwd=public,
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
        )
        if completed.returncode:
            raise RuntimeError(completed.stdout + "\n" + completed.stderr)
        receipt = json.loads(completed.stdout.strip().splitlines()[-1])
    assert hashlib.sha256((ROOT / "main.ipynb").read_bytes()).hexdigest() == before
    receipt["saved_notebook_sha256"] = before
    receipt["saved_first_execution_unchanged_by_validation"] = True
    output = ROOT / "reports/portability/replay-validation.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
