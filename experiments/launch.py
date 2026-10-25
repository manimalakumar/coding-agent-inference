"""Explicit Harbor launch boundary shared by notebooks and the command line."""

import hashlib
import json
import os
import re
import subprocess
from pathlib import Path
from uuid import uuid4

from experiments.configuration_paths import (
    configuration_path,
    resolve_configuration_path,
)
from experiments.storage import atomic_write


def verify_freeze(
    project,
    manifest="configs/harbor-submission.sha256",
    setting_file="configs/LC-TIMEOUT-001.harbor-submission.frozen-setting.json",
):
    project = Path(project).resolve()
    for line in resolve_configuration_path(project / manifest).read_text().splitlines():
        expected, relative = line.split("  ", 1)
        path = (project / relative).resolve()
        if (
            not path.is_relative_to(project)
            or hashlib.sha256(path.read_bytes()).hexdigest() != expected
        ):
            raise ValueError("Frozen configuration drift: " + relative)
    return json.loads(resolve_configuration_path(project / setting_file).read_text())[
        "setting_id"
    ]


def launch(
    project,
    *,
    mode="replay",
    confirmed=False,
    job_name=None,
    credentials=None,
    study_id=None,
):
    """Replay only returns a plan; execute requires explicit opt-in and never retries."""
    if mode not in ("replay", "execute"):
        raise ValueError("Mode must be replay or execute")
    project = Path(project).resolve()
    if study_id and study_id.startswith("public:"):
        from public_preparation.launch import launch as public_launch

        if mode == "replay":
            return {
                "mode": "replay",
                "launched": False,
                "task_id": study_id.split(":", 1)[1],
            }
        return public_launch(
            project,
            study_id.split(":", 1)[1],
            confirmed=confirmed,
            job_name=job_name,
            credentials=credentials,
        )
    catalog_path = configuration_path(project, "configs/study.json")
    catalog = json.loads(catalog_path.read_text()) if catalog_path.is_file() else {}
    selected_study = study_id or catalog.get("active_study")
    selection = (
        catalog["studies"][selected_study]
        if selected_study
        else {
            "job_config": "configs/harbor-model.json",
            "freeze_manifest": "configs/harbor-submission.sha256",
            "frozen_setting": "configs/LC-TIMEOUT-001.harbor-submission.frozen-setting.json",
        }
    )
    if mode == "execute":
        from public_preparation.launch import launch as public_launch

        task_ids = selection.get("task_ids", ["LC-TIMEOUT-001"])
        if len(task_ids) != 1:
            raise ValueError(
                "Choose an explicit public:TASK_ID for a fresh public protocol"
            )
        template = json.loads((project / selection["job_config"]).read_text())
        model_id = template["agents"][0]["kwargs"]["policy"]["model"]["model"]
        return public_launch(
            project,
            task_ids[0],
            model="glm5" if model_id.startswith("z-ai/") else "minimax",
            confirmed=confirmed,
            job_name=job_name,
            credentials=credentials,
        )
    job_name = job_name or "notebook-" + uuid4().hex[:12]
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,100}", job_name):
        raise ValueError("Use a simple unique job name")
    credentials = (
        Path(credentials).expanduser()
        if credentials
        else Path.home() / ".config/openrouter/credentials.env"
    )
    command = [
        str(project / ".venv/bin/harbor"),
        "run",
        "-c",
        str(project / selection["job_config"]),
        "--env-file",
        str(credentials),
        "--job-name",
        job_name,
    ]
    plan = {
        "mode": mode,
        "command": command,
        "job_name": job_name,
        "result_directory": str(project / "runs/harbor" / job_name),
        "launched": False,
    }
    if mode == "replay":
        return plan  # No subprocess, Docker, credential read, or provider request.
    if not confirmed:
        raise ValueError("Execute requires confirmed=True (CLI: --confirm-paid-run)")
    from experiments.portable_prepare import execution_readiness

    execution_readiness(project, selection.get("task_ids", ["LC-TIMEOUT-001"]))
    setting = verify_freeze(
        project, selection["freeze_manifest"], selection["frozen_setting"]
    )
    if Path(plan["result_directory"]).exists():
        raise FileExistsError("Job already exists; choose a fresh job name")
    if not credentials.is_file():
        raise FileNotFoundError("Host credential file is missing")
    if not Path(command[0]).is_file():
        raise FileNotFoundError(
            "Install the frozen Harbor environment with uv sync --frozen"
        )
    env = os.environ | {
        "MSWEA_SILENT_STARTUP": "1",
        "MSWEA_GLOBAL_CONFIG_DIR": str(project / ".state/mini"),
    }
    from experiments.bookkeeping import finalize_job, reconcile

    # Reconcile past partial launches without inference before starting another run.
    reconcile(project, pending_only=True)
    manifests = project / "reports/launches"
    manifests.mkdir(exist_ok=True)
    manifest = manifests / (job_name + ".json")
    # Exclusive creation prevents a notebook rerun from repeating the same paid launch.
    with manifest.open("x") as output:
        json.dump(
            plan | {"setting_id": setting, "status": "launching"}, output, indent=2
        )
    failure = None
    status = "process_finished"
    try:
        result = subprocess.run(command, cwd=project, env=env, check=False)
        plan.update(launched=True, returncode=result.returncode, setting_id=setting)
    except BaseException as exc:
        failure = type(exc).__name__
        status = (
            "interrupted"
            if isinstance(exc, (KeyboardInterrupt, SystemExit))
            else "launch_failed"
        )
        raise
    finally:
        metadata = plan | {"setting_id": setting, "status": status}
        if failure:
            metadata["error_type"] = failure
        atomic_write(manifest, json.dumps(metadata, indent=2))
        try:
            plan["bookkeeping"] = finalize_job(project, job_name)
            metadata["bookkeeping"] = plan["bookkeeping"]
        except Exception as exc:
            metadata["bookkeeping"] = {
                "status": "failed",
                "error_type": type(exc).__name__,
            }
            if failure is None:
                raise RuntimeError(
                    "Run artifacts preserved, but bookkeeping failed. Run 'python -m experiments reconcile'; do not repeat inference."
                ) from exc
        finally:
            atomic_write(manifest, json.dumps(metadata, indent=2))
    return plan
