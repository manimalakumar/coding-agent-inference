"""Deliberate fresh launches consume validated public resources, with no retries."""

import json
import os
import re
import subprocess
from pathlib import Path

from experiments.configuration_paths import configuration_path
from public_preparation.runtime import digest


def readiness(root, task):
    state = root / ".local/public-v1" / task
    path = state / "validated.json"
    if not path.is_file():
        raise RuntimeError(task + ": public preparation controls have not passed")
    receipt = json.loads(path.read_text())
    if (
        receipt.get("model_requests") != 0
        or not receipt.get("baseline_rejected")
        or not receipt.get("reference_accepted")
        or not receipt.get("public_checks_preserved")
    ):
        raise RuntimeError("Incomplete public preparation receipt: " + task)
    modes = {c["fixture"]: c for c in receipt.get("controls", [])}
    if set(modes) != {
        "baseline",
        "reference",
        "reference_budget",
        "reference_error",
    } or not all(c["passed"] and c["model_requests"] == 0 for c in modes.values()):
        raise RuntimeError("Missing terminal capture/evaluation controls: " + task)
    if not receipt.get("wrapper_controls") or not all(
        c["passed"] for c in receipt["wrapper_controls"]
    ):
        raise RuntimeError("Actual command wrapper controls missing: " + task)
    from public_replay.bundle import safe_path

    if not receipt.get("files"):
        raise RuntimeError("Resource fingerprints missing")
    for name, expected in receipt["files"].items():
        if digest(safe_path(root, name).read_bytes()) != expected:
            raise RuntimeError("Validated public resource changed: " + name)
    return receipt


def records(root):
    from experiments.records import harbor_record

    result = []
    for f in (root / ".local/public-launches").glob("*.json"):
        if f.stem.endswith("-config"):
            continue
        meta = json.loads(f.read_text())
        for trial in (root / "runs/public-models" / f.stem).glob("*/config.json"):
            record = harbor_record(trial.parent)
            if record and record["kind"] == "model_attempt":
                record.update(
                    task_id=meta["task_id"],
                    public_protocol=meta["protocol_id"],
                    public_setting_id=meta["setting_id"],
                )
                result.append(record)
    return result


def launch(
    root, task, *, confirmed=False, job_name=None, credentials=None, model="minimax"
):
    if confirmed is not True:
        raise ValueError("Fresh inference requires explicit confirmation")
    if not job_name or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,100}", job_name):
        raise ValueError("Choose a fresh explicit attempt ID")
    receipt = readiness(root, task)
    prepared = json.loads(
        (root / ".local/public-v1" / task / "prepared.json").read_text()
    )
    for image in prepared["images"].values():
        subprocess.run(
            ["docker", "image", "inspect", image],
            check=True,
            capture_output=True,
            timeout=30,
        )
    credentials = (
        Path(credentials).expanduser()
        if credentials
        else Path.home() / ".config/openrouter/credentials.env"
    )
    if not credentials.is_file():
        raise FileNotFoundError("Host-only inference credential file missing")
    out = root / "runs/public-models" / job_name
    ledger = root / ".local/public-launches"
    ledger.mkdir(parents=True, exist_ok=True)
    if out.exists() or (ledger / (job_name + ".json")).exists():
        raise FileExistsError(
            "Attempt already exists; replay it without further inference"
        )
    config = json.loads((root / ".local/public-v1" / task / "model.json").read_text())
    config["jobs_dir"] = str(root / "runs/public-models")
    if model == "glm5":
        original = json.loads(
            (
                configuration_path(
                    root,
                    "configs/held-out-remainder-round1-v1-LC-INT-TEMPERATURE-001-glm5-model.json",
                )
            ).read_text()
        )
        config["agents"][0]["model_name"] = original["agents"][0]["model_name"]
        config["agents"][0]["kwargs"]["policy"]["model"] = original["agents"][0][
            "kwargs"
        ]["policy"]["model"]
    elif model != "minimax":
        raise ValueError("Use a pinned model: minimax or glm5")
    inference = config["agents"][0]["kwargs"]["policy"]["model"]
    pinned = {
        "minimax": ("minimax/minimax-m2.5", "minimax/fp8"),
        "glm5": ("z-ai/glm-5", "z-ai/fp8"),
    }[model]
    provider = inference.get("provider", {})
    if (
        inference.get("model") != pinned[0]
        or provider.get("only") != [pinned[1]]
        or provider.get("allow_fallbacks") is not False
        or inference.get("endpoint") != "https://openrouter.ai/api/v1/chat/completions"
        or inference.get("temperature", 0) != 0
        or (
            model == "glm5"
            and inference.get("reasoning", {}).get("enabled") is not False
        )
    ):
        raise RuntimeError(
            "Pinned model/provider policy changed; fresh protocol validation is required"
        )
    # Store the derived per-launch policy with the validated protocol identity;
    # source, evaluator, capture policy and budgets retain the validated values.
    configpath = ledger / (job_name + "-config.json")
    with configpath.open("x") as f:
        json.dump(config, f, indent=2)
    meta = {
        "task_id": task,
        "protocol_id": receipt["protocol_id"],
        "setting_id": receipt["setting_id"],
        "configuration_sha256": digest(configpath.read_bytes()),
        "status": "launching",
    }
    with (ledger / (job_name + ".json")).open("x") as f:
        json.dump(meta, f, indent=2)
    env = os.environ | {
        "MSWEA_SILENT_STARTUP": "1",
        "MSWEA_GLOBAL_CONFIG_DIR": str(root / ".local/mini"),
    }
    try:
        done = subprocess.run(
            [
                str(root / ".venv/bin/harbor"),
                "run",
                "-c",
                str(configpath),
                "--env-file",
                str(credentials),
                "--job-name",
                job_name,
            ],
            cwd=root,
            env=env,
            check=False,
        )
        meta.update(
            status="completed" if done.returncode == 0 else "failed",
            returncode=done.returncode,
        )
    finally:
        (ledger / (job_name + ".json")).write_text(json.dumps(meta, indent=2) + "\n")
    found = [r for r in records(root) if r["run_id"].startswith(job_name + "/")]
    if len(found) != 1:
        raise RuntimeError(
            "Reconcile saved artifacts; do not retry inference automatically"
        )
    return {
        "mode": "execute",
        "launched": True,
        "record": found[0],
        "public_protocol": receipt["protocol_id"],
    }
