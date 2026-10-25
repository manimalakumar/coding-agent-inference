"""Resolve recorded launch identities through checked relative configuration paths."""

import hashlib
import json
from pathlib import Path
from typing import Any

from experiments.artifact_paths import resolve_artifact_path
from experiments.configuration_paths import resolve_configuration_path
from experiments.records import fingerprint
from experiments.records import load_runs as historical_load_runs
from public_replay.bundle import safe_path


def launch_index(project: Path) -> dict[str, Any]:
    """Create bounded relative references while the original root is available."""
    project = project.resolve()
    entries = {}
    for path in sorted((project / "reports/launches").glob("*.json")):
        launch = json.loads(path.read_text())
        command = launch.get("command", [])
        if "-c" not in command:
            continue
        candidate = Path(command[command.index("-c") + 1])
        if not candidate.is_absolute():
            candidate = safe_path(project, str(candidate))
        candidate = resolve_configuration_path(candidate)
        if not candidate.resolve().is_relative_to(project):
            raise ValueError(
                "External launch configuration: cannot migrate automatically"
            )
        relative = str(candidate.relative_to(project))
        if candidate.parent != project / "configs" or candidate.suffix != ".json":
            raise ValueError("Launch configuration is outside the bounded configs root")
        entries[path.stem] = {
            "configuration_ref": relative,
            "sha256": hashlib.sha256(candidate.read_bytes()).hexdigest(),
        }
    return {"schema_version": 1, "launches": entries}


def resolve_config(project: Path, entry: dict[str, Any]) -> Path:
    path = resolve_configuration_path(safe_path(project, entry["configuration_ref"]))
    if path.parent != project / "configs" or path.suffix != ".json":
        raise ValueError(
            "Configuration must be an exact direct-child configs reference"
        )
    if hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
        raise ValueError("Relocated launch configuration checksum mismatch")
    return path


def load_runs(project: Path, index: dict[str, Any], include_controls: bool = False):
    project = project.resolve()
    records = historical_load_runs(project, include_controls=include_controls)
    for record in records:
        if record["source"] != "harbor_launch":
            continue
        job = record["run_id"].split("/", 1)[0]
        entry = index.get("launches", {}).get(job)
        if entry is None:
            raise ValueError(
                "Missing explicit launch identity migration; regenerate the trusted launch index at the original root"
            )
        config = json.loads(resolve_config(project, entry).read_text())
        agent: dict[str, Any] = next(iter(config.get("agents", [])), {})
        policy = agent.get("kwargs", {}).get("policy", {})
        model = policy.get("model", {})
        record.update(
            model=model.get("model", agent.get("model_name", "unrecorded_model")),
            task_id=Path(config["tasks"][0]["path"]).name,
            provider=model.get("provider"),
            budgets=policy.get("budgets"),
            inference_id=fingerprint(model),
            commit=policy.get("source_provenance", {}).get("starting_commit"),
            baseline_checksum=policy.get("source_provenance", {}).get(
                "baseline_checksum"
            ),
            configuration_ref=entry["configuration_ref"],
            identity_migration="relative-launch-index-v1",
        )
        incident = (
            project
            / "reports"
            / (job.replace("-minimax-attempt-001", "") + "-startup-incident.json")
        )
        incident = resolve_artifact_path(incident)
        if incident.exists():
            audit = json.loads(incident.read_text())
            record["startup_audit"] = {
                key: audit.get(key)
                for key in [
                    "classification",
                    "phase",
                    "measured_coding_attempt",
                    "model_requests",
                    "verified_inference_spend_usd",
                    "reported_cost_usd",
                ]
            }
    return records
