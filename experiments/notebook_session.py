"""Notebook display adapters for the existing validated public execution path."""

import copy
import json

import pandas as pd

from experiments import notebook_workflow as workflow
from experiments.configuration_paths import configuration_path
from public_preparation.launch import readiness


def preparation(root, task_id, model):
    """Fail on unavailable resources; never substitute a completed attempt."""
    receipt = readiness(root, task_id)
    prepared = workflow.public_preparation(root, task_id)
    if model == "glm5":
        config = json.loads(
            configuration_path(
                root,
                "configs/held-out-remainder-round1-v1-LC-INT-TEMPERATURE-001-glm5-model.json",
            ).read_text()
        )
        selected = config["agents"][0]
        prepared["job"]["agents"][0]["model_name"] = selected["model_name"]
        prepared["policy"]["model"] = copy.deepcopy(
            selected["kwargs"]["policy"]["model"]
        )
    elif model != "minimax":
        raise ValueError("Use a pinned model: minimax or glm5")
    # Adapt the receipt's public schema for the existing teaching displays.
    # The launch gate continues to read and verify the original receipt itself.
    prepared["controls"] = {
        **receipt,
        "controls": {
            control["fixture"]: {
                **control,
                "patch_accepted": control["accepted"],
                "trial": control["artifact_dir"],
            }
            for control in receipt["controls"]
        },
    }
    return prepared


def issue_outputs(root, prepared):
    """Read actual local fixture results, with no model/service invocation."""
    from experiments.study_views import issue_outputs as annotated_outputs

    if prepared["selection"]["task_ids"] == ["LC-ANNOTATED-TOOLS-001"]:
        return annotated_outputs(root, prepared)
    controls = prepared["controls"]["controls"]
    baseline = controls["baseline"]
    reference = controls["reference"]
    checks = sorted(set(baseline["checks"]) | set(reference["checks"]))
    frame = pd.DataFrame(
        [
            {
                "Check": check,
                "Baseline exit": baseline["checks"].get(check, {}).get("exit_code"),
                "Reference exit": reference["checks"].get(check, {}).get("exit_code"),
            }
            for check in checks
        ]
    )
    return frame, {
        "recorded_exit_code": baseline["checks"]["reproducer"]["exit_code"],
        "observed_failure": prepared["frozen"]["expected_failure"],
        "evidence": baseline["artifact_dir"] + "/verifier/checks.json",
    }


def display_summary(value):
    """Short display labels; evidence payloads and provenance remain untouched."""
    if isinstance(value, dict):
        return {display_summary(k): display_summary(v) for k, v in value.items()}
    if isinstance(value, list):
        return [display_summary(item) for item in value]
    if isinstance(value, str):
        for old, new in [
            ("Author explanation:", "Explanation:"),
            ("Author-written historical summary", "Explanation"),
            ("historical_request_summary", "request"),
            (
                "Historical request summary (original wording privately retained)",
                "Request",
            ),
            (
                "historical request summarized; original first response and numeric usage retained",
                "Live call; request summarized, response and usage retained",
            ),
            ("Historical request summary", "Request"),
            ("Recorded request summary", "Request"),
            ("Request summary and recorded controls", "Request"),
            ("Request summary", "Request"),
            ("request_summary", "request"),
            ("Saved request", "Request"),
            ("saved request", "request"),
            ("Recorded request", "Request"),
            ("recorded request", "request"),
            ("author-written", "explanatory"),
            ("historical", "recorded"),
            ("Historical", "Recorded"),
        ]:
            value = value.replace(old, new)
    return value
