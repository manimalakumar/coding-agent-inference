"""Freeze GLM-5 calibration using the existing response-ceiling task/runtime."""

import copy
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

from experiments.configuration_paths import configuration_path
from experiments.launch import verify_freeze
from harbor_integration.host_response_ceiling import ResponseCeilingAgent
from sandbox.docker import digest

LABEL = "context-counter-glm5-calibration-v1"
PARENT = "context-counter-response-6000-v2"
COMPATIBILITY = "reports/compatibility/glm-5-z-ai-fp8-response6000-v1"


def prepare(project):
    project = Path(project)
    parent_id = verify_freeze(
        project, f"configs/{PARENT}.sha256", f"configs/{PARENT}.frozen-setting.json"
    )
    parent = json.loads(
        (
            configuration_path(project, f"configs/{PARENT}.frozen-setting.json")
        ).read_text()
    )
    path = configuration_path(project, f"configs/{LABEL}.frozen-setting.json")
    if path.exists():
        raise FileExistsError("Never overwrite a frozen calibration")
    job_path = f"configs/{LABEL}-model.json"
    job = json.loads((project / job_path).read_text())
    original = json.loads(
        (configuration_path(project, f"configs/{PARENT}-model.json")).read_text()
    )
    policy = job["agents"][0]["kwargs"]["policy"]
    model = json.loads(
        (configuration_path(project, "configs/glm-5-comparison-v1.json")).read_text()
    )
    assert policy["model"] == model
    check = copy.deepcopy(job)
    check["job_name"] = original["job_name"]
    check["agents"][0]["model_name"] = original["agents"][0]["model_name"]
    check["agents"][0]["kwargs"]["policy"]["model"] = original["agents"][0]["kwargs"][
        "policy"
    ]["model"]
    assert check == original, "Only model/provider/reasoning configuration may differ"
    ResponseCeilingAgent(
        policy=policy, logs_dir=Path("/tmp/glm-calibration-constructor-check")
    )
    compatibility = json.loads((project / COMPATIBILITY / "result.json").read_text())
    assert compatibility["status"] == "passed" and compatibility["task_ids_used"] == []
    assert compatibility["model_config"] == model
    events = [
        json.loads(line)
        for line in (project / COMPATIBILITY / "requests.jsonl")
        .read_text()
        .splitlines()
    ]
    requests = [e for e in events if e["event"] == "model_request"]
    assert len(requests) == 2
    assert all(e["body"]["max_tokens"] == 6000 for e in requests)
    assert all(e["body"].get("reasoning") == {"enabled": False} for e in requests)
    files = set(parent["source_and_config_sha256"]) | {
        job_path,
        "configs/glm-5-comparison.json",
        "experiments/glm_feature_calibration.py",
        "experiments/compatibility_ceiling.py",
    }
    files |= {
        str(p.relative_to(project))
        for p in (project / COMPATIBILITY).rglob("*")
        if p.is_file()
    }
    hashes = {
        f: hashlib.sha256((project / f).read_bytes()).hexdigest() for f in sorted(files)
    }
    frozen = {
        **parent,
        "experimental_setting": LABEL,
        "parent_setting_id": parent_id,
        "frozen_at": datetime.now(UTC).isoformat(),
        "policy": policy,
        "source_and_config_sha256": hashes,
        "intervention": "Fresh GLM-5 calibration; model/provider/reasoning policy changes only.",
        "comparison_scope": "Single feature-development calibration, same effective limits/instructions/resources/checks as the 6000-cap MiniMax attempt; not held-out or a reliability estimate.",
        "task_bundle_reused_unchanged": True,
        "setup_model_requests": compatibility["model_requests"],
        "compatibility_report": COMPATIBILITY + "/result.json",
        "setup_reported_cost_usd": compatibility["reported_cost_usd"],
        "previous_patch_or_diagnostic_feedback_supplied": False,
    }
    frozen.pop("setting_id", None)
    frozen["setting_id"] = digest(frozen)
    path.write_text(json.dumps(frozen, indent=2) + "\n")
    manifest = f"configs/{LABEL}.sha256"
    (project / manifest).write_text(
        "".join(f"{sha}  {f}\n" for f, sha in sorted(hashes.items()))
        + f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(project)}\n"
    )
    catalog_path = configuration_path(project, "configs/study.json")
    catalog = json.loads(catalog_path.read_text())
    catalog["studies"][LABEL] = {
        "task_ids": ["LC-CONTEXT-COUNTER-001"],
        "job_config": job_path,
        "freeze_manifest": manifest,
        "frozen_setting": str(path.relative_to(project)),
    }
    catalog_path.write_text(json.dumps(catalog, indent=2) + "\n")
    assert (
        verify_freeze(project, manifest, str(path.relative_to(project)))
        == frozen["setting_id"]
    )
    report = {
        "setting": LABEL,
        "setting_id": frozen["setting_id"],
        "parent_setting_id": parent_id,
        "model": model,
        "budgets": policy["budgets"],
        "only_inference_policy_changed": True,
        "same_task_instructions": True,
        "same_tools_images_permissions_checks": True,
        "new_snapshot_and_context": True,
        "prior_patch_or_feedback_supplied": False,
        "compatibility": compatibility,
        "task_bundle": frozen["task_bundle"],
    }
    (project / f"reports/{LABEL}-preparation.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    prepare(Path(__file__).resolve().parents[1])
