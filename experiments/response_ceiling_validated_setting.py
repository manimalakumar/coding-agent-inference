"""Freeze the 6000-token setting after a zero-inference runtime/evaluator control."""

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

from experiments.configuration_paths import configuration_path
from experiments.launch import verify_freeze
from sandbox.docker import digest

LABEL = "context-counter-response-6000-v2"


def prepare(project):
    project = Path(project)
    parent_label = "context-counter-response-6000-v1"
    parent_id = verify_freeze(
        project,
        f"configs/{parent_label}.sha256",
        f"configs/{parent_label}.frozen-setting.json",
    )
    parent = json.loads(
        (
            configuration_path(project, f"configs/{parent_label}.frozen-setting.json")
        ).read_text()
    )
    destination = configuration_path(project, f"configs/{LABEL}.frozen-setting.json")
    if destination.exists():
        raise FileExistsError("Never overwrite a frozen setting")
    job_path = f"configs/{LABEL}-model.json"
    job = json.loads((project / job_path).read_text())
    policy = job["agents"][0]["kwargs"]["policy"]
    stripped = json.loads(json.dumps(policy))
    for path in [
        "harbor_integration/host_response_ceiling.py",
        "configs/completion-response-6000.json",
    ]:
        stripped["code_fingerprints"].pop(path)
    assert stripped == parent["policy"], (
        "Only versioned constructor validation may change from v1"
    )
    control_job = (
        project
        / "reports/harbor-validation/context-counter-response-6000-v2-reference-budget-control"
    )
    results = list(control_job.glob("LC-*/result.json"))
    assert len(results) == 1
    result = json.loads(results[0].read_text())
    assert result["verifier_result"]["rewards"]["reward"] == 1.0, result
    assert result.get("exception_info") is None, result.get("exception_info")
    paths = set(parent["source_and_config_sha256"]) | {
        job_path,
        "experiments/response_ceiling_validated_setting.py",
        "harbor_integration/host_response_ceiling.py",
        "configs/completion-response-6000.json",
        "tests/test_response_ceiling_runtime.py",
        str(results[0].relative_to(project)),
    }
    files = {
        path: hashlib.sha256((project / path).read_bytes()).hexdigest()
        for path in sorted(paths)
    }
    frozen = {
        **parent,
        "experimental_setting": LABEL,
        "parent_setting_id": parent_id,
        "frozen_at": datetime.now(UTC).isoformat(),
        "policy": policy,
        "source_and_config_sha256": files,
        "startup_guard_note": "v1 stopped before inference; v2 versions only constructor policy validation. Runtime/setup/capture/evaluation inherited unchanged.",
        "zero_inference_capture_control": str(results[0].relative_to(project)),
    }
    frozen.pop("setting_id", None)
    frozen["setting_id"] = digest(frozen)
    destination.write_text(json.dumps(frozen, indent=2) + "\n")
    manifest = f"configs/{LABEL}.sha256"
    (project / manifest).write_text(
        "".join(f"{sha}  {path}\n" for path, sha in sorted(files.items()))
        + f"{hashlib.sha256(destination.read_bytes()).hexdigest()}  {destination.relative_to(project)}\n"
    )
    catalog_path = configuration_path(project, "configs/study.json")
    catalog = json.loads(catalog_path.read_text())
    catalog["studies"][LABEL] = {
        "task_ids": ["LC-CONTEXT-COUNTER-001"],
        "job_config": job_path,
        "freeze_manifest": manifest,
        "frozen_setting": str(destination.relative_to(project)),
    }
    catalog_path.write_text(json.dumps(catalog, indent=2) + "\n")
    assert (
        verify_freeze(project, manifest, str(destination.relative_to(project)))
        == frozen["setting_id"]
    )
    report = {
        "setting": LABEL,
        "setting_id": frozen["setting_id"],
        "parent_setting_id": parent_id,
        "budgets": policy["budgets"],
        "model": policy["model"],
        "constructor_and_budget_tests_passed": 2,
        "zero_inference_capture_control": str(results[0].relative_to(project)),
        "unchanged_agent_loop": True,
        "unchanged_isolation": True,
        "unchanged_checks_images_dependencies": True,
        "setup_model_requests": 0,
        "headroom_report": parent["headroom_report"],
    }
    (project / f"reports/{LABEL}-preparation.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    prepare(Path(__file__).resolve().parents[1])
