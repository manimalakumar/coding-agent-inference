"""Freeze exactly one fresh attempt per model against the repaired feature setup."""

import copy
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

from experiments.configuration_paths import configuration_path
from experiments.launch import verify_freeze
from harbor_integration.host_response_ceiling import ResponseCeilingAgent
from sandbox.docker import digest

LABEL = "context-counter-repaired-calibration-v1"
PARENT = "context-counter-environment-v2"


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
    output = configuration_path(project, f"configs/{LABEL}.frozen-setting.json")
    if output.exists():
        raise FileExistsError("Never overwrite a calibration freeze")
    validation = json.loads((project / parent["validation_report"]).read_text())
    assert validation["status"] == "passed" and validation["model_requests"] == 0
    original = json.loads(
        (configuration_path(project, f"configs/{PARENT}-reference.json")).read_text()
    )
    model_settings = json.loads(
        (
            configuration_path(project, "configs/comparison-4x2x3-v1.plan.json")
        ).read_text()
    )["models"]
    jobs = {}
    paths = set(parent["source_and_config_sha256"]) | {
        "experiments/repaired_feature_calibration.py"
    }
    policies = {}
    for model_key, model in model_settings.items():
        job = copy.deepcopy(original)
        job["job_name"] = f"{LABEL}-{model_key}-attempt-001"
        job["jobs_dir"] = str(project / "runs/harbor")
        spec = job["agents"][0]
        spec["kwargs"].pop("fixture")
        spec["model_name"] = "openrouter/" + model["model"]
        policy = spec["kwargs"]["policy"]
        policy["model"] = model
        policy["environment_revision"]["paid_calibration_paused"] = False
        policy["environment_revision"]["authorized_scope"] = (
            "One fresh calibration per model; stop after both outcomes; no automatic repair or retry."
        )
        ResponseCeilingAgent(
            policy=policy,
            logs_dir=Path("/tmp/repaired-calibration-constructor-" + model_key),
        )
        policies[model_key] = policy
        config = f"configs/{LABEL}-{model_key}-model.json"
        (project / config).write_text(json.dumps(job, indent=2) + "\n")
        jobs[model_key] = {
            "study_id": LABEL + "-" + model_key,
            "job_config": config,
            "job_name": job["job_name"],
        }
        paths.add(config)
    a = copy.deepcopy(policies["minimax"])
    b = copy.deepcopy(policies["glm5"])
    a.pop("model")
    b.pop("model")
    assert a == b, "Both models must receive the same effective policy"
    frozen = {
        **parent,
        "experimental_setting": LABEL,
        "parent_setting_id": parent_id,
        "frozen_at": datetime.now(UTC).isoformat(),
        "models": model_settings,
        "policy": policies["glm5"],
        "shared_non_model_policy": a,
        "jobs": jobs,
        "source_and_config_sha256": {
            p: hashlib.sha256((project / p).read_bytes()).hexdigest()
            for p in sorted(paths)
        },
        "execution_status": "two_fresh_attempts_authorized_not_started",
        "paid_calibration_paused": False,
        "authorized_measured_attempts": 2,
        "automatic_retries": 0,
        "prior_patch_or_diagnostic_feedback_supplied": False,
        "intervention": "One fresh calibration per model using the exact repaired environment, task instructions, tools and effective limits.",
        "comparison_scope": "Two feature-development calibration observations, not the four-task comparison or held-out evaluation. Stop and assess after both, regardless of success.",
    }
    frozen.pop("setting_id", None)
    frozen["setting_id"] = digest(frozen)
    output.write_text(json.dumps(frozen, indent=2) + "\n")
    manifest = f"configs/{LABEL}.sha256"
    (project / manifest).write_text(
        "".join(
            f"{sha}  {p}\n"
            for p, sha in sorted(frozen["source_and_config_sha256"].items())
        )
        + f"{hashlib.sha256(output.read_bytes()).hexdigest()}  {output.relative_to(project)}\n"
    )
    catalog_path = configuration_path(project, "configs/study.json")
    catalog = json.loads(catalog_path.read_text())
    for key, job in jobs.items():
        catalog["studies"][job["study_id"]] = {
            "task_ids": ["LC-CONTEXT-COUNTER-001"],
            "job_config": job["job_config"],
            "freeze_manifest": manifest,
            "frozen_setting": str(output.relative_to(project)),
        }
    catalog_path.write_text(json.dumps(catalog, indent=2) + "\n")
    assert (
        verify_freeze(project, manifest, str(output.relative_to(project)))
        == frozen["setting_id"]
    )
    report = {
        "setting": LABEL,
        "setting_id": frozen["setting_id"],
        "parent_environment_id": parent_id,
        "models": model_settings,
        "budgets": a["budgets"],
        "same_non_model_policy": True,
        "task_bundle_reused_unchanged": parent["task_bundle"],
        "verifier_image": parent["verifier_image"],
        "zero_inference_validation_reused": parent["validation_report"],
        "new_setup_model_requests": 0,
        "jobs": jobs,
        "fresh_snapshot_and_empty_context": True,
        "prior_patch_diagnostics_hidden_feedback_supplied": False,
    }
    (project / f"reports/{LABEL}-preparation.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    prepare(Path(__file__).resolve().parents[1])
