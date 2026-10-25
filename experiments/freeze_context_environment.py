"""Version and freeze zero-inference environment controls without launching models."""

import argparse
import copy
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import toml

from experiments.configuration_paths import configuration_path
from experiments.launch import verify_freeze
from sandbox.docker import digest

LABEL = "context-counter-environment-v2"
PARENT = "context-counter-glm5-calibration-v1"


def controls(project):
    project = Path(project)
    prep = json.loads((project / f"reports/{LABEL}-preparation.json").read_text())
    task = Path(prep["task_bundle"])
    config = toml.loads((task / "task.toml").read_text())
    config["verifier"]["environment"]["docker_image"] = prep["verifier_image"]
    (task / "task.toml").write_text(toml.dumps(config))
    original = json.loads(
        (configuration_path(project, f"configs/{PARENT}-model.json")).read_text()
    )
    for fixture in ["baseline", "reference"]:
        job = copy.deepcopy(original)
        job["job_name"] = LABEL + "-" + fixture + "-control-001"
        job["jobs_dir"] = str(project / "reports/harbor-validation")
        job["tasks"] = [{"path": str(task)}]
        job["agents"][0]["kwargs"]["fixture"] = fixture
        policy = job["agents"][0]["kwargs"]["policy"]
        policy["environment_revision"] = {
            "version": LABEL,
            "regression_location": "libs/langchain_v1/tests/unit_tests/test_*.py (direct children only)",
            "type_command": "mypy --cache-dir=/dev/null $CHANGED_SOURCE",
            "type_analysis": "unchanged full strict analysis; cache writes disabled",
            "paid_calibration_paused": True,
        }
        path = configuration_path(project, f"configs/{LABEL}-{fixture}.json")
        if path.exists():
            raise FileExistsError("Control configurations are never overwritten")
        path.write_text(json.dumps(job, indent=2) + "\n")
    print("Prepared two fixture-only Harbor jobs; model client is never constructed.")


def freeze(project):
    project = Path(project)
    parent_id = verify_freeze(
        project, f"configs/{PARENT}.sha256", f"configs/{PARENT}.frozen-setting.json"
    )
    parent = json.loads(
        (
            configuration_path(project, f"configs/{PARENT}.frozen-setting.json")
        ).read_text()
    )
    prep = json.loads((project / f"reports/{LABEL}-preparation.json").read_text())
    output = configuration_path(project, f"configs/{LABEL}.frozen-setting.json")
    if output.exists():
        raise FileExistsError("Never overwrite a frozen environment")
    summary = {}
    native_files = set()
    for fixture in ["baseline", "reference"]:
        job = project / "reports/harbor-validation" / f"{LABEL}-{fixture}-control-001"
        trial = next(job.glob("LC-*"))
        result = json.loads((trial / "result.json").read_text())
        checks = json.loads((trial / "verifier/checks.json").read_text())
        receipt = json.loads((trial / "agent/receipt.json").read_text())["receipt"]
        rewards = result["verifier_result"]["rewards"]
        assert result.get("exception_info") is None
        assert receipt["model_requests"] == 0 and receipt["reported_cost"] == 0
        assert rewards["task_success"] == (1 if fixture == "reference" else 0)
        assert checks["checks"]["file_policy"]["exit_code"] == 0
        if fixture == "reference":
            assert checks["checks"]["regression_on_baseline"]["exit_code"] == 1
            assert all(
                v["exit_code"] == 0
                for k, v in checks["checks"].items()
                if k != "regression_on_baseline"
            )
            assert len(checks["checks"]) == 9
            assert checks["checks"]["types"]["command"][1] == "--cache-dir=/dev/null"
        else:
            assert checks["checks"]["reproducer"]["exit_code"] == 1
            assert checks["checks"]["existing_tests"]["exit_code"] == 0
            assert checks["checks"]["hidden_acceptance"]["exit_code"] == 1
        summary[fixture] = {
            "run_id": str(trial.relative_to(project)),
            "status": "passed_control",
            "expected_task_success": fixture == "reference",
            "rewards": rewards,
            "model_requests": 0,
            "inference_cost_usd": 0,
            "checks": {k: v["exit_code"] for k, v in checks["checks"].items()},
        }
        native_files |= {
            str(p.relative_to(project)) for p in job.rglob("*") if p.is_file()
        }
    combined_path = f"reports/{LABEL}-combined-tool-control.json"
    combined = json.loads((project / combined_path).read_text())
    assert combined["returncode"] == 0 and combined["model_requests"] == 0
    assert combined["probe"]["status"] == "passed"
    reference_checks = json.loads(
        (project / summary["reference"]["run_id"] / "verifier/checks.json").read_text()
    )
    reg = combined["probe"]["regression_file"]
    assert reg in reference_checks["files_changed"]
    assert "/workspace/" + reg in reference_checks["checks"]["regression"]["command"]
    summary_report = {
        "setting": LABEL,
        "status": "passed",
        "model_requests": 0,
        "inference_cost_usd": 0,
        "combined_tool_control": combined_path,
        "permitted_regression_file": reg,
        "controls": summary,
        "baseline_and_reference_type_checks_passed_under_original_tool_limit": True,
        "acceptance_tests_unchanged": True,
        "old_model_attempts_reclassified": False,
        "paid_calibration_paused": True,
    }
    report = f"reports/{LABEL}-validation.json"
    (project / report).write_text(json.dumps(summary_report, indent=2) + "\n")
    task = Path(prep["task_bundle"])
    files = (
        set(parent["source_and_config_sha256"])
        | native_files
        | {
            report,
            combined_path,
            f"reports/{LABEL}-preparation.json",
            "experiments/prepare_context_environment.py",
            "experiments/context_environment_control.py",
            "experiments/freeze_context_environment.py",
            f"reports/{LABEL}-build-provenance.json",
        }
    )
    files |= {str(p.relative_to(project)) for p in task.rglob("*") if p.is_file()}
    files |= {
        f"configs/{LABEL}-{fixture}.json" for fixture in ["baseline", "reference"]
    }
    files |= {
        str(p.relative_to(project))
        for p in (project / "reports").glob(LABEL + "-mypy-*.log")
    }
    hashes = {
        name: hashlib.sha256((project / name).read_bytes()).hexdigest()
        for name in sorted(files)
    }
    policy = json.loads(
        (configuration_path(project, f"configs/{LABEL}-reference.json")).read_text()
    )["agents"][0]["kwargs"]["policy"]
    frozen = {
        **parent,
        "experimental_setting": LABEL,
        "parent_setting_id": parent_id,
        "frozen_at": datetime.now(UTC).isoformat(),
        "policy": policy,
        "source_and_config_sha256": hashes,
        "task_bundle": str(task),
        "task_bundle_reused_unchanged": False,
        "verifier_image": prep["verifier_image"],
        "parent_verifier_image": prep["required_parent_verifier_image"],
        "validation_report": report,
        "setup_model_requests": 0,
        "setup_reported_cost_usd": 0,
        "intervention": "Root-only new regression files; strict mypy with cache writes disabled; matching evaluator budget metadata. Dependencies, acceptance tests, source baseline, agent image, tool limits and total budgets unchanged.",
        "comparison_scope": "Zero-inference setup repair only; not an agent success or measured comparison. Original model verdicts and costs preserved.",
        "execution_status": "controls_only_no_paid_runs",
        "paid_calibration_paused": True,
        "regression_location": summary_report["permitted_regression_file"],
        "type_command": "mypy --cache-dir=/dev/null $CHANGED_SOURCE",
    }
    frozen.pop("setting_id", None)
    frozen.pop("compatibility_report", None)
    frozen["setting_id"] = digest(frozen)
    output.write_text(json.dumps(frozen, indent=2) + "\n")
    manifest = f"configs/{LABEL}.sha256"
    (project / manifest).write_text(
        "".join(f"{sha}  {name}\n" for name, sha in sorted(hashes.items()))
        + f"{hashlib.sha256(output.read_bytes()).hexdigest()}  {output.relative_to(project)}\n"
    )
    assert (
        verify_freeze(project, manifest, str(output.relative_to(project)))
        == frozen["setting_id"]
    )
    print(json.dumps({"setting_id": frozen["setting_id"], **summary_report}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["controls", "freeze"])
    args = parser.parse_args()
    globals()[args.phase](Path(__file__).resolve().parents[1])
