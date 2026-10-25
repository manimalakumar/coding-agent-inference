import json
import subprocess
from pathlib import Path
from unittest.mock import Mock

import pytest

from experiments.bookkeeping import audit_trial
from experiments.configuration_paths import configuration_path
from experiments.launch import launch
from experiments.records import events, load_runs
from experiments.report import write_reports
from tests.test_experiments import frozen_project

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "failure,status",
    [(OSError("local error"), "launch_failed"), (KeyboardInterrupt(), "interrupted")],
)
def test_failed_and_interrupted_launches_finalize_without_retry(
    monkeypatch, tmp_path, failure, status
):
    credentials = frozen_project(tmp_path)
    process = Mock(side_effect=failure)
    monkeypatch.setattr(subprocess, "run", process)
    with pytest.raises(type(failure)):
        launch(
            tmp_path,
            mode="execute",
            confirmed=True,
            credentials=credentials,
            job_name="broken",
        )
    assert process.call_count == 1
    manifest = json.loads((tmp_path / "reports/launches/broken.json").read_text())
    assert manifest["status"] == status
    assert manifest["bookkeeping"]["status"] == "incomplete"
    audit = json.loads((tmp_path / manifest["bookkeeping"]["audit"]).read_text())
    assert audit["model_requests_made_by_bookkeeping"] == 0
    row = json.loads((tmp_path / "reports/notebook-run-records.json").read_text())[
        "records"
    ][0]
    assert row["failure_category"] == "missing_run_artifacts"
    assert row["cost_usd"] is None and not row["token_usage_complete"]
    assert not row["success"] and row["patch_accepted"] is None


def test_normal_exit_refreshes_reports_even_when_harbor_returns_failure(
    monkeypatch, tmp_path
):
    credentials = frozen_project(tmp_path)
    monkeypatch.setattr(
        subprocess, "run", Mock(return_value=subprocess.CompletedProcess([], 1))
    )
    result = launch(
        tmp_path,
        mode="execute",
        confirmed=True,
        credentials=credentials,
        job_name="nonzero",
    )
    assert result["returncode"] == 1 and result["bookkeeping"]["status"] == "incomplete"
    assert (tmp_path / "reports/EXPERIMENT-RESULTS.md").is_file()
    assert (tmp_path / "reports/experiment-results.jsonl").is_file()
    assert list((tmp_path / "reports/bookkeeping/snapshots").glob("*.json"))


def test_missing_trial_placeholder_is_replaced_by_recovered_native_artifacts(tmp_path):
    credentials = frozen_project(tmp_path)
    launches = tmp_path / "reports/launches"
    launches.mkdir()
    (launches / "job.json").write_text(
        json.dumps(
            {
                "setting_id": "test",
                "status": "interrupted",
                "command": [
                    "harbor",
                    "-c",
                    str(configuration_path(tmp_path, "configs/harbor-model.json")),
                ],
            }
        )
    )
    records = load_runs(tmp_path)
    assert len(records) == 1 and records[0]["source"] == "harbor_launch"
    trial = tmp_path / "runs/harbor/job/trial"
    trial.mkdir(parents=True)
    (trial / "config.json").write_text(
        json.dumps({"task": {"path": "task"}, "agent": {"kwargs": {}}})
    )
    records = load_runs(tmp_path)
    assert len(records) == 1 and records[0]["source"] == "harbor"
    assert credentials.is_file()


def test_reconciliation_counts_raw_cost_and_recovers_final_denied_tool_observation():
    records = load_runs(ROOT)
    old = next(
        r for r in records if r["run_id"].endswith("LC-ANNOTATED-TOOLS-001__CFC988s")
    )
    audited = audit_trial(Path(old["artifact_dir"]), old)
    assert audited["status"] == "complete"
    assert audited["raw_minus_native_cost"] == 0.00198711
    assert audited["record"]["cost_usd"] == 0.03044406
    focused = next(
        r for r in records if r["run_id"].endswith("LC-ANNOTATED-TOOLS-001__UEiZNLt")
    )
    audited = audit_trial(Path(focused["artifact_dir"]), focused)
    assert audited["status"] == "complete"
    assert audited["native_tool_results_supplemented"] == 1
    assert not audited["missing_tool_results"]
    assert audited["record"]["patch_accepted"] and not audited["record"]["task_success"]


def test_live_partial_trace_is_tolerated_but_complete_corruption_is_not(tmp_path):
    trace = tmp_path / "trace.jsonl"
    trace.write_text('{"event":"model_request"}\n{"event":')
    assert events(trace) == [{"event": "model_request"}]
    trace.write_text('{"event":\n')
    with pytest.raises(json.JSONDecodeError):
        events(trace)


def test_reports_preserve_content_addressed_snapshots_and_active_study_defaults(
    tmp_path,
):
    write_reports(tmp_path)
    write_reports(tmp_path)
    assert len(list((tmp_path / "reports/bookkeeping/snapshots").glob("*.json"))) == 1
    catalog = json.loads((configuration_path(ROOT, "configs/study.json")).read_text())
    plan = launch(ROOT, mode="replay")
    assert catalog["studies"][catalog["active_study"]]["job_config"] in " ".join(
        plan["command"]
    )
    assert not plan["launched"]


@pytest.mark.parametrize("interrupted", [False, True])
def test_bookkeeping_failure_is_visible_and_never_masks_original_interrupt(
    monkeypatch, tmp_path, interrupted
):
    credentials = frozen_project(tmp_path)
    process = (
        Mock(side_effect=KeyboardInterrupt())
        if interrupted
        else Mock(return_value=subprocess.CompletedProcess([], 0))
    )
    monkeypatch.setattr(subprocess, "run", process)
    monkeypatch.setattr(
        "experiments.bookkeeping.finalize_job",
        Mock(side_effect=ValueError("bad artifacts")),
    )
    expected = KeyboardInterrupt if interrupted else RuntimeError
    with pytest.raises(expected):
        launch(
            tmp_path,
            mode="execute",
            confirmed=True,
            credentials=credentials,
            job_name="audit-error",
        )
    manifest = json.loads((tmp_path / "reports/launches/audit-error.json").read_text())
    assert manifest["bookkeeping"]["status"] == "failed"
    assert process.call_count == 1
