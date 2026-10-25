"""Demo replay safety, economics and native Harbor artifact compatibility."""

import hashlib
import json
import subprocess
from pathlib import Path
from unittest.mock import Mock

import pytest

from experiments.configuration_paths import configuration_path
from experiments.launch import launch, verify_freeze
from experiments.presentation import details, table
from experiments.records import aggregate, harbor_record, load_runs, walkthrough

ROOT = Path(__file__).resolve().parents[1]


def sample(**updates):
    return (
        dict(
            kind="model_attempt",
            configuration_id="setting-a",
            model="model",
            inference_id="routing-a",
            task_id="task",
            success=False,
            calls=3,
            input_tokens=100,
            output_tokens=40,
            token_usage_complete=True,
            known_cost_usd=0.25,
            cost_usd=0.25,
            elapsed_seconds=10,
            classification="budget_exhausted",
        )
        | updates
    )


def test_failed_attempt_costs_count_and_controls_do_not():
    records = [
        sample(),
        sample(success=True, known_cost_usd=0.75, cost_usd=0.75),
        sample(kind="local_control", success=True, cost_usd=0, known_cost_usd=0),
    ]
    group = aggregate(records)[0]
    assert group["attempts"] == 2
    assert group["accepted_completions"] == 1
    assert group["total_spend_usd"] == 1
    assert group["cost_per_accepted_completion_usd"] == 1
    assert group["calls"] == 6


def test_unknown_costs_and_zero_success_are_not_free():
    group = aggregate([sample(), sample(success=True, cost_usd=None)])[0]
    assert group["known_spend_usd"] == 0.5
    assert group["total_spend_usd"] is None
    assert group["cost_per_accepted_completion_usd"] is None
    assert group["economics_status"] == "incomplete_cost_records"
    zero = aggregate([sample()])[0]
    assert zero["economics_status"] == "undefined_no_accepted_completions"
    assert zero["cost_per_accepted_completion_usd"] is None


def test_repeats_tasks_and_policies_never_merge_implicitly():
    records = [
        sample(),
        sample(task_id="task-two"),
        sample(configuration_id="setting-b"),
        sample(inference_id="routing-b"),
    ]
    assert len(aggregate(records)) == 4
    across = aggregate(records, across_tasks=True)
    assert len(across) == 3
    assert next(r for r in across if r["attempts"] == 2)["tasks"] == 2


def test_replay_never_launches_or_reads_secrets(monkeypatch, tmp_path):
    process = Mock(side_effect=AssertionError("Replay launched a process"))
    monkeypatch.setattr(subprocess, "run", process)
    result = launch(tmp_path, credentials=tmp_path / "nonexistent-secret")
    assert not result["launched"]
    process.assert_not_called()
    with pytest.raises(ValueError, match="confirmed"):
        launch(tmp_path, mode="execute")
    process.assert_not_called()


def frozen_project(tmp_path):
    (tmp_path / "configs").mkdir()
    config = configuration_path(tmp_path, "configs/harbor-model.json")
    config.write_text("{}")
    (configuration_path(tmp_path, "configs/harbor-v1.sha256")).write_text(
        hashlib.sha256(config.read_bytes()).hexdigest()
        + "  configs/harbor-model.json\n"
    )
    (
        configuration_path(
            tmp_path, "configs/LC-TIMEOUT-001.harbor-v1.frozen-setting.json"
        )
    ).write_text('{"setting_id":"test"}')
    (tmp_path / ".venv/bin").mkdir(parents=True)
    (tmp_path / ".venv/bin/harbor").touch()
    (tmp_path / "reports").mkdir()
    credentials = tmp_path / "secret.env"
    credentials.write_text("OPENROUTER_API_KEY=not-a-real-key\n")
    return credentials


def test_execute_uses_native_cli_and_prevents_accidental_repeat(monkeypatch, tmp_path):
    credentials = frozen_project(tmp_path)
    process = Mock(return_value=subprocess.CompletedProcess([], 0))
    monkeypatch.setattr(subprocess, "run", process)
    result = launch(
        tmp_path,
        mode="execute",
        confirmed=True,
        job_name="one",
        credentials=credentials,
    )
    assert result["launched"] and result["setting_id"] == "test"
    argv = process.call_args.args[0]
    assert argv[0].endswith("/.venv/bin/harbor") and argv[1] == "run"
    assert "not-a-real-key" not in json.dumps(argv)
    manifest = (tmp_path / "reports/launches/one.json").read_text()
    assert "not-a-real-key" not in manifest
    with pytest.raises(FileExistsError):
        launch(
            tmp_path,
            mode="execute",
            confirmed=True,
            job_name="one",
            credentials=credentials,
        )
    assert process.call_count == 1


def test_execute_rejects_frozen_drift_before_process(monkeypatch, tmp_path):
    credentials = frozen_project(tmp_path)
    (configuration_path(tmp_path, "configs/harbor-model.json")).write_text(
        '{"changed":true}'
    )
    process = Mock()
    monkeypatch.setattr(subprocess, "run", process)
    with pytest.raises(ValueError, match="drift"):
        launch(tmp_path, mode="execute", confirmed=True, credentials=credentials)
    process.assert_not_called()


def test_saved_timeout_results_preserve_failures_and_separate_diagnostics():
    records = load_runs(ROOT)
    run = next(r for r in records if r["run_id"] == "LC-TIMEOUT-001-dc59aa1a8c5f")
    assert run["calls"] == 6 and len(run["request_failures"]) == 1
    assert run["cost_usd"] is None and run["known_cost_usd"] > 0
    run = next(r for r in records if r["run_id"] == "LC-TIMEOUT-001-dc26d992137f")
    assert not run["success"] and run["submitted"]
    assert walkthrough(run)["diagnostics"]  # Never counted as another model attempt.
    assert all("diagnostics/" not in r["run_id"] for r in records)
    assert any(r["kind"] == "local_control" and r["success"] for r in records)


def test_harbor_incomplete_trace_cost_is_unknown_and_policy_ids_differ(tmp_path):
    folder = tmp_path / "job/trial"
    folder.mkdir(parents=True)
    config = {
        "task": {"path": "task"},
        "agent": {"kwargs": {"policy": {"budgets": {"model_calls": 30}}}},
    }
    (folder / "config.json").write_text(json.dumps(config))
    (folder / "result.json").write_text(
        json.dumps(
            {
                "task_name": "task",
                "task_checksum": "digest",
                "agent_result": {"metadata": {"model_requests": 1}, "cost_usd": 0.1},
                "exception_info": {"exception_type": "TransportError"},
            }
        )
    )
    original = harbor_record(folder)
    assert original["cost_usd"] is None
    assert not original["token_usage_complete"]
    assert original["classification"] == "infrastructure_failure"
    config["agent"]["kwargs"]["policy"]["budgets"]["model_calls"] = 31
    (folder / "config.json").write_text(json.dumps(config))
    assert harbor_record(folder)["configuration_id"] != original["configuration_id"]


def test_notebook_defaults_to_replay_and_views_escape_artifact_html():
    notebook = json.loads((ROOT / "main.ipynb").read_text())
    source = "\n".join(
        "".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code"
    )
    assert 'MODE = "replay"' in source
    assert "CONFIRM_PAID_EXECUTION = False" in source
    assert "load_runs(ROOT)" in source
    assert "<script>" not in details("x", "<script>")
    assert "&lt;script&gt;" in table([{"x": "<script>"}], ["x"])
    assert verify_freeze(ROOT)


def test_harbor_reward_cannot_override_missing_submission(tmp_path):
    folder = tmp_path / "job/trial"
    folder.mkdir(parents=True)
    (folder / "config.json").write_text(
        json.dumps({"task": {"path": "task"}, "agent": {"kwargs": {}}})
    )
    (folder / "result.json").write_text(
        json.dumps(
            {
                "task_name": "task",
                "verifier_result": {
                    "rewards": {
                        "reward": 1,
                        "patch_acceptance": 1,
                        "agent_submission": 0,
                    }
                },
            }
        )
    )
    assert harbor_record(folder)["success"] is False


def test_exploratory_spend_keeps_failed_pilots_and_labels_mixed_policies():
    from experiments.records import exploratory_economics

    rows = [
        sample(),
        sample(
            configuration_id="setting-b",
            success=True,
            known_cost_usd=0.75,
            cost_usd=0.75,
        ),
        sample(kind="local_control", success=True, cost_usd=0, known_cost_usd=0),
    ]
    assert len(aggregate(rows)) == 2  # Controlled comparisons stay separate.
    total = exploratory_economics(rows)[0]
    assert total["scope"] == "exploratory_lifetime_not_a_controlled_comparison"
    assert total["configuration_ids"] == ["setting-a", "setting-b"]
    assert total["attempts"] == 2
    assert total["total_spend_usd"] == 1
    assert total["cost_per_accepted_completion_usd"] == 1


def test_failed_format_responses_remain_in_core_attempt_cost_and_tokens():
    records = load_runs(ROOT)
    row = next(
        r
        for r in records
        if r["run_id"]
        == "annotated-tools-minimax-attempt-001/LC-ANNOTATED-TOOLS-001__CFC988s"
    )
    assert row["calls"] == 26
    assert row["output_tokens"] == 9872
    assert row["cost_usd"] == 0.03044406
    assert not row["success"] and not row["submitted"]
    assert row["termination_reason"] == "repair_round_budget_exhausted"
    assert row["task_python"] == "3.13.16"
    native = json.loads(Path(row["result_path"]).read_text())
    derived = native["agent_result"]["model_usage"]["minimax/minimax-m2.5"]
    assert derived["n_output_tokens"] < row["output_tokens"]
    assert derived["cost_usd"] < row["cost_usd"]
    assert row["failure_category"] == "destructive_edit_failure"
    assert row["contributing_factors"] == [
        "output_ceiling_tool_format_recovery_failure"
    ]
    assert row["tool_format_repair_rounds"] == 2
    assert row["code_repair_rounds"] is None
    native_trajectory = json.loads(
        (Path(row["artifact_dir"]) / "agent/mini-swe-agent.trajectory.json").read_text()
    )
    assert native_trajectory["info"]["model_stats"]["instance_cost"] == 0.02845695
    audit = json.loads(
        (
            ROOT
            / "reports/run-reviews"
            / "annotated-tools-minimax-attempt-001--LC-ANNOTATED-TOOLS-001__CFC988s.json"
        ).read_text()
    )
    assert audit["accounting_reconciliation"]["difference_usd"] == 0.00198711
    for relative, digest in audit["original_run_sha256"].items():
        assert (
            hashlib.sha256(
                (Path(row["artifact_dir"]) / relative).read_bytes()
            ).hexdigest()
            == digest
        )
