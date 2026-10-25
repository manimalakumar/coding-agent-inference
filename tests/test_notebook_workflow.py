"""Zero-inference controls for the notebook's real Harbor entry point."""

import hashlib
from pathlib import Path
from unittest.mock import Mock

import nbformat
import pytest

from experiments import notebook_workflow as w
from experiments.configuration_paths import configuration_path

ROOT = Path(__file__).resolve().parents[1]


def test_replay_does_not_prepare_or_launch(monkeypatch, tmp_path):
    record = {"run_id": "fixed"}
    monkeypatch.setattr(w, "exact_record", lambda project, run_id: record)
    forbidden = Mock(side_effect=AssertionError("Replay must not execute"))
    monkeypatch.setattr(w, "preparation", forbidden)
    monkeypatch.setattr(w, "preflight_execute", forbidden)
    monkeypatch.setattr(w, "launch", forbidden)
    result = w.execute_or_replay(tmp_path, saved_run_id="fixed")
    assert result == {"mode": "replay", "launched": False, "record": record}
    forbidden.assert_not_called()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"mode": "bad"},
        {"mode": "execute"},
        {"mode": "execute", "confirmed": True},
        {"mode": "execute", "confirmed": True, "job_name": "../escape"},
    ],
)
def test_invalid_execution_never_prepares_or_launches(monkeypatch, tmp_path, kwargs):
    forbidden = Mock(side_effect=AssertionError("Invalid execution must not launch"))
    monkeypatch.setattr(w, "preparation", forbidden)
    monkeypatch.setattr(w, "launch", forbidden)
    with pytest.raises(ValueError):
        w.execute_or_replay(tmp_path, **kwargs)
    forbidden.assert_not_called()


@pytest.mark.parametrize("relative", ["reports/launches/once.json", "runs/harbor/once"])
def test_existing_attempt_cannot_spend_again(monkeypatch, tmp_path, relative):
    path = tmp_path / relative
    path.parent.mkdir(parents=True)
    path.touch()
    forbidden = Mock(side_effect=AssertionError("No repeat inference"))
    monkeypatch.setattr(w, "launch", forbidden)
    with pytest.raises(FileExistsError):
        w.execute_or_replay(tmp_path, mode="execute", confirmed=True, job_name="once")
    forbidden.assert_not_called()


def test_execute_routes_once_to_validated_public_launcher(monkeypatch, tmp_path):
    import json

    from public_preparation import launch as public_launch

    (tmp_path / "configs").mkdir()
    (configuration_path(tmp_path, "configs/study.json")).write_text(
        json.dumps({"studies": {}})
    )
    record = {"run_id": "fresh/native-trial"}
    launcher = Mock(return_value={"launched": True, "record": record})
    monkeypatch.setattr(public_launch, "launch", launcher)
    result = w.execute_or_replay(
        tmp_path,
        mode="execute",
        confirmed=True,
        study_id="public:LC-INT-TEMPERATURE-001",
        job_name="fresh",
    )
    assert result["record"] is record
    launcher.assert_called_once_with(
        tmp_path,
        "LC-INT-TEMPERATURE-001",
        model="minimax",
        confirmed=True,
        job_name="fresh",
        credentials=None,
    )


def test_setup_failure_prevents_inference(monkeypatch, tmp_path):
    import json

    from public_preparation import launch as public_launch

    (tmp_path / "configs").mkdir()
    (configuration_path(tmp_path, "configs/study.json")).write_text(
        json.dumps({"studies": {}})
    )
    process = Mock(side_effect=AssertionError("Unprepared task must not execute"))
    monkeypatch.setattr(public_launch.subprocess, "run", process)
    with pytest.raises(
        RuntimeError, match="public preparation controls have not passed"
    ):
        w.execute_or_replay(
            tmp_path,
            mode="execute",
            confirmed=True,
            study_id="public:LC-INT-TEMPERATURE-001",
            job_name="fresh",
        )
    process.assert_not_called()


def test_launch_error_is_not_retried(monkeypatch, tmp_path):
    import json

    from public_preparation import launch as public_launch

    (tmp_path / "configs").mkdir()
    (configuration_path(tmp_path, "configs/study.json")).write_text(
        json.dumps({"studies": {}})
    )
    launcher = Mock(side_effect=RuntimeError("record preserved"))
    monkeypatch.setattr(public_launch, "launch", launcher)
    with pytest.raises(RuntimeError, match="record preserved"):
        w.execute_or_replay(
            tmp_path,
            mode="execute",
            confirmed=True,
            study_id="public:LC-INT-TEMPERATURE-001",
            job_name="fresh",
        )
    launcher.assert_called_once()


def test_public_first_invocation_display_preserved():
    from tests.public_preservation import assert_preserved_first_cell

    assert_preserved_first_cell(ROOT)


def test_saved_exchange_and_cost_come_from_reviewed_public_evidence():
    from experiments import notebook_presentation as presentation

    record = w.exact_record(ROOT, w.DEMO_RUN_ID)
    path = ROOT / "evidence/tutorial.json"
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    first = w.first_exchange(record)
    assert first["request"]["model"] == "minimax/minimax-m2.5"
    assert (
        "Historical request summary"
        in first["request"]["historical_request_summary"]["label"]
    )
    assert first["response"]["id"]
    context, _, _ = presentation.context_demo(record, w.public_evidence(ROOT))
    assert (
        context.iloc[0]["Provider-reported input tokens"]
        == first["response"]["usage"]["prompt_tokens"]
    )
    ledger, summary = w.accounting(ROOT, record)
    assert len(ledger) == 21
    assert summary["complete reported spend USD"] == 0.02148405
    assert summary["cost complete"] is True
    assert "Historical request summary" in presentation.exchange_html(ROOT, record)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before


def test_zero_request_incident_is_not_fabricated():
    record = next(
        r for r in w.records(ROOT) if r["run_id"].endswith("/unrecorded-trial")
    )
    assert w.first_exchange(record)["request"] is None
    assert "no request recorded" in w.first_exchange(record)["status"]


def test_unknown_request_cost_is_not_zero(monkeypatch, tmp_path):
    from experiments import openrouter_accounting

    monkeypatch.setattr(
        openrouter_accounting,
        "request_ledger",
        lambda *args: [
            {
                "run_id": "partial",
                "reported_cost_usd": None,
                "effective_cost_usd": None,
                "accounting_conflict": False,
                "usage": None,
            }
        ],
    )
    _, summary = w.accounting(
        tmp_path,
        {
            "run_id": "partial",
            "calls": 1,
            "patch_accepted": True,
            "explicit_submission": False,
        },
    )
    assert summary["cost complete"] is False
    assert summary["complete reported spend USD"] is None
    assert summary["cost per accepted patch USD"] is None


def test_initial_request_preview_matches_native_without_old_context(monkeypatch):
    prepared = w.preparation(ROOT)
    forbidden = Mock(
        side_effect=AssertionError(
            "Preview must not read a prior run or invoke a provider"
        )
    )
    with monkeypatch.context() as patcher:
        patcher.setattr(w, "first_exchange", forbidden)
        patcher.setattr(w, "native_trace", forbidden)
        patcher.setattr(w, "launch", forbidden)
        preview = w.request_preview(ROOT, prepared)
    actual = w.first_exchange(w.exact_record(ROOT, w.DEMO_RUN_ID))["request"]
    assert preview == actual
    assert not any(m["role"] in {"assistant", "tool"} for m in preview["messages"])
    forbidden.assert_not_called()


def test_milestones_follow_observed_commands_not_call_numbers(monkeypatch):
    trace = []
    for number, command in [
        (2, "python /task/reproduce.py"),
        (6, "python - <<'PY'\np.write_text('change')\nPY"),
        (7, "python -m pytest test_new.py -q"),
        (9, "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"),
    ]:
        trace += [
            {"event": "model_request", "number": number},
            {
                "event": "tool_result",
                "tool_call": {"command": command},
                "result": {"returncode": 0, "output": "observed"},
            },
        ]
    monkeypatch.setattr(w, "native_trace", lambda r: (None, trace))
    selected = w.milestone_actions({})
    assert selected["request"].tolist() == [2, 6, 7, 9]
    assert "last observed action" in selected.iloc[-1]["milestone (heuristic)"]
    assert "editing command" in selected.iloc[1]["milestone (heuristic)"]


def test_historical_context_keeps_counts_and_labels_summarized_observation():
    from experiments import notebook_presentation as presentation

    record = w.exact_record(ROOT, w.DEMO_RUN_ID)
    frame, added, receipt = presentation.context_demo(record, w.public_evidence(ROOT))
    assert frame["Request number"].tolist() == [1, 2]
    assert frame["Message count"].tolist() == [3, 5]
    assert frame["Provider-reported input tokens"].tolist() == [2923, 3118]
    assert any(m["role"] == "assistant" and m["tool_calls"] for m in added)
    assert any(
        m["role"] == "tool" and "Author explanation:" in m["content"] for m in added
    )
    assert receipt["added_messages"] == added


@pytest.mark.parametrize("requests", [0, 1])
def test_missing_or_single_request_remains_inspectable(monkeypatch, tmp_path, requests):
    trace = (
        [
            {
                "event": "model_request",
                "number": 1,
                "body": {"messages": [{"role": "user", "content": "issue"}]},
            }
        ]
        if requests
        else []
    )
    monkeypatch.setattr(
        w, "native_trace", lambda r: (tmp_path / "missing.jsonl", trace)
    )
    record = {
        "artifact_dir": str(tmp_path),
        "source": "harbor",
        "run_id": "partial",
        "calls": requests,
        "model": "test",
        "classification": "error",
        "patch_accepted": None,
        "explicit_submission": False,
    }
    assert w.context_change(record)["added_messages"] == []
    assert w.milestone_actions(record).empty
    assert w.checks_frame(record).iloc[0]["status"] == "not run / not recorded"
    assert "not recorded" in w.candidate_evidence(record)["patch"]
    assert "Acceptance is unknown" in w.outcome_learning(record)
    assert w.first_exchange(record)["response"] is None


def test_rejected_or_stopped_outcome_does_not_claim_success():
    assert "not accepted" in w.outcome_learning(
        {"patch_accepted": False, "termination_reason": "budget_exhausted"}
    )
    assert "one observation" in w.outcome_learning({"patch_accepted": True})


def test_timing_medians_require_valid_audit_and_no_interruption(tmp_path):
    audits = tmp_path / "reports/run-reviews"
    audits.mkdir(parents=True)
    attempts = []
    for name, valid, interrupted, seconds in [
        ("good", True, False, 12),
        ("bad", False, False, 900),
        ("interrupted", True, True, 999),
        ("unknown", None, False, 3),
        ("negative", True, False, -1),
    ]:
        (audits / (name + ".json")).write_text(
            __import__("json").dumps(
                {"timing_audit": {"controlled_completion_time_valid": valid}}
            )
        )
        attempts.append(
            {
                "run_id": name,
                "model": "test",
                "task_id": "test",
                "infrastructure_interrupted": interrupted,
                "elapsed_seconds": seconds,
            }
        )
    timing = w.timing_eligibility(tmp_path, attempts)
    eligible = timing[timing["timing_status"] == "eligible non-interrupted"]
    assert eligible["run_id"].tolist() == ["good"]
    assert eligible["eligible_agent_seconds"].median() == 12
    assert (
        timing[timing["infrastructure_interrupted"]]["eligible_agent_seconds"]
        .isna()
        .all()
    )


def test_real_heldout_timing_exclusions_are_applied():
    from public_replay.analysis import measured_attempts

    bundle = w.public_evidence(ROOT)
    ids = set(bundle.read("study.json")["scopes"]["held_out_six"]["run_ids"])
    attempts = [r for r in measured_attempts(bundle) if r["run_id"] in ids]
    timing = w.timing_eligibility(ROOT, attempts)
    assert len(timing) == 36
    assert timing["infrastructure_interrupted"].sum() == 3
    assert timing["timing_status"].eq("clock invalid").sum() == 2
    assert timing["timing_status"].eq("eligible non-interrupted").sum() == 31
    assert (
        timing.loc[timing["clock_valid"].eq(False), "eligible_agent_seconds"]
        .isna()
        .all()
    )


def test_live_section_one_cells_handle_no_request_outcome(tmp_path, monkeypatch):
    import pandas as pd
    from IPython.display import HTML

    from experiments import notebook_presentation as presentation
    from experiments.tutorial import patch_html

    monkeypatch.setattr(
        presentation, "preserve_invocation", lambda *a: "test: archived"
    )

    record = {
        "artifact_dir": str(tmp_path),
        "source": "harbor",
        "run_id": "startup-failure",
        "kind": "model_attempt",
        "calls": 0,
        "calls_complete": False,
        "model": "test",
        "classification": "infrastructure_failure",
        "patch_accepted": None,
        "explicit_submission": False,
        "termination_reason": "error",
    }
    namespace = {
        "workflow": w,
        "presentation": presentation,
        "run": record,
        "ROOT": ROOT,
        "MODE": "execute",
        "preview": {},
        "display": Mock(),
        "HTML": HTML,
        "pd": pd,
        "patch_html": patch_html,
        "bundle": w.public_evidence(ROOT),
    }
    notebook = nbformat.read(ROOT / "main.ipynb", as_version=4)
    starts = (
        "exchange = workflow.first_exchange(run)",
        "context_requests, added_messages, context_receipt = presentation.context_demo(run, bundle)",
        "candidate = workflow.candidate_evidence(run)",
        "ledger, attempt_economics = workflow.accounting(ROOT, run)",
    )
    cells = [
        c
        for c in notebook.cells
        if c.cell_type == "code" and c.source.startswith(starts)
    ]
    assert len(cells) == 4
    for cell in cells:
        exec(compile(cell.source, "<live-section-one>", "exec"), namespace)
    assert namespace["attempt_economics"]["cost complete"] is False
    assert namespace["milestones"].empty
