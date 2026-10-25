"""Display changes must not alter source evidence or experimental policy."""

import copy
import json
from pathlib import Path

import nbformat
import pytest

from experiments import notebook_presentation as p
from experiments import notebook_workflow as w
from experiments import task_rosters as old
from experiments.notebook_charts import cumulative_cost_points, request_axis_range

ROOT = Path(__file__).resolve().parents[1]


def test_tutorial_structure_and_public_first_cell_remain_preserved():
    from tests.public_preservation import assert_preserved_first_cell

    assert_preserved_first_cell(ROOT)
    notebook = nbformat.read(ROOT / "main.ipynb", 4)
    headings = "\n".join(c.source for c in notebook.cells if c.cell_type == "markdown")
    assert "## 1. One task end to end" in headings
    assert "Historical" in headings or "historical" in headings


def test_stack_reads_saved_provider_configuration_without_claiming_measured_precision():
    study = w.public_evidence(ROOT).read("study.json")
    before = copy.deepcopy(study)
    frame = p.stack_models(study)
    assert frame["Pinned provider"].tolist() == ["minimax/fp8", "z-ai/fp8"]
    assert frame["Precision"].eq("advertised precision: FP8").all()
    study["model_preparation"]["minimax/minimax-m2.5"]["provider"]["only"] = [
        "other/provider"
    ]
    assert p.stack_models(study).iloc[0]["Precision"] == "unknown"
    assert before["model_preparation"]["minimax/minimax-m2.5"]["provider"]["only"] == [
        "minimax/fp8"
    ]


def test_all_thirteen_author_ratings_are_separate_from_original_metadata():
    bundle = w.public_evidence(ROOT)
    expected = json.loads((ROOT / "notebooks/author-task-metadata.json").read_text())[
        "ratings"
    ]
    assert len(expected) == 13
    for build in [p.development_roster, p.heldout_roster]:
        result = build(ROOT, bundle)
        rows, audit = result[:2]
        for row, receipt in zip(rows.to_dict("records"), audit, strict=True):
            assert "Recorded difficulty" not in row
            assert (
                row["Author-assigned difficulty"]
                == receipt["author_assigned_difficulty"]
            )
            number = row["Issue / PR"].split('">')[0].rsplit("/", 1)[1]
            assert row["Author-assigned difficulty"] == expected[number]
            assert "difficulty_recorded" in receipt
    original, _ = old.development_roster(ROOT, bundle)
    assert original["Recorded difficulty"].tolist() == [
        "not recorded",
        "Level 1",
        "not_assigned",
        "Level 2",
        "not recorded",
    ]


def test_held_roster_preserves_all_forty_launches_and_uses_plain_labels():
    rows, audit, counts = p.heldout_roster(ROOT, w.public_evidence(ROOT))
    assert counts == {"held_out_two": 4, "held_out_six": 36, "held_out_total": 40}
    assert len({rid for r in audit for rid in r["run_ids"]}) == 40
    assert not any("n / A / I" in col for col in rows.columns)
    assert (
        rows["MiniMax M2.5 results"]
        .str.contains("launches; .* accepted; .* interrupted")
        .all()
    )


def test_economics_are_still_separate_with_only_display_scope_names_changed():
    bundle = w.public_evidence(ROOT)
    original, original_audit = old.separate_economics(bundle)
    display, display_audit = p.separate_economics(bundle)
    assert len(display) == 6
    assert display.drop(columns="Scope").equals(original.drop(columns="Scope"))
    assert display_audit == original_audit
    assert set(display.Scope) == {p.SCOPE_LABELS[s] for s in original.Scope}
    assert (
        p.spending_reconciliation(bundle)[1] == old.spending_reconciliation(bundle)[1]
    )


def test_context_uses_actual_requests_one_and_first_tool_feedback_and_native_usage():
    record = w.exact_record(ROOT, w.DEMO_RUN_ID)
    rows, added, receipt = p.context_demo(record, w.public_evidence(ROOT))
    assert rows["Request number"].tolist() == [1, 2]
    assert rows["Message count"].tolist() == [3, 5]
    assert rows["Provider-reported input tokens"].tolist() == [2923, 3118]
    assert any(m["role"] == "tool" for m in added)
    assert any(m["role"] == "assistant" for m in added)
    assert "usage.prompt_tokens" in receipt["input_tokens_source"]
    assert "estimate" in receipt["estimates_source"]


def test_context_missing_usage_remains_unknown_instead_of_becoming_estimates(
    monkeypatch,
):
    trace = [
        {
            "event": "model_request",
            "number": 1,
            "body": {"messages": [{"role": "user", "content": "task"}]},
        },
        {
            "event": "model_request",
            "number": 2,
            "body": {"messages": [{"role": "tool", "content": "failure"}]},
        },
    ]
    monkeypatch.setattr(w, "native_trace", lambda record: (None, trace))
    rows, _, _ = p.context_demo({"run_id": "not-saved"}, w.public_evidence(ROOT))
    assert rows["Provider-reported input tokens"].eq("unknown").all()
    assert rows["Estimated visible-payload tokens (estimate)"].eq("unknown").all()


@pytest.mark.parametrize(
    "missing",
    ["reported_cost_usd", "cumulative_reported_cost_usd", "cumulative_cost_complete"],
)
def test_exact_cumulative_cost_never_bridges_an_unknown_charge(missing):
    points = [
        {
            "reported_cost_usd": 1,
            "cumulative_reported_cost_usd": i,
            "cumulative_cost_complete": True,
        }
        for i in [1, 2, 3]
    ]
    points[1][missing] = None
    before = copy.deepcopy(points)
    assert cumulative_cost_points(points) == [1, None, None]
    assert points == before


def test_chart_receipts_labels_percentages_budgets_and_no_optional_notebook_plot():
    root, receipt = p.assets(ROOT)
    outcome = (root / "main-outcomes.svg").read_text()
    assert "2/18 accepted (11.1%)" in outcome
    assert "11/18 accepted (61.1%)" in outcome
    assert [
        [r[k] for k in ["Accepted", "Unsuccessful", "Interrupted"]]
        for r in receipt["outcome_counts"]
    ] == [[2, 15, 1], [11, 5, 2]]
    distributions = (root / "distributions.svg").read_text()
    assert "Budget: 30" in distributions and "Budget: 10,000" in distributions
    assert (
        "Input tokens per request, including cached tokens"
        in (root / "matched-context.svg").read_text()
    )
    assert (
        "Cumulative reported inference cost (USD)"
        in (root / "matched-cost.svg").read_text()
    )
    assert receipt["model_colors"] == p.MODEL_COLORS
    code = "\n".join(
        c.source
        for c in nbformat.read(ROOT / "main.ipynb", 4).cells
        if c.cell_type == "code"
    )
    assert "optional-minimax" not in code
    assert "notebook_presentation as presentation" in code


def test_checks_display_removes_scope_without_changing_results():
    record = w.exact_record(ROOT, w.DEMO_RUN_ID)
    original = w.checks_frame(record, split="development")
    display = p.checks_frame(record, split="development")
    assert "scope" not in display.columns
    assert display.equals(original.drop(columns="scope"))


def test_trajectory_axes_use_observed_range_within_saved_limit():
    assert request_axis_range([1, 2, 29, 30], 30) == (1, 30)
    assert request_axis_range([1, 2, 7], 30) == (1, 7)
    with pytest.raises(ValueError):
        request_axis_range([1, 31], 30)
    _, receipt = p.assets(ROOT)
    assert receipt["trajectory_request_range"] == [1, 30]
    assert receipt["trajectory_request_limit"] == 30


def test_results_tables_keep_full_descriptions_and_issue_numbers_under_short_display_options():
    import pandas as pd

    bundle = w.public_evidence(ROOT)
    first_two = p.first_two_results(bundle)
    six, _ = p.acceptance_tables(bundle)
    with pd.option_context("display.max_colwidth", 10):
        for frame in [first_two, six]:
            output = p.results_table_html(frame)
            for description in frame.Task:
                assert description in output
                assert description.startswith("Issue #")
            assert "white-space:normal" in output
    notebook = nbformat.read(ROOT / "main.ipynb", 4)
    code = "\n".join(c.source for c in notebook.cells if c.cell_type == "code")
    assert "display(per_task_acceptance)" not in code
    assert "display(presentation.first_two_results(bundle))" not in code
