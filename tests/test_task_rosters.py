import json
from pathlib import Path

import nbformat
import pandas as pd

from experiments import notebook_workflow as w
from experiments import task_rosters as p

ROOT = Path(__file__).resolve().parents[1]


def test_all_five_development_tasks_have_recorded_links_metadata_and_outcomes():
    rows, audit = p.development_roster(ROOT, w.public_evidence(ROOT))
    assert len(rows) == 5 and len(audit) == 5
    assert {r["task_id"] for r in audit} == set(p.DEVELOPMENT_TASKS)
    assert rows["Issue / PR"].str.contains("github.com/langchain-ai/langchain/").all()
    assert rows["Outcome summary"].str.contains("original successes").all()
    assert rows.iloc[0]["Recorded difficulty"] == "not recorded"
    assert rows.iloc[2]["Recorded difficulty"] == "not_assigned"
    assert rows.iloc[1]["Recorded difficulty"] == "Level 1"
    assert (
        "1/3 original successes; 2 patches accepted" in rows.iloc[2]["Outcome summary"]
    )
    assert all((ROOT / r["metadata_source"]).is_file() for r in audit)


def test_feature_table_retained_and_startup_not_counted_as_paid():
    bundle = w.public_evidence(ROOT)
    activity, incidents = p.feature_activity(ROOT, bundle)
    assert activity["paid_attempts"] == 6
    assert activity["paid_attempts_by_model"] == {"MiniMax M2.5": 4, "GLM-5": 2}
    assert activity["recorded_environment_repair_model_calls"] == 0
    assert activity["startup_incidents_not_paid_attempts"] == 1
    assert activity["startup_model_calls"] == 0
    assert len(incidents) == 1 and incidents[0]["native_cost_usd"] is None
    view = json.loads((ROOT / "notebooks/tutorial-study.json").read_text())
    records = {r["run_id"]: r for r in w.records(ROOT)}
    table, audit = p.feature_transitions(records, view)
    assert len(table) == len(audit) == 6
    assert all(a["run_id"] != incidents[0]["run_id"] for a in audit)


def test_eight_heldout_tasks_reconcile_four_plus_thirty_six():
    bundle = w.public_evidence(ROOT)
    rows, audit, counts = p.heldout_roster(ROOT, bundle)
    assert len(rows) == len(audit) == 8
    assert counts == {"held_out_two": 4, "held_out_six": 36, "held_out_total": 40}
    first = [r for r in audit if r["scope"] == "held_out_two"]
    later = [r for r in audit if r["scope"] == "held_out_six"]
    assert len(first) == 2 and len(later) == 6
    assert all(r["difficulty_recorded"] == "L2" for r in first)
    assert all(r["difficulty_recorded"] == "not independently rated" for r in later)
    assert all(len(r["run_ids"]) == 2 for r in first)
    assert all(len(r["run_ids"]) == 6 for r in later)
    assert len({rid for task in audit for rid in task["run_ids"]}) == 40


def test_roster_counts_match_exact_original_task_model_records():
    bundle = w.public_evidence(ROOT)
    rows, audit, _ = p.heldout_roster(ROOT, bundle)
    for i, task in enumerate(audit):
        source = [
            r
            for r in p.scope_records(bundle, task["scope"])
            if r["task_id"] == task["task_id"]
        ]
        for model, label in p.MODELS.items():
            group = [r for r in source if r["model"] == model]
            assert (
                rows.iloc[i][label + " n / A / I"]
                == f"{len(group)} / {sum(r['patch_accepted'] is True for r in group)} / {sum(r['infrastructure_interrupted'] is True for r in group)}"
            )


def test_first_two_results_are_visible_and_keep_interruption_separate():
    results = p.first_two_results(w.public_evidence(ROOT))
    assert len(results) == 4 and results["Attempt"].eq(1).all()
    assert results["Patch verdict"].str.contains("interrupted").sum() == 1
    assert results["Patch verdict"].eq("not accepted").sum() == 3


def test_three_economics_scopes_remain_separate_with_unknown_charges():
    rows, audit = p.separate_economics(w.public_evidence(ROOT))
    assert len(rows) == 6
    assert set(rows.Scope) == {"development", "held_out_two", "held_out_six"}
    held = rows[rows.Scope != "development"]
    assert held["Paid attempts / launches"].sum() == 40
    dev = rows[rows.Scope == "development"]
    assert dev["Paid attempts / launches"].sum() == 14
    assert dev["Startup incidents"].sum() == 1
    assert dev["Cost / accepted patch"].eq("not pooled across configurations").all()
    assert rows[rows.Scope == "held_out_six"]["Complete spend USD"].eq("unknown").all()
    assert (
        rows[rows.Scope == "held_out_six"]["Cost / accepted patch"].eq("unknown").all()
    )
    assert set(r["scope"] for r in audit) == set(rows.Scope)
    assert all(r["run_ids"] and r["configuration_id"] for r in audit)


def test_planned_runs_are_not_added_to_any_roster_launch_count():
    bundle = w.public_evidence(ROOT)
    assert len(bundle.read("study.json")["planned_development"]["schedule"]) == 24
    assert (
        len(p.scope_records(bundle, "held_out_two"))
        + len(p.scope_records(bundle, "held_out_six"))
        == 40
    )
    assert len(p.scope_records(bundle, "development")) == 20


def test_html_escapes_problem_text_but_keeps_only_constructed_link():
    frame = pd.DataFrame(
        [
            {
                "Issue / PR": p._link(
                    "https://github.com/langchain-ai/langchain/issues/40205",
                    "Issue #40205",
                ),
                "Problem": "<script>alert(1)</script>",
            }
        ]
    )
    markup = p.roster_html(frame)
    assert "<a href=" in markup
    assert "<script>" not in markup
    assert "&lt;script&gt;" in markup


def test_notebook_keeps_model_call_cell_exact_and_adds_only_compact_views():
    n = nbformat.read(ROOT / "main.ipynb", as_version=4)
    cell = next(
        c for c in n.cells if "selected-first-invocation" in c.metadata.get("tags", [])
    )
    from tests.public_preservation import assert_preserved_first_cell

    assert_preserved_first_cell(ROOT)
    assert cell.outputs
    assert len(n.cells) == 37
    text = "\n".join(c.source for c in n.cells)
    for statement in [
        "presentation.development_roster(ROOT, bundle)",
        "presentation.heldout_roster(ROOT, bundle)",
        "presentation.first_two_results(bundle)",
        "presentation.separate_economics(bundle)",
    ]:
        assert statement in text
    assert "Selected transitions below show a client-value failure" not in text
