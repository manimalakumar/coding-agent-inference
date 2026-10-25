import copy
import json
from decimal import Decimal
from pathlib import Path

import nbformat
import pytest

from experiments import notebook_workflow as w
from experiments import study_views as p
from experiments.configuration_paths import configuration_path

ROOT = Path(__file__).resolve().parents[1]


def test_main_comparison_is_exact_six_task_block():
    bundle = w.public_evidence(ROOT)
    held = p.held_frame(bundle)
    task, overall = p.acceptance_tables(bundle)
    assert len(held) == 36 and len(task) == 6
    assert held.groupby(["task_id", "model"]).size().eq(3).all()
    assert overall["Accepted / launched"].tolist() == ["2/18", "11/18"]
    assert overall["Interruptions"].tolist() == [1, 2]
    assert p.distribution_table(bundle)["Attempts (failures retained)"].tolist() == [
        18,
        18,
    ]


def test_spending_headline_matches_all_disjoint_native_accounting_scopes():
    bucket, summary = p.spending_reconciliation(w.public_evidence(ROOT))
    assert summary["known_operational_spend_usd"] == "2.76420370"
    assert sum(Decimal(v) for v in bucket["Known spend USD"]) == Decimal("2.76420370")
    assert summary["complete_operational_spend_usd"] is None
    assert len(bucket) == 5
    assert (
        bucket[bucket.Scope == "held_out_six"]["Known spend USD"].iloc[0]
        == "1.91816912"
    )


def test_paths_are_relative_only_in_display_projection():
    original = {
        "path": str(configuration_path(ROOT, "configs/study.json")),
        "other": str(ROOT.parent / "private/a.json"),
    }
    before = copy.deepcopy(original)
    view = p.relative_paths(ROOT, original)
    assert view == {"path": "configs/study.json", "other": "../private/a.json"}
    assert original == before
    assert "Codex" not in p.details(ROOT, "audit", original)


def test_setup_labels_and_actual_expected_outputs():
    prepared = w.preparation(ROOT)
    controls = p.controls(prepared)
    assert len(controls) == 4 and controls["Model calls"].eq(0).all()
    assert controls.Status.eq("matches expectation").all()
    from experiments.notebook_presentation import issue_outputs

    outputs, receipt = issue_outputs(ROOT, prepared)
    assert outputs.iloc[0]["Actual, unchanged code"] == "missing"
    assert outputs.iloc[0]["Expected"] == "Fragments to evaluate"
    assert receipt["recorded_exit_code"] == 1


def test_messages_have_readable_roles_and_escape_html():
    markup = p.messages_html(
        ROOT,
        [
            {"role": "system", "content": "safe"},
            {"role": "user", "content": "issue"},
            {"role": "user", "content": "budget"},
            {"role": "tool", "content": "<script>bad</script>"},
        ],
    )
    for heading in ["System", "Task", "Budget", "Tool result"]:
        assert "<h4>" + heading + "</h4>" in markup
    assert "<script>" not in markup


def test_previewed_diff_keeps_production_visible_and_full_patch_unchanged():
    patch = "diff --git a/src/a.py b/src/a.py\n+fix\ndiff --git a/tests/test_x.py b/tests/test_x.py\n+new test\n"
    assert "+fix" in p.production_diff(patch)
    assert "+new test" not in p.production_diff(patch)
    assert "+new test" in patch


def test_fresh_first_invocation_display_is_archived_exclusively(monkeypatch, tmp_path):
    original = {
        "request": {"model": "test"},
        "response": {"id": "generation-1"},
        "trace_sha256": "abc",
    }
    monkeypatch.setattr(w, "first_exchange", lambda record: original)
    record = {"run_id": "fresh/task"}
    assert "Replay" in p.preserve_invocation(tmp_path, record, "html", "replay")
    assert not list(tmp_path.iterdir())
    relative = p.preserve_invocation(tmp_path, record, "actual display", "execute")
    path = tmp_path / relative
    before = path.read_bytes()
    assert json.loads(before)["first_response"] == original["response"]
    assert (
        p.preserve_invocation(tmp_path, record, "actual display", "execute") == relative
    )
    with pytest.raises(RuntimeError, match="refusing to overwrite"):
        p.preserve_invocation(tmp_path, record, "different", "execute")
    assert path.read_bytes() == before


def test_selected_run_exchange_and_matching_plot_selections():
    native = {r["run_id"]: r for r in w.records(ROOT)}
    assert native[p.MATCHED_RUNS[0]]["patch_accepted"] is False
    assert native[p.MATCHED_RUNS[1]]["patch_accepted"] is True
    assert all(r.split("/")[0].endswith("attempt-001") for r in p.MATCHED_RUNS)
    root, manifest = p.assets(ROOT)
    assert manifest["matched_runs"] == p.MATCHED_RUNS
    assert manifest["matched_task"] == "LC-INT-TEMPERATURE-001"
    assert (
        "MiniMax: accepted versus failed attempts on the same task"
        in (root / "optional-minimax.svg").read_text()
    )


def test_notebook_has_one_live_selected_exchange_and_no_removed_subsections():
    n = nbformat.read(ROOT / "main.ipynb", as_version=4)
    selected = [
        c for c in n.cells if "selected-first-invocation" in c.metadata.get("tags", [])
    ]
    assert len(selected) == 1 and "run)" in selected[0].source
    assert "skip-execution" not in selected[0].metadata.get("tags", [])
    source = "\n".join(c.source for c in n.cells)
    for removed in [
        "fixed_invocation",
        "Further held-out evaluation",
        "Development plan stays separate",
        "suggested extensions",
        "for saved in all_records",
    ]:
        assert removed not in source
    assert "workflow.first_exchange(by_id[workflow.DEMO_RUN_ID])" not in source
    from tests.public_preservation import assert_preserved_first_cell

    assert_preserved_first_cell(ROOT)
