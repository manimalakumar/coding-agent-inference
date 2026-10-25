import json
from pathlib import Path

import nbformat
import pytest

from experiments import intervention_views as p
from experiments import notebook_workflow as w

ROOT = Path(__file__).resolve().parents[1]


def test_first_model_cell_is_identical_including_outputs():
    from tests.public_preservation import assert_preserved_first_cell

    assert_preserved_first_cell(ROOT)


def test_actual_demo_test_creation_and_verification_are_distinct():
    r = w.exact_record(ROOT, w.DEMO_RUN_ID)
    from experiments.notebook_presentation import milestone_actions

    selected = milestone_actions(r).set_index("request")
    assert selected.loc[13, "Milestone"] == "test creation"
    assert selected.loc[14, "Milestone"] == "verification execution"
    assert selected.loc[14, "exit"] == 0
    assert selected.loc[11, "Milestone"] == "source edit"


@pytest.mark.parametrize(
    "command",
    [
        "cat <<'EOF' > /workspace/test_new.py\nimport pytest\n# run python -m pytest later\nEOF",
        'echo "python -m pytest tests/test_x.py"',
        "python - <<'PY'\nimport pytest\nprint(\"pytest\")\nPY",
    ],
)
def test_test_text_or_printed_check_does_not_count_as_execution(command):
    assert "verification execution" not in p.action_labels(command)


@pytest.mark.parametrize(
    "command",
    [
        "cd /workspace && python -m pytest -q tests/test_x.py",
        "ruff check --no-cache file.py && ruff format --check file.py",
        "python -m mypy --cache-dir=/dev/null file.py",
        "cat <<'EOF' > test_new.py\nimport pytest\nEOF\npytest test_new.py",
    ],
)
def test_real_shell_check_execution_is_recognized(command):
    assert "verification execution" in p.action_labels(command)


def test_main_graph_outcome_partition_matches_saved_results():
    counts = p.outcome_counts(w.public_evidence(ROOT))
    assert counts[["Accepted", "Unsuccessful", "Interrupted"]].values.tolist() == [
        [2, 15, 1],
        [11, 5, 2],
    ]
    assert counts[["Accepted", "Unsuccessful", "Interrupted"]].sum(axis=1).tolist() == [
        18,
        18,
    ]


def test_interventions_and_readable_feature_settings_have_receipts():
    records = {r["run_id"]: r for r in w.records(ROOT)}
    view = json.loads((ROOT / "notebooks/tutorial-study.json").read_text())
    development, audit = p.development_transitions(records, view)
    assert list(development.columns) == [
        "Problem",
        "Change introduced",
        "Next outcome",
        "Learning",
    ]
    assert all(r["run_id"] in records for r in audit)
    feature, audit = p.feature_transitions(records, view)
    labels = " ".join(feature["Change introduced"])
    for text in [
        "Incremental edits",
        "Larger response allowance",
        "Environment repair",
    ]:
        assert text in labels
    assert len(feature) == 6
    assert audit[-1]["patch_accepted"] is True
    assert audit[-2]["patch_accepted"] is False


def test_issue_link_stack_and_recorded_setup_are_explicit():
    n = nbformat.read(ROOT / "main.ipynb", as_version=4)
    text = "\n".join(c.source for c in n.cells)
    assert "https://github.com/langchain-ai/langchain/issues/31051" in text
    assert "The evaluated agent never receives this reference" in text
    assert "Public preparation:" in text
    for component in [
        "Harbor",
        "mini-SWE-agent",
        "OpenRouter",
        "Docker",
        "Project code",
    ]:
        assert "| " + component + " |" in n.cells[0].source
    assert (
        text.index("main-outcomes.svg")
        < text.index("distributions.svg")
        < text.index("matched-context.svg")
    )


def test_asset_purposes_final_values_and_model_colors():
    root, manifest = p.assets(ROOT)
    assert manifest["model_colors"] == p.MODEL_COLORS
    assert manifest["readable_task"] == "Integer-temperature metadata bug"
    assert manifest["matched_attempt"] == 1
    context = (root / "matched-context.svg").read_text()
    cost = (root / "matched-cost.svg").read_text()
    assert "Input context carried into successive requests" in context
    assert "Cumulative reported inference charge" in cost
    for model in ["MiniMax M2.5", "GLM-5"]:
        assert model + ":" in context
        assert model + ": $" in cost
