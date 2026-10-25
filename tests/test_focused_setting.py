import json
from pathlib import Path

import toml
from harbor.models.job.config import JobConfig

from experiments.configuration_paths import configuration_path
from experiments.launch import verify_freeze
from experiments.records import load_runs


def test_editing_only_setting_preserves_budget_model_checks_and_isolation():
    root = Path(__file__).resolve().parents[1]
    old = json.loads(
        (
            configuration_path(root, "configs/harbor-annotated-tools-v1-model.json")
        ).read_text()
    )
    new_path = configuration_path(
        root, "configs/harbor-annotated-tools-focused-editing-v1-model.json"
    )
    new = json.loads(new_path.read_text())
    JobConfig.model_validate(new)
    before = old["agents"][0]["kwargs"]["policy"]
    after = new["agents"][0]["kwargs"]["policy"]
    for key in ("budgets", "model", "source_provenance", "reproduction"):
        assert before[key] == after[key]
    assert after["budgets"]["response_tokens"] == 1500
    assert old["agents"][0]["import_path"] == new["agents"][0]["import_path"]
    task_before, task_after = [Path(j["tasks"][0]["path"]) for j in (old, new)]
    original = toml.loads((task_before / "task.toml").read_text())
    revised = toml.loads((task_after / "task.toml").read_text())
    assert original["verifier"] == revised["verifier"]
    for path in task_before.rglob("*"):
        if path.is_file() and path.name != "task.toml":
            assert (
                path.read_bytes()
                == (task_after / path.relative_to(task_before)).read_bytes()
            )
    original_template = before["mini_config"]["agent"]["instance_template"]
    new_template = after["mini_config"]["agent"]["instance_template"]
    assert new_template.startswith(original_template)
    added = new_template[len(original_template) :]
    assert "workspace_edit.py restore PATH" in added
    assert "immediately" in added.lower()
    assert "get_type_hints" not in added


def test_focused_attempt_preserves_patch_and_submission_outcomes_separately():
    root = Path(__file__).resolve().parents[1]
    run = next(
        r
        for r in load_runs(root)
        if r["run_id"]
        == "annotated-tools-focused-editing-minimax-attempt-001/LC-ANNOTATED-TOOLS-001__UEiZNLt"
    )
    assert run["patch_acceptance"] == 1
    assert not run["success"] and not run["submitted"]
    assert run["calls"] == 30 and run["output_tokens"] == 5878
    assert run["cost_usd"] == 0.02652759
    assert run["tool_format_repair_rounds"] == 0
    assert run["code_repair_rounds"] == 2
    assert run["failure_category"] == "submission_protocol_failure"
    assert (
        verify_freeze(
            root,
            "configs/harbor-annotated-tools-focused-editing.sha256",
            "configs/LC-ANNOTATED-TOOLS-001.focused-editing.frozen-setting.json",
        )
        == run["declared_frozen_setting_id"]
    )
