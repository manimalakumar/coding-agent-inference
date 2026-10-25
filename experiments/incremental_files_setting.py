"""Freeze a prompt-only incremental-file-writing development intervention."""

import copy
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import yaml

from experiments.configuration_paths import configuration_path
from experiments.launch import verify_freeze
from sandbox.docker import digest

LABEL = "context-counter-incremental-files-v1"
INSTRUCTION = """

## Incremental file creation

Create new files through short, incremental edits rather than a single large
command. Write a small initial skeleton, then append or patch one small logical
block at a time in separate tool calls. Do not send an entire large file as one
tool-call payload. Finish each command and its JSON arguments completely before
continuing in a later call.
"""


def prepare(project):
    project = Path(project)
    parent_label = "context-counter-feature-v1"
    parent_id = verify_freeze(
        project,
        f"configs/{parent_label}.sha256",
        f"configs/{parent_label}.frozen-setting.json",
    )
    output = configuration_path(project, f"configs/{LABEL}.frozen-setting.json")
    if output.exists():
        raise FileExistsError("Never overwrite a frozen intervention")
    control_path = "reports/context-counter-test-directories-control-v1.json"
    control = json.loads((project / control_path).read_text())
    assert control["status"] == "passed" and control["model_requests"] == 0
    assert all(control["probe"]["checks"].values())
    parent = json.loads(
        (
            configuration_path(project, f"configs/{parent_label}.frozen-setting.json")
        ).read_text()
    )
    before = json.loads(
        (configuration_path(project, f"configs/{parent_label}-model.json")).read_text()
    )
    after = copy.deepcopy(before)
    after["job_name"] = LABEL + "-model"
    policy = after["agents"][0]["kwargs"]["policy"]
    original = before["agents"][0]["kwargs"]["policy"]
    policy["mini_config"]["agent"]["instance_template"] += INSTRUCTION
    prompt = "configs/mini-swe-incremental-edits.yaml"
    (project / prompt).write_text(
        yaml.safe_dump(policy["mini_config"], sort_keys=False)
    )
    policy["prompt_config"] = Path(prompt).name
    policy["code_fingerprints"][prompt] = hashlib.sha256(
        (project / prompt).read_bytes()
    ).hexdigest()
    assert (
        policy["budgets"] == original["budgets"]
        and policy["budgets"]["response_tokens"] == 1500
    )
    assert policy["model"] == original["model"]
    assert (
        policy["mini_config"]["agent"]["instance_template"]
        == original["mini_config"]["agent"]["instance_template"] + INSTRUCTION
    )
    stripped = copy.deepcopy(policy)
    stripped["mini_config"]["agent"]["instance_template"] = original["mini_config"][
        "agent"
    ]["instance_template"]
    stripped["prompt_config"] = original["prompt_config"]
    stripped["code_fingerprints"].pop(prompt)
    assert stripped == original, (
        "Only the generic file-creation instruction may affect agent behavior"
    )
    job_path = f"configs/{LABEL}-model.json"
    (project / job_path).write_text(json.dumps(after, indent=2) + "\n")
    instruction_path = f"configs/{LABEL}-instruction.md"
    (project / instruction_path).write_text(INSTRUCTION.lstrip())
    recipe = "experiments/incremental_files_setting.py"
    paths = set(parent["source_and_config_sha256"]) | {
        job_path,
        prompt,
        instruction_path,
        recipe,
        "experiments/test_directory_control.py",
        control_path,
    }
    files = {
        p: hashlib.sha256((project / p).read_bytes()).hexdigest() for p in sorted(paths)
    }
    frozen = {
        **parent,
        "experimental_setting": LABEL,
        "parent_setting_id": parent_id,
        "frozen_at": datetime.now(UTC).isoformat(),
        "policy": policy,
        "intervention": "generic short incremental new-file edits only",
        "intervention_instruction": INSTRUCTION,
        "source_and_config_sha256": files,
        "directory_control": control_path,
        "parent_evaluator_validation": parent["validation_report"],
        "task_bundle_reused_unchanged": after["tasks"][0]["path"],
        "comparison_scope": "Development tuning; fresh context/snapshot; prompt-only intervention, original 1500-token response ceiling and all total budgets retained.",
    }
    frozen.pop("setting_id", None)
    frozen["setting_id"] = digest(frozen)
    output.write_text(json.dumps(frozen, indent=2) + "\n")
    setting_relative = str(output.relative_to(project))
    manifest = f"configs/{LABEL}.sha256"
    (project / manifest).write_text(
        "".join(f"{sha}  {p}\n" for p, sha in sorted(files.items()))
        + f"{hashlib.sha256(output.read_bytes()).hexdigest()}  {setting_relative}\n"
    )
    catalog_path = configuration_path(project, "configs/study.json")
    catalog = json.loads(catalog_path.read_text())
    catalog["studies"][LABEL] = {
        "task_ids": ["LC-CONTEXT-COUNTER-001"],
        "job_config": job_path,
        "freeze_manifest": manifest,
        "frozen_setting": setting_relative,
    }
    catalog_path.write_text(json.dumps(catalog, indent=2) + "\n")
    assert verify_freeze(project, manifest, setting_relative) == frozen["setting_id"]
    report = {
        "setting": LABEL,
        "setting_id": frozen["setting_id"],
        "parent_setting_id": parent_id,
        "intervention_instruction": INSTRUCTION,
        "model": policy["model"],
        "budgets": policy["budgets"],
        "agent_image": frozen["agent_image"],
        "verifier_image": frozen["verifier_image"],
        "baseline_checksum": frozen["baseline_checksum"],
        "starting_commit": frozen["starting_commit"],
        "task_instruction_unchanged": True,
        "test_directory_permissions_unchanged": True,
        "hidden_acceptance_unchanged": True,
        "evaluator_validation_reused": parent["validation_report"],
        "zero_inference_directory_control": control_path,
        "setup_model_requests": 0,
        "fresh_snapshot_and_empty_context_required": True,
    }
    (project / f"reports/{LABEL}-preparation.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    prepare(Path(__file__).resolve().parents[1])
