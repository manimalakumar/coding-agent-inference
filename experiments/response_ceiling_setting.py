"""Freeze a response-cap-only development setting, retaining incremental-file guidance."""

import copy
import hashlib
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

import toml
import yaml

from agent.budget import Budget, BudgetExhausted
from agent.schemas import Budgets
from experiments.configuration_paths import configuration_path
from experiments.launch import verify_freeze
from sandbox.docker import digest

LABEL = "context-counter-response-6000-v1"
PARENT = "context-counter-incremental-files-v1"
RESPONSE_TOKENS = 6000


def prepare(project):
    project = Path(project)
    parent_id = verify_freeze(
        project, f"configs/{PARENT}.sha256", f"configs/{PARENT}.frozen-setting.json"
    )
    output = configuration_path(project, f"configs/{LABEL}.frozen-setting.json")
    if output.exists():
        raise FileExistsError("Never overwrite a frozen response-ceiling experiment")
    headroom_path = f"reports/{LABEL}-headroom.json"
    headroom = json.loads((project / headroom_path).read_text())
    assert (
        headroom["planned_response_tokens"] == RESPONSE_TOKENS
        and headroom["model_requests"] == 0
    )
    assert all(
        r["headroom_to_6000"] > 0 for r in headroom["proxy_measurements"].values()
    )
    parent = json.loads(
        (
            configuration_path(project, f"configs/{PARENT}.frozen-setting.json")
        ).read_text()
    )
    before = json.loads(
        (configuration_path(project, f"configs/{PARENT}-model.json")).read_text()
    )
    after = copy.deepcopy(before)
    after["job_name"] = LABEL + "-model"
    policy = after["agents"][0]["kwargs"]["policy"]
    original = before["agents"][0]["kwargs"]["policy"]
    policy["budgets"]["response_tokens"] = RESPONSE_TOKENS
    policy["completion_policy"]["agent_budgets"]["response_tokens"] = RESPONSE_TOKENS
    template = policy["mini_config"]["agent"]["instance_template"]
    assert template.count("1,500") == 1
    policy["mini_config"]["agent"]["instance_template"] = template.replace(
        "1,500", "6,000"
    )
    prompt = f"configs/mini-swe-{LABEL}.yaml"
    (project / prompt).write_text(
        yaml.safe_dump(policy["mini_config"], sort_keys=False)
    )
    policy["prompt_config"] = Path(prompt).name
    policy["code_fingerprints"][prompt] = hashlib.sha256(
        (project / prompt).read_bytes()
    ).hexdigest()
    old_task = Path(before["tasks"][0]["path"])
    task = project / "harbor_tasks/settings" / LABEL / old_task.name
    assert not task.exists()
    shutil.copytree(old_task, task)
    instructions = (task / "instruction.md").read_text()
    assert instructions.count("1,500") == 1
    (task / "instruction.md").write_text(instructions.replace("1,500", "6,000"))
    cfg = toml.loads((task / "task.toml").read_text())
    cfg["metadata"]["experimental_setting"] = LABEL
    (task / "task.toml").write_text(toml.dumps(cfg))
    after["tasks"] = [{"path": str(task)}]
    # Reuse exact images/checks: the verifier's embedded agent-budget description
    # is historical metadata; it never controls model output. Host policy does.
    for p in (old_task / "tests").rglob("*"):
        if p.is_file():
            assert (
                p.read_bytes()
                == (task / "tests" / p.relative_to(old_task / "tests")).read_bytes()
            )
    stripped = copy.deepcopy(policy)
    stripped["budgets"]["response_tokens"] = 1500
    stripped["completion_policy"]["agent_budgets"]["response_tokens"] = 1500
    stripped["mini_config"]["agent"]["instance_template"] = template
    stripped["prompt_config"] = original["prompt_config"]
    stripped["code_fingerprints"].pop(prompt)
    assert stripped == original, "No independent policy change is allowed"
    # Controller-only, zero-inference test: the larger cap cannot overrun totals.
    budget = Budget(Budgets(**policy["budgets"]))
    assert budget.reserve("probe") == 6000
    budget.account("probe", {"completion_tokens": 6000}, 6000)
    assert budget.reserve("probe") == 4000
    budget.account("probe", {"completion_tokens": 4000}, 4000)
    try:
        budget.reserve("probe")
    except BudgetExhausted:
        pass
    else:
        raise AssertionError("Cumulative output limit was not enforced")
    job_path = f"configs/{LABEL}-model.json"
    (project / job_path).write_text(json.dumps(after, indent=2) + "\n")
    paths = set(parent["source_and_config_sha256"]) | {
        job_path,
        prompt,
        headroom_path,
        "experiments/response_ceiling_setting.py",
    }
    paths |= {str(p.relative_to(project)) for p in task.rglob("*") if p.is_file()}
    recipe = (
        project
        / "reports/task-preparation/LC-CONTEXT-COUNTER-001/check_response_headroom.py"
    )
    shutil.copyfile(project / ".local/setup-inputs/check_response_headroom.py", recipe)
    paths.add(str(recipe.relative_to(project)))
    files = {
        p: hashlib.sha256((project / p).read_bytes()).hexdigest() for p in sorted(paths)
    }
    frozen = {
        **parent,
        "experimental_setting": LABEL,
        "parent_setting_id": parent_id,
        "frozen_at": datetime.now(UTC).isoformat(),
        "policy": policy,
        "intervention": "per-response ceiling 1500 to 6000 only; matching budget descriptions updated",
        "intervention_instruction": "No new workflow instruction; incremental-file guidance retained unchanged.",
        "source_and_config_sha256": files,
        "task_bundle_reused_unchanged": False,
        "task_bundle": str(task),
        "headroom_report": headroom_path,
        "comparison_scope": "Separate development tuning; fresh snapshot/context. Only per-response ceiling changes relative to incremental-files-v1. Total calls/output/time and all other behavior stay fixed.",
        "evaluator_budget_metadata_note": "Exact verifier image reused: embedded 1500 response budget is descriptive legacy metadata, unused by scoring. Signed host policy/actual API requests enforce 6000 or remaining cumulative tokens.",
    }
    frozen.pop("setting_id", None)
    frozen["setting_id"] = digest(frozen)
    output.write_text(json.dumps(frozen, indent=2) + "\n")
    setting_path = str(output.relative_to(project))
    manifest = f"configs/{LABEL}.sha256"
    (project / manifest).write_text(
        "".join(f"{sha}  {p}\n" for p, sha in sorted(files.items()))
        + f"{hashlib.sha256(output.read_bytes()).hexdigest()}  {setting_path}\n"
    )
    catalog_path = configuration_path(project, "configs/study.json")
    catalog = json.loads(catalog_path.read_text())
    catalog["studies"][LABEL] = {
        "task_ids": [old_task.name],
        "job_config": job_path,
        "freeze_manifest": manifest,
        "frozen_setting": setting_path,
    }
    catalog_path.write_text(json.dumps(catalog, indent=2) + "\n")
    assert verify_freeze(project, manifest, setting_path) == frozen["setting_id"]
    report = {
        "setting": LABEL,
        "setting_id": frozen["setting_id"],
        "parent_setting_id": parent_id,
        "response_tokens_before": 1500,
        "response_tokens_after": RESPONSE_TOKENS,
        "model": policy["model"],
        "budgets": policy["budgets"],
        "incremental_file_instruction_unchanged": True,
        "agent_image": frozen["agent_image"],
        "verifier_image": frozen["verifier_image"],
        "baseline_checksum": frozen["baseline_checksum"],
        "starting_commit": frozen["starting_commit"],
        "acceptance_files_byte_identical": True,
        "permissions_and_dependencies_unchanged": True,
        "controller_cumulative_budget_simulation": "6000 + 4000 accepted, third reservation denied; zero model calls",
        "headroom_report": headroom_path,
        "setup_model_requests": 0,
        "prior_patch_hidden_feedback_in_agent_context": False,
    }
    (project / f"reports/{LABEL}-preparation.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    prepare(Path(__file__).resolve().parents[1])
