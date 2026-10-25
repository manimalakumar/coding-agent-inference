"""Prepare and freeze a new Harbor completion policy without rewriting old settings."""

import argparse
import copy
import hashlib
import json
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import toml
import yaml
from harbor.models.job.config import JobConfig
from harbor.models.task.config import TaskConfig

from experiments.configuration_paths import configuration_path
from harbor_integration.prepare_resources import build

ROOT = Path(__file__).resolve().parents[1]
PREFIX = "harbor-workspace-deliverable-v1"
TASK_ID = "LC-ANNOTATED-TOOLS-001"
SETTING = f"configs/{TASK_ID}.workspace-deliverable-v1.frozen-setting.json"
COMMON = "configs/completion-workspace-capture.json"

COMPLETION = """## Workspace delivery and stopping

The final workspace is your deliverable. Harbor automatically captures it when you
submit, exhaust a budget, or stop with an execution error. Trusted evaluation then
runs on that frozen candidate in a separate environment. Task success means all
required functional, regression, lint, format, type and file-policy checks pass.
Explicit submission is recorded separately and is not required for patch acceptance.

No model request or output tokens are reserved for submission. The last permitted
request may edit source or execute development checks. Budget exhaustion stops
agent execution; there are no extra model calls or repairs afterward. Tool actions
authorized by the final allowed response may execute within the remaining time.
The agent's limits remain 30 model requests, 10,000 cumulative output tokens and
20 minutes; each response is capped at 1,500 tokens and each command at 90 seconds.
Malformed tool-response recovery is limited to two retries within these same budgets.
Trusted evaluation has a separate five-minute limit after capture and grants no
agent actions, inference or repairs.

Once required checks pass, finish early to save inference. The standalone bash
command `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT` remains available for that
purpose. Do not combine it with other commands. The agent cannot continue after
submission. Passing checks means checks_passed_pending_review, with no authorization
to commit, push, publish or merge.

Run tests directly where practical. For pipelines use `set -o pipefail`; the harness
also enables it. Always run format --check after formatting and inspect exit codes.

"""


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare():
    old = json.loads(
        (
            configuration_path(
                ROOT, f"configs/{TASK_ID}.focused-editing-v1.frozen-setting.json"
            )
        ).read_text()
    )
    job = json.loads(
        (
            configuration_path(
                ROOT, "configs/harbor-annotated-tools-focused-editing-v1-model.json"
            )
        ).read_text()
    )
    original_task = Path(job["tasks"][0]["path"])
    task = ROOT / "harbor_tasks/settings" / PREFIX / TASK_ID
    if (ROOT / f"reports/{PREFIX}-preparation.json").exists():
        raise FileExistsError("Never overwrite a prepared setting")
    shutil.copytree(original_task, task, dirs_exist_ok=True)
    common = json.loads((ROOT / COMMON).read_text())
    policy = copy.deepcopy(job["agents"][0]["kwargs"]["policy"])
    assert policy["budgets"] == common["agent_budgets"]
    policy["completion_policy"] = common
    prompt = policy["mini_config"]["agent"]["instance_template"]
    start = prompt.index("## Stopping and reliable test results")
    end = prompt.index("Follow the task-specific completion checklist", start)
    prompt = prompt[:start] + COMPLETION + prompt[end:]
    policy["mini_config"]["agent"]["instance_template"] = prompt
    policy["prompt_config"] = "mini-swe-workspace-capture.yaml"
    (configuration_path(ROOT, policy["prompt_config"])).write_text(
        yaml.safe_dump(policy["mini_config"], sort_keys=False)
    )
    instruction = (task / "instruction.md").read_text()
    instruction = (
        instruction[: instruction.index("## Budget and submission")] + COMPLETION
    )
    instruction = instruction.replace(
        "reserve calls for implementation, verification and final submission",
        "budget calls for implementation and verification",
    )
    (task / "instruction.md").write_text(instruction)
    tests = task / "tests"
    shutil.copyfile(
        ROOT / "evaluator/workspace_policy.py", tests / "workspace_policy.py"
    )
    shutil.copyfile(ROOT / COMMON, tests / "completion-policy.json")
    verify = (tests / "verify.py").read_text()
    verify = verify.replace(
        "BASE = Path", "from workspace_policy import verdict\n\nBASE = Path", 1
    )
    verify = verify.replace(
        '    rewards = {"reward": 0, "patch_acceptance": 0, "agent_submission": 0}',
        '    rewards = {"reward": 0, "patch_acceptance": 0, "agent_submission": 0, "patch_accepted": 0, "explicit_submission": 0, "task_success": 0}\n    common = json.loads(Path("/tests/completion-policy.json").read_text())',
    )
    verify = verify.replace(
        '        deadline = receipt["deadline"]',
        """        if receipt.get("completion_policy") != common["policy_id"] or not receipt.get("workspace_quiesced"):
            raise ValueError("Candidate not frozen under the expected completion policy")
        details["termination_reason"] = receipt["termination_reason"]
        details["explicit_submission"] = receipt["explicit_submission"]
        details["completion_policy"] = common["policy_id"]
        details["evaluation_started_at"] = time.time()
        deadline = details["evaluation_started_at"] + common["evaluation_seconds"]
        details["evaluation_deadline"] = deadline
        details["candidate_sha256"] = hashlib.sha256(Path("/logs/artifacts/candidate.tar").read_bytes()).hexdigest()""",
    )
    verify = verify.replace(
        "        restrict(WORK)\n",
        '        restrict(WORK)\n        details["checks"]["file_policy"] = {"exit_code": 0, "output": "Allowed source changes and new regression files only; existing tests preserved."}\n',
        1,
    )
    verify = verify.replace(
        "Shared 20-minute workflow deadline exhausted",
        "Trusted five-minute evaluation deadline exhausted",
    )
    verify = verify.replace(
        "        reproduced = bool(reg and reg.returncode == 1)",
        '        reproduced = bool(reg and reg.returncode == 1)\n        details["regression_signal"] = reproduced',
    )
    start = verify.index("        development_ok = bool(")
    end = verify.index("    except Exception as exc:", start)
    verify = (
        verify[:start]
        + """        rewards = verdict(
            details["checks"], required=common["required_checks"],
            changed=changed, new_tests=new_tests,
            explicit_submission=receipt["explicit_submission"],
        )
        details["task_success"] = bool(rewards["task_success"])
        details["patch_accepted"] = bool(rewards["patch_accepted"])
        details["status"] = "checks_passed_pending_review" if rewards["task_success"] else "checks_failed"
"""
        + verify[end:]
    )
    verify = verify.replace(
        '        details["error"] = type(exc).__name__ + ": " + str(exc)',
        '        details["error"] = type(exc).__name__ + ": " + str(exc)\n        details["task_success"] = False\n        details["patch_accepted"] = False',
    )
    verify = verify.replace(
        '        rewards["agent_submission"] = int(receipt["agent_submission"])',
        '        rewards["agent_submission"] = int(receipt["explicit_submission"])\n        rewards["explicit_submission"] = int(receipt["explicit_submission"])',
    )
    assert (
        'receipt["deadline"]' not in verify
        and "development_ok and receipt" not in verify
    )
    (tests / "verify.py").write_text(verify)
    subprocess.run(
        [
            str(ROOT / ".venv/bin/ruff"),
            "check",
            "--fix",
            str(tests / "verify.py"),
            str(tests / "workspace_policy.py"),
        ],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        [
            str(ROOT / ".venv/bin/ruff"),
            "format",
            str(tests / "verify.py"),
            str(tests / "workspace_policy.py"),
        ],
        check=True,
        capture_output=True,
    )
    state = Path(policy["receipt_key"]).parents[1] / PREFIX
    state.mkdir(exist_ok=True)
    shutil.copytree(tests, state / "tests", dirs_exist_ok=True)
    shutil.copyfile(policy["receipt_key"], state / "tests/receipt.key")
    base = "maintenance-verifier-base:" + old["verifier_image"].split(":")[1][:16]
    subprocess.run(["docker", "tag", old["verifier_image"], base], check=True)
    assert (
        subprocess.check_output(
            ["docker", "image", "inspect", base, "--format", "{{.Id}}"], text=True
        ).strip()
        == old["verifier_image"]
    )
    (state / "Dockerfile").write_text(
        f"FROM {base}\nUSER root\nCOPY tests /tests\nRUN chmod -R a-w /tests && chmod 400 /tests/receipt.key\n"
    )
    verifier_image = build(state, "maintenance-workspace-verifier:v1")
    config = toml.loads((task / "task.toml").read_text())
    config["metadata"].pop("shared_workflow_seconds", None)
    config["metadata"].update(
        experimental_setting=PREFIX,
        completion_policy=common["policy_id"],
        agent_budget_seconds=1200,
        evaluation_budget_seconds=300,
    )
    # Outer grace permits usage finalization and quiescence, never more agent work.
    config["agent"]["timeout_sec"] = 1280
    config["verifier"]["timeout_sec"] = common["evaluation_seconds"]
    config["verifier"]["environment"]["docker_image"] = verifier_image
    TaskConfig.model_validate(config)
    (task / "task.toml").write_text(toml.dumps(config))
    policy["code_fingerprints"] |= {
        p: sha(ROOT / p)
        for p in [
            COMMON,
            "harbor_integration/host_workspace.py",
            "harbor_integration/workspace_setting.py",
            "evaluator/workspace_policy.py",
        ]
    }
    for mode in (
        "baseline",
        "reference",
        "reference_budget",
        "reference_error",
        "model",
    ):
        new = copy.deepcopy(job)
        new["job_name"] = PREFIX + "-" + mode
        new["tasks"] = [{"path": str(task)}]
        new["agents"][0]["import_path"] = (
            "harbor_integration.host_workspace:WorkspaceAgent"
        )
        new["agents"][0]["kwargs"] = {"policy": policy}
        if mode != "model":
            new["agents"][0]["kwargs"]["fixture"] = mode
            new["jobs_dir"] = str(ROOT / "reports/harbor-validation")
        JobConfig.model_validate(new)
        (configuration_path(ROOT, f"configs/{PREFIX}-{mode}.json")).write_text(
            json.dumps(new, indent=2) + "\n"
        )
    result = {
        "task_directory": str(task),
        "trusted_setup_directory": str(state),
        "agent_image": old["agent_image"],
        "verifier_image": verifier_image,
        "model_requests": 0,
        "parent_setting_id": old["setting_id"],
    }
    (ROOT / f"reports/{PREFIX}-preparation.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    return result


def freeze():
    old = json.loads(
        (
            configuration_path(
                ROOT, f"configs/{TASK_ID}.focused-editing-v1.frozen-setting.json"
            )
        ).read_text()
    )
    preparation = json.loads((ROOT / f"reports/{PREFIX}-preparation.json").read_text())
    validation = f"reports/{PREFIX}-validation-summary.json"
    assert (ROOT / validation).is_file(), "Run all trusted controls before freezing"
    job = json.loads(
        (configuration_path(ROOT, f"configs/{PREFIX}-model.json")).read_text()
    )
    record = copy.deepcopy(old)
    record.update(
        experimental_setting=PREFIX,
        parent_setting_id=old["setting_id"],
        frozen_at=datetime.now(UTC).isoformat(),
        verifier_image=preparation["verifier_image"],
        policy=job["agents"][0]["kwargs"]["policy"],
        validation_report=validation,
    )
    record["termination_policy"] = json.loads((ROOT / COMMON).read_text())
    record["completion_policy_sha256"] = sha(ROOT / COMMON)
    record["comparison_policy"] = (
        "Model-independent workspace-deliverable-v1; previous outcomes never reclassified. All compared models receive identical completion semantics."
    )
    paths = set(old["source_and_config_sha256"])
    paths |= {
        str(p.relative_to(ROOT))
        for p in Path(preparation["task_directory"]).rglob("*")
        if p.is_file()
    }
    paths |= {
        f"configs/{PREFIX}-{mode}.json"
        for mode in (
            "baseline",
            "reference",
            "reference_budget",
            "reference_error",
            "model",
        )
    }
    paths |= {
        COMMON,
        "configs/mini-swe-workspace-capture.yaml",
        "harbor_integration/host_workspace.py",
        "harbor_integration/workspace_setting.py",
        "evaluator/workspace_policy.py",
    }
    record["source_and_config_sha256"] = {p: sha(ROOT / p) for p in sorted(paths)}
    record.pop("setting_id", None)
    record["setting_id"] = hashlib.sha256(
        json.dumps(record, sort_keys=True).encode()
    ).hexdigest()
    (ROOT / SETTING).write_text(json.dumps(record, indent=2) + "\n")
    paths.add(SETTING)
    (configuration_path(ROOT, f"configs/{PREFIX}.sha256")).write_text(
        "".join(f"{sha(ROOT / p)}  {p}\n" for p in sorted(paths))
    )
    catalog = json.loads((configuration_path(ROOT, "configs/study.json")).read_text())
    catalog["active_study"] = "annotated-tools-workspace"
    catalog["studies"][catalog["active_study"]] = {
        "task_ids": [TASK_ID],
        "job_config": f"configs/{PREFIX}-model.json",
        "freeze_manifest": f"configs/{PREFIX}.sha256",
        "frozen_setting": SETTING,
    }
    (configuration_path(ROOT, "configs/study.json")).write_text(
        json.dumps(catalog, indent=2) + "\n"
    )
    return {
        "setting_id": record["setting_id"],
        "completion_policy_sha256": record["completion_policy_sha256"],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "freeze"])
    args = parser.parse_args()
    print(json.dumps(prepare() if args.command == "prepare" else freeze(), indent=2))
