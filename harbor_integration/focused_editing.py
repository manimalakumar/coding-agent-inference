"""Prepare a separate editing-only setting, preserving the original task and logs."""

import argparse
import copy
import hashlib
import json
import shutil
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

import toml
import yaml
from harbor.models.job.config import JobConfig
from harbor.models.task.config import TaskConfig

from experiments.configuration_paths import configuration_path
from harbor_integration.prepare_resources import build

ROOT = Path(__file__).resolve().parents[1]
PREFIX = "harbor-annotated-tools-focused-editing-v1"
TASK_ID = "LC-ANNOTATED-TOOLS-001"
SETTING = f"configs/{TASK_ID}.focused-editing-v1.frozen-setting.json"
INSTRUCTIONS = """

## Focused editing and recovery

Favor small, focused replacements or patches to existing source files. For a
scripted replacement, assert that the old text occurs exactly once before writing.
Preserve the rest of the file; avoid reconstructing entire existing files.
Immediately after each source edit, inspect its diff against the unchanged
snapshot and check that unrelated imports, definitions and content remain intact.

A read-only copy of the permitted production source from this attempt's unchanged
pre-fix snapshot is available at /task/unchanged. It contains no Git history.
Use repository-relative paths with these commands:

  python /task/workspace_edit.py diff PATH
  python /task/workspace_edit.py restore PATH

Diff prints a unified diff for the named production file (an empty diff means no
change). Restore replaces that file with its unchanged pre-fix contents, discarding
your edits to that file. Restore and retry a focused edit if you damage a file;
do not reconstruct it from memory or a partial display. These commands run inside
the same sandbox, with the same source permissions as bash.

Tool-format repair means recovery from a malformed model tool-call response.
It is distinct from repairing code after a failed check, and from Ruff formatting.
The existing allowance of two tool-format repairs is unchanged.
"""


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare():
    started = time.monotonic()
    old = json.loads(
        (
            configuration_path(ROOT, f"configs/{TASK_ID}.harbor-v1.frozen-setting.json")
        ).read_text()
    )
    source_task = ROOT / "harbor_tasks" / TASK_ID
    task = ROOT / "harbor_tasks/settings" / PREFIX / TASK_ID
    if (ROOT / f"reports/{PREFIX}-preparation.json").exists():
        raise FileExistsError("Setting already prepared; never overwrite a frozen task")
    shutil.copytree(source_task, task, dirs_exist_ok=True)
    prior_job = json.loads(
        (
            configuration_path(ROOT, "configs/harbor-annotated-tools-v1-model.json")
        ).read_text()
    )
    policy = copy.deepcopy(prior_job["agents"][0]["kwargs"]["policy"])
    setup_dir = Path(policy["receipt_key"]).parents[1]
    baseline = setup_dir / "baseline"
    saved = json.loads((setup_dir / "setup.json").read_text())
    from sandbox.docker import digest, manifest

    assert digest(manifest(baseline)) == old["baseline_checksum"]
    state = setup_dir / PREFIX
    state.mkdir(exist_ok=True)
    allowed = saved["task"]["source_paths"]
    for source in allowed:
        destination = state / "unchanged" / source
        if destination.exists():
            assert manifest(destination) == manifest(baseline / source)
        else:
            shutil.copytree(baseline / source, destination)
    shutil.copyfile(ROOT / "sandbox/workspace_edit.py", state / "workspace_edit.py")
    (state / "edit-resources.json").write_text(json.dumps({"source_paths": allowed}))
    base_tag = "maintenance-focused-base:" + old["agent_image"].split(":")[1][:16]
    subprocess.run(["docker", "tag", old["agent_image"], base_tag], check=True)
    assert (
        subprocess.check_output(
            ["docker", "image", "inspect", base_tag, "--format", "{{.Id}}"], text=True
        ).strip()
        == old["agent_image"]
    )
    (state / "Dockerfile").write_text(
        f"FROM {base_tag}\nUSER root\n"
        "COPY unchanged /task/unchanged\n"
        "COPY workspace_edit.py /task/workspace_edit.py\n"
        "COPY edit-resources.json /task/edit-resources.json\n"
        "RUN chmod -R a-w /task\nUSER 65532:65532\n"
    )
    agent_image = build(state, "maintenance-focused-editing:annotated-tools-v1")
    task_config = toml.loads((task / "task.toml").read_text())
    task_config["environment"]["docker_image"] = agent_image
    task_config["metadata"]["experimental_setting"] = PREFIX
    TaskConfig.model_validate(task_config)
    (task / "task.toml").write_text(toml.dumps(task_config))
    policy["mini_config"]["agent"]["instance_template"] += INSTRUCTIONS
    policy["prompt_config"] = "mini-swe-focused-editing.yaml"
    prompt = configuration_path(ROOT, policy["prompt_config"])
    prompt.write_text(yaml.safe_dump(policy["mini_config"], sort_keys=False))
    policy["editing_strategy"] = {
        "version": "focused-editing-v1",
        "baseline_source": "/task/unchanged",
        "helper": "/task/workspace_edit.py",
        "permissions": "read-only baseline; same writable production paths",
    }
    policy["repair_semantics"] = {
        "repair_rounds": "tool-format recovery only; never Ruff formatting",
        "code_repair_rounds": "reported separately, nullable unless audited",
    }
    policy["code_fingerprints"] |= {
        name: sha(ROOT / name)
        for name in [
            "sandbox/workspace_edit.py",
            "harbor_integration/focused_editing.py",
        ]
    }
    for mode in ("baseline", "reference", "model"):
        job = copy.deepcopy(prior_job)
        job["job_name"] = PREFIX + "-" + mode
        job["tasks"] = [{"path": str(task)}]
        job["agents"][0]["kwargs"] = {"policy": policy}
        if mode != "model":
            job["agents"][0]["kwargs"]["fixture"] = mode
            job["jobs_dir"] = str(ROOT / "reports/harbor-validation")
        JobConfig.model_validate(job)
        (configuration_path(ROOT, f"configs/{PREFIX}-{mode}.json")).write_text(
            json.dumps(job, indent=2) + "\n"
        )
    record = {
        "setting": PREFIX,
        "parent_setting_id": old["setting_id"],
        "task_id": TASK_ID,
        "agent_image": agent_image,
        "verifier_image": old["verifier_image"],
        "trusted_setup_directory": str(state),
        "task_directory": str(task),
        "model_requests": 0,
        "reported_inference_cost_usd": 0,
        "setup_seconds": time.monotonic() - started,
        "changes": [
            "focused-editing instructions",
            "read-only pre-fix source and diff/restore helper",
        ],
        "unchanged": [
            "model",
            "provider",
            "budgets",
            "response ceiling",
            "verifier image",
            "acceptance checks",
            "source baseline",
            "writable paths",
        ],
    }
    (ROOT / f"reports/{PREFIX}-preparation.json").write_text(
        json.dumps(record, indent=2) + "\n"
    )
    return record


def freeze():
    old = json.loads(
        (
            configuration_path(ROOT, f"configs/{TASK_ID}.harbor-v1.frozen-setting.json")
        ).read_text()
    )
    preparation = json.loads((ROOT / f"reports/{PREFIX}-preparation.json").read_text())
    validation = ROOT / f"reports/{PREFIX}-validation-summary.json"
    assert validation.is_file(), (
        "Validate isolation and baseline/reference controls first"
    )
    job = json.loads(
        (configuration_path(ROOT, f"configs/{PREFIX}-model.json")).read_text()
    )
    record = copy.deepcopy(old)
    record.update(
        experimental_setting=PREFIX,
        parent_setting_id=old["setting_id"],
        frozen_at=datetime.now(UTC).isoformat(),
        agent_image=preparation["agent_image"],
        policy=job["agents"][0]["kwargs"]["policy"],
        validation_report=str(validation.relative_to(ROOT)),
        comparison_policy="Editing strategy only; response ceiling remains 1500. Fresh snapshot/context. Separate exploratory setting.",
    )
    paths = set(old["source_and_config_sha256"])
    paths |= {
        str(p.relative_to(ROOT))
        for p in Path(job["tasks"][0]["path"]).rglob("*")
        if p.is_file()
    }
    paths |= {
        f"configs/{PREFIX}-{mode}.json" for mode in ("baseline", "reference", "model")
    }
    paths |= {
        "configs/mini-swe-focused-editing.yaml",
        "sandbox/workspace_edit.py",
        "harbor_integration/focused_editing.py",
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
    catalog["active_study"] = "annotated-tools-focused-editing"
    catalog["studies"][catalog["active_study"]] = {
        "task_ids": [TASK_ID],
        "job_config": f"configs/{PREFIX}-model.json",
        "freeze_manifest": f"configs/{PREFIX}.sha256",
        "frozen_setting": SETTING,
    }
    (configuration_path(ROOT, "configs/study.json")).write_text(
        json.dumps(catalog, indent=2) + "\n"
    )
    return {"setting_id": record["setting_id"], "setting_file": SETTING}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "freeze"])
    arguments = parser.parse_args()
    print(
        json.dumps(prepare() if arguments.command == "prepare" else freeze(), indent=2)
    )
