"""Prepare zero-inference controls: root regression files and cacheless strict typing."""

import hashlib
import json
import shutil
from pathlib import Path

import toml

from experiments.configuration_paths import configuration_path
from experiments.launch import verify_freeze

LABEL = "context-counter-environment-v2"
PARENT = "context-counter-glm5-calibration-v1"


def prepare(project):
    project = Path(project)
    verify_freeze(
        project, f"configs/{PARENT}.sha256", f"configs/{PARENT}.frozen-setting.json"
    )
    parent = json.loads(
        (
            configuration_path(project, f"configs/{PARENT}.frozen-setting.json")
        ).read_text()
    )
    old = Path(parent["task_bundle"])
    task = project / "harbor_tasks/settings" / LABEL / "LC-CONTEXT-COUNTER-001"
    if task.exists():
        raise FileExistsError("Never overwrite an environment revision")
    shutil.copytree(old, task)
    instruction = (task / "instruction.md").read_text()
    old_location = "Add regressions only in new `test_*.py` files at `tests/unit_tests/` or inside a newly created subdirectory beneath it. Existing test subdirectories are read-only."
    new_location = "Add regressions only in new `test_*.py` files directly at `tests/unit_tests/`, for example `tests/unit_tests/test_context_editing_token_counter.py`. Do not create test subdirectories or `__init__.py` files. Existing test subdirectories are read-only. This existing test-root package is the validated location for pytest, Ruff and file-policy checks together."
    assert instruction.count(old_location) == 1
    instruction = instruction.replace(old_location, new_location)
    assert instruction.count("mypy --cache-dir=/tmp/mypy") == 1
    instruction = instruction.replace(
        "mypy --cache-dir=/tmp/mypy", "mypy --cache-dir=/dev/null"
    )
    instruction = instruction.replace(
        "8. Run type checking on changed production files:",
        "8. Run full strict type checking on changed production files with cache writes disabled (the sandbox file-size limit is unchanged):",
    )
    (task / "instruction.md").write_text(instruction)
    verify = task / "tests/verify.py"
    s = verify.read_text()
    needle = "                    new_tests.append(path)"
    assert s.count(needle) == 1
    s = s.replace(
        needle,
        '                    if Path(name).parent != Path("."):\n                        raise ValueError("Regression files must be directly in tests/unit_tests")\n'
        + needle,
    )
    assert s.count('"--cache-dir=/tmp/mypy"') == 1
    s = s.replace('"--cache-dir=/tmp/mypy"', '"--cache-dir=/dev/null"')
    verify.write_text(s)
    # Match descriptive budget metadata to the already enforced host completion policy.
    (task / "tests/completion-policy.json").write_text(
        json.dumps(parent["policy"]["completion_policy"], indent=2) + "\n"
    )
    for name in ["test_acceptance.py", "workspace_policy.py"]:
        assert (task / "tests" / name).read_bytes() == (
            old / "tests" / name
        ).read_bytes()
    cfg = toml.loads((task / "task.toml").read_text())
    cfg["metadata"]["experimental_setting"] = LABEL
    cfg["metadata"]["configuration_revision"] = 3
    (task / "task.toml").write_text(toml.dumps(cfg))
    trusted = project / ".local/resources/LC-CONTEXT-COUNTER-001/environment-v2"
    trusted.mkdir(parents=True, exist_ok=False)
    context = trusted / "verifier"
    (context / "tests").mkdir(parents=True)
    for name in ["verify.py", "completion-policy.json"]:
        shutil.copyfile(task / "tests" / name, context / "tests" / name)
    (context / "Dockerfile").write_text(
        "FROM maintenance-context-verifier:v1\n"
        "COPY tests/verify.py /tests/verify.py\n"
        "COPY tests/completion-policy.json /tests/completion-policy.json\n"
        "RUN chmod a-w /tests/verify.py /tests/completion-policy.json\n"
    )
    (context / ".dockerignore").write_text(
        "**\n!Dockerfile\n!tests\n!tests/verify.py\n!tests/completion-policy.json\n"
    )
    report = {
        "setting": LABEL,
        "parent_setting_id": parent["setting_id"],
        "task_bundle": str(task),
        "build_context": str(context),
        "dependency_image": parent["dependency_image"],
        "agent_image": parent["agent_image"],
        "required_parent_verifier_image": parent["verifier_image"],
        "dependency_changes": False,
        "agent_tool_file_limit_unchanged": True,
        "type_command": "mypy --cache-dir=/dev/null $CHANGED_SOURCE",
        "regression_location": "libs/langchain_v1/tests/unit_tests/test_*.py (direct children only)",
        "acceptance_tests_sha256": hashlib.sha256(
            (task / "tests/test_acceptance.py").read_bytes()
        ).hexdigest(),
        "model_requests": 0,
        "inference_cost_usd": 0,
    }
    (project / f"reports/{LABEL}-preparation.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    prepare(Path(__file__).resolve().parents[1])
