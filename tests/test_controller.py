"""Budget, permission, submission, and secret-boundary regression tests."""

import json
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from agent.budget import Budget, BudgetExhausted
from agent.controller import Trace, launch
from agent.schemas import Budgets, Resource, SandboxConfig, Task
from evaluator.qualify import collect_changes, qualify
from sandbox.docker import InfrastructureError, Sandbox, manifest


@pytest.fixture
def task():
    return Task(
        task_id="test",
        commit="a" * 40,
        package="libs/openai",
        instructions="Fix a bug",
        source_paths=("src",),
        test_paths=("tests",),
        development_checks=(),
    )


@pytest.fixture
def box(tmp_path, task):
    baseline = tmp_path / "baseline"
    for directory in ["src", "tests"]:
        (baseline / directory).mkdir(parents=True)
    (baseline / "src/code.py").write_text("value = 1\n")
    (baseline / "tests/test_existing.py").write_text("assert True\n")
    task_file = tmp_path / "task.json"
    task_file.write_text(task.model_dump_json())
    box = Sandbox(
        SandboxConfig(writable=("src", "tests")),
        task,
        baseline,
        tmp_path / "workspace",
        task_file,
        "sha256:" + "a" * 64,
        tmp_path / "resources",
    )
    box.prepare()
    return box


def test_shared_call_and_token_budget():
    budget = Budget(Budgets(model_calls=2, generated_tokens=100, response_tokens=80))
    assert budget.reserve("reproduce") == 80
    budget.account(
        "reproduce",
        {
            "completion_tokens": 70,
            "completion_tokens_details": {"reasoning_tokens": 20},
        },
        80,
    )
    assert budget.reserve("fix") == 30
    budget.account("fix", {"completion_tokens": 30}, 30)
    assert budget.generated == 100  # Reasoning is already included; never double count.
    assert budget.roles["reproduce"]["generated_tokens"] == 70
    with pytest.raises(BudgetExhausted):
        budget.reserve("review")


def test_missing_usage_fails_closed():
    budget = Budget(Budgets())
    reserved = budget.reserve("fix")
    with pytest.raises(ValueError):
        budget.account("fix", {}, reserved)
    assert budget.generated == reserved


def test_deadline_and_repairs():
    budget = Budget(Budgets(repair_rounds=2))
    budget.repair()
    budget.repair()
    with pytest.raises(BudgetExhausted, match="repair_round"):
        budget.repair()
    budget.started -= 601
    with pytest.raises(BudgetExhausted, match="wall_clock"):
        budget.reserve("fix")


@pytest.mark.parametrize("path", ["../host", "/etc", ".", "src/../../etc", ".git"])
def test_reject_unsafe_writable_paths(path):
    with pytest.raises(ValidationError):
        SandboxConfig(writable=(path,))


def test_reject_overlapping_writable_paths():
    with pytest.raises(ValidationError):
        SandboxConfig(writable=("src", "src/internal"))


@pytest.mark.parametrize(
    "resource,target",
    [
        ("../secrets", "/resources/data"),
        ("data", "/opt/venv"),
        ("data", "/resources/../../etc"),
    ],
)
def test_resource_mount_cannot_escape(resource, target):
    with pytest.raises(ValidationError):
        Resource(name="bad", resource=resource, target=target)


def test_cannot_expand_task_permissions(tmp_path, task):
    with pytest.raises(InfrastructureError, match="permissions"):
        Sandbox(
            SandboxConfig(writable=("unrelated",)),
            task,
            tmp_path,
            tmp_path / "work",
            tmp_path / "task",
            "sha256:test",
            tmp_path,
        )


def test_command_has_real_container_boundary_and_clean_environment(box):
    with patch.dict("os.environ", {"OPENROUTER_API_KEY": "never-in-container"}):
        command = box.command("test", "python --version")
    assert "--network=none" in command
    assert "--read-only" in command
    assert "--user=65532:65532" in command
    assert "--cap-drop=ALL" in command
    assert "--security-opt=no-new-privileges" in command
    assert "-i" in command
    assert "never-in-container" not in " ".join(command)
    assert "OPENAI_API_KEY=sk-offline-placeholder" in command
    mounts = [command[i + 1] for i, arg in enumerate(command) if arg == "--mount"]
    assert len(mounts) == 4
    assert sum(m.endswith(",readonly") for m in mounts) == 2
    assert not any("docker.sock" in mount for mount in mounts)


def test_fresh_workspace_and_baseline_unchanged(box):
    (box.workspace / "src/code.py").write_text("value = 2\n")
    assert (box.baseline / "src/code.py").read_text() == "value = 1\n"
    with pytest.raises(InfrastructureError, match="fresh"):
        box.prepare()


def test_reject_symlink_submissions(box, tmp_path):
    (box.workspace / "src/leak.py").symlink_to("/etc/passwd")
    with pytest.raises(InfrastructureError, match="Symlink"):
        collect_changes(box, tmp_path / "patch")


def test_reject_existing_test_modification(box, tmp_path):
    (box.workspace / "tests/test_existing.py").write_text("# removed\n")
    with pytest.raises(InfrastructureError, match="Existing tests"):
        collect_changes(box, tmp_path / "patch")


def test_patch_contains_only_changes(box, tmp_path):
    (box.workspace / "src/code.py").write_text("value = 2\n")
    (box.workspace / "tests/test_regression.py").write_text("assert 1 == 1\n")
    output = tmp_path / "submission.patch"
    changes = collect_changes(box, output)
    assert changes == ["src/code.py", "tests/test_regression.py"]
    assert "-value = 1" in output.read_text()
    assert "+++ b/tests/test_regression.py" in output.read_text()


def test_trace_redacts_key(tmp_path):
    path = tmp_path / "trace.jsonl"
    Trace(path, "private-key").emit("test", response={"echo": "private-key"})
    assert "private-key" not in path.read_text()
    assert json.loads(path.read_text())["response"]["echo"] == "[REDACTED]"


def test_invalid_setup_prevents_model_request(tmp_path):
    (tmp_path / "setup.json").write_text(
        json.dumps({"setup_status": "blocked_by_environment"})
    )
    with patch("subprocess.run") as process:
        with pytest.raises(InfrastructureError, match="not validated"):
            launch(tmp_path, tmp_path / "model.json")
        process.assert_not_called()


def test_unchanged_resource_manifest_is_deterministic(box):
    assert manifest(box.baseline) == manifest(box.baseline)


def test_hidden_pass_cannot_override_failed_development_check(box, tmp_path):
    (box.workspace / "src/code.py").write_text("value = 2\n")
    (box.workspace / "tests/test_regression.py").write_text("assert True\n")
    run = tmp_path / "qualification"
    run.mkdir()

    def execute(instance, command, timeout, extra_mounts=()):
        if instance.workspace.name == "regression-workspace":
            code, output = 1, "reported bug"
        elif command.startswith("mypy"):
            code, output = 1, "type check failed"
        else:
            code, output = 0, "passed"
        return {"exit_code": code, "output": output, "termination": None}

    with patch.object(Sandbox, "execute", execute):
        result = qualify(box, run, tmp_path / "hidden", float("inf"))
    assert result["hidden_acceptance"]["exit_code"] == 0
    assert result["development"]["checks"]["types"]["exit_code"] == 1
    assert result["status"] == "checks_failed"


def test_standard_docker_uses_trusted_mounts_and_clean_interpreter(box, tmp_path):
    from agent.mini_runtime import ScopedEnvironment

    with patch("agent.mini_runtime.DockerEnvironment") as native:
        env = ScopedEnvironment(
            box,
            Budget(box.config.budgets),
            Trace(tmp_path / "native.jsonl", ""),
            tmp_path,
        )
    config = native.call_args.kwargs
    native.return_value.config.cwd = config["cwd"]
    assert "--network=none" in config["run_args"]
    assert "--read-only" in config["run_args"]
    assert config["forward_env"] == []
    assert config["env"] == {}
    assert config["image"] == box.image
    assert config["interpreter"][:2] == ["/usr/bin/env", "-i"]
    assert config["interpreter"][-2:] == ["/bin/bash", "-c"]
    assert not any("docker.sock" in arg for arg in config["run_args"])
    assert env.get_template_vars()["system"] == "Linux"
    assert str(box.baseline) not in json.dumps(env.get_template_vars())


def test_native_reproduction_gate_prevents_any_command_before_reproducer(box, tmp_path):
    from agent.mini_runtime import ScopedEnvironment

    box.task = box.task.model_copy(
        update={
            "reproduction_script": "reproduce.py",
            "reproduction_command": "python /task/reproduce.py",
            "expected_failure": "unhashable type: 'Timeout'",
        }
    )
    (box.task_file.parent / "reproduce.py").write_text("import httpx\n")
    with patch("agent.mini_runtime.DockerEnvironment") as native:
        env = ScopedEnvironment(
            box,
            Budget(box.config.budgets),
            Trace(tmp_path / "native.jsonl", ""),
            tmp_path,
        )
    assert env.execute({"command": "cat source.py"})["returncode"] == 2
    native.return_value.execute.assert_not_called()
    native.return_value.execute.return_value = {
        "returncode": 1,
        "output": "TypeError: unhashable type: 'Timeout'",
        "exception_info": "",
    }
    env.execute({"command": "python /task/reproduce.py"})
    assert env.reproduction_verified
    assert (tmp_path / "reproduction-evidence.json").exists()


def test_native_timeout_removes_whole_container(box, tmp_path):
    from agent.mini_runtime import ScopedEnvironment

    with patch("agent.mini_runtime.DockerEnvironment") as native:
        env = ScopedEnvironment(
            box,
            Budget(box.config.budgets),
            Trace(tmp_path / "native.jsonl", ""),
            tmp_path,
        )
    native.return_value.container_id = "fake-container"
    native.return_value.execute.return_value = {
        "returncode": -1,
        "output": "",
        "extra": {"exception_type": "TimeoutExpired"},
    }
    with patch("agent.mini_runtime.subprocess.run") as remove:
        with pytest.raises(BudgetExhausted, match="tool_timeout"):
            env.execute({"command": "sleep 99"})
    assert remove.call_args.args[0] == ["docker", "rm", "-f", "fake-container"]
    assert native.return_value.container_id is None


def test_submission_only_guard_and_real_pipeline_exit_code(box, tmp_path):
    import subprocess

    from agent.mini_runtime import ScopedEnvironment

    box.config = box.config.model_copy(
        update={"stopping_protocol": "submit_priority_v1"}
    )
    budget = Budget(box.config.budgets)
    with patch("agent.mini_runtime.DockerEnvironment") as native:
        env = ScopedEnvironment(
            box, budget, Trace(tmp_path / "native.jsonl", ""), tmp_path
        )
    assert native.call_args.kwargs["interpreter"][-4:] == [
        "/bin/bash",
        "-o",
        "pipefail",
        "-c",
    ]

    def shell(action, **kwargs):
        # Trusted fixed test command only; no model-selected host execution.
        completed = subprocess.run(
            ["/bin/bash", "-o", "pipefail", "-c", action["command"]],
            capture_output=True,
            text=True,
        )
        return {
            "returncode": completed.returncode,
            "output": completed.stdout,
            "exception_info": "",
        }

    native.return_value.execute.side_effect = shell
    result = env.execute({"command": "false | tail -1"})
    assert result["returncode"] == 1
    budget.submission_only = True
    native.return_value.execute.reset_mock()
    assert env.execute({"command": "python -m pytest"})["returncode"] == 2
    native.return_value.execute.assert_not_called()


def test_controller_records_submission_separately_from_acceptance(box, tmp_path):
    from agent.controller import Controller
    from agent.schemas import ModelConfig

    box.config = box.config.model_copy(
        update={"stopping_protocol": "submit_priority_v1"}
    )
    with (
        patch("agent.controller.ScopedEnvironment") as environment,
        patch("agent.controller.DefaultAgent") as agent,
        patch("agent.controller.qualify") as evaluate,
        patch("agent.controller.MeteredModel") as metered,
    ):
        environment.return_value.isolation_checks.return_value = {}
        metered.return_value.usage = []
        metered.return_value.providers = []
        agent.return_value.run.return_value = {"exit_status": "Submitted"}
        evaluate.return_value = {"status": "checks_failed"}
        result = Controller(
            box, ModelConfig(), tmp_path, "", {"provider": {}, "parameters": {}}
        ).execute()
    assert metered.call_args.kwargs["submission_priority"] is True
    assert result["agent_submission"]["submitted"] is True
    assert result["patch_acceptance"] == "failed"
    assert (
        result["termination_policy"]["automatic_evaluation_at_budget_exhaustion"]
        is False
    )


def test_existing_test_entries_are_read_only_mounts(box):
    box.config = box.config.model_copy(
        update={"test_policy": "read_only_existing_new_test_files"}
    )
    (box.baseline / "tests/nested").mkdir()
    (box.baseline / "tests/nested/test_nested.py").write_text("assert True\n")
    command = box.command("test", "true")
    mounts = [command[i + 1] for i, arg in enumerate(command) if arg == "--mount"]
    assert any("target=/workspace/tests/test_existing.py,readonly" in m for m in mounts)
    assert any("target=/workspace/tests/nested,readonly" in m for m in mounts)
    assert any(
        "target=/workspace/tests" in m and not m.endswith(",readonly") for m in mounts
    )


def test_new_regressions_must_have_test_filename(box, tmp_path):
    box.config = box.config.model_copy(
        update={"test_policy": "read_only_existing_new_test_files"}
    )
    (box.workspace / "tests/helper.py").write_text("pass\n")
    with pytest.raises(InfrastructureError, match=r"new test_\*\.py"):
        collect_changes(box, tmp_path / "patch")
