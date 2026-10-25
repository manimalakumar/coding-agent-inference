"""Harbor owns execution/configuration; validate the thin integration boundaries."""

import json
from pathlib import Path
from unittest.mock import Mock

import pytest
from harbor.models.job.config import JobConfig
from harbor.models.task.config import TaskConfig

from agent.budget import Budget, BudgetExhausted
from agent.schemas import Budgets
from harbor_integration.host_mini import HarborTools


def test_native_harbor_schema_and_separate_verifier():
    task = TaskConfig.model_validate_toml(
        Path("harbor_tasks/LC-TIMEOUT-001/task.toml").read_text()
    )
    assert task.verifier.environment_mode.value == "separate"
    assert task.environment.network_mode.value == "no-network"
    assert task.verifier.environment.network_mode.value == "no-network"
    assert task.environment.docker_image.startswith("sha256:")
    assert task.verifier.environment.docker_image != task.environment.docker_image
    assert task.verifier.collect  # Native export hook, no custom container controller.
    for name in ["model", "baseline", "reference"]:
        job = JobConfig.model_validate_json(
            Path(f"configs/harbor-{name}.json").read_text()
        )
        assert job.n_attempts == 1
        assert job.retry.max_retries == 0
        assert (
            job.agents[0].import_path == "harbor_integration.host_mini:HostMiniSweAgent"
        )


def test_model_commands_cannot_bypass_reproduction_or_budget():
    env = Mock()
    budget = Budget(Budgets())
    tools = HarborTools(env, Mock(), budget, Mock())
    assert tools.execute({"command": "cat /etc/passwd"})["returncode"] == 2
    env.exec.assert_not_called()
    budget.started -= 601
    with pytest.raises(BudgetExhausted):
        tools.execute({"command": "python /task/reproduce.py"})
    env.exec.assert_not_called()


def test_submission_only_slot_cannot_execute_development_commands():
    env = Mock()
    budget = Budget(Budgets())
    budget.submission_only = True
    tools = HarborTools(env, Mock(), budget, Mock())
    assert tools.execute({"command": "pytest"})["returncode"] == 2
    env.exec.assert_not_called()


def test_harbor_prompt_policy_matches_runtime():
    prompt = Path("configs/mini-swe-harbor-initial-instructions.yaml").read_text()
    assert "verification after termination" in prompt
    assert "Automatic evaluation at budget exhaustion is disabled" not in prompt
    assert "Existing test files are read-only." in prompt
    assert "connect_timeout" not in prompt  # No previous diagnosis supplied.
    job = json.loads(Path("configs/harbor-model.json").read_text())
    policy = job["agents"][0]["kwargs"]["policy"]
    assert policy["budgets"]["model_calls"] == 30
    assert policy["budgets"]["generated_tokens"] == 10000
    assert policy["budgets"]["seconds"] == 1200
    assert policy["model"]["provider"]["only"] == ["minimax/fp8"]
    assert policy["model"]["provider"]["allow_fallbacks"] is False
