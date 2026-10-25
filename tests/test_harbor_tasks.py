"""Reusable task gates preserve the pre-fix boundary and fixed pilot policy."""

import asyncio
import json
from concurrent.futures import Future
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from agent.budget import Budget
from agent.schemas import Budgets, Task
from harbor_integration.host_task import HarborTools


@pytest.mark.parametrize(
    "output,accepted",
    [
        ("AssertionError: Responses API usage metadata mismatch", True),
        ("NameError: name 'httpx' is not defined", False),
        ("ImportError: missing dependency", False),
    ],
)
def test_task_specific_reproduction_never_accepts_unrelated_errors(
    monkeypatch, output, accepted
):
    future = Future()
    future.set_result(SimpleNamespace(return_code=1, stdout=output, stderr=""))
    monkeypatch.setattr(asyncio, "run_coroutine_threadsafe", lambda *_: future)
    tools = HarborTools(
        Mock(),
        Mock(),
        Budget(Budgets()),
        Mock(),
        "Responses API usage metadata mismatch",
    )
    if accepted:
        assert (
            tools.execute({"command": "python /task/reproduce.py"})["returncode"] == 1
        )
        assert tools.reproduced
    else:
        with pytest.raises(RuntimeError, match="did not reproduce"):
            tools.execute({"command": "python /task/reproduce.py"})
        assert not tools.reproduced


def test_usage_task_keeps_fixed_model_policy_without_timeout_or_fix_context():
    task = Task.model_validate_json(
        Path("tasks/LC-RESPONSES-USAGE-001/task.json").read_text()
    )
    assert task.commit == "9aa75eaef387459ce7e2beaad25c4607ac626d24"
    config = json.loads(Path("configs/harbor-responses-usage-model.json").read_text())
    policy = config["agents"][0]["kwargs"]["policy"]
    assert config["agents"][0]["kwargs"].get("fixture") is None
    assert policy["budgets"]["model_calls"] == 30
    assert policy["budgets"]["generated_tokens"] == 10000
    assert policy["budgets"]["seconds"] == 1200
    assert policy["model"]["model"] == "minimax/minimax-m2.5"
    assert policy["model"]["provider"]["only"] == ["minimax/fp8"]
    assert policy["source_provenance"]["starting_commit"] == task.commit
    context = task.instructions + policy["mini_config"]["agent"]["instance_template"]
    assert "httpx.Timeout" not in context
    assert "Existing test files are read-only." in context
    assert "0189c50570913d73afe8c854f95dcfd5e7d6dd71" not in context
    assert "31756" not in context


def test_core_task_uses_same_agent_and_budget_with_core_only_write_mounts():
    import tomllib

    job = json.loads(
        Path("configs/harbor-annotated-tools-submission-model.json").read_text()
    )
    policy = job["agents"][0]["kwargs"]["policy"]
    prior = json.loads(Path("configs/harbor-responses-usage-model.json").read_text())[
        "agents"
    ][0]["kwargs"]["policy"]
    assert (
        job["agents"][0]["import_path"]
        == "harbor_integration.host_mini_v2:HostMiniSweAgent"
    )
    assert policy["budgets"] == prior["budgets"]
    assert policy["model"] == prior["model"]
    assert policy["mini_config"] == prior["mini_config"]
    task = tomllib.loads(
        Path("harbor_tasks/LC-ANNOTATED-TOOLS-001/task.toml").read_text()
    )
    assert task["environment"]["workdir"] == "/workspace/libs/core"
    compose = Path(
        "harbor_tasks/LC-ANNOTATED-TOOLS-001/environment/docker-compose.yaml"
    ).read_text()
    assert "production:/workspace/libs/core/langchain_core/tools" in compose
    assert "development-tests:/workspace/libs/core/tests/unit_tests" in compose
    assert "partners/openai" not in compose
    context = Path("harbor_tasks/LC-ANNOTATED-TOOLS-001/instruction.md").read_text()
    assert "ae3551c96b590790bc7779bd2609f5cb425bdbcd" not in context
    assert "get_type_hints" not in context
    assert "Existing test files are read-only." in context
