"""Versioned ceiling policy uses the existing runtime and total-budget boundary."""

import copy
import json
from pathlib import Path

import pytest

from agent.budget import BudgetExhausted
from agent.schemas import Budgets
from experiments.configuration_paths import configuration_path
from harbor_integration.host_comparison import ComparisonAgent, DeadlineBudget
from harbor_integration.host_response_ceiling import ResponseCeilingAgent
from harbor_integration.host_workspace import WorkspaceAgent


def test_versioned_policy_and_unchanged_runtime(tmp_path):
    root = Path(__file__).resolve().parents[1]
    policy = json.loads(
        (
            configuration_path(
                root, "configs/context-counter-response-6000-v2-model.json"
            )
        ).read_text()
    )["agents"][0]["kwargs"]["policy"]
    ResponseCeilingAgent(policy=policy, logs_dir=tmp_path)
    assert ResponseCeilingAgent.run is ComparisonAgent.run
    assert ResponseCeilingAgent.setup is WorkspaceAgent.setup
    assert ResponseCeilingAgent.quiesce is WorkspaceAgent.quiesce
    for field in [
        "model_calls",
        "generated_tokens",
        "response_tokens",
        "seconds",
        "repair_rounds",
        "tool_seconds",
    ]:
        bad = copy.deepcopy(policy)
        bad["budgets"][field] += 1
        with pytest.raises(ValueError, match="exact frozen completion policy"):
            ResponseCeilingAgent(policy=bad, logs_dir=tmp_path)


def test_response_cap_shrinks_to_remaining_total():
    budget = DeadlineBudget(Budgets(response_tokens=6000, generated_tokens=10000))
    assert budget.reserve("probe") == 6000
    budget.account("probe", {"completion_tokens": 6000}, 6000)
    assert budget.reserve("probe") == 4000
    budget.account("probe", {"completion_tokens": 4000}, 4000)
    with pytest.raises(BudgetExhausted):
        budget.reserve("probe")
