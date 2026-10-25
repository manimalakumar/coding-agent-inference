"""Response-ceiling setting: retain the frozen runtime; version its policy guard."""

import json
from pathlib import Path

from experiments.configuration_paths import configuration_path
from harbor_integration.host_comparison import ComparisonAgent
from harbor_integration.host_task import HostMiniSweAgent
from harbor_integration.host_workspace import POLICY_ID


class ResponseCeilingAgent(ComparisonAgent):
    def __init__(self, *, policy, fixture=None, **kwargs):
        HostMiniSweAgent.__init__(self, policy=policy, fixture=None, **kwargs)
        if fixture not in (
            None,
            "baseline",
            "reference",
            "reference_budget",
            "reference_error",
        ):
            raise ValueError("Unsupported trusted fixture")
        expected = json.loads(
            (
                configuration_path(
                    Path(__file__).resolve().parents[1],
                    "configs/completion-response-6000-v1.json",
                )
            ).read_text()
        )
        if (
            policy["completion_policy"] != expected
            or expected["policy_id"] != POLICY_ID
            or policy["budgets"] != expected["agent_budgets"]
        ):
            raise ValueError(
                "Response-ceiling comparison requires its exact frozen completion policy"
            )
        self.fixture = fixture
