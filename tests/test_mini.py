"""Exercise upstream agent/provider paths with mocked HTTP: no model API access."""

import json
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import requests
from minisweagent.agents.default import DefaultAgent
from minisweagent.exceptions import FormatError, Submitted

from agent.budget import Budget, BudgetExhausted
from agent.controller import Trace
from agent.mini_runtime import MeteredModel, openrouter_model, stock_config
from agent.schemas import Budgets, ModelConfig


def response(command="echo hello", tokens=10, cost=0.01):
    calls = (
        [
            {
                "id": "call1",
                "type": "function",
                "function": {
                    "name": "bash",
                    "arguments": json.dumps({"command": command}),
                },
            }
        ]
        if command
        else []
    )
    return {
        "provider": "mock-only",
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "Local scripted fixture.",
                    "tool_calls": calls,
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {
            "prompt_tokens": 20,
            "completion_tokens": tokens,
            "cost": cost,
            "completion_tokens_details": {"reasoning_tokens": 2},
        },
    }


@pytest.fixture
def metered(tmp_path):
    model = openrouter_model(
        ModelConfig(reasoning=None),
        {"provider": {}, "parameters": {}},
        "not-a-real-key",
    )
    return MeteredModel(
        model,
        Budget(Budgets(model_calls=2, generated_tokens=15)),
        Trace(tmp_path / "trace.jsonl", "not-a-real-key"),
    )


def test_native_prompt_effective_content_matches_pinned_package():
    import minisweagent
    import yaml

    native = Path(minisweagent.__file__).parent / "config/mini.yaml"
    assert yaml.safe_load(native.read_text()) == stock_config()
    assert stock_config()["agent"]["system_template"].startswith(
        "You are a helpful assistant"
    )


def test_stock_transport_and_parser_receive_remaining_token_cap(metered):
    with patch("requests.post", return_value=Mock(json=lambda: response())) as post:
        first = metered.query([])
        assert first["extra"]["actions"][0]["command"] == "echo hello"
        assert json.loads(post.call_args.kwargs["data"])["max_tokens"] == 15
    with patch(
        "requests.post", return_value=Mock(json=lambda: response(tokens=5))
    ) as post:
        metered.query([])
        assert json.loads(post.call_args.kwargs["data"])["max_tokens"] == 5
    with patch("requests.post") as post:
        with pytest.raises(BudgetExhausted):
            metered.query([])
        post.assert_not_called()
    assert metered.budget.generated == 15  # Reasoning is already in completion_tokens.


def test_failed_http_is_one_attempt_and_cost_remains_unknown(metered):
    failed = Mock(status_code=404, text="unsupported route")
    failed.raise_for_status.side_effect = requests.HTTPError("404")
    with patch("requests.post", return_value=failed) as post:
        with pytest.raises(Exception, match="404"):
            metered.query([])
        assert post.call_count == 1
    assert metered.budget.calls == 1
    assert metered.usage == []
    assert metered.budget.generated == 15


def test_format_errors_preserve_billed_usage_and_consume_repair(metered):
    with patch("requests.post", return_value=Mock(json=lambda: response(command=None))):
        with pytest.raises(FormatError):
            metered.query([])
    assert metered.usage[0]["cost"] == 0.01
    assert metered.budget.generated == 10
    assert metered.budget.repairs == 1


def test_missing_usage_stops_attempt(metered):
    raw = response()
    raw["usage"] = {"cost": 0.01}
    with patch("requests.post", return_value=Mock(json=lambda: raw)):
        with pytest.raises(ValueError, match="omitted completion"):
            metered.query([])
    assert metered.budget.generated == 15


def test_standard_agent_saves_native_trajectory_and_submits(metered, tmp_path):
    class FixtureEnvironment:
        def get_template_vars(self):
            return dict(system="Linux", release="", version="", machine="")

        def serialize(self):
            return {}

        def execute(self, action):
            assert action["command"] == "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
            raise Submitted(
                {"role": "exit", "content": "", "extra": {"exit_status": "Submitted"}}
            )

    config = stock_config()["agent"]
    agent = DefaultAgent(
        metered,
        FixtureEnvironment(),
        system_template=config["system_template"],
        instance_template=config["instance_template"],
        cost_limit=0,
        output_path=tmp_path / "trajectory.json",
    )
    with patch(
        "requests.post",
        return_value=Mock(
            json=lambda: response("echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT")
        ),
    ):
        assert agent.run("Local fixture")["exit_status"] == "Submitted"
    trajectory = json.loads((tmp_path / "trajectory.json").read_text())
    assert (
        trajectory["info"]["config"]["agent_type"]
        == "minisweagent.agents.default.DefaultAgent"
    )
    assert trajectory["trajectory_format"] == "mini-swe-agent-1.1"
    assert len(trajectory["messages"]) == 4


def test_rejected_request_retains_reported_usage_and_http_status(metered, tmp_path):
    failed = Mock(status_code=400)
    failed.json.return_value = {
        "error": {"message": "rejected"},
        "usage": {"prompt_tokens": 4, "completion_tokens": 3, "cost": 0.002},
    }
    failed.raise_for_status.side_effect = requests.HTTPError(
        "bad request", response=failed
    )
    with patch("requests.post", return_value=failed):
        with pytest.raises(Exception, match="400"):
            metered.query([])
    assert metered.usage[0]["cost"] == 0.002
    assert metered.budget.generated == 3
    events = [
        json.loads(line) for line in (tmp_path / "trace.jsonl").read_text().splitlines()
    ]
    assert events[-1]["response"]["status"] == 400
    assert events[-1]["response"]["usage"]["cost"] == 0.002


def test_revised_invoke_prompt_resolves_tool_order_and_keeps_budget(metered):
    from agent.provider import parameter_plan
    from agent.schemas import SandboxConfig

    config = SandboxConfig.model_validate_json(
        Path("configs/sandbox-mini-provider-repair.json").read_text()
    )
    model = ModelConfig.model_validate_json(
        Path("configs/qwen3-coder-provider-repair.json").read_text()
    )
    endpoint = {
        "provider_name": "DeepInfra",
        "tag": "deepinfra/turbo",
        "supported_parameters": ["tools", "tool_choice", "temperature", "max_tokens"],
    }
    plan = parameter_plan(model, {"data": {"endpoints": [endpoint]}})
    assert plan["provider"]["only"] == ["deepinfra/turbo"]
    assert plan["provider"]["allow_fallbacks"] is False
    with pytest.raises(Exception, match="excludes"):
        parameter_plan(
            model, {"data": {"endpoints": [{**endpoint, "tag": "google/vertex"}]}}
        )
    assert config.budgets.model_calls == 20
    assert config.budgets.generated_tokens == 6000
    assert config.budgets.seconds == 1200
    standard = stock_config(config.prompt_config)["agent"]

    class FixtureEnvironment:
        def get_template_vars(self):
            return dict(system="Linux", release="", version="", machine="")

        def serialize(self):
            return {}

        def execute(self, action):
            raise Submitted(
                {"role": "exit", "content": "", "extra": {"exit_status": "Submitted"}}
            )

    agent = DefaultAgent(
        metered,
        FixtureEnvironment(),
        system_template=standard["system_template"],
        instance_template=standard["instance_template"],
        cost_limit=0,
    )
    with patch(
        "requests.post",
        return_value=Mock(
            json=lambda: response("echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT")
        ),
    ) as post:
        agent.run("Issue and supplied reproduction script")
    body = json.loads(post.call_args.kwargs["data"])
    prompt = body["messages"][1]["content"]
    assert (
        "After a failed patch test, inspect the failure and the changed execution path before further broad searches."
        in prompt
    )
    assert "1. Execute the supplied read-only reproduction script first" in prompt
    assert "Create a script to reproduce" not in prompt
    assert "sandbox_write_file" not in prompt
    assert "The only available tool is `bash`" in prompt
    assert [tool["function"]["name"] for tool in body["tools"]] == ["bash"]


def test_submission_slot_and_token_reserve_reach_actual_request(metered):
    metered.submission_priority = True
    metered.budget = Budget(
        Budgets(model_calls=2, generated_tokens=300, response_tokens=300)
    )
    original = [{"role": "user", "content": "Issue"}]
    with patch(
        "requests.post", return_value=Mock(json=lambda: response(tokens=172))
    ) as post:
        metered.query(original)
    first = json.loads(post.call_args.kwargs["data"])
    assert first["max_tokens"] == 172  # Preserve 128 tokens for handoff.
    assert original == [{"role": "user", "content": "Issue"}]
    with patch(
        "requests.post",
        return_value=Mock(
            json=lambda: response(
                "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT", tokens=25
            )
        ),
    ) as post:
        metered.query(original)
    final = json.loads(post.call_args.kwargs["data"])
    assert final["max_tokens"] == 128
    assert "final request for submission" in final["messages"][-1]["content"]
    assert metered.budget.submission_only
    with patch("requests.post") as post:
        with pytest.raises(BudgetExhausted):
            metered.query(original)
        post.assert_not_called()


def test_token_reserve_can_trigger_submission_before_call_ceiling(metered):
    metered.submission_priority = True
    metered.budget = Budget(Budgets(model_calls=20, generated_tokens=128))
    with patch("requests.post", return_value=Mock(json=lambda: response(tokens=20))):
        metered.query([])
    assert metered.budget.submission_only
    assert metered.budget.calls == 1
