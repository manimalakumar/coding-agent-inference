"""Result accounting, provider compatibility, and first-command enforcement."""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from agent.provider import ConfigurationError, parameter_plan
from agent.results import refresh, totals, usage
from agent.schemas import ModelConfig, Task
from experiments.configuration_paths import configuration_path


def endpoint(*parameters):
    return {
        "data": {
            "endpoints": [
                {
                    "provider_name": "test",
                    "supported_parameters": [
                        "tools",
                        "tool_choice",
                        "temperature",
                        "max_tokens",
                        *parameters,
                    ],
                }
            ]
        }
    }


def test_unsupported_reasoning_is_never_sent():
    plan = parameter_plan(ModelConfig(), endpoint())
    assert "reasoning" not in plan["parameters"]
    assert plan["omitted_unsupported_parameters"] == ["reasoning"]
    assert plan["provider"]["require_parameters"] is True


def test_supported_reasoning_is_preserved():
    plan = parameter_plan(ModelConfig(), endpoint("reasoning"))
    assert plan["parameters"]["reasoning"] == {"enabled": False}


def test_requested_reasoning_or_tools_cannot_silently_disappear():
    with pytest.raises(ConfigurationError, match="reasoning"):
        parameter_plan(ModelConfig(reasoning={"enabled": True}), endpoint())
    with pytest.raises(ConfigurationError, match="required agent"):
        parameter_plan(ModelConfig(), {"data": {"endpoints": []}})


def test_excluded_provider_fails_preflight():
    with pytest.raises(ConfigurationError, match="excludes"):
        parameter_plan(ModelConfig(provider={"only": ["other"]}), endpoint())


def test_routing_uses_provider_slugs_instead_of_display_names():
    metadata = endpoint()
    metadata["data"]["endpoints"][0]["tag"] = "test/region"
    plan = parameter_plan(ModelConfig(), metadata)
    assert plan["provider"]["only"] == ["test/region"]


def test_twelve_calls_do_not_expand_generation_budget():
    from agent.budget import Budget, BudgetExhausted
    from agent.schemas import Budgets

    budget = Budget(Budgets())
    for _ in range(12):
        reserved = budget.reserve("agent")
        budget.account("agent", {"completion_tokens": 10}, reserved)
    with pytest.raises(BudgetExhausted, match="model_call"):
        budget.reserve("agent")
    assert budget.generated == 120
    tokens = Budget(Budgets(response_tokens=6000))
    reserved = tokens.reserve("agent")
    tokens.account("agent", {"completion_tokens": 6000}, reserved)
    with pytest.raises(BudgetExhausted, match="generated_token"):
        tokens.reserve("agent")


def test_failed_and_successful_attempts_both_count_in_cost_numerator():
    rows = [
        {
            "model": "m",
            "run_id": "failure",
            "requests": 3,
            "success": False,
            "known_reported_cost_usd": 0.6,
            "requests_without_cost_record": 0,
        },
        {
            "model": "m",
            "run_id": "success",
            "requests": 2,
            "success": True,
            "known_reported_cost_usd": 0.4,
            "requests_without_cost_record": 0,
        },
    ]
    result = totals(rows)["m"]
    assert result["cost_per_successful_task_usd"] == 1.0
    assert result["requests_including_unsuccessful"] == 5
    assert result["successful_tasks"] == 1


def test_zero_success_and_missing_cost_do_not_look_free():
    rows = [
        {
            "model": "m",
            "run_id": "failure",
            "requests": 1,
            "success": False,
            "known_reported_cost_usd": 0,
            "requests_without_cost_record": 1,
        }
    ]
    result = totals(rows)["m"]
    assert result["cost_per_successful_task_usd"] is None
    assert result["total_cost_all_attempts_usd"] is None
    assert result["metric_status"] == "undefined_no_successes"
    rows[0]["success"] = True
    assert totals(rows)["m"]["metric_status"] == "incomplete_cost_records"


def test_ledger_splits_routing_failure_without_double_counting(tmp_path):
    run = tmp_path / "runs/test-run"
    run.mkdir(parents=True)
    (run / "metadata.json").write_text(
        json.dumps({"model": {"model": "m"}, "task_id": "T"})
    )
    (run / "result.json").write_text(
        json.dumps({"termination_reason": "model_call_budget_exhausted"})
    )
    events = [
        {"event": "model_request", "number": 1, "body": {"model": "m"}},
        {
            "event": "model_response",
            "response": {"error": "provider_http_error", "status": 404},
        },
        {"event": "model_request", "number": 2, "body": {"model": "m"}},
        {
            "event": "model_response",
            "response": {
                "choices": [{}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "cost": 0.2},
            },
        },
    ]
    (run / "trace.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events))
    first = refresh(tmp_path)
    second = refresh(tmp_path)
    assert first == second
    assert first["rows"] == 2
    assert first["models"]["m"]["requests_including_unsuccessful"] == 2
    assert first["models"]["m"]["known_reported_cost_all_attempts_usd"] == 0.2


def test_reasoning_subset_is_not_double_counted():
    result = usage(
        [
            {
                "response": {
                    "usage": {
                        "prompt_tokens": 10,
                        "completion_tokens": 100,
                        "cost": 0.1,
                        "completion_tokens_details": {"reasoning_tokens": 40},
                    }
                }
            }
        ]
    )
    assert result["generated_tokens_reported_including_reasoning"] == 100
    assert result["reasoning_tokens_reported"] == 40


def test_task_reproducer_must_be_mount_local_and_command_exact():
    with pytest.raises(ValidationError):
        Task(
            task_id="T",
            commit="a" * 40,
            package="p",
            instructions="",
            source_paths=(),
            test_paths=(),
            development_checks=(),
            reproduction_script="../escape.py",
        )


def test_pilot2_changes_only_call_cap_and_schema():
    root = Path(__file__).resolve().parents[1]
    old = json.loads((configuration_path(root, "configs/sandbox-v1.json")).read_text())
    new = json.loads((configuration_path(root, "configs/sandbox-v2.json")).read_text())
    assert new["budgets"]["model_calls"] == 12
    assert new["budgets"]["generated_tokens"] == 6000
    new["budgets"]["model_calls"] = 6
    new["schema_version"] = 1
    assert old == new


def test_costs_are_separate_for_harness_and_inference_configurations():
    from agent.results import configuration_totals

    common = {
        "model": "m",
        "requests": 1,
        "success": False,
        "requests_without_cost_record": 0,
    }
    rows = [
        {
            **common,
            "run_id": "a",
            "setting_checksum": "setting-a",
            "model_configuration_checksum": "provider-a",
            "known_reported_cost_usd": 0.02,
        },
        {
            **common,
            "run_id": "b",
            "setting_checksum": "setting-b",
            "model_configuration_checksum": "provider-a",
            "known_reported_cost_usd": 0.03,
        },
        {
            **common,
            "run_id": "c",
            "setting_checksum": "setting-b",
            "model_configuration_checksum": "provider-b",
            "known_reported_cost_usd": 0.04,
        },
    ]
    groups = configuration_totals(rows)
    assert len(groups) == 3
    assert [g["known_reported_cost_all_attempts_usd"] for g in groups] == [
        0.02,
        0.03,
        0.04,
    ]
    assert totals(rows)["m"]["known_reported_cost_all_attempts_usd"] == 0.09
    assert all(g["cost_per_successful_task_usd"] is None for g in groups)
