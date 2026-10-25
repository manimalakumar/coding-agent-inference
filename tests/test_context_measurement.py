"""Native accounting and proxy breakdowns stay distinct, including incomplete receipts."""

import json

import pytest

from experiments.context_measurement import breakdown, measure_trace, plot_paths


class WordProxy:
    def count(self, text):
        return len(text.split())


def request(n, messages):
    return {
        "event": "model_request",
        "number": n,
        "body": {"messages": messages, "max_tokens": 6000},
    }


def response(usage, **choice):
    return {
        "event": "model_response",
        "response": {"usage": usage, "choices": [choice]},
    }


def test_missing_charge_stays_unknown_and_cached_tokens_stay_in_context():
    events = [
        request(1, []),
        response(
            {
                "prompt_tokens": 10000,
                "prompt_tokens_details": {"cached_tokens": 8000},
                "cost": 0.1,
            }
        ),
        request(2, []),
        response({"prompt_tokens": 11000}),
        request(3, []),
        response({"prompt_tokens": 12000, "cost": 0.2}, finish_reason="length"),
    ]
    rows, _ = measure_trace(events, "", WordProxy())
    assert rows[0]["native_input_tokens"] == 10000
    assert rows[0]["native_cached_input_tokens"] == 8000
    assert rows[0]["output_ceiling_tokens"] == 6000  # Output cap does not limit input.
    assert [r["cumulative_reported_cost_usd"] for r in rows] == [0.1, None, None]
    assert rows[-1]["cumulative_known_reported_cost_usd"] == 0.3
    assert rows[-1]["native_cached_input_tokens"] is None


def test_malformed_response_is_billed_and_missing_response_retained():
    events = [
        request(1, []),
        response(
            {"cost": 0.25},
            finish_reason="tool_calls",
            message={"tool_calls": [{"function": {"arguments": '{"bad'}}]},
        ),
        request(2, []),
    ]
    rows, _ = measure_trace(events, "", WordProxy())
    assert len(rows) == 2
    assert rows[0]["reported_cost_usd"] == 0.25
    assert rows[1]["response_received"] is False
    assert rows[1]["native_input_tokens"] is None
    assert rows[1]["cumulative_reported_cost_usd"] is None
    assert rows[1]["cumulative_known_reported_cost_usd"] == 0.25


def test_task_tool_and_workflow_are_separate_estimates():
    body = {
        "messages": [
            {"role": "system", "content": "system rules"},
            {"role": "user", "content": "prefix task instruction suffix"},
            {
                "role": "assistant",
                "content": "plan words",
                "tool_calls": [{"function": {"arguments": '{"command":"cat file"}'}}],
            },
            {"role": "tool", "content": "source result text"},
        ],
        "tools": [{"name": "bash"}],
    }
    totals, messages = breakdown(body, "task instruction", WordProxy())
    assert totals["system_instructions"] == 2
    assert totals["task_instructions"] == 2
    assert totals["workflow_instructions"] == 2
    assert totals["tool_results"] == 3
    assert totals["assistant_tool_calls"] > 0 and totals["tool_schema"] > 0
    assert messages[-1]["role"] == "tool"
    assert messages[-1]["estimated_tokens_by_category"] == {"tool_results": 3}


def test_zero_cost_is_known_and_tool_event_is_associated_with_prior_request():
    events = [
        request(1, []),
        response({"cost": 0, "prompt_tokens": 0}),
        {
            "event": "tool_result",
            "tool_call": {"tool_call_id": "a"},
            "result": {"returncode": 1},
        },
        request(2, []),
    ]
    rows, tools = measure_trace(events, "", WordProxy())
    assert rows[0]["cumulative_reported_cost_usd"] == 0
    assert rows[0]["cumulative_cost_complete"] is True
    assert tools[0]["request"] == 1


def test_plot_replay_rejects_stale_data(tmp_path):
    folder = tmp_path / "reports/context-measurement-v1"
    folder.mkdir(parents=True)
    (folder / "plots.json").write_text(json.dumps({"data_sha256": "old", "plots": {}}))
    with pytest.raises(ValueError, match="stale"):
        plot_paths(tmp_path, {"data_sha256": "new"})
