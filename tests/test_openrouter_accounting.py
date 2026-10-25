"""Native accounting and metadata reconciliation preserve billed failures and unknowns."""

import httpx
import pytest

from experiments.openrouter_accounting import (
    attempt_totals,
    fetch_generation,
    request_row,
)

RUN = {"run_id": "attempt", "model": "m"}


def response(cost=0.02):
    return {
        "response": {
            "id": "gen-test",
            "provider": "Actual provider",
            "usage": {"prompt_tokens": 10, "completion_tokens": 1500, "cost": cost},
            "choices": [
                {
                    "message": {
                        "tool_calls": [
                            {"function": {"arguments": '{"command":"unfinished'}}
                        ]
                    }
                }
            ],
        }
    }


def test_malformed_response_is_billed_and_raw_identity_preserved():
    r = request_row(RUN, 1, response())
    assert r["tool_format_error"] == "JSONDecodeError"
    assert (
        r["generation_id"] == "gen-test" and r["actual_provider"] == "Actual provider"
    )
    assert r["raw_response"] == response()["response"]
    total = attempt_totals([r, request_row(RUN, 2, {})])[0]
    assert total["requests"] == 2 and total["cost_usd_known"] == 0.02
    assert total["cost_usd"] is None and total["usage_complete"] is False
    assert total["reasoning_tokens"] is None


def test_reconciliation_supplements_missing_cost_without_mutating_response():
    req = response(None)
    metadata = {
        "http_status": 200,
        "body": {"data": {"id": "gen-test", "total_cost": 0.03}},
    }
    r = request_row(RUN, 1, req, metadata)
    assert r["reported_cost_usd"] is None and r["effective_cost_usd"] == 0.03
    assert r["cost_source"] == "generation_api"
    assert req["response"]["usage"]["cost"] is None


def test_conflict_or_wrong_generation_does_not_silently_replace_charge():
    metadata = {
        "http_status": 200,
        "body": {"data": {"id": "gen-test", "total_cost": 0.03}},
    }
    row = request_row(RUN, 1, response(), metadata)
    assert row["accounting_conflict"] and row["effective_cost_usd"] is None
    metadata["body"]["data"]["id"] = "gen-other"
    row = request_row(RUN, 1, response(None), metadata)
    assert row["effective_cost_usd"] is None


def test_fetch_is_get_metadata_only_cached_and_excludes_key(tmp_path):
    calls = []

    def handler(req):
        calls.append(req)
        assert req.method == "GET" and req.url.path == "/api/v1/generation"
        assert req.url.params["id"] == "gen-test"
        return httpx.Response(
            200, json={"data": {"id": "gen-test", "total_cost": 0.02}}
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        fetch_generation("gen-test", tmp_path, "private-test-key", client)
        fetch_generation("gen-test", tmp_path, "private-test-key", client)
    assert len(calls) == 1
    assert "private-test-key" not in (tmp_path / "gen-test.json").read_text()


def test_bad_id_never_reaches_transport(tmp_path):
    with pytest.raises(ValueError):
        fetch_generation("../../secrets", tmp_path, "key")


def test_metadata_404_remains_unknown_not_zero(tmp_path):
    with httpx.Client(
        transport=httpx.MockTransport(
            lambda r: httpx.Response(404, json={"error": {"message": "not found"}})
        )
    ) as client:
        metadata = fetch_generation("gen-test", tmp_path, "key", client)
    assert request_row(RUN, 1, response(None), metadata)["effective_cost_usd"] is None
