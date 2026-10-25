"""Trusted offline checks: Responses API accounting and unaffected chat accounting."""

import copy
from unittest.mock import AsyncMock, patch

import pytest
from langchain_openai import ChatOpenAI
from langchain_openai.chat_models.base import (
    _create_usage_metadata,
    _create_usage_metadata_responses,
)
from openai.types.responses import Response


@pytest.mark.parametrize(
    "cached,reasoning", [(43, 11), (0, 0), (0, 9), (18, 0), (77, 21)]
)
def test_response_details_and_totals(cached, reasoning):
    usage = {
        "input_tokens": 173,
        "input_tokens_details": {"cached_tokens": cached},
        "output_tokens": 61,
        "output_tokens_details": {"reasoning_tokens": reasoning},
        "total_tokens": 234,
    }
    original = copy.deepcopy(usage)
    result = _create_usage_metadata_responses(usage)
    assert result == {
        "input_tokens": 173,
        "output_tokens": 61,
        "total_tokens": 234,
        "input_token_details": {"cache_read": cached},
        "output_token_details": {"reasoning": reasoning},
    }
    assert usage == original


@pytest.mark.parametrize("details", [None, {}])
def test_optional_details(details):
    result = _create_usage_metadata_responses(
        {
            "input_tokens": 17,
            "output_tokens": 4,
            "total_tokens": 21,
            "input_tokens_details": details,
            "output_tokens_details": details,
        }
    )
    assert result["input_tokens"] == 17
    assert result["output_tokens"] == 4
    assert result["total_tokens"] == 21
    assert result.get("input_token_details", {}) == {}
    assert result.get("output_token_details", {}) == {}


def test_absent_details_and_total_fallback():
    result = _create_usage_metadata_responses({"input_tokens": 7, "output_tokens": 8})
    assert result["total_tokens"] == 15
    assert result.get("input_token_details", {}) == {}
    assert result.get("output_token_details", {}) == {}
    empty = _create_usage_metadata_responses({})
    assert (empty["input_tokens"], empty["output_tokens"], empty["total_tokens"]) == (
        0,
        0,
        0,
    )


def test_chat_completions_unchanged():
    result = _create_usage_metadata(
        {
            "prompt_tokens": 30,
            "completion_tokens": 20,
            "total_tokens": 50,
            "prompt_tokens_details": {"cached_tokens": 9, "audio_tokens": 2},
            "completion_tokens_details": {"reasoning_tokens": 7, "audio_tokens": 1},
        }
    )
    assert result == {
        "input_tokens": 30,
        "output_tokens": 20,
        "total_tokens": 50,
        "input_token_details": {"audio": 2, "cache_read": 9},
        "output_token_details": {"audio": 1, "reasoning": 7},
    }


def response_fixture():
    return Response(
        id="resp_offline",
        created_at=1750000000,
        model="gpt-4.1-mini",
        object="response",
        parallel_tool_calls=True,
        tools=[],
        tool_choice="auto",
        output=[],
        usage={
            "input_tokens": 91,
            "input_tokens_details": {"cached_tokens": 14},
            "output_tokens": 37,
            "output_tokens_details": {"reasoning_tokens": 6},
            "total_tokens": 128,
        },
    )


def check_message(message):
    assert message.usage_metadata == {
        "input_tokens": 91,
        "output_tokens": 37,
        "total_tokens": 128,
        "input_token_details": {"cache_read": 14},
        "output_token_details": {"reasoning": 6},
    }


@pytest.mark.parametrize("output_version", ["v0", "responses/v1"])
def test_public_sync_conversion(output_version):
    model = ChatOpenAI(
        api_key="sk-offline-placeholder",
        model="gpt-4.1-mini",
        use_responses_api=True,
        output_version=output_version,
    )
    with patch.object(
        model.root_client.responses, "create", return_value=response_fixture()
    ) as create:
        check_message(model.invoke("offline fixture"))
    create.assert_called_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("output_version", ["v0", "responses/v1"])
async def test_public_async_conversion(output_version):
    model = ChatOpenAI(
        api_key="sk-offline-placeholder",
        model="gpt-4.1-mini",
        use_responses_api=True,
        output_version=output_version,
    )
    with patch.object(
        model.root_async_client.responses,
        "create",
        new=AsyncMock(return_value=response_fixture()),
    ) as create:
        check_message(await model.ainvoke("offline fixture"))
    create.assert_awaited_once()
