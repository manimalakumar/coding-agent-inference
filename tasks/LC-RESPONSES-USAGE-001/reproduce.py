"""Offline reproducer for lost Responses API token details."""

import json

from langchain_openai.chat_models.base import _create_usage_metadata_responses

usage = {
    "input_tokens": 120,
    "input_tokens_details": {"cached_tokens": 40},
    "output_tokens": 30,
    "output_tokens_details": {"reasoning_tokens": 12},
    "total_tokens": 150,
}
expected = {
    "input_tokens": 120,
    "output_tokens": 30,
    "total_tokens": 150,
    "input_token_details": {"cache_read": 40},
    "output_token_details": {"reasoning": 12},
}
actual = _create_usage_metadata_responses(usage)
print(json.dumps({"usage": usage, "actual": actual, "expected": expected}, indent=2))
assert actual == expected, "Responses API usage metadata mismatch"
print("Responses API usage metadata preserved")
