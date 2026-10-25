"""Unknown-aware accounting v2. Completion tokens already include reasoning."""

import math
from decimal import Decimal
from typing import Any


def numeric(value: Any, *, tokens: bool = False) -> bool:
    """Reject nulls, booleans, nonfinite, negative and fractional token counts."""
    if isinstance(value, bool) or not isinstance(value, (int, float, Decimal)):
        return False
    try:
        return (
            math.isfinite(value) and value >= 0 and (not tokens or int(value) == value)
        )
    except OverflowError, ValueError:
        return False


def normalize_usage(
    records: list[dict[str, Any]],
    expected_calls: int | None = None,
    *,
    calls_complete: bool = True,
) -> dict[str, Any]:
    """Keep partial sums separate from complete totals, including paid errors."""
    expected = len(records) if expected_calls is None else expected_calls
    if (
        isinstance(expected, bool)
        or not isinstance(expected, int)
        or expected < len(records)
    ):
        raise ValueError("Expected call count cannot be smaller than observed records")
    values = []
    for record in records:
        response = record.get("response")
        response = response if isinstance(response, dict) else {}
        raw = response.get("provider_response")
        raw = raw if isinstance(raw, dict) else response
        usage = raw.get("usage")
        values.append(usage if isinstance(usage, dict) else {})
    values.extend({} for _ in range(expected - len(values)))
    result: dict[str, Any] = {
        "requests": expected,
        "observed_requests": len(records),
        "calls_complete": calls_complete and expected == len(records),
    }
    fields = {
        "input_tokens": ("prompt_tokens", None),
        "output_tokens": ("completion_tokens", None),
        "reasoning_tokens": ("reasoning_tokens", "completion_tokens_details"),
        "cached_input_tokens": ("cached_tokens", "prompt_tokens_details"),
        "cache_write_tokens": ("cache_write_tokens", "prompt_tokens_details"),
        "cost_usd": ("cost", None),
    }
    for name, (key, parent) in fields.items():
        numbers = []
        for value in values:
            detail = value.get(parent) if parent else value
            detail = detail if isinstance(detail, dict) else {}
            number = detail.get(key)
            numbers.append(
                number if numeric(number, tokens=name != "cost_usd") else None
            )
        complete = calls_complete and all(number is not None for number in numbers)
        known = sum((Decimal(str(v)) for v in numbers if v is not None), Decimal(0))
        known_value = float(known) if name == "cost_usd" else int(known)
        result[name + "_known"] = known_value
        result[name + "_complete"] = complete
        result[name + "_missing_requests"] = sum(number is None for number in numbers)
        result[name] = known_value if complete else None
    result["token_usage_complete"] = (
        result["input_tokens_complete"] and result["output_tokens_complete"]
    )
    return result
