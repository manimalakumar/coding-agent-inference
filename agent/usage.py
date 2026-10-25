"""Shared reporting aliases for the unknown-aware usage normalizer."""

from public_replay.usage import normalize_usage


def usage(records, expected_calls=None, *, calls_complete=True):
    summary = normalize_usage(records, expected_calls, calls_complete=calls_complete)
    return {
        **summary,
        "input_tokens_reported": summary["input_tokens_known"],
        "generated_tokens_reported_including_reasoning": summary["output_tokens_known"],
        "reasoning_tokens_reported": summary["reasoning_tokens_known"],
        "known_reported_cost_usd": summary["cost_usd_known"],
        "cost_complete": summary["cost_usd_complete"],
        "total_cost_usd": summary["cost_usd"],
        "requests_without_cost_record": summary["cost_usd_missing_requests"],
        "completed_requests": sum(
            isinstance(r.get("response"), dict) and bool(r["response"].get("choices"))
            for r in records
        ),
    }
