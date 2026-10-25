"""All published arithmetic comes from per-request and per-attempt evidence."""

from decimal import Decimal
from typing import Any

import pandas as pd

from public_replay.bundle import Bundle
from public_replay.usage import normalize_usage


def measured_attempts(bundle: Bundle) -> list[dict[str, Any]]:
    attempts = bundle.read("attempts.json")
    requests = bundle.read("requests.json")
    ids = [a["run_id"] for a in attempts]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate public run identities")
    keys = [(r["run_id"], r["request_number"]) for r in requests]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate public request identities")
    result = []
    for attempt in attempts:
        rows = [r for r in requests if r["run_id"] == attempt["run_id"]]
        normalized = normalize_usage(
            [{"response": {"usage": r["reported_usage"]}} for r in rows],
            expected_calls=attempt["calls"],
            calls_complete=attempt["calls_complete"],
        )
        known = sum(
            (
                Decimal(
                    str(
                        r["reported_cost_usd"]
                        if r["accounting_conflict"]
                        else r["effective_cost_usd"]
                    )
                )
                for r in rows
                if (
                    r["reported_cost_usd"]
                    if r["accounting_conflict"]
                    else r["effective_cost_usd"]
                )
                is not None
            ),
            Decimal(0),
        )
        complete = (
            attempt["calls_complete"]
            and len(rows) == attempt["calls"]
            and all(r["effective_cost_usd"] is not None for r in rows)
        )
        result.append(
            {
                **attempt,
                "usage": normalized,
                "known_spend_usd": float(known),
                "complete_spend_usd": float(known) if complete else None,
                "cost_complete": complete,
            }
        )
    return result


def comparison(bundle: Bundle, scope: str = "held_out_six") -> pd.DataFrame:
    selected = bundle.read("study.json")["scopes"][scope]["run_ids"]
    rows = {r["run_id"]: r for r in measured_attempts(bundle)}
    if set(selected) - rows.keys():
        raise ValueError("Declared study scope refers to missing public attempts")
    frame = pd.DataFrame([rows[rid] for rid in selected])
    output = []
    if frame.empty:
        return pd.DataFrame()
    for model, group in frame.groupby("model", sort=True):
        accepted = int(group["patch_accepted"].eq(True).sum())
        known = sum((Decimal(str(v)) for v in group["known_spend_usd"]), Decimal(0))
        complete = bool(group["cost_complete"].all())
        output.append(
            {
                "model": model,
                "launches": len(group),
                "accepted": accepted,
                "interruptions": int(group["infrastructure_interrupted"].sum()),
                "explicit_submissions": int(group["explicit_submission"].sum()),
                "known_spend_usd": float(known),
                "complete_spend_usd": float(known) if complete else None,
                "cost_per_accepted_patch_usd": float(known / accepted)
                if complete and accepted
                else None,
                "calls": int(group["calls"].sum()),
            }
        )
    return pd.DataFrame(output)


def outcomes(bundle: Bundle) -> pd.DataFrame:
    ids = set(bundle.read("study.json")["scopes"]["held_out_six"]["run_ids"])
    frame = pd.DataFrame([r for r in measured_attempts(bundle) if r["run_id"] in ids])
    return (
        frame.groupby(["task_id", "model"], sort=True)
        .agg(
            launches=("run_id", "size"),
            accepted=("patch_accepted", "sum"),
            interruptions=("infrastructure_interrupted", "sum"),
        )
        .reset_index()
    )


def operational(bundle: Bundle) -> pd.DataFrame:
    """Expenses remain disjoint from launch denominators, grouped by original scope."""
    requests = bundle.read("requests.json")
    attempts = measured_attempts(bundle)
    output = []
    for scope in sorted({r["accounting_scope"] for r in requests}):
        rows = [r for r in requests if r["accounting_scope"] == scope]
        known = sum(
            (
                Decimal(
                    str(
                        r["reported_cost_usd"]
                        if r["accounting_conflict"]
                        else r["effective_cost_usd"]
                    )
                )
                for r in rows
                if (
                    r["reported_cost_usd"]
                    if r["accounting_conflict"]
                    else r["effective_cost_usd"]
                )
                is not None
            ),
            Decimal(0),
        )
        incomplete_attempts = (
            sum(not a["cost_complete"] for a in attempts)
            if scope == "coding_attempt"
            else 0
        )
        complete = not incomplete_attempts and all(
            r["effective_cost_usd"] is not None for r in rows
        )
        output.append(
            {
                "attempts_with_incomplete_cost": incomplete_attempts,
                "scope": scope,
                "requests": len(rows),
                "known_spend_usd": float(known),
                "complete_spend_usd": float(known) if complete else None,
            }
        )
    return pd.DataFrame(output)


def check_status(
    exit_code: int | None, expected: int, *, present: bool, ran: bool
) -> str:
    if not ran:
        return "not run"
    if not present or exit_code is None:
        return "missing"
    return "pass" if exit_code == expected else "fail"


def request_payload(bundle: Bundle, model: str) -> dict[str, Any]:
    """Show a redacted saved preparation contract; this function never invokes it."""
    config = bundle.read("study.json")["model_preparation"][model]
    return {
        **config,
        "messages": [
            {"role": "system", "content": "[task-specific instructions not bundled]"},
            {
                "role": "user",
                "content": "[fresh issue and permitted pre-fix source only]",
            },
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "bash",
                    "parameters": {
                        "type": "object",
                        "properties": {"command": {"type": "string"}},
                        "required": ["command"],
                    },
                },
            }
        ],
        "usage": {"include": True},
    }
