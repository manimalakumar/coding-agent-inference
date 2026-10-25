"""OpenRouter-native accounting views and opt-in, host-only Generation API audits.

Original traces, controller receipts and task verdicts are never rewritten.
"""

import argparse
import hashlib
import json
import re
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

from agent.results import request_records
from experiments.records import events, load_runs, read
from experiments.setup_accounting import setup_entries
from experiments.storage import atomic_write

ENDPOINT = "https://openrouter.ai/api/v1/generation"


def numeric(value):
    return (
        value
        if isinstance(value, (int, float)) and not isinstance(value, bool)
        else None
    )


def request_row(run, index, request, generation=None):
    response = request.get("response") or {}
    raw = response.get("provider_response") or response
    usage = raw.get("usage") or response.get("usage") or {}
    generation_id = raw.get("id")
    data = (generation or {}).get("body", {}).get("data", {})
    usable = (generation or {}).get("http_status") == 200 and data.get(
        "id"
    ) == generation_id
    meta_cost = numeric(data.get("total_cost")) if usable else None
    cost = numeric(usage.get("cost"))
    conflict = (
        cost is not None
        and meta_cost is not None
        and abs(Decimal(str(cost)) - Decimal(str(meta_cost))) > Decimal("0.00000001")
    )
    choice = next(iter(raw.get("choices") or []), {})
    format_error = None
    for call in choice.get("message", {}).get("tool_calls") or []:
        try:
            json.loads(call.get("function", {}).get("arguments", ""))
        except (ValueError, TypeError) as error:
            format_error = type(error).__name__
    prompt = usage.get("prompt_tokens_details") or {}
    completion = usage.get("completion_tokens_details") or {}
    return {
        "request_key": f"{run['run_id']}:{index}",
        "run_id": run["run_id"],
        "scope": run.get("accounting_scope", "coding_attempt"),
        "task_id": run.get("task_id"),
        "model": run["model"],
        "frozen_configuration": run.get("declared_frozen_setting_id"),
        "configuration_id": run.get("configuration_id"),
        "inference_id": run.get("inference_id"),
        "request_number": request.get("number", index),
        "generation_id": generation_id,
        "actual_provider": raw.get("provider"),
        "requested_provider": run.get("provider"),
        "usage": usage,
        "reported_cost_usd": cost,
        "input_tokens": numeric(usage.get("prompt_tokens")),
        "output_tokens_including_reasoning": numeric(usage.get("completion_tokens")),
        "reasoning_tokens": numeric(completion.get("reasoning_tokens")),
        "cached_input_tokens": numeric(prompt.get("cached_tokens")),
        "cache_write_tokens": numeric(prompt.get("cache_write_tokens")),
        "error": response.get("error")
        or raw.get("error")
        or (
            "provider finish_reason=error"
            if choice.get("finish_reason") == "error"
            else None
        ),
        "tool_format_error": format_error,
        "http_status": response.get("status"),
        "raw_response": request.get("response"),
        "generation_metadata": generation,
        "generation_reported_cost_usd": meta_cost,
        "accounting_conflict": conflict,
        "effective_cost_usd": None
        if conflict
        else cost
        if cost is not None
        else meta_cost,
        "cost_source": "conflict"
        if conflict
        else "completion_response"
        if cost is not None
        else "generation_api"
        if meta_cost is not None
        else "unknown",
    }


def request_ledger(project, records=None):
    project = Path(project)
    records = (
        records if records is not None else load_runs(project, include_controls=False)
    )
    sources = [dict(r) for r in records if r["kind"] == "model_attempt"]
    for entry in setup_entries(project):
        result = project / entry["result"]
        sources.append(
            {
                "run_id": str(result.parent.relative_to(project)),
                "artifact_dir": str(result.parent),
                "source": "harbor",
                "trace_path": str(result.parent / "requests.jsonl"),
                "model": entry["model"],
                "calls": entry["model_requests"],
                "accounting_scope": "setup",
            }
        )
    for entry in read(project / "reports/infrastructure-costs.json", {}).get(
        "incidents", []
    ):
        decision = read(project / entry["incident_record"])
        folder = project / Path(entry["incident_record"]).parent / "native-job"
        native = list(folder.glob("LC-*"))
        if len(native) == 1:
            sources.append(
                {
                    **decision["original_result"],
                    "artifact_dir": str(native[0]),
                    "accounting_scope": "excluded_infrastructure",
                }
            )
    ledger = []
    for run in sources:
        path = Path(run["artifact_dir"]) / (
            "trace.jsonl" if run.get("source") == "legacy" else "agent/requests.jsonl"
        )
        requests = request_records(events(run.get("trace_path", path)))
        for index in range(max(len(requests), run.get("calls") or 0)):
            request = requests[index] if index < len(requests) else {}
            raw = request.get("response") or {}
            raw = raw.get("provider_response") or raw
            generation_id = raw.get("id")
            cache = (
                project
                / "reports/openrouter-accounting/generations"
                / f"{generation_id}.json"
            )
            # IDs are data, not arbitrary paths. Never follow an unvalidated ID.
            generation = (
                read(cache)
                if isinstance(generation_id, str)
                and re.fullmatch(r"gen-[A-Za-z0-9-]{1,124}", generation_id)
                else None
            )
            ledger.append(request_row(run, index + 1, request, generation))
    return ledger


def attempt_totals(ledger):
    groups = {}
    for row in ledger:
        groups.setdefault(row["run_id"], []).append(row)
    result = []
    for run_id, rows in groups.items():
        total = {
            k: rows[0][k]
            for k in [
                "scope",
                "task_id",
                "model",
                "frozen_configuration",
                "configuration_id",
                "inference_id",
            ]
        }
        total |= {"run_id": run_id, "requests": len(rows)}
        for name, field in [
            ("cost_usd", "effective_cost_usd"),
            ("input_tokens", "input_tokens"),
            ("output_tokens", "output_tokens_including_reasoning"),
            ("reasoning_tokens", "reasoning_tokens"),
            ("cached_input_tokens", "cached_input_tokens"),
            ("cache_write_tokens", "cache_write_tokens"),
        ]:
            values = [r[field] for r in rows]
            observed = (
                [
                    r["reported_cost_usd"] if r["accounting_conflict"] else r[field]
                    for r in rows
                ]
                if name == "cost_usd"
                else values
            )
            known = sum(
                (Decimal(str(v)) for v in observed if v is not None), Decimal(0)
            )
            complete = all(v is not None for v in values)
            total[name] = float(known) if complete else None
            total[name + "_known"] = float(known)
            total[name + "_complete"] = complete
        total["usage_complete"] = (
            total["input_tokens_complete"] and total["output_tokens_complete"]
        )
        total["actual_providers"] = sorted(
            {r["actual_provider"] for r in rows if r["actual_provider"]}
        )
        total["generation_ids_missing"] = sum(r["generation_id"] is None for r in rows)
        total["accounting_conflicts"] = sum(r["accounting_conflict"] for r in rows)
        result.append(total)
    return result


def pandas_comparison(attempts, original_records):
    """Ordinary pandas groupby; no new scoring framework or verdict inference."""
    import pandas as pd

    frame = pd.DataFrame([a for a in attempts if a["scope"] == "coding_attempt"])
    if frame.empty:
        return frame
    outcomes = pd.DataFrame(
        [
            {
                "run_id": r["run_id"],
                "patch_accepted": r.get("patch_accepted"),
                "task_success": r.get("task_success"),
                "explicit_submission": r.get("explicit_submission"),
            }
            for r in original_records
            if r["kind"] == "model_attempt"
        ]
    )
    frame = frame.merge(outcomes, on="run_id", validate="one_to_one")
    keys = [
        "frozen_configuration",
        "configuration_id",
        "model",
        "inference_id",
        "task_id",
    ]
    result = (
        frame.groupby(keys, dropna=False)
        .agg(
            attempts=("run_id", "size"),
            known_spend_usd=("cost_usd_known", "sum"),
            cost_complete=("cost_usd_complete", "all"),
            usage_complete=("usage_complete", "all"),
            accepted_completions=("patch_accepted", lambda s: int(s.eq(True).sum())),
            acceptance_complete=("patch_accepted", lambda s: bool(s.notna().all())),
            original_successes=("task_success", "sum"),
        )
        .reset_index()
    )
    result["total_spend_usd"] = result["known_spend_usd"].where(result["cost_complete"])
    eligible = (
        result["cost_complete"]
        & result["acceptance_complete"]
        & result["accepted_completions"].gt(0)
    )
    result["cost_per_accepted_completion_usd"] = (
        result["total_spend_usd"] / result["accepted_completions"].where(eligible)
    ).where(eligible)
    result.loc[~result["acceptance_complete"], "accepted_completions"] = float("nan")
    return result


def fetch_generation(generation_id, directory, api_key, client=None):
    """GET metadata only; immutable cache; never generate or repair model output."""
    if not re.fullmatch(r"gen-[A-Za-z0-9-]{1,124}", generation_id):
        raise ValueError("Invalid generation ID")
    path = Path(directory) / f"{generation_id}.json"
    if path.exists():
        return read(path)
    import httpx

    owned = client is None
    client = client or httpx.Client(timeout=20, trust_env=False, follow_redirects=False)
    record = {
        "generation_id": generation_id,
        "endpoint": ENDPOINT,
        "fetched_at": datetime.now(UTC).isoformat(),
    }
    try:
        response = client.get(
            ENDPOINT,
            params={"id": generation_id},
            headers={"Authorization": f"Bearer {api_key}"},
        )
        record["http_status"] = response.status_code
        try:
            record["body"] = response.json()
        except ValueError:
            record["body"] = {
                "unparsed_sha256": hashlib.sha256(response.content).hexdigest()
            }
        if (
            response.status_code == 200
            and record["body"].get("data", {}).get("id") != generation_id
        ):
            record["error"] = "generation_id_mismatch"
    except httpx.HTTPError as error:
        record["error"] = type(error).__name__
    finally:
        if owned:
            client.close()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as output:
        json.dump(record, output, indent=2)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["export", "reconcile"])
    parser.add_argument(
        "--project", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--generation-id", action="append", default=[])
    parser.add_argument("--confirm-metadata-fetch", action="store_true")
    args = parser.parse_args()
    destination = args.project / "reports/openrouter-accounting"
    if args.command == "reconcile":
        if not args.confirm_metadata_fetch or not args.generation_id:
            parser.error(
                "Metadata reconciliation requires generation IDs and --confirm-metadata-fetch"
            )
        from dotenv import dotenv_values

        values = dotenv_values(Path.home() / ".config/openrouter/credentials.env")
        key = values.get("OPENROUTER_API_KEY") or values.get("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("Host OpenRouter key unavailable")
        for identifier in args.generation_id:
            result = fetch_generation(identifier, destination / "generations", key)
            print(
                json.dumps(
                    {
                        k: result.get(k)
                        for k in ["generation_id", "http_status", "error"]
                    }
                )
            )
    else:
        records = load_runs(args.project, include_controls=False)
        ledger = request_ledger(args.project, records)
        attempts = attempt_totals(ledger)
        comparison = pandas_comparison(attempts, records)
        atomic_write(
            destination / "requests.jsonl",
            "".join(json.dumps(r) + "\n" for r in ledger),
        )
        atomic_write(
            destination / "attempts.json", json.dumps(attempts, indent=2) + "\n"
        )
        atomic_write(
            destination / "comparison.json",
            comparison.to_json(orient="records", indent=2) + "\n",
        )
        print(
            json.dumps(
                {
                    "requests": len(ledger),
                    "attempts_and_expense_groups": len(attempts),
                    "generation_api_calls": 0,
                    "inference_calls": 0,
                }
            )
        )


if __name__ == "__main__":
    main()
