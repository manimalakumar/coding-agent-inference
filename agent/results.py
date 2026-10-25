"""Rebuild the result ledger from raw traces; never discard failed-request costs."""

import hashlib
import json
from decimal import Decimal
from pathlib import Path

from agent.usage import usage

PROJECT = Path(__file__).resolve().parents[1]


def load(path: Path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def request_records(events: list[dict]) -> list[dict]:
    records = []
    current = None
    for event in events:
        if event["event"] == "model_request":
            current = {
                "number": event.get("number"),
                "role": event.get("role"),
                "at": event.get("at"),
                "model": event["body"]["model"],
                "response": None,
            }
            records.append(current)
        elif event["event"] == "model_response" and current is not None:
            current["response"] = event["response"]
    return records


def classify(result: dict, review: dict) -> tuple[str, str]:
    if review.get("kind", "").startswith("trusted_post_run_audit") and review.get(
        "classification"
    ):
        return review["classification"], review.get(
            "reason", "See trusted audit and raw trace."
        )
    if review.get("classification_correct") is False:
        return (
            "Agent execution and diagnosis failure",
            "Omitted an import, repeated the mistake, then misclassified NameError as a missing dependency.",
        )
    reason = result.get("termination_reason", "interrupted_or_incomplete")
    if reason == "checks_passed_pending_review":
        return reason, "Executable checks passed; human review is still required."
    if reason.endswith("budget_exhausted"):
        return (
            "Budget exhausted during investigation",
            "Workflow budget reached; inspect the retained trace for progress and submitted changes.",
        )
    if reason == "configuration_failure":
        return (
            "Configuration failure",
            "Model/API preflight rejected unsupported request settings before inference.",
        )
    if reason in {"infrastructure_failure", "blocked_by_environment"}:
        return (
            "Infrastructure or environment failure",
            "Review the recorded failure separately from coding performance.",
        )
    return reason, "See the raw result and trace."


def rows_for_run(run: Path) -> list[dict]:
    metadata = load(run / "metadata.json", {})
    if not metadata.get("model"):
        return []
    result = load(run / "result.json", {})
    review = load(run / "review.json", {})
    trace = run / "trace.jsonl"
    events = (
        [json.loads(line) for line in trace.read_text().splitlines() if line.strip()]
        if trace.exists()
        else []
    )
    records = request_records(events)
    common = {
        "run_id": run.name,
        "task_id": metadata.get("task_id"),
        "model": metadata["model"]["model"],
        "workflow_label": metadata.get("workflow_label"),
        "setting_checksum": metadata.get("setting_checksum"),
        "model_configuration_checksum": hashlib.sha256(
            json.dumps(metadata["model"], sort_keys=True).encode()
        ).hexdigest(),
        "raw_termination_reason": result.get("termination_reason"),
        "raw_agent_classification": (result.get("submission") or {}).get(
            "classification"
        ),
        "result_path": str(run / "result.json"),
    }
    rows = []
    # Request errors are their own rows and are excluded from the execution row,
    # so each request contributes exactly once to model cost totals.
    errors = [r for r in records if (r.get("response") or {}).get("error")]
    for record in errors:
        response = record["response"]
        status = response.get("status")
        configuration = status in {400, 404, 422}
        rows.append(
            {
                **common,
                "phase": "request_failure",
                "request_number": record["number"],
                "classification": "Configuration failure"
                if configuration
                else "Inference request failure",
                "what_it_shows": f"Provider rejected request (HTTP {status}); recorded separately from coding performance.",
                "success": False,
                **usage([record]),
            }
        )
    execution = [r for r in records if r not in errors]
    if execution or not records:
        classification, explanation = classify(result, review)
        if (
            len(records) == 6
            and len(errors) == 1
            and len(execution) == 5
            and not result.get("files_changed")
        ):
            explanation = "Five completed source-inspection calls; no reproduction test or edit before the shared request limit."
        rows.append(
            {
                **common,
                "phase": "execution" if execution else "preflight",
                "classification": classification,
                "what_it_shows": explanation,
                "success": result.get("termination_reason")
                == "checks_passed_pending_review",
                "elapsed_seconds": result.get("elapsed_seconds"),
                **usage(execution),
            }
        )
    return rows


def totals(rows: list[dict]) -> dict:
    result = {}
    for model in sorted({row["model"] for row in rows}):
        entries = [row for row in rows if row["model"] == model]
        successful = {row["run_id"] for row in entries if row["success"]}
        known = sum(
            (Decimal(str(r["known_reported_cost_usd"])) for r in entries), Decimal(0)
        )
        missing = sum(r["requests_without_cost_record"] for r in entries)
        success_count = len(successful)
        # Legacy rows predate cost_complete; retain their missing-record contract.
        # An explicit incomplete flag also covers truncated request streams.
        complete = all(
            r.get("cost_complete", r["requests_without_cost_record"] == 0)
            and r["requests_without_cost_record"] == 0
            for r in entries
        )
        result[model] = {
            "attempts": len({r["run_id"] for r in entries}),
            "requests_including_unsuccessful": sum(r["requests"] for r in entries),
            "successful_tasks": success_count,
            "success_criterion": "Attempt reached checks_passed_pending_review; no merge implied.",
            "known_reported_cost_all_attempts_usd": float(known),
            "total_cost_all_attempts_usd": float(known) if complete else None,
            "cost_complete": complete,
            "requests_without_cost_record": missing,
            "cost_per_successful_task_usd": float(known / success_count)
            if success_count and complete
            else None,
            "metric_status": "undefined_no_successes"
            if not success_count
            else "incomplete_cost_records"
            if not complete
            else "available",
            "formula": "Cost of ALL inference requests for this model, including failures and repairs, divided by qualifying successful attempts.",
        }
    return result


def configuration_totals(rows: list[dict]) -> list[dict]:
    """Keep controlled setting/model configurations separate from lifetime totals."""
    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        key = (
            row.get("setting_checksum") or "unrecorded",
            row["model"],
            row.get("model_configuration_checksum") or "unrecorded",
        )
        groups.setdefault(key, []).append(row)
    return [
        {
            "setting_checksum": setting,
            "model": model,
            "model_configuration_checksum": inference,
            "workflow_label": next(
                (r.get("workflow_label") for r in entries if r.get("workflow_label")),
                None,
            ),
            **totals(entries)[model],
        }
        for (setting, model, inference), entries in sorted(groups.items())
    ]


def refresh(project: Path = PROJECT) -> dict:
    rows = []
    for run in sorted((project / "runs").iterdir()):
        if run.is_dir():
            rows.extend(rows_for_run(run))
    summary = totals(rows)
    per_configuration = configuration_totals(rows)
    reports = project / "reports"
    reports.mkdir(exist_ok=True)
    (reports / "result-log.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows)
    )
    (reports / "model-totals.json").write_text(json.dumps(summary, indent=2) + "\n")
    (reports / "configuration-totals.json").write_text(
        json.dumps(per_configuration, indent=2) + "\n"
    )
    lines = [
        "# Result log",
        "",
        "Generated from raw run artifacts. Failed requests and unsuccessful coding attempts remain in each model's cost numerator.",
        "",
        "| Run | Classification | What it shows | Reported cost |",
        "| --- | --- | --- | --- |",
    ]
    for row in rows:
        label = row["model"] + (
            f": rejected request {row['request_number']}"
            if row["phase"] == "request_failure"
            else ": execution"
        )
        cost = (
            f"${row['known_reported_cost_usd']:.8f}"
            if row["cost_complete"]
            else "Unknown; not treated as free"
        )
        lines.append(
            f"| [{label}]({row['result_path']}) | {row['classification']} | {row['what_it_shows']} | {cost} |"
        )
    lines += [
        "",
        "## Exploratory lifetime totals across configurations",
        "",
        "| Model | Attempts | Successful tasks | Known cost across all attempts | Cost per successful task |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for model, data in summary.items():
        metric = (
            "Undefined: no successes"
            if data["metric_status"] == "undefined_no_successes"
            else "Unknown: incomplete costs"
            if data["metric_status"] != "available"
            else f"${data['cost_per_successful_task_usd']:.8f}"
        )
        lines.append(
            f"| {model} | {data['attempts']} | {data['successful_tasks']} | ${data['known_reported_cost_all_attempts_usd']:.8f} | {metric} |"
        )
    lines += [
        "",
        "## Per-configuration economics",
        "",
        "Configurations are grouped by frozen harness checksum, model and inference-configuration checksum. Lifetime totals above are exploratory spend, not a frozen baseline estimate.",
        "",
        "| Setting | Model | Inference configuration | Attempts | Successes | Known inference cost | Cost per success |",
        "| --- | --- | --- | ---: | ---: | ---: | --- |",
    ]
    for entry in per_configuration:
        metric = (
            "Undefined: no successes"
            if not entry["successful_tasks"]
            else "Unknown: incomplete costs"
            if entry["total_cost_all_attempts_usd"] is None
            else f"${entry['cost_per_successful_task_usd']:.8f}"
        )
        lines.append(
            f"| {entry['setting_checksum'][:12]} | {entry['model']} | "
            f"{entry['model_configuration_checksum'][:12]} | {entry['attempts']} | "
            f"{entry['successful_tasks']} | ${entry['known_reported_cost_all_attempts_usd']:.8f} | {metric} |"
        )
    lines += [
        "",
        "Unknown usage/cost is retained explicitly, never replaced by zero. Successful tasks means qualifying attempts, including repeats of a task; hidden checks must pass and human review remains pending. Configuration failures remain separate rows but their inference costs count. Raw artifacts are not rewritten. Rebuilding this ledger is idempotent; phase rows never double-count the same request.",
        "",
    ]
    (reports / "RESULTS.md").write_text("\n".join(lines))
    return {"rows": len(rows), "models": summary, "configurations": per_configuration}


if __name__ == "__main__":
    print(json.dumps(refresh(), indent=2))
