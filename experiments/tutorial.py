"""Read-only teaching views; execution and historical normalization stay unchanged."""

import ast
import html
from collections import defaultdict
from decimal import Decimal
from pathlib import Path

from agent.results import request_records
from experiments.presentation import details, table
from experiments.records import events, read, walkthrough
from experiments.setup_accounting import setup_entries

NOT_RUN = "not run"


def source_text(project, relative, symbol):
    """Read the actual implementation without importing its execution dependencies."""
    text = (Path(project) / relative).read_text()
    nodes = ast.parse(text).body
    node = None
    for name in symbol.split("."):
        node = next(n for n in nodes if getattr(n, "name", None) == name)
        nodes = getattr(node, "body", [])
    return "\n".join(text.splitlines()[node.lineno - 1 : node.end_lineno])


def selected_records(records, models):
    return [r for r in records if r["kind"] == "model_attempt" and r["model"] in models]


def exact_run(records, run_id):
    matches = [r for r in records if r["run_id"] == run_id]
    if len(matches) > 1:
        raise ValueError("Ambiguous run ID: " + run_id)
    return matches[0] if matches else None


def redacted(value):
    """Redact structured credentials; credentials are never loaded for Replay."""
    if isinstance(value, dict):
        return {
            k: "<REDACTED>"
            if k.lower()
            in {"authorization", "api_key", "_api_key", "x-api-key", "secret"}
            else redacted(v)
            for k, v in value.items()
        }
    if isinstance(value, list):
        return [redacted(v) for v in value]
    return value


def saved_request(path):
    item = next((e for e in events(path) if e["event"] == "model_request"), None)
    return redacted(item["body"]) if item else None


def compact_request(body):
    if body is None:
        return {"status": NOT_RUN}
    return {
        **{k: v for k, v in body.items() if k not in {"messages", "tools"}},
        "messages": [
            {"role": m["role"], "characters": len(str(m.get("content", "")))}
            for m in body.get("messages", [])
        ],
        "tools": [t.get("function", {}).get("name") for t in body.get("tools", [])],
    }


def action_rows(record, requests):
    folder = Path(record["artifact_dir"])
    trace = events(
        folder
        / ("agent/requests.jsonl" if record["source"] == "harbor" else "trace.jsonl")
    )
    number = None
    rows = []
    for event in trace:
        if event["event"] == "model_request":
            number = event.get("number")
        if event["event"] == "tool_result" and number in requests:
            result = event.get("result", {})
            command = event.get("tool_call", {}).get("command", "")
            rows.append(
                {
                    "request": number,
                    "command (excerpt)": command[:180],
                    "exit": result.get("returncode", result.get("exit_code")),
                    "result (tail)": str(result.get("output", ""))[-400:],
                }
            )
    return rows


def check_rows(record):
    data = walkthrough(record)
    evaluation = data.get("evaluation") or {}
    rows = []
    for name, check in evaluation.get("checks", {}).items():
        if not isinstance(check, dict):
            continue
        rows.append(
            {
                "stage": "trusted acceptance (development task)"
                if name == "hidden_acceptance"
                else "development / policy check",
                "check": name,
                "exit": check.get("exit_code"),
                "evidence (tail)": str(check.get("output", ""))[-240:],
            }
        )
    if not rows and data.get("review"):
        review = data["review"]
        for key in [
            "classification",
            "focused_test_results",
            "development_lint_format_types",
            "hidden_acceptance",
        ]:
            if key in review:
                rows.append(
                    {
                        "stage": "original rejection / later trusted audit",
                        "check": key,
                        "exit": None,
                        "evidence (tail)": str(review[key]),
                    }
                )
    return rows


def patch_html(patch, limit=32):
    lines = patch.splitlines()
    focused = []
    for line in lines[:limit]:
        color = (
            "#147d38"
            if line.startswith("+")
            else "#b42318"
            if line.startswith("-")
            else "inherit"
        )
        focused.append(f'<span style="color:{color}">{html.escape(line)}</span>')
    tail = (
        f"\n… {len(lines) - limit} more lines in full patch below"
        if len(lines) > limit
        else ""
    )
    return (
        '<pre style="white-space:pre-wrap;font-size:12px">'
        + "\n".join(focused)
        + html.escape(tail)
        + "</pre>"
    )


def walkthrough_view(record, requests):
    if record is None:
        return "<p>not run</p>"
    data = walkthrough(record)
    summary = [
        {
            "original verdict": record["classification"],
            "audit category": record.get("failure_category"),
            "task success": record["task_success"],
            "patch accepted": record["patch_accepted"],
            "explicit submission": record["explicit_submission"],
        }
    ]
    reproduction = data.get("reproduction") or {}
    output = reproduction.get("result", reproduction)
    checks = check_rows(record)
    focused_checks = [
        row
        for row in checks
        if row["check"] in {"regression", "existing_tests", "hidden_acceptance"}
        or (row["exit"] not in (0, None) and row["check"] != "regression_on_baseline")
    ]
    if not focused_checks:
        focused_checks = checks
    diagnostic_rows = [
        {
            "label": d.get("label"),
            "scope": "post-run diagnostic; excluded from success",
            "functional status": d.get("diagnostic_functional_status"),
            "acceptance exit": (d.get("hidden_acceptance") or {}).get("exit_code"),
        }
        for d in data["diagnostics"]
    ]
    return "".join(
        [
            table(summary, list(summary[0])),
            "<b>1 · Reproduction</b>",
            table(
                [
                    {
                        "exit": output.get("returncode", output.get("exit_code")),
                        "observed output (tail)": str(output.get("output", output))[
                            -450:
                        ],
                    }
                ],
                ["exit", "observed output (tail)"],
            ),
            details("Saved reproduction evidence", output),
            "<b>2 · Selected executed actions (request numbers fixed in the view manifest)</b>",
            table(
                action_rows(record, requests),
                ["request", "command (excerpt)", "exit", "result (tail)"],
            ),
            "<b>3 · Frozen candidate diff (excerpt)</b>",
            patch_html(data["patch"]),
            details("Full saved patch", data["patch"]),
            "<b>4 · Trusted verdict evidence</b>",
            table(focused_checks, ["stage", "check", "exit", "evidence (tail)"]),
            details(
                "All recorded checks · baseline regression is expected to exit 1",
                checks,
            ),
            details(
                "Original record, receipts and evaluation",
                {
                    "record": record,
                    "evaluation": data["evaluation"],
                    "review": data["review"],
                },
            ),
            table(
                diagnostic_rows,
                ["label", "scope", "functional status", "acceptance exit"],
            )
            if diagnostic_rows
            else "",
            details(
                "Later audit / diagnostics — original verdict unchanged",
                {"audit": data.get("audit"), "diagnostics": data["diagnostics"]},
            ),
        ]
    )


def learning(record):
    if record is None:
        return "**Learning:** not run; no outcome can be inferred."
    if record["patch_accepted"] is True and not record["task_success"]:
        return "**Learning:** patch acceptance and the original task verdict differ under this historical policy. Preserve both; do not retroactively score the old attempt under today's stopping rule."
    if record["task_success"]:
        return "**Learning:** this attempt completed its required checks under its recorded setting. One success is evidence of feasibility, not a reliability estimate or a controlled model comparison."
    return "**Learning:** the original unsuccessful verdict remains. Read the failed or missing checks and separate the audited cause from the native termination label. Extra diagnostic evidence never rewrites this outcome."


def development_slots(plan, records, frozen_id):
    """Only exact planned jobs with the matching frozen setting enter the round."""
    rows = []
    for slot in plan["schedule"]:
        matches = [
            r
            for r in records
            if r["kind"] == "model_attempt"
            and r["run_id"].split("/", 1)[0] == slot["job_name"]
        ]
        if len(matches) > 1:
            raise ValueError("Multiple trials for a single planned slot")
        record = matches[0] if matches else None
        eligible = (
            record is not None
            and record.get("declared_frozen_setting_id") == frozen_id
            and record["task_id"] == slot["task_id"]
            and record["model"] == slot["model_id"]
        )
        rows.append(
            {
                **slot,
                "status": NOT_RUN
                if record is None
                else "recorded"
                if eligible
                else "configuration mismatch",
                "record": record if eligible else None,
            }
        )
    return rows


def development_matrix(slots):
    groups = defaultdict(list)
    for slot in slots:
        groups[slot["task_id"], slot["model_id"]].append(slot)
    return [
        {
            "task": task,
            "model": model,
            "planned": len(group),
            "recorded": sum(s["record"] is not None for s in group),
            "status": NOT_RUN
            if all(s["status"] == NOT_RUN for s in group)
            else ", ".join(sorted({s["status"] for s in group})),
        }
        for (task, model), group in groups.items()
    ]


def raw_usage(record):
    folder = Path(record["artifact_dir"])
    trace = folder / (
        "agent/requests.jsonl" if record.get("source") != "legacy" else "trace.jsonl"
    )
    return usage_requests(request_records(events(trace)), record.get("calls"))


def usage_requests(requests, expected=None):
    values = [(r.get("response") or {}).get("usage") or {} for r in requests]
    expected = len(values) if expected is None else expected
    paths = {
        "input": ("prompt_tokens",),
        "output_including_reasoning": ("completion_tokens",),
        "reasoning_subset": ("completion_tokens_details", "reasoning_tokens"),
        "cached_input_subset": ("prompt_tokens_details", "cached_tokens"),
        "cache_write_input": ("prompt_tokens_details", "cache_write_tokens"),
    }
    result = {"requests": expected, "raw_request_records": len(values)}
    for label, path in paths.items():
        measurements = []
        for usage in values:
            value = usage
            for key in path:
                value = value.get(key) if isinstance(value, dict) else None
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                measurements.append(value)
        complete = bool(expected) and len(measurements) == expected == len(values)
        result[label] = sum(measurements) if complete else None
        result[label + "_known"] = sum(measurements)
        result[label + "_reported_requests"] = len(measurements)
    return result


def economics(records):
    if not records:
        return {
            "attempts": 0,
            "status": NOT_RUN,
            "known_spend_usd": None,
            "total_spend_usd": None,
            "cost_per_accepted_patch_usd": None,
            "cost_per_successful_task_usd": None,
            "accepted_patches": None,
            "successful_attempts": None,
            "explicit_completion_rate": None,
        }
    known = sum((Decimal(str(r["known_cost_usd"])) for r in records), Decimal(0))
    complete = all(r.get("cost_usd") is not None for r in records)
    accepted = sum(r.get("patch_accepted") is True for r in records)
    acceptance_known = all(
        r.get("patch_accepted") in (True, False) and r.get("patch_accepted") is not None
        for r in records
    )
    success = sum(r.get("task_success") is True for r in records)
    return {
        "attempts": len(records),
        "successful_attempts": success,
        "accepted_patches": accepted if acceptance_known else None,
        "known_spend_usd": float(known),
        "total_spend_usd": float(known) if complete else None,
        "cost_per_accepted_patch_usd": float(known / accepted)
        if complete and acceptance_known and accepted
        else None,
        "cost_per_successful_task_usd": float(known / success)
        if complete and success
        else None,
        "explicit_completion_rate": sum(bool(r["explicit_submission"]) for r in records)
        / len(records),
        "status": "complete cost records"
        if complete
        else "unknown total; known spend is a lower bound",
        "patch_status": "unknown acceptance"
        if not acceptance_known
        else "undefined: no accepted patches"
        if not accepted
        else "available"
        if complete
        else "unknown cost",
    }


def operational_rows(project, all_records, models):
    """Disjoint expense buckets; exclusions never vanish from operational totals."""
    project = Path(project)
    attempts = [r for r in all_records if r["kind"] == "model_attempt"]
    incidents = read(project / "reports/infrastructure-costs.json", {}).get(
        "incidents", []
    )
    setup = setup_entries(project)
    rows = []
    for label, entries, known_key in [
        (
            "Selected models · coding attempts across settings (exploratory)",
            [r for r in attempts if r["model"] in models],
            "known_cost_usd",
        ),
        (
            "Other archived experiments · expense only",
            [r for r in attempts if r["model"] not in models],
            "known_cost_usd",
        ),
        (
            "User-excluded infrastructure incidents · expense only",
            [r for r in incidents if r.get("include_in_operational_spend")],
            "known_cost_usd",
        ),
        ("Provider compatibility / setup · expense only", setup, "cost_usd"),
    ]:
        known = sum((Decimal(str(r.get(known_key) or 0)) for r in entries), Decimal(0))
        rows.append(
            {
                "scope": label,
                "entries": len(entries),
                "known_spend_usd": float(known),
                "total_spend_usd": float(known)
                if all(r.get("cost_usd") is not None for r in entries)
                else None,
            }
        )
    total = sum((Decimal(str(r["known_spend_usd"])) for r in rows), Decimal(0))
    rows.append(
        {
            "scope": "All recorded project inference · operational total",
            "entries": sum(r["entries"] for r in rows),
            "known_spend_usd": float(total),
            "total_spend_usd": float(total)
            if all(r["total_spend_usd"] is not None for r in rows)
            else None,
        }
    )
    return rows
