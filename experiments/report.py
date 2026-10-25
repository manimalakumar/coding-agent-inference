"""Rebuild the central log and per-setting economics from structured run artifacts."""

import fcntl
import hashlib
import json
from pathlib import Path

from experiments.records import aggregate, exploratory_economics, load_runs
from experiments.storage import atomic_write


def write_reports(project):
    project = Path(project).resolve()
    lock = project / ".state/report.lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    with lock.open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        return _write_reports(project)


def _write_reports(project):
    project = Path(project).resolve()
    records = load_runs(project)
    data = {
        "schema_version": 1,
        "mode": "replay",
        "records": records,
        "per_task": aggregate(records),
        "across_tasks": aggregate(records, across_tasks=True),
        "exploratory_lifetime": exploratory_economics(records),
    }
    reports = project / "reports"
    reports.mkdir(exist_ok=True)
    serialized = json.dumps(data, indent=2) + "\n"
    digest = hashlib.sha256(serialized.encode()).hexdigest()
    snapshot = reports / "bookkeeping/snapshots" / (digest + ".json")
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    if not snapshot.exists():
        with snapshot.open("x") as output:
            output.write(serialized)
    atomic_write(reports / "notebook-run-records.json", serialized)
    atomic_write(
        reports / "experiment-results.jsonl",
        "".join(json.dumps(r) + "\n" for r in records),
    )
    lines = [
        "# Experiment result log",
        "",
        "Generated from saved artifacts. Original classifications remain unchanged; post-run diagnostics are separate.",
        "",
        "| Task | Run/model | Configuration | Primary outcome | What it shows | Calls | Output tokens | Reported cost |",
        "| --- | --- | --- | --- | --- | ---: | ---: | ---: |",
    ]
    for row in records:
        if row["kind"] != "model_attempt":
            continue
        cost = (
            f"${row['cost_usd']:.8f}"
            if row["cost_usd"] is not None
            else f"Unknown (known: ${row['known_cost_usd']:.8f})"
        )
        lines.append(
            f"| {row['task_id']} | [{row['model']} — {row['run_id']}]({row['result_path']}) | {row['configuration_id']} | {row.get('failure_category') or row['classification']} | {str(row.get('what_it_shows') or row.get('termination_reason') or '').replace(chr(10), ' ')} | {row['calls']} | {row['output_tokens']} | {cost} |"
        )
    lines += [
        "",
        "## Patch economics and explicit workflow completion",
        "",
        "All failed inference stays in cost. Patch acceptance and explicit submission are separate measurements. Earlier success classifications retain their original rules; unknown historical acceptance is not inferred from later diagnostics.",
        "",
        "| Configuration | Model | Attempts | Accepted patches | Cost per accepted patch | Explicit submissions | Explicit workflow-completion rate |",
        "| --- | --- | ---: | ---: | --- | ---: | ---: |",
    ]
    for row in data["across_tasks"]:
        metric = (
            f"${row['cost_per_accepted_patch_usd']:.8f}"
            if row["patch_economics_status"] == "available"
            else row["patch_economics_status"]
        )
        lines.append(
            f"| {row['configuration_id']} | {row['model']} | {row['attempts']} | {row['accepted_patches'] if row['accepted_patches'] is not None else 'Unknown'} | {metric} | {row['explicit_submissions']} | {row['explicit_workflow_completion_rate']:.1%} |"
        )
    lines += [
        "",
        "## Per-configuration economics",
        "",
        "All inference, including unsuccessful requests and failed attempts, stays in the numerator. Different settings remain separate.",
        "",
        "| Configuration | Model | Attempts | Accepted completions | Known spend | Total spend | Cost per accepted completion |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in data["across_tasks"]:
        total = (
            f"${row['total_spend_usd']:.8f}"
            if row["total_spend_usd"] is not None
            else "Unknown"
        )
        metric = (
            f"${row['cost_per_accepted_completion_usd']:.8f}"
            if row["economics_status"] == "available"
            else row["economics_status"]
        )
        lines.append(
            f"| {row['configuration_id']} | {row['model']} | {row['attempts']} | {row['accepted_completions']} | ${row['known_spend_usd']:.8f} | {total} | {metric} |"
        )
    lines += [
        "",
        "## Exploratory lifetime spend across policies",
        "",
        "Spend accounting only; these mixed settings are not a controlled economics comparison.",
        "",
        "| Model | Attempts | Calls | Accepted completions | Known spend | Total spend | Cost per accepted completion |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in data["exploratory_lifetime"]:
        total = (
            f"${row['total_spend_usd']:.8f}"
            if row["total_spend_usd"] is not None
            else "Unknown"
        )
        metric = (
            f"${row['cost_per_accepted_completion_usd']:.8f}"
            if row["economics_status"] == "available"
            else row["economics_status"]
        )
        lines.append(
            f"| {row['model']} | {row['attempts']} | {row['calls']} | {row['accepted_completions']} | ${row['known_spend_usd']:.8f} | {total} | {metric} |"
        )
    lines += [
        "",
        "Trusted controls and post-run diagnostics never count as model successes. Accepted completions retain each run's original success policy; accepted patches are a separate evaluator metric. Repeated qualifying attempts count as completions, not distinct solved issues. A pass means pending human review; no merge is authorized.",
        "",
    ]
    atomic_write(reports / "EXPERIMENT-RESULTS.md", "\n".join(lines))
    return data
