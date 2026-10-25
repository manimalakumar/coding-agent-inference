"""Explicit saved-record rosters; plans never enter launch denominators."""

import html
import json
import re
from decimal import Decimal

import pandas as pd

from experiments import intervention_views as prior
from public_replay.analysis import measured_attempts

DEVELOPMENT_TASKS = [
    "LC-TIMEOUT-001",
    "LC-RESPONSES-USAGE-001",
    "LC-ANNOTATED-TOOLS-001",
    "LC-CHROMA-NONE-001",
    "LC-CONTEXT-COUNTER-001",
]
DEVELOPMENT_FREEZES = {
    "LC-TIMEOUT-001": "configs/LC-TIMEOUT-001.harbor-submission.frozen-setting.json",
    "LC-RESPONSES-USAGE-001": "configs/LC-RESPONSES-USAGE-001.harbor-submission.frozen-setting.json",
    "LC-ANNOTATED-TOOLS-001": "configs/LC-ANNOTATED-TOOLS-001.workspace-fresh-start.frozen-setting.json",
    "LC-CHROMA-NONE-001": "configs/LC-CHROMA-NONE-001.workspace-capture.frozen-setting.json",
    "LC-CONTEXT-COUNTER-001": "configs/context-counter-repaired-environment.frozen-setting.json",
}
HELD_FREEZES = {
    "held_out_two": "configs/held-out-two-tasks.frozen-setting.json",
    "held_out_six": "configs/held-out-six-tasks.frozen-setting.json",
}
# Short reader summaries of the cited task instructions, not new acceptance rules.
PROBLEMS = {
    "LC-TIMEOUT-001": "Initialize ChatOpenAI with custom timeouts without losing values or client reuse.",
    "LC-RESPONSES-USAGE-001": "Retain cached-input and reasoning-token usage details in Responses API results.",
    "LC-ANNOTATED-TOOLS-001": "Retain tool parameter descriptions when annotations are stored as strings.",
    "LC-CHROMA-NONE-001": "Skip missing Chroma document content without discarding valid documents.",
    "LC-CONTEXT-COUNTER-001": "Let callers supply the counter used to trigger and perform context editing.",
    "LC-ASYNC-TOOLS-001": "Support asynchronous tool declarations and preserve their metadata.",
    "LC-SUMMARY-OVERFLOW-001": "Avoid identical overflow retries; recover using reduced input.",
    "LC-FIREWORKS-RERANK-001": "Add a Fireworks document reranker.",
    "LC-OPENROUTER-PROVIDER-001": "Preserve the actual serving provider in response metadata.",
    "LC-HF-ACCELERATOR-001": "Support non-CUDA accelerator discovery for Hugging Face pipelines.",
    "LC-JSON-INPUT-001": "Reject unsupported JSON-splitter input rather than silently discarding it.",
    "LC-INT-TEMPERATURE-001": "Keep integer temperature values in tracing metadata.",
    "LC-PARALLEL-TOOL-MERGE-001": "Keep distinct parallel tool-call chunks from being merged together.",
}


def __getattr__(name):
    return getattr(prior, name)


def _read(project, relative):
    return json.loads((project / relative).read_text())


def startup_incident(record):
    audit = record.get("startup_audit") or {}
    return (
        audit.get("measured_coding_attempt") is False
        and audit.get("model_requests") == 0
    )


def scope_records(bundle, scope):
    ids = bundle.read("study.json")["scopes"][scope]["run_ids"]
    by_id = {r["run_id"]: r for r in measured_attempts(bundle)}
    if len(ids) != len(set(ids)) or set(ids) - by_id.keys():
        raise ValueError(
            "Scope identities must be unique and have saved attempt records"
        )
    return [by_id[rid] for rid in ids]


def _link(url, label):
    if not re.fullmatch(
        r"https://github\.com/langchain-ai/langchain/(?:issues|pull)/\d+", url
    ):
        raise ValueError("Expected a recorded LangChain issue/PR URL")
    return '<a href="' + html.escape(url) + '">' + html.escape(label) + "</a>"


def _outcome_summary(group):
    paid = [r for r in group if not startup_incident(r)]
    if not paid:
        return "not run"
    successes = sum(r["task_success"] is True for r in paid)
    accepted = sum(r["patch_accepted"] is True for r in paid)
    unknown = sum(r["patch_accepted"] is None for r in paid)
    result = f"{successes}/{len(paid)} original successes; {accepted} patches accepted"
    if unknown:
        result += f"; {unknown} acceptance records unknown"
    return result


def development_roster(project, bundle):
    records = scope_records(bundle, "development")
    # These four references are explicitly recorded in the preserved round report.
    report = "reports/COMPARISON-ROUND.md"
    references = re.findall(
        r"^\| #(\d+) —[^|]+\|", (project / report).read_text(), re.M
    )
    if len(references) != 4:
        raise ValueError("The four-task reference roster changed")
    rows, audits = [], []
    for task in DEVELOPMENT_TASKS:
        freeze_path = DEVELOPMENT_FREEZES[task]
        frozen = _read(project, freeze_path)
        if task == "LC-CONTEXT-COUNTER-001":
            url = frozen["issue_url"]
            label = "Issue #" + url.rsplit("/", 1)[1]
            family = frozen["task_type"].replace("_", " ")
        else:
            number = references[DEVELOPMENT_TASKS.index(task)]
            url, label = (
                "https://github.com/langchain-ai/langchain/pull/" + number,
                "PR #" + number,
            )
            family = "bug fix"
        group = [
            r for r in records if r["task_id"] == task and r["model"] in prior.MODELS
        ]
        row = {
            "Issue / PR": _link(url, label),
            "Problem": PROBLEMS[task],
            "Family": family,
            "Recorded difficulty": frozen.get("difficulty", "not recorded"),
            "Outcome summary": " · ".join(
                prior.MODELS[m]
                + ": "
                + _outcome_summary([r for r in group if r["model"] == m])
                for m in prior.MODELS
            ),
        }
        rows.append(row)
        audits.append(
            {
                "task_id": task,
                "metadata_source": freeze_path,
                "reference_source": freeze_path
                if task == "LC-CONTEXT-COUNTER-001"
                else report,
                "family_source": freeze_path
                if task == "LC-CONTEXT-COUNTER-001"
                else "README.md: Earlier bug pilots",
                "summary_source": "tasks/" + task + "/task.json",
                "difficulty_recorded": frozen.get("difficulty"),
                "records": [
                    {
                        k: r.get(k)
                        for k in [
                            "run_id",
                            "model",
                            "configuration_id",
                            "classification",
                            "task_success",
                            "patch_accepted",
                            "calls",
                            "startup_audit",
                        ]
                    }
                    for r in group
                ],
            }
        )
    return pd.DataFrame(rows), audits


def roster_html(frame):
    # Escape every cell before inserting the one validated GitHub hyperlink column.
    safe = frame.map(lambda v: html.escape(str(v)))
    safe["Issue / PR"] = frame["Issue / PR"]
    return (
        "<style>.study-roster td,.study-roster th{white-space:normal;text-align:left;vertical-align:top;padding:6px;max-width:24em;font-size:12px}</style>"
        + safe.to_html(index=False, border=0, classes="study-roster", escape=False)
    )


def feature_activity(project, bundle):
    records = [
        r
        for r in scope_records(bundle, "development")
        if r["task_id"] == "LC-CONTEXT-COUNTER-001" and r["model"] in prior.MODELS
    ]
    paid = [r for r in records if not startup_incident(r)]
    incidents = [r for r in records if startup_incident(r)]
    repair = _read(
        project, "configs/context-counter-repaired-environment.frozen-setting.json"
    )
    return {
        "paid_attempts": len(paid),
        "paid_attempts_by_model": {
            prior.MODELS[m]: sum(r["model"] == m for r in paid) for m in prior.MODELS
        },
        "recorded_environment_repair_model_calls": repair.get("setup_model_requests"),
        "startup_incidents_not_paid_attempts": len(incidents),
        "startup_model_calls": sum(r["calls"] for r in incidents),
    }, [
        {
            k: r.get(k)
            for k in [
                "run_id",
                "classification",
                "startup_audit",
                "calls",
                "native_cost_usd",
                "cost_complete",
            ]
        }
        for r in incidents
    ]


def heldout_roster(project, bundle):
    rows, audit, totals = [], [], {}
    for scope, freeze in HELD_FREEZES.items():
        metadata = _read(project, freeze)["tasks"]
        records = scope_records(bundle, scope)
        totals[scope] = len(records)
        if {r["task_id"] for r in records} != set(metadata):
            raise ValueError("Frozen task roster and recorded scope differ")
        for task, spec in metadata.items():
            group = [r for r in records if r["task_id"] == task]
            url = spec["issue_url"]
            row = {
                "Issue / PR": _link(url, "Issue #" + url.rsplit("/", 1)[1]),
                "Problem": PROBLEMS[task],
                "Family": spec["family"].replace("_", " "),
                "Recorded difficulty": spec.get("difficulty", "not recorded"),
            }
            for model, name in prior.MODELS.items():
                selected = [r for r in group if r["model"] == model]
                row[name + " n / A / I"] = (
                    f"{len(selected)} / {sum(r['patch_accepted'] is True for r in selected)} / {sum(r['infrastructure_interrupted'] is True for r in selected)}"
                )
            rows.append(row)
            audit.append(
                {
                    "task_id": task,
                    "scope": scope,
                    "metadata_source": freeze,
                    "family_recorded": spec["family"],
                    "difficulty_recorded": spec.get("difficulty"),
                    "summary_source": spec["task_bundle"] + "/instruction.md",
                    "run_ids": [r["run_id"] for r in group],
                }
            )
    if totals != {"held_out_two": 4, "held_out_six": 36}:
        raise ValueError("Expected 4 earlier + 36 repeated held-out launches")
    return pd.DataFrame(rows), audit, {**totals, "held_out_total": sum(totals.values())}


def first_two_results(bundle):
    return pd.DataFrame(
        [
            {
                "Task": r["task_id"],
                "Model": prior.MODELS[r["model"]],
                "Attempt": r["attempt_number"],
                "Patch verdict": "interrupted; original rejection retained"
                if r["infrastructure_interrupted"]
                else "accepted"
                if r["patch_accepted"] is True
                else "not accepted",
                "Explicit submission": r["explicit_submission"],
            }
            for r in scope_records(bundle, "held_out_two")
        ]
    )


def separate_economics(bundle):
    rows, audit = [], []
    for scope in ["development", "held_out_two", "held_out_six"]:
        records = [
            r for r in scope_records(bundle, scope) if r["model"] in prior.MODELS
        ]
        for model, label in prior.MODELS.items():
            group = [r for r in records if r["model"] == model]
            paid = [r for r in group if not startup_incident(r)]
            accepted = sum(r["patch_accepted"] is True for r in paid)
            known = sum((Decimal(str(r["known_spend_usd"])) for r in group), Decimal(0))
            complete = all(r["cost_complete"] for r in group)
            ratio = (
                "not pooled across configurations"
                if scope == "development"
                else "undefined: no accepted patches"
                if not accepted
                else "unknown"
                if not complete
                else "$" + str(known / accepted)
            )
            rows.append(
                {
                    "Scope": scope,
                    "Model": label,
                    "Paid attempts / launches": len(paid),
                    "Startup incidents": len(group) - len(paid),
                    "Configurations": len({r["configuration_id"] for r in group}),
                    "Accepted patches": accepted,
                    "Known spend USD": str(known),
                    "Complete spend USD": str(known) if complete else "unknown",
                    "Cost / accepted patch": ratio,
                }
            )
            # Every configuration gets its own receipt; no development success rate
            # or cost-per-success figure is pooled across its exploratory variants.
            for config in sorted({r["configuration_id"] for r in group}):
                items = [r for r in group if r["configuration_id"] == config]
                audit.append(
                    {
                        "scope": scope,
                        "model": model,
                        "configuration_id": config,
                        "run_ids": [r["run_id"] for r in items],
                        "task_success": [r["task_success"] for r in items],
                        "patch_accepted": [r["patch_accepted"] for r in items],
                        "cost_complete": all(r["cost_complete"] for r in items),
                        "known_spend_usd": str(
                            sum(
                                (Decimal(str(r["known_spend_usd"])) for r in items),
                                Decimal(0),
                            )
                        ),
                    }
                )
    return pd.DataFrame(rows), audit
