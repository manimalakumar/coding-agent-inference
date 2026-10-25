"""Current notebook displays, including reviewed historical summaries."""

import json
import re
from pathlib import Path

import pandas as pd

from experiments import notebook_workflow as workflow
from experiments import task_rosters as prior
from experiments.notebook_charts import assets as assets

SCOPE_LABELS = {
    "development": "Development tuning",
    "held_out_two": "First two held-out tasks (4 launches)",
    "held_out_six": "Six-task repeated evaluation (36 launches)",
    "excluded_infrastructure": "Excluded infrastructure incidents",
    "setup": "Trusted setup",
}


def __getattr__(name):
    # In particular, the protected first-response display stays byte-for-byte equal.
    return getattr(prior, name)


def stack_models(study):
    rows = []
    for model, settings in study["model_preparation"].items():
        providers = (settings.get("provider") or {}).get("only", [])
        rows.append(
            {
                "Model": prior.MODELS[model],
                "Pinned provider": ", ".join(providers) or "unknown",
                "Precision": "advertised precision: FP8"
                if providers and all(p.endswith("/fp8") for p in providers)
                else "unknown",
            }
        )
    return pd.DataFrame(rows)


def budget_table(budgets):
    labels = {
        "seconds": "Wall-clock limit (seconds)",
        "model_calls": "Model request limit",
        "generated_tokens": "Cumulative output token limit",
        "repair_rounds": "Tool-format repair limit",
        "response_tokens": "Per-response output token limit",
        "tool_seconds": "Command limit (seconds)",
    }
    return pd.DataFrame([{labels.get(k, k): v for k, v in budgets.items()}])


def _author_ratings(project, frame, audit):
    metadata = json.loads((project / "notebooks/author-task-metadata.json").read_text())
    frame = frame.copy()
    ratings = []
    for link, receipt in zip(frame["Issue / PR"], audit, strict=True):
        number = re.search(
            r"github.com/langchain-ai/langchain/(?:issues|pull)/(\d+)", link
        )
        rating = metadata["ratings"].get(number.group(1)) if number else None
        ratings.append(rating or "not assigned")
        receipt["author_assigned_difficulty"] = rating
        receipt["author_metadata_source"] = "notebooks/author-task-metadata.json"
    frame["Author-assigned difficulty"] = ratings
    return frame.drop(columns=["Recorded difficulty"]), audit


def development_roster(project, bundle):
    frame, audit = prior.development_roster(project, bundle)
    return _author_ratings(project, frame, audit)


def heldout_roster(project, bundle):
    frame, audit, totals = prior.heldout_roster(project, bundle)
    frame, audit = _author_ratings(project, frame, audit)
    for name in prior.MODELS.values():
        column = name + " n / A / I"
        frame[name + " results"] = frame[column].map(
            lambda value: "{} launches; {} accepted; {} interrupted".format(
                *value.split(" / ")
            )
        )
        frame = frame.drop(columns=[column])
    return frame, audit, totals


def _task_names():
    names = dict(prior.PROBLEMS)
    project = Path(__file__).resolve().parents[1]
    for freeze in prior.HELD_FREEZES.values():
        tasks = json.loads((project / freeze).read_text())["tasks"]
        for task, spec in tasks.items():
            number = spec["issue_url"].rsplit("/", 1)[1]
            names[task] = f"Issue #{number} — {prior.PROBLEMS[task]}"
    return names


def task_label(task):
    return _task_names()[task]


def results_table_html(frame):
    """Wrap complete text without pandas' notebook representation truncation."""
    with pd.option_context(
        "display.max_colwidth",
        None,
        "display.max_rows",
        None,
        "display.max_columns",
        None,
    ):
        return prior.transition_table_html(frame)


def checks_frame(record, split="development"):
    # Check names and expected/observed exits already identify what was tested.
    # Remove the old, redundant category rather than misclassify trusted checks.
    return workflow.checks_frame(record, split=split).drop(
        columns="scope", errors="ignore"
    )


def first_two_results(bundle):
    frame = prior.first_two_results(bundle)
    frame["Task"] = frame["Task"].map(_task_names())
    return frame


def acceptance_tables(bundle):
    tasks, overall = prior.acceptance_tables(bundle)
    tasks["Task"] = tasks["Task"].map(_task_names())
    overall["Accepted launches (%)"] = overall["Accepted / launched"].map(
        lambda value: round(
            100 * int(value.split("/")[0]) / int(value.split("/")[1]), 1
        )
    )
    return tasks, overall


def separate_economics(bundle):
    frame, audit = prior.separate_economics(bundle)
    frame["Scope"] = frame["Scope"].map(SCOPE_LABELS)
    return frame, audit


def spending_reconciliation(bundle):
    frame, receipt = prior.spending_reconciliation(bundle)
    frame["Scope"] = frame["Scope"].map(lambda value: SCOPE_LABELS.get(value, value))
    return frame, receipt


def _trace_context_demo(record, bundle):
    """Exact request/message counts, native response usage, optional saved estimates."""
    _, trace = workflow.native_trace(record)
    requests = [e for e in trace if e.get("event") == "model_request"]
    columns = [
        "Request number",
        "Message count",
        "Provider-reported input tokens",
        "Estimated visible-payload tokens (estimate)",
    ]
    if not requests:
        return pd.DataFrame(columns=columns), [], {"status": "not run"}
    first = requests[0]
    later = next(
        (
            e
            for e in requests[1:]
            if any(m.get("role") == "tool" for m in e["body"].get("messages", []))
        ),
        None,
    )
    usage = {
        row["request"]: row
        for row in workflow.request_context(record).to_dict("records")
    }
    saved_context = {
        row["request"]: row
        for row in bundle.read("context.json")
        .get(record["run_id"], {})
        .get("requests", [])
    }
    rows = []
    for event in [first] + ([later] if later else []):
        number = event.get("number")
        native = usage.get(number, {}).get("input tokens")
        if pd.isna(native):
            native = saved_context.get(number, {}).get("native_input_tokens")
        estimate = saved_context.get(number, {}).get("estimated_visible_payload_tokens")
        rows.append(
            {
                "Request number": number,
                "Message count": len(event["body"].get("messages", [])),
                "Provider-reported input tokens": int(native)
                if pd.notna(native)
                else "unknown",
                "Estimated visible-payload tokens (estimate)": estimate
                if estimate is not None
                else "unknown",
            }
        )
    before = first["body"].get("messages", [])
    added = (
        [m for m in later["body"].get("messages", []) if m not in before]
        if later
        else []
    )
    receipt = {
        "status": "saved request comparison"
        if later
        else "no subsequent tool-result request recorded",
        "run_id": record["run_id"],
        "request_numbers": [
            e.get("number") for e in [first] + ([later] if later else [])
        ],
        "input_tokens_source": "saved provider response usage.prompt_tokens, or saved context.json native_input_tokens when missing; includes cached input",
        "estimates_source": "saved context.json; tokenizer estimates, not provider accounting",
        "added_messages": added,
    }
    return pd.DataFrame(rows, columns=columns), added, receipt


_original_milestone_actions = prior.milestone_actions


def context_demo(record, bundle):
    if record["run_id"] == workflow.DEMO_RUN_ID:
        saved = workflow._historical(Path(__file__).resolve().parents[1])[
            "context_demo"
        ]
        return pd.DataFrame(saved[0]), saved[1], saved[2]
    return _trace_context_demo(record, bundle)


def milestone_actions(record):
    if record["run_id"] == workflow.DEMO_RUN_ID:
        return pd.DataFrame(
            workflow._historical(Path(__file__).resolve().parents[1])["milestones"]
        )
    return _original_milestone_actions(record)


_original_issue_outputs = prior.issue_outputs


def issue_outputs(project, prepared):
    if prepared["study_id"] == workflow.DEMO_STUDY:
        saved = workflow._historical(project)["issue_outputs"]
        return pd.DataFrame(saved[0]), saved[1]
    return _original_issue_outputs(project, prepared)
