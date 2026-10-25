"""Read-only teaching views and explicit six-task MiniMax/GLM comparison."""

import copy
import hashlib
import html
import json
import re
from decimal import Decimal
from pathlib import Path

import pandas as pd

from experiments import notebook_workflow as workflow
from public_replay.analysis import comparison, measured_attempts

MODELS = {"minimax/minimax-m2.5": "MiniMax M2.5", "z-ai/glm-5": "GLM-5"}
MATCHED_RUNS = [
    "held-out-remainder-round1-v1-LC-INT-TEMPERATURE-001-minimax-attempt-001/LC-INT-TEMPERATURE-001__387WVka",
    "held-out-remainder-round1-v1-LC-INT-TEMPERATURE-001-glm5-attempt-001/LC-INT-TEMPERATURE-001__CFAXfDX",
]


def relative_paths(project, value):
    """Presentation projection only; native traces and frozen settings stay intact."""
    if isinstance(value, dict):
        return {
            relative_paths(project, k): relative_paths(project, v)
            for k, v in value.items()
        }
    if isinstance(value, list):
        return [relative_paths(project, v) for v in value]
    if isinstance(value, str):
        return value.replace(str(project.resolve()) + "/", "").replace(
            str(project.resolve().parent) + "/", "../"
        )
    return value


def details(project, title, value, opened=False):
    return workflow.details(title, relative_paths(project, value), opened=opened)


def model_settings(body):
    return pd.DataFrame(
        [
            {
                "Model": body.get("model"),
                "Pinned provider": ", ".join(
                    (body.get("provider") or {}).get("only", [])
                ),
                "Fallbacks": (body.get("provider") or {}).get("allow_fallbacks"),
                "Sampling temperature": body.get("temperature"),
                "Reasoning": json.dumps(body["reasoning"])
                if "reasoning" in body
                else "parameter omitted",
                "Output ceiling": body.get("max_tokens"),
            }
        ]
    )


def messages_html(project, messages):
    pieces = []
    for i, message in enumerate(messages):
        role = message.get("role")
        title = (
            "System"
            if role == "system"
            else "Tool result"
            if role == "tool"
            else "Agent action"
            if role == "assistant"
            else "Task"
            if i == 1
            else "Budget"
        )
        text = str(message.get("content", ""))
        if role == "assistant":
            text += "\n" + json.dumps(message.get("tool_calls", []), indent=2)
        if len(text) > 1800:
            text = text[:1800] + "\n… [excerpt; complete message in expandable JSON]"
        pieces.append(
            "<h4>"
            + html.escape(title)
            + '</h4><pre style="white-space:pre-wrap;max-height:19em;overflow:auto">'
            + html.escape(relative_paths(project, text))
            + "</pre>"
        )
    return "".join(pieces)


def exchange_html(project, record):
    first = workflow.first_exchange(record)
    if first["request"] is None:
        return "<p>No model request recorded for this outcome.</p>"
    raw = first["response"] or {}
    raw = copy.deepcopy(raw.get("provider_response") or raw)
    for choice in raw.get("choices") or []:
        message = choice.get("message", {})
        for field in ["reasoning", "reasoning_details"]:
            message.pop(field, None)
    compact = {
        "run_id": record["run_id"],
        "generation_id": raw.get("id"),
        "actual_serving_provider": raw.get("provider"),
        "response_status": first["response_status"],
        "usage": raw.get("usage"),
        "error": raw.get("error"),
    }
    answer = [
        {"role": "assistant", **c.get("message", {})}
        for c in (raw.get("choices") or [])
    ]
    return (
        "<h4>Actual first response</h4>"
        + details(project, "Generation and native usage", compact, opened=True)
        + messages_html(project, answer)
        + details(
            project,
            "Historical request summary (original wording privately retained)"
            if first.get("summarized")
            else "Actual first request JSON",
            first["request"],
        )
        + details(project, "Actual first response JSON (raw reasoning omitted)", raw)
    )


def controls(prepared):
    labels = {
        "baseline": ("Unchanged code → expected failure", False),
        "reference": ("Reference fix → expected pass", True),
        "reference_budget": ("Valid workspace at budget stop → still accepted", True),
        "reference_error": ("Valid workspace at error stop → still accepted", True),
    }
    result = []
    for key, (label, expected) in labels.items():
        control = prepared["controls"].get("controls", {}).get(key)
        result.append(
            {
                "Setup check": label,
                "Expected patch accepted": expected,
                "Observed patch accepted": control.get("patch_accepted")
                if control
                else None,
                "Model calls": control.get("model_requests") if control else None,
                "Status": "matches expectation"
                if control and control.get("patch_accepted") is expected
                else "not recorded / differs",
            }
        )
    return pd.DataFrame(result)


def production_diff(patch):
    sections = re.split(r"(?=^diff --git )", patch, flags=re.M)
    return (
        "".join(
            s
            for s in sections
            if s.startswith("diff --git ") and "/tests/" not in s.splitlines()[0]
        )
        or "No production diff recorded."
    )


def held_frame(bundle):
    selected = set(bundle.read("study.json")["scopes"]["held_out_six"]["run_ids"])
    frame = pd.DataFrame(
        [a for a in measured_attempts(bundle) if a["run_id"] in selected]
    )
    if len(frame) != 36 or set(frame["model"]) != set(MODELS):
        raise ValueError(
            "Expected the exact frozen six-task, two-model, three-attempt block"
        )
    return frame


def acceptance_tables(bundle):
    frame = held_frame(bundle)
    records = []
    for task, group in frame.groupby("task_id"):
        row = {"Task": task}
        for model, label in MODELS.items():
            selected = group[group["model"] == model]
            n = int(selected["patch_accepted"].eq(True).sum())
            interruptions = int(selected["infrastructure_interrupted"].sum())
            row[label] = f"{n}/{len(selected)}" + (
                f" · {interruptions} interrupted" if interruptions else ""
            )
        records.append(row)
    overall = comparison(bundle, "held_out_six")
    overall["Model"] = overall["model"].map(MODELS)
    overall["Accepted / launched"] = (
        overall["accepted"].astype(str) + "/" + overall["launches"].astype(str)
    )
    return pd.DataFrame(records), overall[
        ["Model", "Accepted / launched", "interruptions", "explicit_submissions"]
    ].rename(
        columns={
            "interruptions": "Interruptions",
            "explicit_submissions": "Explicit submissions",
        }
    )


def spending_reconciliation(bundle):
    """Disjoint request scopes, summed with Decimal; unknown cost is never zeroed."""
    study = bundle.read("study.json")
    scope_by_id = {
        rid: name for name, scope in study["scopes"].items() for rid in scope["run_ids"]
    }
    groups = {}
    for r in bundle.read("requests.json"):
        scope = (
            scope_by_id[r["run_id"]]
            if r["accounting_scope"] == "coding_attempt"
            else r["accounting_scope"]
        )
        groups.setdefault(scope, []).append(r)
    rows = []
    for scope, requests in sorted(groups.items()):
        costs = [
            r["reported_cost_usd"]
            if r["accounting_conflict"]
            else r["effective_cost_usd"]
            for r in requests
        ]
        total = sum((Decimal(str(c)) for c in costs if c is not None), Decimal(0))
        rows.append(
            {
                "Scope": scope,
                "Requests": len(requests),
                "Known spend USD": str(total),
                "Unpriced request rows": sum(c is None for c in costs),
            }
        )
    known = sum((Decimal(r["Known spend USD"]) for r in rows), Decimal(0))
    attempts = measured_attempts(bundle)
    return pd.DataFrame(rows), {
        "known_operational_spend_usd": str(known),
        "complete_operational_spend_usd": None
        if any(not a["cost_complete"] for a in attempts)
        or any(r["Unpriced request rows"] for r in rows)
        else str(known),
        "evidence_content_id": bundle.manifest["content_id"],
        "earlier_2_60_claim": "No matching dated/scope receipt identified; do not use as a reconciled project total.",
    }


def distribution_table(bundle):
    held = held_frame(bundle)
    rows = []
    for model, group in held.groupby("model"):
        rows.append(
            {
                "Model": MODELS[model],
                "Attempts (failures retained)": len(group),
                "Known calls": int(group.calls.sum()),
                "Incomplete call records": int((~group.calls_complete).sum()),
                "Known output tokens": sum(
                    g["output_tokens_known"] for g in group.usage
                ),
                "Incomplete output usage": sum(
                    not g["output_tokens_complete"] for g in group.usage
                ),
            }
        )
    return pd.DataFrame(rows)


def build_assets(project):
    """Explicit zero-inference asset build; Replay reads these checked SVG artifacts."""
    import matplotlib as mpl
    from matplotlib.backends.backend_svg import FigureCanvasSVG
    from matplotlib.figure import Figure

    bundle = workflow.public_evidence(project)
    frame = held_frame(bundle)
    context = bundle.read("context.json")
    destination = project / "notebooks/assets/study"
    destination.mkdir(parents=True, exist_ok=True)

    def figure():
        f = Figure(figsize=(9, 3.6))
        FigureCanvasSVG(f)
        return f

    with mpl.rc_context(
        {
            "font.family": "DejaVu Sans",
            "svg.fonttype": "none",
            "svg.hashsalt": "notebook-comparison-v3",
        }
    ):
        for field, name, ylabel in [
            ("native_input_tokens", "matched-context.svg", "Reported input tokens"),
            (
                "cumulative_reported_cost_usd",
                "matched-cost.svg",
                "Cumulative reported cost (USD)",
            ),
        ]:
            f = figure()
            ax = f.subplots()
            for rid, color in zip(MATCHED_RUNS, ["#2463a5", "#b35716"], strict=True):
                record = frame[frame.run_id == rid].iloc[0]
                points = context[rid]["requests"]
                label = MODELS[record.model] + (
                    " · accepted" if record.patch_accepted else " · not accepted"
                )
                ax.plot(
                    [r["request"] for r in points],
                    [
                        r[field] if r[field] is not None else float("nan")
                        for r in points
                    ],
                    label=label,
                    color=color,
                )
            ax.set(
                title="MiniMax versus GLM-5: same task, attempt 001",
                xlabel="Model request",
                ylabel=ylabel,
                ylim=(0, None),
            )
            ax.legend(fontsize=9)
            ax.grid(alpha=0.2)
            f.tight_layout()
            f.savefig(destination / name, metadata={"Date": None})
        f = Figure(figsize=(9, 4.2))
        FigureCanvasSVG(f)
        axes = f.subplots(1, 2)
        for index, (model, label) in enumerate(MODELS.items()):
            records = frame[frame.model == model].to_dict("records")
            for j, r in enumerate(records):
                x = index + (j - (len(records) - 1) / 2) * 0.018
                for ax, value, complete in [
                    (axes[0], r["calls"], r["calls_complete"]),
                    (
                        axes[1],
                        r["usage"]["output_tokens_known"],
                        r["usage"]["output_tokens_complete"],
                    ),
                ]:
                    ax.scatter(
                        x,
                        value,
                        marker="o" if complete else "x",
                        s=24,
                        color=["#2463a5", "#b35716"][index],
                        alpha=0.8,
                    )
        for ax, ylabel in zip(
            axes, ["Model requests", "Reported output tokens"], strict=True
        ):
            ax.set(
                xticks=[0, 1],
                xticklabels=list(MODELS.values()),
                ylabel=ylabel,
                ylim=(0, None),
            )
            ax.grid(axis="y", alpha=0.2)
        f.suptitle(
            "Six-task block: all attempts, including failures\n× = incomplete record (known lower bound); ● = complete for that measure"
        )
        f.tight_layout()
        f.savefig(destination / "distributions.svg", metadata={"Date": None})
        f.savefig(
            project / "reports/notebook-workflow-v3/distributions-preview.png", dpi=150
        )
        # Optional same-model diagnostic uses fixed saved accepted/failure selections.
        selections = bundle.read("study.json")["selections"]
        f = figure()
        ax = f.subplots()
        for kind, color in [("accepted", "#2463a5"), ("failure", "#b35716")]:
            points = context[selections[kind]]["requests"]
            ax.plot(
                [r["request"] for r in points],
                [
                    r["native_input_tokens"]
                    if r["native_input_tokens"] is not None
                    else float("nan")
                    for r in points
                ],
                label=kind,
                color=color,
            )
        ax.set(
            title="MiniMax: accepted versus failed attempts on the same task",
            xlabel="Model request",
            ylabel="Reported input tokens",
            ylim=(0, None),
        )
        ax.legend()
        f.tight_layout()
        f.savefig(destination / "optional-minimax.svg", metadata={"Date": None})
    receipt = {
        "evidence_content_id": bundle.manifest["content_id"],
        "matched_task": "LC-INT-TEMPERATURE-001",
        "matched_attempt": 1,
        "matched_runs": MATCHED_RUNS,
        "task_meaning": "Integer-temperature metadata bug; not a sampling-temperature intervention",
        "files": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in destination.glob("*.svg")
        },
    }
    (destination / "manifest.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def assets(project):
    root = project / "notebooks/assets/study"
    manifest = json.loads((root / "manifest.json").read_text())
    if (
        manifest["evidence_content_id"]
        != workflow.public_evidence(project).manifest["content_id"]
    ):
        raise ValueError("Plot evidence identity mismatch")
    for name, expected in manifest["files"].items():
        if (
            Path(name).name != name
            or hashlib.sha256((root / name).read_bytes()).hexdigest() != expected
        ):
            raise ValueError("Plot artifact checksum mismatch")
    return root, manifest


def issue_outputs(project, prepared):
    baseline = prepared["controls"]["controls"]["baseline"]
    path = project / baseline["trial"] / "verifier/checks.json"
    check = json.loads(path.read_text())["checks"]["reproducer"]
    schema, _ = json.JSONDecoder().raw_decode(check["output"])
    actual = schema["parts"]
    return pd.DataFrame(
        [
            {
                "Property": "description",
                "Actual, unchanged code": actual.get("description", "missing"),
                "Expected": "Fragments to evaluate",
            },
            {
                "Property": "type",
                "Actual, unchanged code": actual.get("type"),
                "Expected": "array",
            },
            {
                "Property": "items.type",
                "Actual, unchanged code": actual.get("items", {}).get("type"),
                "Expected": "string",
            },
        ]
    ), {
        "recorded_exit_code": check["exit_code"],
        "observed_failure": check["output"].splitlines()[-1],
        "evidence": str(path.relative_to(project)),
    }


def preserve_invocation(project, record, rendered, mode):
    """Save each fresh first invocation before a future run can replace its display."""
    if mode == "replay":
        return "Replay: original native trace and historical notebook archives retained; no new file written."
    if mode != "execute":
        raise ValueError("Unknown mode")
    first = workflow.first_exchange(record)
    receipt = {
        "run_id": record["run_id"],
        "first_request": first["request"],
        "first_response": first["response"],
        "original_trace_sha256": first["trace_sha256"],
        "notebook_display": {"text/html": rendered},
    }
    raw = json.dumps(receipt, sort_keys=True, indent=2, ensure_ascii=False) + "\n"
    directory = project / "reports/notebook-first-invocations"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / (hashlib.sha256(record["run_id"].encode()).hexdigest() + ".json")
    try:
        with path.open("x") as output:
            output.write(raw)
    except FileExistsError:
        if path.read_text() != raw:
            raise RuntimeError(
                "A preserved first-invocation display differs; refusing to overwrite it"
            ) from None
    return str(path.relative_to(project))


if __name__ == "__main__":
    print(json.dumps(build_assets(Path(__file__).resolve().parents[1]), indent=2))
