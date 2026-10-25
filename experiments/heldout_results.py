"""Offline receipts, context measurements and economics for the held-out freeze."""

import hashlib
import json
from decimal import Decimal
from pathlib import Path

from experiments.configuration_paths import configuration_path
from experiments.launch import verify_freeze
from experiments.records import events, harbor_record, read

ROOT = Path(__file__).resolve().parents[1]
LABEL = "held-out-two-task-v1"


def write(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def report():
    frozen = read(configuration_path(ROOT, f"configs/{LABEL}.frozen-setting.json"))
    verify_freeze(
        ROOT, f"configs/{LABEL}.sha256", f"configs/{LABEL}.frozen-setting.json"
    )
    from experiments.context_measurement import ProxyTokenizer, measure_run

    tokenizer = ProxyTokenizer(ROOT)
    attempts = []
    contexts = []
    out = ROOT / "reports" / LABEL
    for study in frozen["run_order"]:
        job = frozen["jobs"][study]
        folders = list(
            (ROOT / "runs/harbor" / job["job_name"]).glob(job["task_id"] + "__*")
        )
        if len(folders) != 1 or not (folders[0] / "result.json").exists():
            raise RuntimeError(
                "Expected one completed explicit held-out attempt: " + study
            )
        folder = folders[0]
        record = harbor_record(folder)
        checks = read(folder / "verifier/checks.json", {})
        receipt = read(folder / "agent/receipt.json", {}).get("receipt", {})
        failures = [
            k
            for k in frozen["completion_policy"]["required_checks"]
            if checks.get("checks", {}).get(k, {}).get("exit_code")
            != (1 if k == "regression_on_baseline" else 0)
        ]
        category = (
            "checks_passed_pending_review"
            if record["task_success"]
            else "infrastructure_failure"
            if record["classification"] == "infrastructure_failure"
            else "file_policy_failure"
            if "file_policy" in failures
            else "functional_or_regression_failure"
            if any(
                k in failures
                for k in [
                    "reproducer",
                    "regression",
                    "existing_tests",
                    "hidden_acceptance",
                ]
            )
            else "lint_and_format_failure"
            if any(k in failures for k in ["lint", "format"])
            else "type_check_failure"
            if "types" in failures
            else "missing_regression_evidence"
            if "regression_on_baseline" in failures
            else record["classification"]
        )
        infrastructure_interrupted = (
            receipt.get("termination_detail")
            in ("OpenRouterAPIError", "ConfigurationError")
            or record["classification"] == "infrastructure_failure"
        )
        if infrastructure_interrupted:
            category = "infrastructure_interruption"
        elif not record["task_success"] and not checks.get("files_changed"):
            category = "no_final_changes"
        monotonic = receipt.get("monotonic_elapsed_seconds")
        civil = receipt.get("civil_elapsed_seconds")
        timing_valid = (
            monotonic is not None and civil is not None and abs(civil - monotonic) < 5
        )
        original = {
            str(p.relative_to(folder)): sha(p)
            for p in sorted(folder.rglob("*"))
            if p.is_file()
        }
        review = {
            "schema_version": 1,
            "failure_category": category,
            "original_verdict_preserved": True,
            "split": "held_out",
            "frozen_setting_id": frozen["setting_id"],
            "failed_required_checks": failures,
            "reason": "Classification derived from the frozen final candidate, controller receipt and trusted checks; no additional inference, edits or repairs. Infrastructure interruptions are not diagnosed as coding failures; native patch verdicts remain intact.",
            "code_repair_rounds": None,
            "infrastructure_interrupted": infrastructure_interrupted,
            "timing_audit": {
                "controlled_completion_time_valid": timing_valid,
                "monotonic_seconds": monotonic,
                "civil_seconds": civil,
            },
            "original_run_sha256": original,
        }
        review_path = (
            ROOT
            / "reports/run-reviews"
            / (folder.parent.name + "--" + folder.name + ".json")
        )
        if review_path.exists():
            old = read(review_path)
            assert old == review, "Do not replace an original audit receipt"
        else:
            write(review_path, json.dumps(review, indent=2))
        record = harbor_record(folder)
        trace = events(folder / "agent/requests.jsonl")
        # First reproduction is auditable from raw tool results. Other milestones
        # are added only after an explicit command/exit-code audit, never guessed.
        tool_request = 0
        tools = []
        for e in trace:
            if e["event"] == "model_request":
                tool_request += 1
            elif e["event"] == "tool_result":
                tools.append((tool_request, e))
        milestones = []
        for request, e in tools:
            if (
                e["tool_call"]["command"].strip() == "python /task/reproduce.py"
                and e["result"]["returncode"] == 1
            ):
                peers = [x for n, x in tools if n == request]
                milestones = [
                    {
                        "stage": "reproduction",
                        "label": "reported failure reproduced",
                        "request": request,
                        "tool_index": peers.index(e),
                        "command_sha256": hashlib.sha256(
                            e["tool_call"]["command"].encode()
                        ).hexdigest(),
                        "expected_exit_code": 1,
                    }
                ]
                break
        milestones += read(out / "milestones.json", {}).get(record["run_id"], [])
        context = measure_run(ROOT, record["run_id"], tokenizer, milestones)
        contexts.append(context)
        item = {
            **record,
            "split": "held_out",
            "frozen_setting_id": frozen["setting_id"],
            "infrastructure_interrupted": infrastructure_interrupted,
            "coding_performance_eligible": not infrastructure_interrupted,
            "controlled_completion_time_valid": timing_valid,
            "trusted_check_exit_codes": {
                k: v["exit_code"] for k, v in checks.get("checks", {}).items()
            },
            "failed_required_checks": failures,
            "missing_required_checks": [
                k
                for k in frozen["completion_policy"]["required_checks"]
                if k not in checks.get("checks", {})
            ],
            "files_changed": checks.get("files_changed", []),
            "agent_monotonic_seconds": monotonic,
            "agent_civil_seconds": civil,
            "tool_format_repair_rounds": receipt.get("tool_format_repair_rounds"),
            "context_summary": context["summary"],
            "native_artifacts_sha256": original,
        }
        attempts.append(item)
    import pandas as pd

    frame = pd.DataFrame(attempts)
    economics = []
    for model, group in frame.groupby("model", sort=True):
        costs = group["cost_usd"].tolist()
        complete = all(v is not None and not pd.isna(v) for v in costs)
        known = sum((Decimal(str(v)) for v in group["known_cost_usd"]), Decimal(0))
        accepted = int(group["patch_accepted"].eq(True).sum())
        acceptance_complete = all(
            isinstance(v, bool) for v in group["patch_accepted"].tolist()
        )
        metric_status = (
            "unknown_acceptance_records"
            if not acceptance_complete
            else "undefined_no_accepted_patches"
            if not accepted
            else "unknown_incomplete_cost"
            if not complete
            else "available"
        )
        economics.append(
            {
                "model": model,
                "attempts": len(group),
                "accepted_patches": accepted if acceptance_complete else None,
                "acceptance_records_complete": acceptance_complete,
                "explicit_submissions": int(group["explicit_submission"].sum()),
                "known_reported_spend_usd": float(known),
                "reported_spend_usd": float(known) if complete else None,
                "cost_complete": complete,
                "cost_per_accepted_patch_usd": float(known / accepted)
                if metric_status == "available"
                else None,
                "cost_per_accepted_patch_status": metric_status,
                "explicit_workflow_completion_rate": float(
                    group["explicit_submission"].mean()
                ),
                "calls": int(group["calls"].sum()),
                "output_tokens": int(group["output_tokens"].sum()),
            }
        )
        token_complete = bool(group["token_usage_complete"].eq(True).all())
        economics[-1].update(
            known_output_tokens=int(group["output_tokens"].sum()),
            output_tokens=int(group["output_tokens"].sum()) if token_complete else None,
            token_usage_complete=token_complete,
            coding_attempts_with_complete_execution=int(
                group["coding_performance_eligible"].sum()
            ),
            infrastructure_interruptions=int(group["infrastructure_interrupted"].sum()),
        )
    data = {
        "schema_version": 1,
        "split": "held_out",
        "setting": LABEL,
        "setting_id": frozen["setting_id"],
        "scope": "Two held-out tasks, one fresh attempt per model per task; no reliability estimate",
        "setup_inference_calls": 0,
        "setup_inference_spend_usd": 0,
        "attempts": attempts,
        "economics": economics,
        "additional_attempts_authorized": 0,
        "historical_development_results_preserved": True,
    }
    write(out / "results.json", json.dumps(data, indent=2) + "\n")
    measurement = {
        "analysis_version": "held-out-context-v1",
        "split": "held_out",
        "setting_id": frozen["setting_id"],
        "context_intervention": False,
        "native_metrics": "OpenRouter usage.prompt_tokens (cached included), completion/reasoning tokens, usage.cost",
        "estimate_method": "Local o200k_base proxy on visible request fields; estimates are not a native token partition and exclude provider chat framing.",
        "tokenizer": tokenizer.metadata,
        "runs": contexts,
    }
    write(out / "context-measurement.json", json.dumps(measurement, indent=2) + "\n")
    lines = [
        "# Two-task held-out evaluation",
        "",
        f"Frozen setting: `{LABEL}` / `{frozen['setting_id']}`.",
        "",
        "One fresh attempt per task per model. Both baselines reproduce, references pass, and budget termination captures qualifying patches without requiring submission. Setup inference: zero. Held-out results remain separate from development tuning and the unrun four-task development round.",
        "",
        "| Task | Model | Patch accepted | Explicit submission | Termination | Failed required checks | Calls | Output | Reported cost |",
        "| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: |",
    ]
    for r in attempts:
        cost = f"${r['cost_usd']:.8f}" if r["cost_usd"] is not None else "unknown"
        outcome = (
            "infrastructure interruption; original rejection preserved"
            if r["infrastructure_interrupted"]
            else str(r["patch_accepted"])
        )
        output = str(r["output_tokens"]) + (
            " known; incomplete" if not r["token_usage_complete"] else ""
        )
        lines.append(
            f"| {r['task_id']} | {r['model']} | {outcome} | {r['explicit_submission']} | {r['termination_reason']} | {', '.join(r['failed_required_checks']) or 'none'} | {r['calls']} | {output} | {cost} |"
        )
    lines += [
        "",
        "## Inference economics",
        "",
        "All requests, malformed responses and unsuccessful attempts are included. These figures cover this held-out freeze only; total operational/development spend is reported separately. Missing required checks are included in the failure list above. An infrastructure interruption is not a completed coding-performance observation.",
        "",
        "| Model | Attempts | Accepted | Known reported spend | Complete spend | Cost per accepted patch |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for e in economics:
        metric = (
            f"${e['cost_per_accepted_patch_usd']:.8f}"
            if e["cost_per_accepted_patch_usd"] is not None
            else e["cost_per_accepted_patch_status"]
        )
        total = (
            f"${e['reported_spend_usd']:.8f}"
            if e["reported_spend_usd"] is not None
            else "unknown"
        )
        lines.append(
            f"| {e['model']} | {e['attempts']} | {e['accepted_patches']} | ${e['known_reported_spend_usd']:.8f} | {total} | {metric} |"
        )
    lines += ["", "## Saved-artifact diagnosis", ""]
    for note in read(out / "diagnostics.json", {}).get("attempts", []):
        lines.append("- `" + note["run_id"] + "`: " + note["explanation"])
    lines += [
        "",
        "## Limits",
        "",
        "Only two held-out tasks and one attempt per model/task are observed. This does not establish repeatability, a stable ranking, or causation between context length and outcome. Public upstream tasks may have appeared in training data. Accepted candidates are checks_passed_pending_review; no commit, push, PR or merge is authorized.",
        "",
        "## Artifacts",
        "",
        "Exact run IDs, native configuration IDs, request costs/usage, check exits, candidate patches, source checksums, context estimates and immutable artifact receipts are recorded in results.json and context-measurement.json. Original native files are never rewritten.",
    ]
    write(out / "README.md", "\n".join(lines) + "\n")
    print(json.dumps({"attempts": len(attempts), "economics": economics}, indent=2))
    return data, measurement


def plots(data, measurement):
    import matplotlib as mpl
    from matplotlib.backends.backend_svg import FigureCanvasSVG
    from matplotlib.figure import Figure

    out = ROOT / "reports" / LABEL
    files = []
    with mpl.rc_context(
        {
            "font.size": 10,
            "svg.fonttype": "none",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    ):
        for tid in data["attempts"]:
            task = tid["task_id"]
            if any(p["task_id"] == task for p in files):
                continue
            runs = [r for r in measurement["runs"] if r["task_id"] == task]
            for key, name, ylabel in [
                (
                    "native_input_tokens",
                    "input-context",
                    "Reported input tokens (cached included)",
                ),
                (
                    "cumulative_reported_cost_usd",
                    "cumulative-cost",
                    "Cumulative reported cost (USD)",
                ),
            ]:
                fig = Figure(figsize=(10, 4.8))
                FigureCanvasSVG(fig)
                ax = fig.subplots()
                fig.subplots_adjust(left=0.12, right=0.96, bottom=0.19, top=0.87)
                for run in runs:
                    label = "MiniMax M2.5" if "minimax" in run["model"] else "GLM-5"
                    color = "#2463a5" if "minimax" in run["model"] else "#b35716"
                    x = [r["request"] for r in run["requests"]]
                    y = [
                        r[key] if r[key] is not None else float("nan")
                        for r in run["requests"]
                    ]
                    outcome = (
                        "interrupted"
                        if run["failure_category"] == "infrastructure_interruption"
                        else "accepted"
                        if run["patch_accepted"]
                        else "not accepted"
                    )
                    ax.plot(
                        x, y, label=label + " · " + outcome, marker=".", color=color
                    )
                    for m in run["milestones"]:
                        n = m["visible_in_next_request"]
                        if n is not None:
                            row = next(
                                (r for r in run["requests"] if r["request"] == n), None
                            )
                            if row and row[key] is not None:
                                ax.scatter(
                                    [n],
                                    [row[key]],
                                    marker={
                                        "reproduction": "o",
                                        "editing": "s",
                                        "verification": "^",
                                    }[m["stage"]],
                                    color=color,
                                    s=50,
                                )
                ax.set_title(
                    task
                    + " · "
                    + (
                        "request context"
                        if name == "input-context"
                        else "reported inference spend"
                    )
                )
                ax.set_xlabel("Model request number")
                ax.set_ylabel(ylabel)
                ax.set_ylim(bottom=0)
                ax.grid(alpha=0.2)
                ax.legend(loc="upper left")
                fig.text(
                    0.5,
                    0.035,
                    "No context intervention. Tool results enter later requests: circle=reproduce; square=edit attempt; triangle=verify.",
                    ha="center",
                    fontsize=8,
                )
                path = out / (task + "-" + name + ".svg")
                fig.savefig(path, metadata={"Date": None})
                files.append(
                    {
                        "task_id": task,
                        "kind": name,
                        "path": str(path.relative_to(ROOT)),
                        "sha256": sha(path),
                    }
                )
                preview = out / "previews" / (task + "-" + name + ".png")
                preview.parent.mkdir(exist_ok=True)
                fig.savefig(preview, dpi=130)
    write(
        out / "plots.json",
        json.dumps(
            {
                "results_sha256": sha(out / "results.json"),
                "measurement_sha256": sha(out / "context-measurement.json"),
                "source_sha256": sha(Path(__file__)),
                "files": files,
            },
            indent=2,
        ),
    )


if __name__ == "__main__":
    data, measurement = report()
    plots(data, measurement)
