"""Compact tutorial and static fallback, both derived from one public manifest."""

import hashlib
import html
import json
import re
import shutil
from pathlib import Path

import nbformat

from public_replay.analysis import (
    comparison,
    measured_attempts,
    operational,
    outcomes,
    request_payload,
)
from public_replay.bundle import Bundle


def plots(bundle: Bundle, output: Path) -> None:
    import matplotlib as mpl
    from matplotlib.backends.backend_svg import FigureCanvasSVG
    from matplotlib.figure import Figure

    context = bundle.read("context.json")
    selections = bundle.read("study.json")["selections"]
    output.mkdir(parents=True, exist_ok=True)
    with mpl.rc_context(
        {
            "svg.hashsalt": "public-replay-v1",
            "svg.fonttype": "none",
            "font.family": "DejaVu Sans",
            "font.size": 10,
        }
    ):
        for field, name, ylabel in [
            (
                "native_input_tokens",
                "context.svg",
                "Reported input tokens (cached included)",
            ),
            (
                "cumulative_reported_cost_usd",
                "cost.svg",
                "Cumulative reported cost (USD)",
            ),
        ]:
            fig = Figure(figsize=(8, 3.7))
            FigureCanvasSVG(fig)
            ax = fig.subplots()
            fig.subplots_adjust(left=0.13, bottom=0.19, right=0.97, top=0.87)
            for kind, color in [("accepted", "#2463a5"), ("failure", "#b35716")]:
                evidence = context[selections[kind]]
                rows = evidence["requests"]
                ax.plot(
                    [r["request"] for r in rows],
                    [r[field] if r[field] is not None else float("nan") for r in rows],
                    label=kind + " · fixed MiniMax temperature example",
                    color=color,
                )
                for milestone in evidence["milestones"]:
                    n = milestone["visible_in_next_request"]
                    row = next((r for r in rows if r["request"] == n), None)
                    if row and row[field] is not None:
                        ax.scatter(
                            n,
                            row[field],
                            color=color,
                            marker={
                                "editing": "s",
                                "verification": "^",
                                "reproduction": "o",
                            }[milestone["stage"]],
                            s=30,
                        )
            ax.set(xlabel="Model request", ylabel=ylabel, ylim=(0, None))
            ax.legend(fontsize=8)
            ax.grid(alpha=0.2)
            fig.savefig(output / name, metadata={"Date": None})


def build_demo(data: Path, output: Path) -> None:
    bundle = Bundle(data)
    output.mkdir(parents=True, exist_ok=True)
    plots(bundle, output / "assets")
    if data.resolve() != (output / "public_replay/data").resolve():
        target = output / "evidence"
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(data, target)
        href = "evidence"
    else:
        href = "public_replay/data"
    cells = []

    def md(text):
        cells.append(nbformat.v4.new_markdown_cell(text))

    def code(text):
        cells.append(nbformat.v4.new_code_cell(text.strip()))

    md("""# Coding-agent inference economics: MiniMax M2.5 and GLM-5

**Question:** what do independently repeated maintenance tasks cost when failures count too?

**Objectives:** trace the model/harness boundary; distinguish patch acceptance from submission; reproduce held-out outcomes and unknown-aware economics.

**Prerequisites:** Python 3.14, the installed public Replay dependencies and this reviewed export. Setup installation may use network. Replay uses saved evidence: no Docker, credentials, provider access or paid execution. A *held-out task* was excluded from harness tuning; it may still have appeared in a model's training data.

**Workflow:** validate manifest → inspect preparation → learn from development/calibration → compare held-out repeats → calculate economics. The optional four-task development schedule is **unrun**, not a prerequisite. Public source-heavy audit evidence is **not bundled**; public projections have new hashes mapped to original run IDs.
""")
    code("""from pathlib import Path
import os, json
from IPython.display import display, HTML, SVG
from public_replay.bundle import Bundle
from public_replay.analysis import comparison, outcomes, measured_attempts, operational, request_payload
MODE = "replay"
assert MODE == "replay"  # This public interface has no paid Execute command.
bundle = Bundle(os.environ.get("PUBLIC_EVIDENCE_DIR"))
study = bundle.read("study.json")
records = {r["run_id"]: r for r in measured_attempts(bundle)}
def show(frame):
    display(HTML(frame.to_html(index=False, na_rep="unknown", float_format=lambda v: f"{v:.8f}")))
print("Validated public evidence:", bundle.manifest["content_id"])
""")
    md("""## Model and harness preparation

**Question:** what was sent and who could access what? The displayed payload uses saved native model/provider/reasoning/output settings. Task-specific messages are explicit placeholders because raw task/source traces are not distributed. It is preparation evidence, not a fresh invocation. `provider.only` pins routing, `allow_fallbacks=False` prevents fallback; `max_tokens` is a response-output limit, not an input-context limit.

The measured workflow invoked Harbor's controller and upstream mini-SWE-agent. Its sole coding tool executed bash in offline Docker. The API client ran on the trusted host. A separate evaluator applied the captured patch to a clean workspace with private checks. All required checks must pass: **checks_passed_pending_review**. Explicit submission can save calls but is a separate outcome.
""")
    code("""for model in study["model_preparation"]:
    print(json.dumps(request_payload(bundle, model), indent=2))
print(json.dumps(study["harness"], indent=2))
print("Historical invocation path: harbor run -c <frozen task/model config> --job-name <unique attempt>")
print("Path placeholders are intentional. Fresh execution requires new resolved configs and validated local task preparation.")
""")
    md("""**Learning:** the harness owns time/call/output limits and captures the final workspace even at budget exhaustion. Task commands remain offline. No previous patch or hidden feedback enters a fresh attempt. Private signing keys are not published; a content hash is an integrity check, not proof of public authenticity.

## Development learning and feature calibration

**Question:** what was learned before held-out evaluation? Development pilots retain their original policies; a passing intermediate workspace cannot replace a failed final verdict. Feature #39757 requested `ContextEditingMiddleware(..., token_counter: TokenCounter | None = None)`. Its calibration and environment-repair controls are distinct from model success. No control enters a measured denominator.
""")
    code("""selected = [study["selections"]["development_bug"], *study["selections"]["feature"].values()]
for run_id in selected:
    r = records[run_id]
    print(json.dumps({k:r.get(k) for k in ["run_id", "configuration_id", "split", "patch_accepted", "task_success", "failure_category", "termination_reason", "reproduction_evidence", "agent_action_summary", "candidate_diff_summary", "required_checks"]}, indent=2))
print("Optional four-task development comparison:", study["planned_development"]["status"],
      "—", study["planned_development"]["planned_launches"], "planned slots, zero observed launches")
""")
    md("""**Learning:** calibration evidence can explain a configuration change without proving that the change caused success. Original configuration IDs remain separate.

## Held-out repeats and two fixed walkthroughs

**Question:** which final workspaces passed? The six-task study has three fresh launches per model/task. The earlier two-task study is kept separate. Accepted means all frozen functional, regression, style, type and file-policy checks passed. Infrastructure interruptions are labelled separately and retain their costs.
""")
    code("""show(outcomes(bundle))
for case in ["accepted", "failure"]:
    r = records[study["selections"][case]]
    print(case, json.dumps({k:r.get(k) for k in ["run_id", "patch_accepted", "explicit_submission", "termination_reason", "failure_category"]}, indent=2))
    print("Diff summary:", r.get("candidate_diff_summary"))
    print("Reproduction and actions:", r.get("reproduction_evidence"), r.get("agent_action_summary"))
    print("Required checks:", [(c["check"],c["status"]) for c in r["required_checks"]])
print(bundle.optional_audit())
""")
    md("""**Learning:** the accepted MiniMax temperature repeat reached its call limit without submitting; automatic workspace evaluation accepted it. The contrasting repeat passed an intermediate regression, then later edits left invalid production syntax. Its final rejection is preserved. These are fixed examples, not a representative sample selected to estimate reliability.

## Context measurement and inference economics

**Question:** how much information and reported spend accumulated? A circle marks reproduction, square an edit and triangle a verification command. Cached input still occupies context. Saved tokenizer category counts are **proxy estimates**, not native counts; no tokenizer vocabulary or raw source is required here. Context growth does **not** demonstrate causation of failure.
""")
    code("""from public_replay.demo import plots
plots(bundle, Path("assets"))
display(SVG(filename="assets/context.svg"))
display(SVG(filename="assets/cost.svg"))
print("Six-task held-out comparison — includes all failed and interrupted launches")
show(comparison(bundle, "held_out_six"))
print("Earlier two-task held-out comparison — separate scope")
show(comparison(bundle, "held_out_two"))
print("Total operational inference — disjoint accounting scopes, not a model-comparison denominator")
show(operational(bundle))
""")
    md(f"""**Learning:** cost per accepted patch = all inference spend in the declared scope ÷ accepted final patches. A missing charge makes complete spend and the exact ratio **unknown**; no accepted patches makes the ratio undefined. Reasoning is a subset of completion tokens, not an extra quantity to add. Failed/malformed responses remain in the numerator. Setup and excluded infrastructure expenses stay in their own scopes.

## Findings, limits and extensions

The tables above are computed from request/attempt records, not entered totals. Three attempts per task/model are a small descriptive study. Hosted model aliases and public tasks limit causal/reliability claims. Keep future context masking or workflow changes in new settings. Fresh inference cannot guarantee identical patches, cost or latency.

Audit inputs: [attempts]({href}/attempts.json), [requests]({href}/requests.json), [context estimates]({href}/context.json), [study manifest]({href}/study.json), [task provenance]({href}/tasks.json). Full patches/raw traces/hidden fixtures: **not bundled**; use original artifact IDs for owner-mediated review. Task reconstruction has a separate doctor/prepare path and separately declared readiness. Owner licensing and Git-history review remain publication gates. No employer endorsement is claimed.
""")
    notebook = nbformat.v4.new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {
                "name": "python3",
                "display_name": "Python 3 (public Replay)",
                "language": "python",
            },
            "language_info": {"name": "python"},
        },
    )
    for i, cell in enumerate(notebook.cells):
        cell["id"] = hashlib.sha256((str(i) + cell.source).encode()).hexdigest()[:12]
    nbformat.write(notebook, output / "demo.ipynb")

    def prose(text):
        value = html.escape(text)
        value = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2">\1</a>', value)
        value = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", value)
        value = re.sub(r"`([^`]+)`", r"<code>\1</code>", value)
        parts = []
        for paragraph in value.split("\n\n"):
            if paragraph.startswith("## "):
                parts.append("<h2>" + paragraph[3:] + "</h2>")
            elif paragraph.startswith("# "):
                parts.append("<h1>" + paragraph[2:] + "</h1>")
            else:
                parts.append("<p>" + paragraph + "</p>")
        return "".join(parts)

    def table(frame):
        return frame.to_html(
            index=False, na_rep="unknown", float_format=lambda v: f"{v:.8f}"
        )

    study = bundle.read("study.json")
    records = {r["run_id"]: r for r in measured_attempts(bundle)}
    import pandas as pd

    snippets = []
    n = 0
    for cell in notebook.cells:
        if cell.cell_type == "markdown":
            snippets.append(prose(cell.source))
            continue
        n += 1
        snippets.append(
            "<details><summary>Executable preparation / analysis code</summary><pre>"
            + html.escape(cell.source)
            + "</pre></details>"
        )
        if n == 1:
            snippets.append(
                "<p>Replay evidence validated: <code>"
                + bundle.manifest["content_id"]
                + "</code></p>"
            )
        elif n == 2:
            payloads = {
                m: request_payload(bundle, m) for m in study["model_preparation"]
            }
            snippets.append(
                "<details><summary>Saved model settings and redacted request preparation</summary><pre>"
                + html.escape(json.dumps(payloads, indent=2))
                + "</pre></details>"
            )
            snippets.append(
                "<details><summary>Frozen harness and budgets</summary><pre>"
                + html.escape(json.dumps(study["harness"], indent=2))
                + "</pre></details>"
            )
        elif n == 3:
            ids = [
                study["selections"]["development_bug"],
                *study["selections"]["feature"].values(),
            ]
            snippets.append(
                table(
                    pd.DataFrame(
                        [
                            {
                                k: records[rid].get(k)
                                for k in [
                                    "run_id",
                                    "patch_accepted",
                                    "failure_category",
                                    "termination_reason",
                                ]
                            }
                            for rid in ids
                        ]
                    )
                )
            )
            snippets.append(
                "<p>Optional four-task development schedule: <strong>not run</strong> (24 planned slots).</p>"
            )
        elif n == 4:
            snippets.append(table(outcomes(bundle)))
            ids = [study["selections"][k] for k in ["accepted", "failure"]]
            snippets.append(
                table(
                    pd.DataFrame(
                        [
                            {
                                k: records[rid].get(k)
                                for k in [
                                    "run_id",
                                    "patch_accepted",
                                    "explicit_submission",
                                    "termination_reason",
                                ]
                            }
                            for rid in ids
                        ]
                    )
                )
            )
        elif n == 5:
            snippets.append(
                "<img src='assets/context.svg' alt='Request context with milestones'><img src='assets/cost.svg' alt='Cumulative native reported spend'>"
            )
            snippets.append(
                "<h3>Six-task held-out comparison</h3>" + table(comparison(bundle))
            )
            snippets.append(
                "<h3>Earlier two-task scope</h3>"
                + table(comparison(bundle, "held_out_two"))
            )
            snippets.append(
                "<h3>Total operational inference — separate scopes</h3>"
                + table(operational(bundle))
            )
    document = (
        "<!doctype html><html lang='en'><meta charset='utf-8'><title>Maintenance economics — saved Replay</title><style>body{max-width:1100px;margin:2em auto;font:17px system-ui;padding:0 1em}pre{white-space:pre-wrap;overflow-wrap:anywhere}table{border-collapse:collapse;font-size:13px;display:block;overflow-x:auto}td,th{padding:.4em;border:1px solid #ccc}td{max-width:440px;overflow-wrap:anywhere}img{max-width:100%}details{margin:1em 0}code{overflow-wrap:anywhere}</style>"
        + "".join(snippets)
        + "</html>"
    )
    (output / "demo.html").write_text(document)
