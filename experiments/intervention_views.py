"""Presentation-only interventions, shell milestones and purpose-labelled plots."""

import hashlib
import json
import re
import shlex
from pathlib import Path

import pandas as pd

from experiments import notebook_workflow as workflow
from experiments import study_views as prior

MODEL_COLORS = {"minimax/minimax-m2.5": "#2463a5", "z-ai/glm-5": "#b35716"}


def __getattr__(name):
    # Retain the exact first-response renderer and exclusive archival semantics.
    return getattr(prior, name)


def development_transitions(by_id, view):
    specs = [
        (
            "Large edits destroyed source structure",
            "Focused replacements, immediate diff checks and Git-free restore",
            view["bug_walkthroughs"][2]["run_id"],
            "The next patch was accepted, but its original submission-based task verdict stayed unsuccessful.",
        ),
        (
            "A usable patch could be discarded for missing submission",
            "Final workspace capture and evaluation; submission scored separately",
            view["bug_walkthroughs"][3]["run_id"],
            "The next fresh attempt passed checks and submitted; this demonstrates an end-to-end path, not a causal estimate.",
        ),
    ]
    return _transition_frames(by_id, specs)


def feature_transitions(by_id, view):
    specs = [
        (
            "Custom-counter API and regression coverage needed",
            "Initial feature attempt: 1,500-token response allowance",
            view["feature_run_id"],
            "Malformed test-writing calls left no new regression file; the candidate was rejected.",
        ),
        (
            "Large file-writing arguments were truncated",
            "Incremental edits; keep the 1,500-token response ceiling",
            view["feature_tuning_run_ids"][0],
            "The short-edit instruction alone did not resolve tool-format recovery in this attempt.",
        ),
        (
            "Tool arguments needed more room per response",
            "Larger response allowance: 6,000; total calls/tokens unchanged",
            view["response_ceiling_run_id"],
            "Complete larger payloads became possible, but regression and lint failures still rejected the candidate.",
        ),
        (
            "Check behavior needed testing with the second model",
            "GLM-5 calibration under the same effective limits",
            view["glm_feature_calibration_run_id"],
            "An intermediate passing workspace did not qualify the final captured candidate; file policy and typing setup still needed repair.",
        ),
        (
            "Test-directory policy and mypy cache limit were inconsistent",
            "Environment repair: validated test root and cacheless typing — MiniMax",
            view["repaired_calibration_runs"]["minimax"],
            "Functional checks passed, but remaining lint/format errors kept this fresh attempt unsuccessful.",
        ),
        (
            "Repeat calibration after zero-inference setup repair",
            "Environment repair: same repaired setup — GLM-5",
            view["repaired_calibration_runs"]["glm5"],
            "The fresh candidate passed every required check. Earlier failures and their costs remain unchanged.",
        ),
    ]
    return _transition_frames(by_id, specs)


def _transition_frames(by_id, specs):
    rows, audit = [], []
    for problem, change, run_id, learning in specs:
        r = by_id[run_id]
        outcome = (
            "accepted"
            if r["patch_accepted"] is True
            else "not accepted"
            if r["patch_accepted"] is False
            else "acceptance unknown"
        )
        if r.get("failure_category"):
            outcome += " · " + r["failure_category"].replace("_", " ")
        rows.append(
            {
                "Problem": problem,
                "Change introduced": change,
                "Next outcome": prior.MODELS[r["model"]] + ": " + outcome,
                "Learning": learning,
            }
        )
        audit.append(
            {
                "setting": change,
                "run_id": run_id,
                "configuration_id": r["configuration_id"],
                "original_classification": r["classification"],
                "task_success": r["task_success"],
                "patch_accepted": r["patch_accepted"],
                "explicit_submission": r["explicit_submission"],
            }
        )
    return pd.DataFrame(rows), audit


def executable_shell_lines(command):
    """Exclude heredoc payloads: imports/test text inside them are not shell runs."""
    lines, heredocs = [], []
    terminator, payload = None, []
    for line in command.splitlines():
        if terminator:
            if line.strip() == terminator:
                heredocs.append("\n".join(payload))
                terminator, payload = None, []
            else:
                payload.append(line)
            continue
        lines.append(line)
        match = re.search(r"<<-?\s*(['\"]?)([A-Za-z_]\w*)\1", line)
        if match:
            terminator = match.group(2)
    if terminator:
        heredocs.append("\n".join(payload))
    return lines, heredocs


def check_commands(lines):
    """Recognize actual shell command heads, not words in strings or imports."""
    found = []
    for line in lines:
        try:
            lexer = shlex.shlex(line, posix=True, punctuation_chars=";&|")
            lexer.whitespace_split = True
            tokens = list(lexer)
        except ValueError:
            continue
        segments, segment = [], []
        for token in tokens:
            if token in {"&&", "||", ";", "|", "&"}:
                segments.append(segment)
                segment = []
            else:
                segment.append(token)
        segments.append(segment)
        for words in segments:
            while words and re.fullmatch(r"[A-Za-z_]\w*=.*", words[0]):
                words = words[1:]
            if words and words[0] in {"env", "command"}:
                words = words[1:]
            if not words:
                continue
            head = Path(words[0]).name
            if head in {"pytest", "ruff", "mypy"} or (
                re.fullmatch(r"python(?:\d+(?:\.\d+)?)?", head)
                and len(words) > 2
                and words[1] == "-m"
                and words[2] in {"pytest", "ruff", "mypy"}
            ):
                found.append(" ".join(words))
    return found


def action_labels(command):
    lines, payloads = executable_shell_lines(command)
    shell = "\n".join(lines)
    writes = bool(
        re.search(
            r"write_text\(|write_bytes\(|apply_patch|sed\s+-i|\bpatch\s|\bcat\b[^\n]*<<|\btee\b|open\([^\n]*['\"](?:w|a)",
            command,
        )
    )
    # Test-file targets are taken from shell headers or Python path literals,
    # while checker execution is parsed only from executable shell command heads.
    test_target = bool(
        re.search(r"(?:/|['\" ])test_[\w-]+\.py", shell + "\n" + "\n".join(payloads))
    )
    labels = []
    if "/task/reproduce.py" in shell:
        labels.append("reproduction")
    if writes:
        labels.append("test creation" if test_target else "source edit")
    if check_commands(lines):
        labels.append("verification execution")
    if "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" in shell:
        labels.append("submission")
    return labels


def milestone_actions(record):
    _, trace = workflow.native_trace(record)
    rows, request = [], None
    for event in trace:
        if event.get("event") == "model_request":
            request = event.get("number")
        if event.get("event") != "tool_result":
            continue
        command = (event.get("tool_call") or {}).get("command", "")
        result = event.get("result") or {}
        rows.append(
            {
                "request": request,
                "labels": action_labels(command),
                "command": command[:700],
                "exit": result.get("exit_code", result.get("returncode")),
                "observation (tail)": result.get("output", "")[-700:],
            }
        )
    seen, selected = set(), []
    for i, row in enumerate(rows):
        labels = [label for label in row["labels"] if label not in seen]
        if i == len(rows) - 1:
            labels.append("last observed action")
        if labels:
            seen.update(row["labels"])
            selected.append(
                {
                    "Milestone": ", ".join(labels),
                    **{k: v for k, v in row.items() if k != "labels"},
                }
            )
    return pd.DataFrame(
        selected,
        columns=["Milestone", "request", "command", "exit", "observation (tail)"],
    )


def outcome_counts(bundle):
    frame = prior.held_frame(bundle)
    rows = []
    for model in prior.MODELS:
        group = frame[frame.model == model]
        interrupted = group.infrastructure_interrupted.eq(True)
        if group.loc[~interrupted, "patch_accepted"].isna().any():
            raise ValueError(
                "Unresolved non-interrupted patch verdict; do not relabel unknown as unsuccessful"
            )
        accepted = group.patch_accepted.eq(True) & ~interrupted
        unsuccessful = group.patch_accepted.eq(False) & ~interrupted
        row = {
            "model": model,
            "Model": prior.MODELS[model],
            "Accepted": int(accepted.sum()),
            "Unsuccessful": int(unsuccessful.sum()),
            "Interrupted": int(interrupted.sum()),
        }
        if row["Accepted"] + row["Unsuccessful"] + row["Interrupted"] != len(group):
            raise ValueError("Outcome partition must retain every launch")
        rows.append(row)
    return pd.DataFrame(rows)


def build_assets(project):
    import matplotlib as mpl
    from matplotlib.backends.backend_svg import FigureCanvasSVG
    from matplotlib.figure import Figure

    bundle = workflow.public_evidence(project)
    frame = prior.held_frame(bundle)
    context = bundle.read("context.json")
    root = project / "notebooks/assets/interventions"
    root.mkdir(parents=True, exist_ok=True)
    qa = project / "reports/notebook-workflow-v4"
    qa.mkdir(parents=True, exist_ok=True)

    def make_figure(height=4.4):
        fig = Figure(figsize=(10.5, height))
        FigureCanvasSVG(fig)
        return fig

    def save(fig, name):
        fig.savefig(root / (name + ".svg"), metadata={"Date": None})
        fig.savefig(qa / (name + "-preview.png"), dpi=120)

    with mpl.rc_context(
        {
            "font.family": "DejaVu Sans",
            "svg.fonttype": "none",
            "svg.hashsalt": "notebook-comparison-v4",
        }
    ):
        counts = outcome_counts(bundle)
        fig = make_figure(3.3)
        ax = fig.subplots()
        left = [0, 0]
        for label, color in [
            ("Accepted", "#25864b"),
            ("Unsuccessful", "#7d8997"),
            ("Interrupted", "#bf684f"),
        ]:
            values = counts[label].tolist()
            ax.barh([0, 1], values, left=left, color=color, label=label, height=0.48)
            for i, (offset, value) in enumerate(zip(left, values, strict=True)):
                if value:
                    ax.text(
                        offset + value / 2,
                        i,
                        str(value),
                        ha="center",
                        va="center",
                        color="white",
                        fontweight="bold",
                    )
            left = [a + b for a, b in zip(left, values, strict=True)]
        ax.set(
            yticks=[0, 1],
            yticklabels=counts.Model.tolist(),
            xlabel="Launches (18 per model)",
            xlim=(0, 18),
            xticks=[0, 3, 6, 9, 12, 15, 18],
            title="Main outcome: six frozen tasks × three attempts per model",
        )
        ax.invert_yaxis()
        ax.legend(
            loc="upper center", bbox_to_anchor=(0.5, -0.27), ncol=3, frameon=False
        )
        for tick, model in zip(ax.get_yticklabels(), prior.MODELS, strict=True):
            tick.set_color(MODEL_COLORS[model])
        fig.tight_layout()
        save(fig, "main-outcomes")
        fig = make_figure()
        axes = fig.subplots(1, 2)
        for index, model in enumerate(prior.MODELS):
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
                        s=25,
                        color=MODEL_COLORS[model],
                        alpha=0.8,
                    )
        for ax, ylabel in zip(
            axes,
            ["Requests consumed per attempt", "Reported output tokens per attempt"],
            strict=True,
        ):
            ax.set(
                xticks=[0, 1],
                xticklabels=list(prior.MODELS.values()),
                ylabel=ylabel,
                ylim=(0, None),
            )
            ax.grid(axis="y", alpha=0.2)
        fig.suptitle(
            "Work consumed across all 36 attempts, including failures\n× = incomplete record (known lower bound); ● = complete for that measure"
        )
        fig.tight_layout()
        save(fig, "distributions")
        matched = [
            (
                rid,
                prior.MODELS[frame[frame.run_id == rid].iloc[0].model],
                MODEL_COLORS[frame[frame.run_id == rid].iloc[0].model],
                "-",
                bool(frame[frame.run_id == rid].iloc[0].patch_accepted),
            )
            for rid in prior.MATCHED_RUNS
        ]

        def trajectory(field, name, title, ylabel, specs, subtitle):
            fig = make_figure()
            ax = fig.subplots()
            max_request = 0
            maximum_value = 0
            for index, (rid, label, color, style, accepted) in enumerate(specs):
                points = context[rid]["requests"]
                max_request = max(max_request, points[-1]["request"])
                outcome = "accepted" if accepted else "unsuccessful"
                ax.plot(
                    [r["request"] for r in points],
                    [
                        r[field] if r[field] is not None else float("nan")
                        for r in points
                    ],
                    color=color,
                    linestyle=style,
                    label=f"{label} · {outcome}",
                )
                valid = [r for r in points if r[field] is not None]
                if valid:
                    final = valid[-1]
                    value = final[field]
                    maximum_value = max(maximum_value, max(r[field] for r in valid))
                    text = (
                        f"${value:.8f}" if "cost" in field else f"{value:,.0f} tokens"
                    )
                    if final is not points[-1]:
                        text += " (last known; final unknown)"
                    ax.scatter(final["request"], value, color=color, s=24)
                    ax.annotate(
                        f"{label}: {text}",
                        (final["request"], value),
                        xytext=(8, 12 if index == 0 else -17),
                        textcoords="offset points",
                        color=color,
                        fontsize=9,
                        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85},
                    )
            ax.set(
                title=title + "\n" + subtitle,
                xlabel="Model request",
                ylabel=ylabel,
                ylim=(0, None),
                xlim=(0, max_request + 14),
            )
            if maximum_value:
                ax.set_ylim(0, maximum_value * 1.2)
            ax.legend(loc="upper left", fontsize=9)
            ax.grid(alpha=0.2)
            fig.tight_layout()
            save(fig, name)

        subtitle = "Integer-temperature metadata bug · attempt 001"
        trajectory(
            "native_input_tokens",
            "matched-context",
            "Input context carried into successive requests",
            "Reported input tokens/request (cached included)",
            matched,
            subtitle,
        )
        trajectory(
            "cumulative_reported_cost_usd",
            "matched-cost",
            "Cumulative reported inference charge",
            "Cumulative reported cost (USD)",
            matched,
            subtitle,
        )
        selections = bundle.read("study.json")["selections"]
        diagnostic = []
        for kind, style in [("accepted", "-"), ("failure", "--")]:
            rid = selections[kind]
            number = frame[frame.run_id == rid].iloc[0].attempt_number
            diagnostic.append(
                (
                    rid,
                    f"MiniMax attempt {number:03d}",
                    MODEL_COLORS["minimax/minimax-m2.5"],
                    style,
                    kind == "accepted",
                )
            )
        trajectory(
            "native_input_tokens",
            "optional-minimax",
            "MiniMax: accepted versus failed attempts on the same task",
            "Reported input tokens/request",
            diagnostic,
            "Integer-temperature metadata bug · separate diagnostic attempts",
        )
    manifest = {
        "evidence_content_id": bundle.manifest["content_id"],
        "matched_task": "LC-INT-TEMPERATURE-001",
        "readable_task": "Integer-temperature metadata bug",
        "matched_attempt": 1,
        "matched_runs": prior.MATCHED_RUNS,
        "model_colors": MODEL_COLORS,
        "outcome_counts": counts.to_dict("records"),
        "files": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in root.glob("*.svg")
        },
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def assets(project):
    root = project / "notebooks/assets/interventions"
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


if __name__ == "__main__":
    print(json.dumps(build_assets(Path(__file__).resolve().parents[1]), indent=2))


def transition_table_html(frame):
    return (
        "<style>.development-transitions td,.development-transitions th{white-space:normal;text-align:left;vertical-align:top;padding:8px;max-width:22em;font-size:13px}</style>"
        + frame.to_html(
            index=False, border=0, classes="development-transitions", escape=True
        )
    )
