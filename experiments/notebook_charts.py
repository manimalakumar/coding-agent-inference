"""Notebook chart assets derived from reviewed historical evidence."""

import hashlib
import json
from pathlib import Path

from experiments import notebook_workflow as workflow
from experiments import study_views as prior
from experiments.intervention_views import MODEL_COLORS, outcome_counts


def cumulative_cost_points(points):
    """Stop a cumulative exact series at the first unknown/unreconciled charge.

    Known lower-bound subtotals are not exact cumulative costs. Keep the missing
    request in the plotted sequence so no line can bridge that gap.
    """
    complete = True
    values = []
    for point in points:
        complete = (
            complete
            and point.get("reported_cost_usd") is not None
            and point.get("cumulative_cost_complete") is True
            and point.get("cumulative_reported_cost_usd") is not None
        )
        values.append(point["cumulative_reported_cost_usd"] if complete else None)
    return values


def request_axis_range(requests, limit):
    """Use observed request numbers without extending beyond the frozen ceiling."""
    if not requests or min(requests) < 1 or max(requests) > limit:
        raise ValueError("Trajectory request numbers must fit the frozen call limit")
    lower, upper = min(requests), max(requests)
    return (lower if lower < upper else max(0, lower - 1), upper)


def build_assets(project):
    import matplotlib as mpl
    from matplotlib.backends.backend_svg import FigureCanvasSVG
    from matplotlib.figure import Figure

    bundle = workflow.public_evidence(project)
    frame = prior.held_frame(bundle)
    context = bundle.read("context.json")
    request_limit = bundle.read("study.json")["harness"]["budgets"]["model_calls"]
    root = project / "notebooks/assets/tutorial"
    root.mkdir(parents=True, exist_ok=True)
    qa = project / ".local/chart-endpoints/previews"
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
            "svg.hashsalt": "notebook-comparison-v6",
        }
    ):
        counts = outcome_counts(bundle)
        if (
            not counts[["Accepted", "Unsuccessful", "Interrupted"]]
            .sum(axis=1)
            .eq(18)
            .all()
        ):
            raise ValueError("Outcome plot requires exactly 18 launches per model")
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
            yticklabels=[
                f"{row.Model}\n{row.Accepted}/18 accepted ({row.Accepted / 18:.1%})"
                for row in counts.itertuples()
            ],
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
        for ax, ylabel, ceiling in zip(
            axes,
            ["Requests consumed per attempt", "Reported output tokens per attempt"],
            [30, 10000],
            strict=True,
        ):
            ax.axhline(
                ceiling,
                color="#555555",
                linestyle="--",
                linewidth=1,
                label=f"Budget: {ceiling:,}",
            )
            ax.set(
                xticks=[0, 1],
                xticklabels=list(prior.MODELS.values()),
                ylabel=ylabel,
                ylim=(0, None),
            )
            ax.grid(axis="y", alpha=0.2)
            ax.legend(loc="lower left", fontsize=8)
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
            fig = make_figure(4.8)
            ax = fig.subplots()
            request_numbers = []
            maximum_value = 0
            for index, (rid, label, color, style, accepted) in enumerate(specs):
                points = context[rid]["requests"]
                request_numbers.extend(r["request"] for r in points)
                outcome = (
                    "final patch passed all required checks"
                    if accepted
                    else "final patch failed required checks"
                )
                values = (
                    cumulative_cost_points(points)
                    if field == "cumulative_reported_cost_usd"
                    else [r[field] for r in points]
                )
                ax.plot(
                    [r["request"] for r in points],
                    [value if value is not None else float("nan") for value in values],
                    color=color,
                    linestyle=style,
                    label=label,
                )
                valid = [
                    dict(r, **{field: value})
                    for r, value in zip(points, values, strict=True)
                    if value is not None
                ]
                if valid:
                    final = valid[-1]
                    value = final[field]
                    maximum_value = max(maximum_value, max(r[field] for r in valid))
                    text = (
                        f"${value:.3f}" if "cost" in field else f"{value:,.0f} tokens"
                    )
                    if final["request"] != points[-1]["request"]:
                        text += " (last known; final unknown)"
                    else:
                        text += " — " + outcome
                    ax.scatter(final["request"], value, color=color, s=24)
                    ax.annotate(
                        text,
                        (final["request"], value),
                        xytext=(-8, 14 if index == 0 else -22),
                        textcoords="offset points",
                        ha="right",
                        color=color,
                        fontsize=9,
                        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85},
                    )
            ax.set(
                title=title + "\n" + subtitle,
                xlabel="Model request",
                ylabel=ylabel,
                ylim=(0, None),
                xlim=request_axis_range(request_numbers, request_limit),
            )
            if maximum_value:
                ax.set_ylim(0, maximum_value * 1.2)
            ax.legend(loc="upper left", fontsize=9)
            ax.grid(alpha=0.2)
            caption = (
                "Spending accumulated with each request. The final evaluation outcome is shown at each line’s endpoint."
                if "cost" in field
                else "Input context varied with each request. The final evaluation outcome is shown at each line’s endpoint."
            )
            fig.text(0.5, 0.025, caption, ha="center", fontsize=9)
            fig.tight_layout(rect=(0, 0.07, 1, 1))
            save(fig, name)

        subtitle = "Integer-temperature metadata bug · attempt 001"
        trajectory(
            "native_input_tokens",
            "matched-context",
            "Input context carried into successive requests",
            "Input tokens per request, including cached tokens",
            matched,
            subtitle,
        )
        trajectory(
            "cumulative_reported_cost_usd",
            "matched-cost",
            "Cumulative reported inference cost (USD)",
            "Cumulative reported inference cost (USD)",
            matched,
            subtitle,
        )
    manifest = {
        "evidence_content_id": bundle.manifest["content_id"],
        "matched_task": "LC-INT-TEMPERATURE-001",
        "readable_task": "Integer-temperature metadata bug",
        "matched_attempt": 1,
        "matched_runs": prior.MATCHED_RUNS,
        "model_colors": MODEL_COLORS,
        "outcome_counts": counts.to_dict("records"),
        "launches_per_model": 18,
        "trajectory_request_limit": request_limit,
        "trajectory_request_range": request_axis_range(
            [
                r["request"]
                for rid in prior.MATCHED_RUNS
                for r in context[rid]["requests"]
            ],
            request_limit,
        ),
        "budget_reference_lines": {"model_requests": 30, "output_tokens": 10000},
        "cost_policy": "Exact cumulative costs stop at first unknown; lower bounds are not plotted as exact totals",
        "files": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in root.glob("*.svg")
        },
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def assets(project):
    root = project / "notebooks/assets/tutorial"
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
