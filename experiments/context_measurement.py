"""Offline context accounting from saved request bodies; never invokes a model."""

import argparse
import base64
import hashlib
import json
from decimal import Decimal
from functools import lru_cache
from pathlib import Path

from experiments.records import events, harbor_record, read

VERSION = "saved-context-v1"
ROOT = Path(__file__).resolve().parents[1]
CATEGORIES = (
    "system_instructions",
    "task_instructions",
    "workflow_instructions",
    "assistant_text",
    "assistant_tool_calls",
    "assistant_reasoning_fields",
    "assistant_other_visible_fields",
    "tool_results",
    "controller_or_other_user_messages",
    "other_messages",
    "tool_schema",
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def number(value):
    return (
        value
        if isinstance(value, (int, float)) and not isinstance(value, bool)
        else None
    )


class ProxyTokenizer:
    """Hash-verified local vocabulary; no tiktoken downloader/cache APIs."""

    def __init__(self, project):
        import tiktoken

        folder = Path(project) / "reports/context-measurement-v1/tokenizer"
        self.metadata = read(folder / "metadata.json")
        asset = folder / "o200k_base.tiktoken"
        if sha(asset) != self.metadata["sha256"]:
            raise ValueError("Tokenizer vocabulary checksum mismatch")
        ranks = {}
        for line in asset.read_bytes().splitlines():
            token, rank = line.split()
            ranks[base64.b64decode(token)] = int(rank)
        self.encoding = tiktoken.Encoding(
            name="context_o200k_proxy_v1",
            pat_str=self.metadata["pattern"],
            mergeable_ranks=ranks,
            special_tokens=self.metadata["special_tokens"],
        )

    @lru_cache(maxsize=16000)
    def count(self, text):
        return len(self.encoding.encode_ordinary(text))


def content_text(value):
    return (
        value if isinstance(value, str) else "" if value is None else canonical(value)
    )


def breakdown(body, task_text, tokenizer):
    totals = dict.fromkeys(CATEGORIES, 0)
    messages = []
    for index, message in enumerate(body.get("messages") or []):
        counts = dict.fromkeys(CATEGORIES, 0)
        role, text = message.get("role"), content_text(message.get("content"))
        if role in ("system", "developer"):
            counts["system_instructions"] = tokenizer.count(text)
        elif role == "user" and task_text and task_text in text:
            # Count components separately: BPE boundaries and chat framing are not native.
            prefix, suffix = text.split(task_text, 1)
            counts["task_instructions"] = tokenizer.count(task_text)
            counts["workflow_instructions"] = tokenizer.count(prefix) + tokenizer.count(
                suffix
            )
        elif role == "tool":
            counts["tool_results"] = tokenizer.count(text)
        elif role == "assistant":
            counts["assistant_text"] = tokenizer.count(text)
            if message.get("tool_calls"):
                counts["assistant_tool_calls"] = tokenizer.count(
                    canonical(message["tool_calls"])
                )
            reasoning = {
                k: message[k]
                for k in ("reasoning", "reasoning_details")
                if message.get(k) is not None
            }
            if reasoning:
                counts["assistant_reasoning_fields"] = tokenizer.count(
                    canonical(reasoning)
                )
            extra = {
                k: v
                for k, v in message.items()
                if k
                not in (
                    "role",
                    "content",
                    "tool_calls",
                    "reasoning",
                    "reasoning_details",
                )
                and v is not None
            }
            if extra:
                counts["assistant_other_visible_fields"] = tokenizer.count(
                    canonical(extra)
                )
        else:
            category = (
                "controller_or_other_user_messages"
                if role == "user"
                else "other_messages"
            )
            counts[category] = tokenizer.count(text)
        for key, value in counts.items():
            totals[key] += value
        messages.append(
            {
                "message_index": index,
                "role": role,
                "content_characters": len(text),
                "estimated_tokens_by_category": {k: v for k, v in counts.items() if v},
            }
        )
    if body.get("tools"):
        totals["tool_schema"] = tokenizer.count(canonical(body["tools"]))
    return totals, messages


def measure_trace(trace_events, task_text, tokenizer):
    """Keep every request, including malformed/missing responses and format repairs."""
    requests, tool_events = [], []
    current = None
    for event in trace_events:
        if event["event"] == "model_request":
            current = {
                "number": event.get("number", len(requests) + 1),
                "body": event["body"],
                "response": None,
            }
            requests.append(current)
        elif event["event"] == "model_response" and current is not None:
            current["response"] = event["response"]
        elif event["event"] == "tool_result" and current is not None:
            tool_events.append({"request": current["number"], **event})
    known_cost, complete = Decimal(0), True
    rows = []
    for request in requests:
        response = request["response"] or {}
        raw = response.get("provider_response") or response
        usage = raw.get("usage") or response.get("usage") or {}
        cost = number(usage.get("cost"))
        if cost is None:
            complete = False
        else:
            known_cost += Decimal(str(cost))
        native_input = number(usage.get("prompt_tokens"))
        cached = number((usage.get("prompt_tokens_details") or {}).get("cached_tokens"))
        estimated, messages = breakdown(request["body"], task_text, tokenizer)
        choice = next(iter(raw.get("choices") or []), {})
        rows.append(
            {
                "request": request["number"],
                "generation_id": raw.get("id"),
                "actual_provider": raw.get("provider"),
                "native_input_tokens": native_input,
                "native_cached_input_tokens": cached,
                "native_output_tokens": number(usage.get("completion_tokens")),
                "native_reasoning_output_tokens": number(
                    (usage.get("completion_tokens_details") or {}).get(
                        "reasoning_tokens"
                    )
                ),
                "reported_cost_usd": cost,
                "cumulative_known_reported_cost_usd": float(known_cost),
                "cumulative_reported_cost_usd": float(known_cost) if complete else None,
                "cumulative_cost_complete": complete,
                "finish_reason": choice.get("finish_reason"),
                "response_received": request["response"] is not None,
                "message_count": len(request["body"].get("messages") or []),
                "output_ceiling_tokens": request["body"].get("max_tokens"),
                "estimated_tokens_by_category": estimated,
                "estimated_visible_payload_tokens": sum(estimated.values()),
                "estimate_minus_native_tokens": sum(estimated.values()) - native_input
                if native_input is not None
                else None,
                "message_estimates": messages,
            }
        )
    for index, row in enumerate(rows):
        previous = rows[index - 1]["native_input_tokens"] if index else None
        row["native_input_delta_from_previous_request"] = (
            row["native_input_tokens"] - previous
            if previous is not None and row["native_input_tokens"] is not None
            else None
        )
    return rows, tool_events


def measure_run(project, run_id, tokenizer, milestone_spec):
    project = Path(project).resolve()
    folder = (project / "runs/harbor" / run_id).resolve()
    if not folder.is_relative_to(project / "runs/harbor"):
        raise ValueError("Run must be inside the saved Harbor artifact directory")
    record = harbor_record(folder)
    config = read(folder / "config.json")
    task_path = Path(config["task"]["path"]) / "instruction.md"
    trace_events = events(folder / "agent/requests.jsonl")
    rows, tools = measure_trace(trace_events, task_path.read_text(), tokenizer)
    sent_requests = [e for e in trace_events if e["event"] == "model_request"]
    milestones = []
    for spec in milestone_spec:
        matching = [e for e in tools if e["request"] == spec["request"]]
        selected = matching[spec["tool_index"]]
        command = selected["tool_call"]["command"]
        if hashlib.sha256(command.encode()).hexdigest() != spec["command_sha256"]:
            raise ValueError("Audited milestone command changed")
        if selected["result"]["returncode"] != spec["expected_exit_code"]:
            raise ValueError("Audited milestone exit code changed")
        milestones.append(
            {
                **spec,
                "tool_call_id": selected["tool_call"].get("tool_call_id"),
                "visible_in_next_request": next(
                    (
                        e["number"]
                        for e in sent_requests
                        if e["number"] > spec["request"]
                        and any(
                            m.get("role") == "tool"
                            and m.get("tool_call_id")
                            == selected["tool_call"].get("tool_call_id")
                            for m in e["body"].get("messages", [])
                        )
                    ),
                    None,
                ),
                "timing": "Tool runs after this request response; its result enters a later request, if any.",
            }
        )
    known_inputs = [
        r["native_input_tokens"] for r in rows if r["native_input_tokens"] is not None
    ]
    known_caches = [
        r["native_cached_input_tokens"]
        for r in rows
        if r["native_cached_input_tokens"] is not None
    ]
    return {
        "run_id": run_id,
        "model": record["model"],
        "configuration_id": record["configuration_id"],
        "task_id": record["task_id"],
        "patch_accepted": record["patch_accepted"],
        "task_success": record["task_success"],
        "explicit_submission": record["explicit_submission"],
        "failure_category": record.get("failure_category"),
        "termination_reason": record["termination_reason"],
        "sources": {
            str(path.relative_to(project)): sha(path)
            for path in [
                folder / "config.json",
                folder / "agent/requests.jsonl",
                folder / "result.json",
                task_path,
            ]
        },
        "first_request_messages": rows[0]["message_count"] if rows else None,
        "first_request_roles": [
            m.get("role") for m in sent_requests[0]["body"].get("messages", [])
        ]
        if sent_requests
        else [],
        "persistent_memory": "No persistent agent-memory resource configured; frozen controller creates a fresh DefaultAgent per attempt. Trace is not evidence about provider-internal cache storage.",
        "summary": {
            "requests": len(rows),
            "first_native_input_tokens": rows[0]["native_input_tokens"]
            if rows
            else None,
            "last_native_input_tokens": rows[-1]["native_input_tokens"]
            if rows
            else None,
            "peak_native_input_tokens": max(known_inputs) if known_inputs else None,
            "native_input_tokens_summed_across_requests": sum(known_inputs),
            "native_input_complete": len(known_inputs) == len(rows),
            "native_cached_input_tokens_summed": sum(known_caches)
            if known_caches
            else None,
            "native_cache_fields_complete": len(known_caches) == len(rows),
            "total_reported_cost_usd": rows[-1]["cumulative_reported_cost_usd"]
            if rows
            else None,
            "known_reported_cost_usd": rows[-1]["cumulative_known_reported_cost_usd"]
            if rows
            else 0,
            "cost_complete": rows[-1]["cumulative_cost_complete"] if rows else False,
        },
        "milestones": milestones,
        "requests": rows,
    }


def measure(project):
    project = Path(project)
    manifest = read(project / "reports/context-measurement-v1/protocol.json")
    tokenizer = ProxyTokenizer(project)
    runs = [
        measure_run(project, spec["run_id"], tokenizer, spec["milestones"])
        for spec in manifest["runs"]
    ]
    result = {
        "analysis_version": VERSION,
        "scope": "Two explicit repaired feature-calibration attempts; descriptive analysis, not the unrun four-task comparison.",
        "native_metrics": "OpenRouter response usage.prompt_tokens, cached_tokens and usage.cost; every saved request is retained.",
        "estimate_method": "o200k_base proxy BPE on visible text segments and serialized tool calls/schema; not model-native, excludes chat framing and unavailable provider details; category sums are NOT a native token partition.",
        "tokenizer": tokenizer.metadata,
        "source_protocol_sha256": sha(
            project / "reports/context-measurement-v1/protocol.json"
        ),
        "inference_calls": 0,
        "context_intervention": False,
        "runs": runs,
    }
    result["data_sha256"] = hashlib.sha256(canonical(result).encode()).hexdigest()
    return result


def render_figures(data, destination):
    """Rebuild standalone scientific plots locally; notebook Replay reads verified SVGs."""
    import matplotlib as mpl
    from matplotlib.backends.backend_svg import FigureCanvasSVG
    from matplotlib.figure import Figure

    destination = Path(destination)
    colors = ["#2463a5", "#b35716"]
    shorts = ["MiniMax", "GLM-5"]
    files = []
    with mpl.rc_context(
        {
            "font.size": 10,
            "svg.fonttype": "none",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    ):
        for kind in ["input-context", "cumulative-cost"]:
            fig = Figure(figsize=(10.5, 5.2))
            FigureCanvasSVG(fig)
            ax = fig.subplots()
            fig.subplots_adjust(left=0.12, right=0.98, bottom=0.20, top=0.89)
            for index, run in enumerate(data["runs"]):
                rows = run["requests"]
                x = [r["request"] for r in rows]
                key = (
                    "native_input_tokens"
                    if kind == "input-context"
                    else "cumulative_reported_cost_usd"
                )
                y = [r[key] if r[key] is not None else float("nan") for r in rows]
                ax.plot(
                    x,
                    y,
                    color=colors[index],
                    marker=".",
                    linewidth=2,
                    label=shorts[index],
                )
                if kind == "input-context":
                    for m in run["milestones"]:
                        request = m["visible_in_next_request"]
                        row = next((r for r in rows if r["request"] == request), None)
                        if not row or row[key] is None:
                            continue
                        markers = {
                            "reproduction": "o",
                            "editing": "s",
                            "verification": "^",
                        }
                        ax.scatter(
                            [request],
                            [row[key]],
                            color=colors[index],
                            marker=markers[m["stage"]],
                            s=55,
                            zorder=4,
                        )
                        ax.annotate(
                            f"{shorts[index]}: {m['label']} → #{request}",
                            (request, row[key]),
                            xytext=(
                                7,
                                13
                                if index == 0 or m["stage"] == "verification"
                                else -28,
                            ),
                            textcoords="offset points",
                            fontsize=8,
                            color=colors[index],
                        )
                else:
                    outcome = (
                        "accepted · pending review"
                        if run["patch_accepted"]
                        else "not accepted · lint/format"
                    )
                    if rows[-1][key] is not None:
                        ax.annotate(
                            f"{shorts[index]}: {outcome}\n${rows[-1][key]:.8f}",
                            (x[-1], y[-1]),
                            xytext=(9, 0),
                            textcoords="offset points",
                            fontsize=9,
                            color=colors[index],
                        )
            ax.set_xlabel("Model request number (within attempt)")
            ax.grid(alpha=0.2)
            last_request = max(r["requests"][-1]["request"] for r in data["runs"])
            ax.set_xticks([1, *range(3, last_request + 1, 3)])
            ax.legend(loc="upper left")
            if kind == "input-context":
                ax.set_ylabel("Provider-reported input tokens (cached included)")
                ax.set_title("Request context across the repaired feature calibration")
                ax.set_xlim(0.5, last_request + 1.5)
                ax.set_ylim(bottom=0)
                fig.text(
                    0.5,
                    0.04,
                    "Milestones mark the first later request that can contain the tool result. Counts are native, not tokenizer estimates.",
                    ha="center",
                    fontsize=8,
                )
            else:
                ax.set_ylabel("Cumulative provider-reported cost (USD)")
                ax.set_title(
                    "Reported spend across the two repaired calibration attempts"
                )
                ax.set_xlim(0.5, max(len(r["requests"]) for r in data["runs"]) + 11)
                maximum_cost = max(
                    r["cumulative_known_reported_cost_usd"]
                    for run in data["runs"]
                    for r in run["requests"]
                )
                ax.set_ylim(0, max(maximum_cost * 1.22, 0.001))
                fig.text(
                    0.5,
                    0.04,
                    "Unknown charges break the complete-cost series; no missing request is treated as free. One attempt per model is not a reliability estimate.",
                    ha="center",
                    fontsize=8,
                )
            name = kind + ".svg"
            fig.savefig(
                destination / name,
                metadata={
                    "Date": None,
                    "Description": "Saved trace analysis " + data["data_sha256"],
                },
            )
            preview = destination / "previews"
            preview.mkdir(exist_ok=True)
            fig.savefig(preview / (kind + ".png"), dpi=130)
            files.append(name)
    receipt = {
        "data_sha256": data["data_sha256"],
        "plots": {name: sha(destination / name) for name in files},
        "inference_calls": 0,
        "native_run_artifacts_modified": False,
        "plot_source_sha256": sha(__file__),
    }
    (destination / "plots.json").write_text(json.dumps(receipt, indent=2) + "\n")


def plot_paths(project, data):
    """Read-only Replay: refuse stale plot data or modified plot artifacts."""
    folder = Path(project) / "reports/context-measurement-v1"
    receipt = read(folder / "plots.json")
    if receipt["data_sha256"] != data["data_sha256"]:
        raise ValueError(
            "Context plot data are stale; rebuild the derived analysis explicitly"
        )
    if receipt.get("plot_source_sha256") != sha(__file__):
        raise ValueError("Plot or measurement code changed; rebuild derived plots")
    paths = []
    for name, expected in receipt["plots"].items():
        path = folder / name
        if sha(path) != expected:
            raise ValueError("Context plot checksum mismatch")
        paths.append(path)
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=ROOT)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()
    data = measure(args.project)
    destination = args.project / "reports/context-measurement-v1"
    (destination / "measurement.json").write_text(json.dumps(data, indent=2) + "\n")
    if args.render:
        render_figures(data, destination)
    print(
        json.dumps(
            {
                "data_sha256": data["data_sha256"],
                "inference_calls": 0,
                "runs": [{"model": r["model"], **r["summary"]} for r in data["runs"]],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
