"""The shared inspection path for main.ipynb Execute and Replay; no agent loop."""

import copy
import hashlib
import html
import json
import os
import re
import shlex
import shutil
import subprocess
import tomllib
from decimal import Decimal
from pathlib import Path

import pandas as pd

from experiments.artifact_paths import resolve_artifact_path
from experiments.configuration_paths import configuration_path
from experiments.launch import launch, verify_freeze
from experiments.records import events, read, walkthrough
from experiments.relocated_records import launch_index, load_runs
from public_replay.bundle import Bundle
from public_replay.sanitize import redact
from public_replay.usage import normalize_usage

DEMO_STUDY = "annotated-tools-workspace-v2"
DEMO_RUN_ID = (
    "annotated-tools-workspace-v2-minimax-attempt-001/LC-ANNOTATED-TOOLS-001__4SxgJND"
)


def records(project: Path):
    return load_runs(project, launch_index(project), include_controls=False)


def exact_record(project: Path, run_id: str):
    matched = [r for r in records(project) if r["run_id"] == run_id]
    if len(matched) != 1:
        raise ValueError("Expected exactly one saved run: " + run_id)
    return matched[0]


def preparation(project: Path, study_id: str = DEMO_STUDY):
    selection = read(configuration_path(project, "configs/study.json"))["studies"][
        study_id
    ]
    config = read(project / selection["job_config"])
    policy = config["agents"][0]["kwargs"]["policy"]
    frozen = read(project / selection["frozen_setting"])
    task = Path(config["tasks"][0]["path"])
    if not task.is_absolute():
        task = project / task
    controls = read(project / frozen["validation_report"], {})
    return {
        "study_id": study_id,
        "selection": selection,
        "job": config,
        "policy": policy,
        "frozen": frozen,
        "task": task,
        "controls": controls,
        "instructions": (task / "instruction.md").read_text()
        if (task / "instruction.md").exists()
        else "not available: task preparation required",
        "environment": tomllib.loads((task / "task.toml").read_text())
        if (task / "task.toml").exists()
        else {},
    }


def control_rows(prepared):
    return [
        {
            "control": name,
            "model requests": row.get("model_requests"),
            "patch accepted": row.get("patch_accepted"),
            "task success": row.get("task_success"),
            "reproducer exit": row.get("check_exit_codes", {}).get("reproducer"),
            "classification": row.get("classification"),
        }
        for name, row in prepared["controls"].get("controls", {}).items()
    ]


def preflight_execute(project: Path, prepared, credentials: Path | None = None):
    """Host readiness only after explicit Execute consent, before any model request."""
    selection = prepared["selection"]
    verify_freeze(project, selection["freeze_manifest"], selection["frozen_setting"])
    controls = prepared["controls"].get("controls", {})
    if (
        controls.get("baseline", {}).get("patch_accepted") is not False
        or controls.get("reference", {}).get("patch_accepted") is not True
    ):
        raise RuntimeError(
            "Missing baseline-reject/reference-pass controls; prepare and validate before inference"
        )
    for path in [project / ".venv/bin/harbor", prepared["task"] / "task.toml"]:
        if not path.exists():
            raise FileNotFoundError("Prepared execution resource missing: " + str(path))
    policy = prepared["policy"]
    for name in ["receipt_key", "reference_fixture"]:
        if not Path(policy[name]).is_file():
            raise FileNotFoundError("Private execution resource missing: " + name)
    credentials = (
        Path(credentials).expanduser()
        if credentials
        else Path.home() / ".config/openrouter/credentials.env"
    )
    if not credentials.is_file():
        raise FileNotFoundError(
            "Host OpenRouter credential file missing; never paste it into a cell"
        )
    docker = shutil.which("docker") or str(Path.home() / ".docker/bin/docker")
    env = os.environ | {
        "PATH": str(Path(docker).parent) + os.pathsep + os.environ.get("PATH", "")
    }
    for image in [
        prepared["environment"]["environment"]["docker_image"],
        prepared["environment"]["verifier"]["environment"]["docker_image"],
    ]:
        result = subprocess.run(
            [docker, "image", "inspect", image],
            env=env,
            capture_output=True,
            timeout=30,
        )
        if result.returncode:
            raise RuntimeError(
                "Prepared image unavailable; trusted setup is required before inference"
            )
    return {
        "status": "ready",
        "credentials": "host-only file exists; value not displayed",
        "images": "agent and separate verifier available",
    }


def execute_or_replay(
    project: Path,
    *,
    mode="replay",
    confirmed=False,
    saved_run_id=DEMO_RUN_ID,
    study_id=DEMO_STUDY,
    job_name=None,
    credentials=None,
):
    """Both modes return the SAME native record for all downstream inspection."""
    if mode not in {"replay", "execute"}:
        raise ValueError("Mode must be replay or execute")
    if mode == "replay":
        return {
            "mode": mode,
            "launched": False,
            "record": exact_record(project, saved_run_id),
        }
    if confirmed is not True:
        raise ValueError("Execute requires CONFIRM_PAID_EXECUTION=True")
    if not job_name or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,100}", job_name):
        raise ValueError(
            "Execute requires an explicit unique FRESH_ATTEMPT_ID; rerunning it must not spend again"
        )
    if (project / "reports/launches" / (job_name + ".json")).exists() or (
        project / "runs/harbor" / job_name
    ).exists():
        raise FileExistsError(
            "Attempt already exists. Replay it; do not repeat inference"
        )
    selection = read(configuration_path(project, "configs/study.json"))["studies"].get(
        study_id
    )
    task_ids = (
        selection.get("task_ids", [])
        if selection
        else [study_id.split(":", 1)[1]]
        if study_id.startswith("public:")
        else []
    )
    if len(task_ids) == 1:
        return execute_public(
            project,
            task_id=task_ids[0],
            confirmed=confirmed,
            job_name=job_name,
            credentials=credentials,
        )
    prepared = preparation(project, study_id)
    readiness = preflight_execute(project, prepared, credentials)
    # Original measured path: Harbor -> host adapter -> upstream DefaultAgent.
    old_path = os.environ.get("PATH", "")
    docker = shutil.which("docker") or str(Path.home() / ".docker/bin/docker")
    try:
        os.environ["PATH"] = str(Path(docker).parent) + os.pathsep + old_path
        receipt = launch(
            project,
            mode="execute",
            confirmed=True,
            study_id=study_id,
            job_name=job_name,
            credentials=credentials,
        )
    finally:
        os.environ["PATH"] = old_path
    found = [r for r in records(project) if r["run_id"].startswith(job_name + "/")]
    if len(found) != 1:
        raise RuntimeError(
            "Launch evidence requires reconciliation; no automatic inference retry"
        )
    return {
        "mode": mode,
        "launched": True,
        "readiness": readiness,
        "launch": receipt,
        "record": found[0],
    }


def native_trace(record):
    path = Path(record["artifact_dir"]) / (
        "trace.jsonl" if record["source"] == "legacy" else "agent/requests.jsonl"
    )
    return path, events(path)


def first_exchange(record):
    path, trace = native_trace(record)
    request = response = None
    for event in trace:
        if event.get("event") == "model_request":
            if request is not None:
                break
            request = event
        elif event.get("event") == "model_response" and request is not None:
            response = event
            break
    return {
        "run_id": record["run_id"],
        "status": "saved native exchange"
        if request is not None
        else "not run: no request recorded",
        "trace_path": str(path),
        "request": request.get("body") if request else None,
        "response": response.get("response") if response else None,
        "response_status": "recorded" if response else "not recorded / interrupted",
        "trace_sha256": hashlib.sha256(path.read_bytes()).hexdigest()
        if path.is_file()
        else None,
    }


def details(title, value, *, opened=False):
    """Small expandable views; explicit links handle the large original artifacts."""
    text = (
        value
        if isinstance(value, str)
        else json.dumps(value, indent=2, ensure_ascii=False)
    )
    return f'<details {"open" if opened else ""}><summary>{html.escape(title)}</summary><pre style="white-space:pre-wrap;max-height:32em;overflow:auto">{html.escape(str(redact(text)))}</pre></details>'


def exchange_html(record):
    exchange = first_exchange(record)
    if exchange["request"] is None:
        return details(
            record["run_id"] + " — " + exchange["status"],
            {"calls": record["calls"], "startup_audit": record.get("startup_audit")},
        )
    response = exchange["response"] or {}
    raw = response.get("provider_response") or response
    visible_response = copy.deepcopy(
        {k: raw.get(k) for k in ["id", "provider", "usage", "error", "choices"]}
    )
    for choice in visible_response.get("choices") or []:
        message = choice.get("message", {})
        # Raw reasoning stays in the linked original trace; the tool action is enough here.
        for field in ["reasoning", "reasoning_details"]:
            message.pop(field, None)
    summary = {
        "original run ID": record["run_id"],
        "trace SHA256": exchange["trace_sha256"],
        "response status": exchange["response_status"],
        "model": record["model"],
        "original verdict": record["classification"],
    }
    inner = details(
        "Original first request — actual instructions, messages and bash schema",
        redact(exchange["request"]),
    ) + details(
        "Original first response — action and native accounting",
        redact(visible_response),
    )
    link = Path(exchange["trace_path"]).as_uri()
    return (
        f"<details><summary>{html.escape(record['run_id'])}</summary>"
        + details("Identity", summary, opened=True)
        + inner
        + f'<p><a href="{html.escape(link)}">Original private trace</a> (source-bearing; do not publish)</p></details>'
    )


def request_context(record):
    """Message counts/characters are exact; native token counts come from responses."""
    _, trace = native_trace(record)
    rows = []
    for event in trace:
        if event["event"] == "model_request":
            body = event["body"]
            messages = body.get("messages", [])
            rows.append(
                {
                    "request": event.get("number", len(rows) + 1),
                    "messages": len(messages),
                    "tool observations": sum(m.get("role") == "tool" for m in messages),
                    "message characters (not tokens)": sum(
                        len(str(m.get("content", ""))) for m in messages
                    ),
                    "input tokens": None,
                    "output tokens": None,
                    "reasoning subset": None,
                    "reported cost": None,
                }
            )
        elif event["event"] == "model_response" and rows:
            raw = event["response"] or {}
            raw = raw.get("provider_response") or raw
            usage = raw.get("usage") or {}
            rows[-1].update(
                {
                    "input tokens": usage.get("prompt_tokens"),
                    "output tokens": usage.get("completion_tokens"),
                    "reasoning subset": (
                        usage.get("completion_tokens_details") or {}
                    ).get("reasoning_tokens"),
                    "reported cost": usage.get("cost"),
                }
            )
    return pd.DataFrame(rows)


def actions(record, numbers=(1, 10, 20, 21)):
    _, trace = native_trace(record)
    request = None
    rows = []
    for event in trace:
        if event["event"] == "model_request":
            request = event.get("number")
        if event["event"] == "tool_result" and request in numbers:
            result = event.get("result") or {}
            rows.append(
                {
                    "request": request,
                    "command": str(
                        redact((event.get("tool_call") or {}).get("command", ""))
                    )[:600],
                    "exit": result.get("exit_code", result.get("returncode")),
                    "observation (tail)": str(redact(result.get("output", "")))[-600:],
                }
            )
    return pd.DataFrame(rows)


def candidate_evidence(record):
    folder = Path(record["artifact_dir"])
    receipt = read(folder / "agent/receipt.json", {})
    checks = read(folder / "verifier/checks.json", {})
    patch = folder / "verifier/submission.patch"
    return {
        "run_id": record["run_id"],
        "record": record,
        "receipt": receipt.get("receipt", receipt),
        "checks": checks.get("checks", {}),
        "patch": patch.read_text()
        if patch.is_file()
        else walkthrough(record).get("patch", "not bundled / not recorded")
        if record.get("source") == "legacy"
        else "not recorded: no captured patch available",
        "patch_path": str(patch),
        "candidate_path": str(folder / "artifacts/logs/artifacts/candidate.tar"),
    }


def checks_frame(record, split="development"):
    evidence = candidate_evidence(record)
    if not evidence["checks"]:
        return pd.DataFrame(
            [
                {
                    "check": "trusted checks",
                    "scope": split,
                    "expected exit": None,
                    "observed exit": None,
                    "status": "not run / not recorded",
                }
            ]
        )
    return pd.DataFrame(
        [
            {
                "check": name,
                "scope": "trusted acceptance (" + split + ")"
                if name == "hidden_acceptance"
                else "development / file policy",
                "expected exit": 1 if name == "regression_on_baseline" else 0,
                "observed exit": c.get("exit_code"),
                "status": "missing"
                if c.get("exit_code") is None
                else "pass"
                if c["exit_code"] == (1 if name == "regression_on_baseline" else 0)
                else "fail",
            }
            for name, c in evidence["checks"].items()
            if isinstance(c, dict)
        ]
    )


def accounting(project: Path, record):
    from experiments.openrouter_accounting import request_ledger

    rows = request_ledger(project, [record])
    rows = [r for r in rows if r["run_id"] == record["run_id"]]
    usage = normalize_usage(
        [{"response": {"usage": r.get("usage")}} for r in rows],
        expected_calls=record["calls"],
        calls_complete=record.get("calls_complete", True),
    )
    known = sum(
        (
            Decimal(
                str(
                    r["reported_cost_usd"]
                    if r["accounting_conflict"]
                    else r["effective_cost_usd"]
                )
            )
            for r in rows
            if (
                r["reported_cost_usd"]
                if r["accounting_conflict"]
                else r["effective_cost_usd"]
            )
            is not None
        ),
        Decimal(0),
    )
    complete = (
        record.get("calls_complete", True)
        and len(rows) == record["calls"]
        and all(r["effective_cost_usd"] is not None for r in rows)
    )
    frame = pd.DataFrame(
        [
            {
                k: r.get(k)
                for k in [
                    "request_number",
                    "generation_id",
                    "actual_provider",
                    "reported_cost_usd",
                    "generation_reported_cost_usd",
                    "effective_cost_usd",
                    "tool_format_error",
                    "accounting_conflict",
                ]
            }
            | {
                "input_tokens": (r.get("usage") or {}).get("prompt_tokens"),
                "output_tokens": (r.get("usage") or {}).get("completion_tokens"),
            }
            for r in rows
        ]
    )
    summary = {
        "requests including failures": len(rows),
        "known reported spend USD": float(known),
        "complete reported spend USD": float(known) if complete else None,
        "cost complete": complete,
        "accepted patch": record["patch_accepted"],
        "explicit submission": record["explicit_submission"],
        "cost per accepted patch USD": float(known)
        if complete and record["patch_accepted"]
        else None,
        "usage": usage,
    }
    return frame, summary


def public_evidence(project: Path):
    return Bundle(project / "notebooks/evidence")


def launch_command(project, prepared, job_name="YOUR_UNIQUE_FRESH_ATTEMPT_ID"):
    return shlex.join(
        [
            str(project / ".venv/bin/harbor"),
            "run",
            "-c",
            str(project / prepared["selection"]["job_config"]),
            "--env-file",
            "<host-only credentials file>",
            "--job-name",
            job_name,
        ]
    )


def request_preview(project: Path, prepared):
    """Render initial inputs offline; never construct a client or read old tool results.

    Uses the frozen mini-SWE Jinja templates, installed BASH_TOOL literal and
    the original WorkspaceModel notice expression. Native first requests remain
    the authority: the notebook checks this projection against the actual body.
    """
    import ast
    from types import SimpleNamespace

    from jinja2 import StrictUndefined, Template

    policy = prepared["policy"]
    templates = policy["mini_config"]["agent"]
    variables = {
        "task": prepared["instructions"],
        "system": "Linux",
        "release": "container",
        "version": "",
        "machine": "",
    }
    messages = [
        {
            "role": role,
            "content": Template(templates[name], undefined=StrictUndefined).render(
                **variables
            ),
        }
        for role, name in [("system", "system_template"), ("user", "instance_template")]
    ]
    source = project / "harbor_integration/host_workspace.py"
    tree = ast.parse(source.read_text())
    model_class = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "WorkspaceModel"
    )
    method = next(
        n
        for n in model_class.body
        if isinstance(n, ast.FunctionDef) and n.name == "query"
    )
    notice = next(
        n.value
        for n in method.body
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "notice" for t in n.targets)
    )
    budget = SimpleNamespace(
        calls=0, generated=0, limits=SimpleNamespace(**policy["budgets"])
    )
    messages.append(
        {
            "role": "user",
            "content": eval(
                compile(ast.Expression(notice), str(source), "eval"),
                {"__builtins__": {}},
                {"self": SimpleNamespace(budget=budget)},
            ),
        }
    )
    paths = list(
        (project / ".venv/lib").glob(
            "python*/site-packages/minisweagent/models/utils/actions_toolcall.py"
        )
    )
    if len(paths) != 1:
        raise FileNotFoundError(
            "Install the frozen Harbor environment to inspect its actual tool schema"
        )
    tool_tree = ast.parse(paths[0].read_text())
    tool = next(
        ast.literal_eval(n.value)
        for n in tool_tree.body
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "BASH_TOOL" for t in n.targets)
    )
    model = policy["model"]
    body = {
        "model": model["model"],
        "provider": model["provider"],
        "temperature": model.get("temperature", 0),
        "max_tokens": min(
            policy["budgets"]["response_tokens"], policy["budgets"]["generated_tokens"]
        ),
        "usage": {"include": True},
        "tools": [tool],
        "messages": messages,
    }
    if model.get("reasoning") is not None:
        body["reasoning"] = model["reasoning"]
    return body


def milestone_actions(record):
    """Select observed milestones by command content, never by fixed request ID.

    Heuristic labels describe commands, not proof that an edit or check succeeded.
    Each category may be absent; include its first event and the final tool result.
    """
    _, trace = native_trace(record)
    rows, request = [], None
    for event in trace:
        if event.get("event") == "model_request":
            request = event.get("number")
        if event.get("event") != "tool_result":
            continue
        command = str((event.get("tool_call") or {}).get("command", ""))
        labels = []
        if "/task/reproduce.py" in command:
            labels.append("reproduction")
        if re.search(
            r"write_text\(|write_bytes\(|apply_patch|sed\s+-i|patch\s|cat\s*<<|\btee\b|open\([^\n]*['\"](?:w|a)",
            command,
        ):
            labels.append("editing command")
        if re.search(r"\bpytest\b|\bruff\b|\bmypy\b", command):
            labels.append("verification command")
        if "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" in command:
            labels.append("submission command")
        result = event.get("result") or {}
        rows.append(
            {
                "request": request,
                "labels": labels,
                "command": str(redact(command))[:700],
                "exit": result.get("exit_code", result.get("returncode")),
                "observation (tail)": str(redact(result.get("output", "")))[-700:],
            }
        )
    seen, selected = set(), []
    for i, row in enumerate(rows):
        fresh = [label for label in row["labels"] if label not in seen]
        if i == len(rows) - 1:
            fresh.append("last observed action")
        if fresh:
            seen.update(row["labels"])
            selected.append(
                {
                    "milestone (heuristic)": ", ".join(fresh),
                    **{k: v for k, v in row.items() if k != "labels"},
                }
            )
    return pd.DataFrame(
        selected,
        columns=[
            "milestone (heuristic)",
            "request",
            "command",
            "exit",
            "observation (tail)",
        ],
    )


def context_change(record):
    """Compare an actual first request to the first later request carrying a tool result."""
    _, trace = native_trace(record)
    requests = [e for e in trace if e.get("event") == "model_request"]
    if not requests:
        return {"status": "not run: no model request recorded", "added_messages": []}
    first = requests[0]
    later = next(
        (
            e
            for e in requests[1:]
            if any(m.get("role") == "tool" for m in e["body"].get("messages", []))
        ),
        None,
    )
    if later is None:
        return {
            "status": "no later request containing tool feedback recorded",
            "first_request": first.get("number"),
            "added_messages": [],
        }
    before, after = first["body"].get("messages", []), later["body"].get("messages", [])
    # Budget notices change on every call: match unchanged messages by full content,
    # then show newly carried assistant actions, tool observations and budget notice.
    unchanged = [m for m in after if m in before]
    added = [m for m in after if m not in before]
    compact = []
    for m in added:
        compact.append(
            {
                "role": m.get("role"),
                "tool_call_id": m.get("tool_call_id"),
                "tool_calls": redact(m.get("tool_calls")),
                "content excerpt": str(redact(m.get("content", "")))[:1200],
            }
        )
    return {
        "status": "saved request comparison",
        "first_request": first.get("number"),
        "later_request": later.get("number"),
        "unchanged_instruction_messages": len(unchanged),
        "tools_unchanged": first["body"].get("tools") == later["body"].get("tools"),
        "added_messages": compact,
    }


def outcome_learning(record):
    if record.get("patch_accepted") is True:
        return "The captured patch was accepted under this attempt's recorded policy. This is one observation, not a reliability estimate. Explicit submission remains a separate metric."
    if record.get("patch_accepted") is False:
        return "The captured patch was not accepted. Read the failed or missing checks and termination reason; agent submission alone does not establish correctness."
    return "Acceptance is unknown or evaluation did not complete. Preserve the partial evidence and reported charges; do not infer a successful patch or a coding failure from missing checks."


def timing_eligibility(project, attempts):
    """Join exact saved audits; unknown/invalid clocks never enter timing medians."""
    import math

    rows = []
    for attempt in attempts:
        audit_path = (
            project
            / "reports/run-reviews"
            / (attempt["run_id"].replace("/", "--") + ".json")
        )
        audit_path = resolve_artifact_path(audit_path)
        audit = read(audit_path, {})
        timing = audit.get("timing_audit") or {}
        valid = timing.get("controlled_completion_time_valid")
        interrupted = attempt.get("infrastructure_interrupted") is True
        seconds = attempt.get("elapsed_seconds")
        numeric = (
            isinstance(seconds, (int, float))
            and not isinstance(seconds, bool)
            and math.isfinite(seconds)
            and seconds >= 0
        )
        if interrupted:
            status = "interrupted: excluded from medians"
        elif valid is not True:
            status = (
                "clock invalid" if valid is False else "timing audit missing: excluded"
            )
        elif not numeric:
            status = "elapsed time missing or invalid: excluded"
        else:
            status = "eligible non-interrupted"
        rows.append(
            {
                "run_id": attempt["run_id"],
                "task_id": attempt["task_id"],
                "model": attempt["model"],
                "timing_status": status,
                "infrastructure_interrupted": interrupted,
                "clock_valid": valid,
                "raw_elapsed_seconds": seconds,
                "eligible_agent_seconds": seconds
                if status == "eligible non-interrupted"
                else None,
                "monotonic_seconds": timing.get("monotonic_seconds"),
                "civil_seconds": timing.get("civil_seconds"),
                "audit_path": str(audit_path.relative_to(project))
                if audit_path.is_file()
                else None,
            }
        )
    return pd.DataFrame(rows)


# Portable saved-evidence access. Fresh attempts retain the original execution path.
_original_records = records
_original_first_exchange = first_exchange
_original_candidate_evidence = candidate_evidence
_original_accounting = accounting
_original_request_preview = request_preview
_original_preflight_execute = preflight_execute


def _historical(project):
    from experiments.portable_evidence import tutorial

    return tutorial(project)


def records(project):
    saved = _historical(project)["records"]
    if (project / "reports/launches").exists():
        saved = saved + _original_records(project)
    from public_preparation.launch import records as public_records

    return saved + public_records(project)


def _is_saved(record):
    return record["run_id"] == DEMO_RUN_ID


def first_exchange(record):
    if _is_saved(record):
        return _historical(Path(__file__).resolve().parents[1])["first_exchange"]
    return _original_first_exchange(record)


def candidate_evidence(record):
    if _is_saved(record):
        return _historical(Path(__file__).resolve().parents[1])["candidate"]
    return _original_candidate_evidence(record)


def accounting(project, record):
    if _is_saved(record):
        saved = _historical(project)
        return pd.DataFrame(saved["ledger"]), saved["accounting"]
    return _original_accounting(project, record)


def request_preview(project, prepared):
    # Replay uses a labelled historical summary without loading inference templates.
    # Fresh requests use the locally expanded, validated configuration.
    if prepared["study_id"] == DEMO_STUDY:
        return _historical(project)["request_preview"]
    return _original_request_preview(project, prepared)


def preflight_execute(project, prepared, credentials=None):
    from experiments.portable_prepare import execution_readiness

    execution_readiness(project, prepared["selection"]["task_ids"])
    return _original_preflight_execute(project, prepared, credentials)


def execute_public(
    project,
    *,
    task_id,
    model="minimax",
    confirmed=False,
    job_name=None,
    credentials=None,
):
    from public_preparation.launch import launch as public_launch

    return public_launch(
        project,
        task_id,
        model=model,
        confirmed=confirmed,
        job_name=job_name,
        credentials=credentials,
    )


def public_preparation(project, task_id):
    state = project / ".local/public-v1" / task_id
    config = read(state / "model.json")
    frozen = read(state / "prepared.json")
    task = Path(config["tasks"][0]["path"])
    return {
        "study_id": "public:" + task_id,
        "selection": {"task_ids": [task_id], "job_config": str(state / "model.json")},
        "job": config,
        "policy": config["agents"][0]["kwargs"]["policy"],
        "frozen": frozen,
        "task": task,
        "instructions": (task / "instruction.md").read_text(),
        "environment": tomllib.loads((task / "task.toml").read_text()),
        "controls": read(state / "validated.json", {}),
    }
