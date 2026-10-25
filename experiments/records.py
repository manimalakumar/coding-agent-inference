"""Normalize saved artifacts without executing agents or rewriting original results."""

import hashlib
import json
from collections import defaultdict
from datetime import datetime
from decimal import Decimal
from pathlib import Path

from experiments.artifact_paths import resolve_artifact_path
from experiments.configuration_paths import resolve_configuration_path


def read(path, default=None):
    path = resolve_artifact_path(resolve_configuration_path(path))
    return json.loads(path.read_text()) if path.exists() else default


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def events(path):
    path = Path(path)
    if not path.exists():
        return []
    content = path.read_text()
    rows = content.splitlines()
    output = []
    for index, line in enumerate(rows):
        if not line.strip():
            continue
        try:
            output.append(json.loads(line))
        except json.JSONDecodeError:
            # A different live trial may currently be appending its final JSON line.
            if index != len(rows) - 1 or content.endswith("\n"):
                raise
    return output


def duration(section):
    if section.get("started_at") and section.get("finished_at"):
        return (
            datetime.fromisoformat(section["finished_at"])
            - datetime.fromisoformat(section["started_at"])
        ).total_seconds()
    return None


def measured_boolean(value):
    return (
        bool(value)
        if isinstance(value, (bool, int, float)) and value in (0, 1)
        else None
    )


def patch_economics(entries, known, complete):
    """Patch acceptance and explicit completion never share a denominator."""
    outcomes = [
        measured_boolean(r.get("patch_accepted", r.get("patch_acceptance")))
        for r in entries
    ]
    accepted = sum(v is True for v in outcomes)
    acceptance_complete = all(v is not None for v in outcomes)
    submissions = sum(
        bool(r.get("explicit_submission", r.get("submitted", False))) for r in entries
    )
    status = (
        "incomplete_patch_acceptance_records"
        if not acceptance_complete
        else "undefined_no_accepted_patches"
        if not accepted
        else "incomplete_cost_records"
        if not complete
        else "available"
    )
    return {
        "known_accepted_patches": accepted,
        "accepted_patches": accepted if acceptance_complete else None,
        "patch_acceptance_records_complete": acceptance_complete,
        "cost_per_accepted_patch_usd": float(known / accepted)
        if status == "available"
        else None,
        "patch_economics_status": status,
        "explicit_submissions": submissions,
        "explicit_workflow_completion_rate": submissions / len(entries),
    }


def legacy_record(folder):
    # Existing read-only importer preserves trusted review classifications and failed requests.
    from agent.results import rows_for_run

    rows = rows_for_run(folder)
    if not rows:
        return None
    meta, result = read(folder / "metadata.json"), read(folder / "result.json", {})
    execution = next((r for r in rows if r["phase"] != "request_failure"), rows[0])
    observed_calls = sum(r["requests"] for r in rows)
    calls = max(observed_calls, result.get("model_requests", 0))
    missing_calls = calls - observed_calls
    missing = sum(r["requests_without_cost_record"] for r in rows) + missing_calls
    known = sum((Decimal(str(r["known_reported_cost_usd"])) for r in rows), Decimal(0))
    submission = result.get("agent_submission")
    submitted = (
        submission.get("submitted") if isinstance(submission, dict) else submission
    )
    if submitted is None:
        submitted = (result.get("submission") or {}).get("exit_status") == "Submitted"
    acceptance = result.get("patch_acceptance")
    return {
        "schema_version": 1,
        "run_id": folder.name,
        "task_id": meta["task_id"],
        "source": "legacy",
        "kind": "model_attempt",
        "model": meta["model"]["model"],
        "configuration_id": meta.get("setting_checksum") or "unrecorded:" + folder.name,
        "inference_id": rows[0]["model_configuration_checksum"],
        "provider": meta["model"].get("provider"),
        "task_python": read(folder / "setting.json", {})
        .get("sandbox", {})
        .get("python"),
        "budgets": read(folder / "setting.json", {}).get("sandbox", {}).get("budgets"),
        "harness": result.get("framework")
        or meta.get("workflow_label")
        or "historical custom controller",
        "commit": meta.get("commit"),
        "baseline_checksum": meta.get("baseline_checksum"),
        "success": result.get("termination_reason") == "checks_passed_pending_review",
        "task_success": result.get("termination_reason")
        == "checks_passed_pending_review",
        "completion_policy_id": "historical_original_rules",
        "explicit_submission": bool(submitted),
        "patch_accepted": measured_boolean(acceptance),
        "submitted": bool(submitted),
        "patch_acceptance": acceptance if acceptance is not None else "not_recorded",
        "classification": execution["classification"],
        "what_it_shows": execution.get("what_it_shows"),
        "termination_reason": result.get("termination_reason"),
        "calls": calls,
        "input_tokens": sum(r["input_tokens_reported"] for r in rows),
        "output_tokens": sum(
            r["generated_tokens_reported_including_reasoning"] for r in rows
        ),
        "reasoning_tokens": sum(r["reasoning_tokens_reported"] for r in rows),
        "token_usage_complete": not missing_calls
        and all(r["token_usage_complete"] for r in rows),
        "known_cost_usd": float(known),
        "cost_usd": float(known) if not missing else None,
        "requests_without_cost": missing,
        "elapsed_seconds": result.get("elapsed_seconds"),
        "setup_seconds": result.get("container_setup_seconds"),
        "request_failures": [r for r in rows if r["phase"] == "request_failure"],
        "artifact_dir": str(folder),
        "result_path": str(folder / "result.json"),
    }


def harbor_record(folder, control=False):
    from agent.results import request_records, usage

    result, config = read(folder / "result.json", {}), read(folder / "config.json", {})
    if not config.get("agent"):
        return None
    agent = config["agent"]
    kwargs = agent.get("kwargs", {})
    policy = kwargs.get("policy", {})
    info = result.get("agent_result") or {}
    receipt = info.get("metadata") or read(folder / "agent/receipt.json", {}).get(
        "receipt", {}
    )
    reward = (result.get("verifier_result") or {}).get("rewards") or read(
        folder / "verifier/reward.json", {}
    )
    control = (
        control or bool(kwargs.get("fixture")) or bool(receipt.get("local_fixture"))
    )
    records = request_records(events(folder / "agent/requests.jsonl"))
    accounting = usage(records)
    calls = max(receipt.get("model_requests", 0), len(records))
    cost = accounting["total_cost_usd"] if records else info.get("cost_usd")
    if control:
        cost = 0
    elif calls > len(records):
        cost = (
            None  # Incomplete traces cannot establish a complete inference numerator.
        )
    trusted_review = read(folder / "review.json", {})
    audit = None
    # Later audit annotations never rewrite native results, traces or old reviews.
    if len(folder.parents) >= 4:
        audit = (
            folder.parents[3]
            / "reports/run-reviews"
            / (folder.parent.name + "--" + folder.name + ".json")
        )
        audit = resolve_artifact_path(audit)
        trusted_review |= read(audit, {})
    exception = result.get("exception_info")
    status = receipt.get("exit_status", "incomplete")
    workspace_policy = (
        policy.get("completion_policy", {}).get("policy_id")
        == "workspace-deliverable-v1"
    )
    success = (
        (reward.get("task_success") == 1 and reward.get("patch_accepted") == 1)
        if workspace_policy
        else (
            reward.get("reward") == 1
            and reward.get("patch_acceptance") == 1
            and reward.get("agent_submission") == 1
            and not exception
        )
    )
    classification = (
        "checks_passed_pending_review"
        if success
        else (
            "infrastructure_failure"
            if exception
            else "budget_exhausted"
            if "budget" in status.lower() or "time" in status.lower()
            else "checks_failed"
            if reward
            else "incomplete"
        )
    )
    # Native task digest and complete policy participate; job names and timestamps do not.
    configuration_id = "harbor:" + fingerprint(
        {
            "policy": policy,
            "task_checksum": result.get("task_checksum"),
            "agent": {
                **agent,
                "kwargs": {k: v for k, v in kwargs.items() if k != "fixture"},
            },
            "environment": config.get("environment"),
            "verifier": config.get("verifier"),
        }
    )
    return {
        "schema_version": 1,
        "run_id": folder.parent.name + "/" + folder.name,
        "task_id": result.get("task_name") or Path(config["task"]["path"]).name,
        "source": "harbor",
        "kind": "local_control" if control else "model_attempt",
        "model": policy.get("model", {}).get("model") or agent.get("model_name"),
        "configuration_id": configuration_id,
        "inference_id": fingerprint(policy.get("model", {})),
        "provider": policy.get("model", {}).get("provider"),
        "budgets": policy.get("budgets"),
        "harness": "Harbor + mini-SWE-agent",
        "task_checksum": result.get("task_checksum"),
        "commit": policy.get("source_provenance", {}).get("starting_commit"),
        "baseline_checksum": policy.get("source_provenance", {}).get(
            "baseline_checksum"
        ),
        "success": success,
        "task_success": success,
        "completion_policy_id": policy.get("completion_policy", {}).get(
            "policy_id", "historical_original_rules"
        ),
        "explicit_submission": bool(
            reward.get(
                "explicit_submission",
                reward.get(
                    "agent_submission",
                    receipt.get("explicit_submission", receipt.get("agent_submission")),
                ),
            )
        ),
        "patch_accepted": measured_boolean(
            reward.get("patch_accepted", reward.get("patch_acceptance"))
        ),
        "submitted": bool(
            reward.get("agent_submission", receipt.get("agent_submission"))
        ),
        "patch_acceptance": reward.get("patch_acceptance", "not_evaluated"),
        "classification": classification,
        "failure_category": trusted_review.get("failure_category"),
        "audit_path": str(audit) if audit and audit.is_file() else None,
        "contributing_factors": trusted_review.get("contributing_factors", []),
        "tool_format_repair_rounds": receipt.get("repair_rounds"),
        "code_repair_rounds": trusted_review.get("code_repair_rounds"),
        "code_repair_measurement": trusted_review.get(
            "code_repair_measurement",
            "not instrumented; never infer from tool-format retries",
        ),
        "what_it_shows": trusted_review.get("reason"),
        "termination_reason": receipt.get("termination_reason", status)
        if workspace_policy
        else status,
        "termination_detail": receipt.get("termination_detail", status),
        "execution_error": receipt.get("execution_error"),
        "calls": calls,
        "input_tokens": accounting["input_tokens_reported"]
        if records
        else info.get("n_input_tokens"),
        "output_tokens": accounting["generated_tokens_reported_including_reasoning"]
        if records
        else info.get("n_output_tokens"),
        "reasoning_tokens": accounting["reasoning_tokens_reported"]
        if records
        else None,
        "token_usage_complete": accounting["token_usage_complete"]
        and len(records) == calls
        if records
        else control,
        "known_cost_usd": accounting["known_reported_cost_usd"]
        if records
        else (info.get("cost_usd") or 0),
        "cost_usd": cost,
        "requests_without_cost": accounting["requests_without_cost_record"]
        + max(0, calls - len(records)),
        "elapsed_seconds": duration(result.get("agent_execution") or {}),
        "workflow_seconds": duration(result),
        "setup_seconds": duration(result.get("environment_setup") or {}),
        "request_failures": [
            r for r in records if (r.get("response") or {}).get("error")
        ],
        "artifact_dir": str(folder),
        "result_path": str(folder / "result.json"),
    }


def load_runs(project, include_controls=True):
    """One record per attempt; diagnostics and fixture controls never become model attempts."""
    project = Path(project).resolve()
    records = []
    for folder in sorted((project / "runs").glob("*")):
        if (folder / "metadata.json").exists():
            record = legacy_record(folder)
            if record:
                records.append(record)
    for config in sorted((project / "runs/harbor").glob("*/*/config.json")):
        record = harbor_record(config.parent)
        if record:
            launch = read(
                project / "reports/launches" / (config.parent.parent.name + ".json"), {}
            )
            record["declared_frozen_setting_id"] = launch.get("setting_id")
            for setting_path in (project / "configs").glob("*.frozen-setting.json"):
                setting = read(setting_path, {})
                if setting.get("setting_id") == record["declared_frozen_setting_id"]:
                    record["task_python"] = setting.get("task_python")
                    record["task_runtime_versions"] = setting.get(
                        "task_runtime_versions"
                    )
                    break
            records.append(record)
    # A launch that died before Harbor wrote trial config must not disappear from
    # the numerator. Zero here means observed lower bound, never measured usage.
    native_jobs = {
        r["run_id"].split("/", 1)[0] for r in records if r["source"] == "harbor"
    }
    for path in sorted((project / "reports/launches").glob("*.json")):
        if path.stem in native_jobs:
            continue
        launch = read(path, {})
        command = launch.get("command", [])
        job: dict = {}
        if "-c" in command:
            config_path = Path(command[command.index("-c") + 1]).resolve()
            if config_path.is_relative_to(project):
                job = read(config_path, {})
        agent: dict = next(iter(job.get("agents", [])), {})
        policy = agent.get("kwargs", {}).get("policy", {})
        model = (
            policy.get("model", {}).get("model")
            or agent.get("model_name")
            or "unrecorded_model"
        )
        setting = launch.get("setting_id")
        template: dict = next(
            (
                r
                for r in records
                if setting
                and r.get("declared_frozen_setting_id") == setting
                and r["model"] == model
            ),
            {},
        )
        task: dict = next(iter(job.get("tasks", [])), {})
        records.append(
            {
                "schema_version": 1,
                "run_id": path.stem + "/unrecorded-trial",
                "source": "harbor_launch",
                "kind": "model_attempt",
                "task_id": Path(task.get("path", "unrecorded_task")).name,
                "model": model,
                "configuration_id": template.get(
                    "configuration_id", "unrecorded_launch:" + (setting or path.stem)
                ),
                "declared_frozen_setting_id": setting,
                "inference_id": fingerprint(policy.get("model", {})),
                "provider": policy.get("model", {}).get("provider"),
                "budgets": policy.get("budgets"),
                "harness": "Harbor + mini-SWE-agent",
                "commit": policy.get("source_provenance", {}).get("starting_commit"),
                "baseline_checksum": policy.get("source_provenance", {}).get(
                    "baseline_checksum"
                ),
                "success": False,
                "task_success": False,
                "submitted": False,
                "explicit_submission": False,
                "patch_acceptance": "not_evaluated",
                "patch_accepted": None,
                "completion_policy_id": policy.get("completion_policy", {}).get(
                    "policy_id", "historical_original_rules"
                ),
                "classification": "infrastructure_failure",
                "failure_category": "missing_run_artifacts",
                "termination_reason": "error",
                "termination_detail": launch.get("status", "unfinalized_launch"),
                "what_it_shows": "Launch has no readable native trial. Calls and known cost are lower bounds; total usage/cost and acceptance remain unknown. Reconcile artifacts; never repeat inference automatically.",
                "calls": 0,
                "calls_complete": False,
                "input_tokens": None,
                "output_tokens": None,
                "reasoning_tokens": None,
                "token_usage_complete": False,
                "cost_usd": None,
                "known_cost_usd": 0,
                "requests_without_cost": 1,
                "elapsed_seconds": None,
                "workflow_seconds": None,
                "setup_seconds": None,
                "request_failures": [],
                "artifact_dir": str(project / "runs/harbor" / path.stem),
                "result_path": str(path),
            }
        )
    if include_controls:
        for summary_path in sorted(
            (project / "reports").glob("*-validation-summary.json")
        ):
            summary = read(summary_path, {})
            for control in summary.get("controls", {}).values():
                folder = project / control["trial"]
                record = harbor_record(folder, control=True)
                if record and record["run_id"] not in {r["run_id"] for r in records}:
                    records.append(record)
    return records


def aggregate(records, across_tasks=False):
    """Configuration-safe repeated-attempt summaries; excludes local fixtures by design."""
    groups = defaultdict(list)
    keys = ["configuration_id", "model", "inference_id"]
    if not across_tasks:
        keys.append("task_id")
    for record in records:
        if record["kind"] == "model_attempt":
            groups[tuple(record[k] for k in keys)].append(record)
    output = []
    for group, entries in sorted(groups.items()):
        accepted = sum(bool(r["success"]) for r in entries)
        known = sum((Decimal(str(r["known_cost_usd"])) for r in entries), Decimal(0))
        complete = all(r["cost_usd"] is not None for r in entries)
        elapsed = [
            r["elapsed_seconds"] for r in entries if r["elapsed_seconds"] is not None
        ]
        output.append(
            dict(zip(keys, group))
            | {
                "attempts": len(entries),
                "tasks": len({r["task_id"] for r in entries}),
                "accepted_completions": accepted,
                "acceptance_rate": accepted / len(entries),
                "calls": sum(r["calls"] for r in entries),
                "input_tokens": sum(r["input_tokens"] or 0 for r in entries),
                "output_tokens": sum(r["output_tokens"] or 0 for r in entries),
                "token_usage_complete": all(r["token_usage_complete"] for r in entries),
                "known_spend_usd": float(known),
                "total_spend_usd": float(known) if complete else None,
                "cost_per_accepted_completion_usd": float(known / accepted)
                if accepted and complete
                else None,
                "economics_status": "undefined_no_accepted_completions"
                if not accepted
                else "available"
                if complete
                else "incomplete_cost_records",
                "mean_agent_seconds": sum(elapsed) / len(elapsed)
                if len(elapsed) == len(entries)
                else None,
                "failure_categories": sorted(
                    {
                        r.get("failure_category") or r["classification"]
                        for r in entries
                        if not r["success"]
                    }
                ),
                **patch_economics(entries, known, complete),
            }
        )
    return output


def walkthrough(record):
    """Read a selected run, keeping trusted diagnostics separate from its original verdict."""
    folder = Path(record["artifact_dir"])
    trace = events(
        folder
        / ("agent/requests.jsonl" if record["source"] == "harbor" else "trace.jsonl")
    )
    actions = [e for e in trace if e["event"] == "tool_result"]
    patches = [
        folder / p
        for p in [
            "submission.patch",
            "rejected-candidate.patch",
            "verifier/submission.patch",
        ]
    ]
    patch = next((p for p in patches if p.exists()), None)
    return {
        "record": record,
        "review": read(folder / "review.json", {}),
        "audit": read(Path(record["audit_path"]), {})
        if record.get("audit_path")
        else {},
        "reproduction": read(folder / "reproduction-evidence.json")
        or next(
            (
                a
                for a in actions
                if "reproduce.py" in json.dumps(a.get("tool_call", {}))
            ),
            None,
        ),
        "actions": actions,
        "patch": patch.read_text()
        if patch
        else "No saved patch. See tool actions for any unsubmitted edits.",
        "patch_path": str(patch) if patch else None,
        "evaluation": read(folder / "verifier/checks.json")
        if record["source"] == "harbor"
        else read(folder / "result.json", {}).get("evaluation"),
        "diagnostics": [
            read(p) | {"artifact_path": str(p)}
            for p in sorted((folder / "diagnostics").glob("*/result.json"))
        ],
    }


def exploratory_economics(records):
    """Lifetime spend accounting across policies, explicitly excluded from comparisons."""
    groups = defaultdict(list)
    for row in records:
        if row["kind"] == "model_attempt":
            groups[row["model"]].append(row)
    result = []
    for model, rows in sorted(groups.items()):
        known = sum((Decimal(str(r["known_cost_usd"])) for r in rows), Decimal(0))
        complete = all(r["cost_usd"] is not None for r in rows)
        accepted = sum(bool(r["success"]) for r in rows)
        result.append(
            {
                "scope": "exploratory_lifetime_not_a_controlled_comparison",
                "model": model,
                "attempts": len(rows),
                "configuration_ids": sorted({r["configuration_id"] for r in rows}),
                "calls": sum(r["calls"] for r in rows),
                "accepted_completions": accepted,
                "known_spend_usd": float(known),
                "total_spend_usd": float(known) if complete else None,
                "cost_per_accepted_completion_usd": float(known / accepted)
                if accepted and complete
                else None,
                "economics_status": "undefined_no_accepted_completions"
                if not accepted
                else "available"
                if complete
                else "incomplete_cost_records",
                **patch_economics(rows, known, complete),
            }
        )
    return result
