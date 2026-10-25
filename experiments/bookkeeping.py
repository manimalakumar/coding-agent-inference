"""Reconcile native Harbor outputs and raw provider usage, without inference."""

import hashlib
import json
import math
import re
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from experiments.records import events, read
from experiments.report import write_reports
from experiments.storage import atomic_write


def audit_trial(folder, record):
    required = [
        "config.json",
        "result.json",
        "agent/requests.jsonl",
        "agent/receipt.json",
        "agent/source-snapshot.json",
        "agent/isolation.json",
        "agent/mini-swe-agent.trajectory.json",
        "verifier/checks.json",
        "verifier/reward.json",
        "verifier/submission.patch",
        "artifacts/logs/artifacts/candidate.tar",
    ]
    if record.get("completion_policy_id") == "workspace-deliverable-v1":
        required += ["agent/quiescence.json"]
    missing = [p for p in required if not (folder / p).is_file()]
    trace = events(folder / "agent/requests.jsonl")
    requests = [e for e in trace if e["event"] == "model_request"]
    responses = [e for e in trace if e["event"] == "model_response"]
    native = read(folder / "agent/mini-swe-agent.trajectory.json", {})
    tool_results = {
        e["tool_call"].get("tool_call_id"): {
            "source": "raw_trace",
            "result": e["result"],
        }
        for e in trace
        if e["event"] == "tool_result" and e["tool_call"].get("tool_call_id")
    }
    supplemented = 0
    valid_actions = []
    for message in native.get("messages", []):
        valid_actions.extend(message.get("extra", {}).get("actions", []))
        call_id = message.get("tool_call_id")
        if message.get("role") == "tool" and call_id and call_id not in tool_results:
            tool_results[call_id] = {
                "source": "native_trajectory",
                "result": message.get("extra", {}),
            }
            supplemented += 1
    missing_tools = [
        a.get("tool_call_id")
        for a in valid_actions
        if a.get("tool_call_id") not in tool_results
    ]
    receipt = read(folder / "agent/receipt.json", {}).get("receipt", {})
    problems = []
    if not (len(requests) == len(responses) == receipt.get("model_requests")):
        problems.append("request_response_receipt_count_mismatch")
    if not record.get("token_usage_complete"):
        problems.append("incomplete_token_usage")
    if record.get("token_usage_complete") and receipt.get(
        "output_tokens"
    ) != record.get("output_tokens"):
        problems.append("raw_output_tokens_receipt_mismatch")
    if record.get("cost_usd") is None:
        problems.append("incomplete_provider_cost")
    signed_cost = receipt.get("reported_cost")
    if (
        isinstance(signed_cost, (int, float))
        and record.get("cost_usd") is not None
        and not math.isclose(signed_cost, record["cost_usd"], abs_tol=1e-8)
    ):
        problems.append("raw_cost_receipt_mismatch")
    if missing_tools:
        problems.append("missing_tool_results")
    candidate = folder / "artifacts/logs/artifacts/candidate.tar"
    checks = read(folder / "verifier/checks.json", {})
    native_reward = (read(folder / "result.json", {}).get("verifier_result") or {}).get(
        "rewards"
    )
    saved_reward = read(folder / "verifier/reward.json", {})
    if native_reward is not None and native_reward != saved_reward:
        problems.append("native_result_verifier_reward_mismatch")
    if (
        candidate.is_file()
        and checks.get("candidate_sha256")
        and hashlib.sha256(candidate.read_bytes()).hexdigest()
        != checks["candidate_sha256"]
    ):
        problems.append("candidate_checksum_mismatch")
    native_cost = native.get("info", {}).get("model_stats", {}).get("instance_cost")
    raw_cost = record.get("cost_usd")
    return {
        "record": record,
        "status": "complete" if not missing and not problems else "incomplete",
        "missing_artifacts": missing,
        "problems": problems,
        "raw_requests": len(requests),
        "raw_responses": len(responses),
        "native_tool_results_supplemented": supplemented,
        "missing_tool_results": missing_tools,
        "raw_usage_authoritative": True,
        "native_trajectory_reported_cost": native_cost,
        "raw_minus_native_cost": round(raw_cost - native_cost, 12)
        if isinstance(native_cost, (float, int)) and raw_cost is not None
        else None,
        "artifact_sha256": {
            p: hashlib.sha256((folder / p).read_bytes()).hexdigest()
            for p in required
            if (folder / p).is_file()
        },
    }


def finalize_job(project, job_name, data=None):
    """Immutable per-finalization audit plus a latest index; never change a verdict."""
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,100}", job_name):
        raise ValueError("Invalid job name")
    project = Path(project).resolve()
    data = data if data is not None else write_reports(project)
    records = [
        r
        for r in data["records"]
        if r["source"] in ("harbor", "harbor_launch")
        and r["run_id"].split("/", 1)[0] == job_name
    ]
    trials = [audit_trial(Path(r["artifact_dir"]), r) for r in records]
    audit = {
        "schema_version": 1,
        "job_name": job_name,
        "checked_at": datetime.now(UTC).isoformat(),
        "status": "complete"
        if trials and all(t["status"] == "complete" for t in trials)
        else "incomplete",
        "problems": []
        if trials
        else ["no_trial_artifacts; inference and acceptance cannot be established"],
        "trials": trials,
        "model_requests_made_by_bookkeeping": 0,
        "native_artifacts_modified": False,
        "reports": [
            "reports/EXPERIMENT-RESULTS.md",
            "reports/experiment-results.jsonl",
            "reports/notebook-run-records.json",
        ],
    }
    destination = project / "reports/bookkeeping/jobs" / job_name
    destination.mkdir(parents=True, exist_ok=True)
    version = destination / (
        datetime.now(UTC).strftime("%Y%m%dT%H%M%S") + "-" + uuid4().hex[:8] + ".json"
    )
    with version.open("x") as output:
        json.dump(audit, output, indent=2)
    index = {"status": audit["status"], "audit": str(version.relative_to(project))}
    atomic_write(destination / "latest.json", json.dumps(index, indent=2) + "\n")
    return index


def reconcile(project, pending_only=False):
    """Recover direct Harbor launches or interrupted finalization, using saved data."""
    project = Path(project).resolve()
    jobs = {p.name for p in (project / "runs/harbor").glob("*") if p.is_dir()}
    jobs |= {p.stem for p in (project / "reports/launches").glob("*.json")}
    if pending_only:
        jobs = {
            j
            for j in jobs
            if read(project / "reports/bookkeeping/jobs" / j / "latest.json", {}).get(
                "status"
            )
            != "complete"
        }
    data = write_reports(project)
    return {j: finalize_job(project, j, data) for j in sorted(jobs)}
