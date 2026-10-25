"""New trusted export command. Reads private evidence; never invokes inference."""

import argparse
import json
import re
import shutil
from pathlib import Path
from typing import Any

from experiments.configuration_paths import configuration_path
from experiments.openrouter_accounting import request_ledger
from experiments.records import events, read
from experiments.relocated_records import launch_index, load_runs
from public_replay.analysis import check_status
from public_replay.bundle import canonical, digest
from public_replay.sanitize import redact
from public_replay.usage import numeric

ATTEMPT_FIELDS = "run_id task_id source kind model configuration_id inference_id provider budgets harness commit baseline_checksum success task_success completion_policy_id explicit_submission patch_accepted classification failure_category termination_reason termination_detail calls token_usage_complete cost_usd known_cost_usd elapsed_seconds workflow_seconds setup_seconds".split()
CONTEXT_FIELDS = "request generation_id actual_provider native_input_tokens native_cached_input_tokens native_output_tokens native_reasoning_output_tokens reported_cost_usd cumulative_reported_cost_usd cumulative_known_reported_cost_usd cumulative_cost_complete finish_reason response_received message_count output_ceiling_tokens estimated_tokens_by_category estimated_visible_payload_tokens estimate_minus_native_tokens native_input_delta_from_previous_request".split()
MODEL_BODY_FIELDS = "model temperature provider reasoning max_tokens".split()
CODE_FILES = [
    "__init__.py",
    "__main__.py",
    "usage.py",
    "sanitize.py",
    "bundle.py",
    "analysis.py",
    "demo.py",
    "validate.py",
    "prepare.py",
    "_historical_tools.py",
]
PUBLIC_DOCS = [
    ".gitignore",
    "README-public.md",
    "CONTRIBUTING.md",
    "SECURITY.md",
    "CITATION.cff",
    "LICENSE-DECISION.md",
    "THIRD_PARTY_NOTICES.md",
]


def put(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical(value))


def projection(project: Path):
    index = launch_index(project)
    records = load_runs(project, index)
    assert len({r["run_id"] for r in records}) == len(records)
    frozen = read(
        configuration_path(
            project, "configs/held-out-remainder-round1-v1.frozen-setting.json"
        )
    )
    six = read(project / "reports/held-out-six-task-repeats/results.json")
    two = read(project / "reports/held-out-two-tasks/results.json")
    original = {r["run_id"]: r for r in six["attempts"] + two["attempts"]}
    six_ids = [r["run_id"] for r in six["attempts"]]
    two_ids = [r["run_id"] for r in two["attempts"]]
    if len(six_ids) != 36 or len(two_ids) != 4 or set(six_ids) & set(two_ids):
        raise ValueError(
            "Historical held-out scopes are not the declared disjoint 36 and 4 launches"
        )
    ledger = request_ledger(project, records)
    requests = []
    for row in ledger:
        usage = row["usage"] or {}
        selected = {}
        for name in ["prompt_tokens", "completion_tokens", "cost"]:
            value = usage.get(name)
            selected[name] = value if numeric(value, tokens=name != "cost") else None
        for parent, fields in [
            ("prompt_tokens_details", ["cached_tokens", "cache_write_tokens"]),
            ("completion_tokens_details", ["reasoning_tokens"]),
        ]:
            detail = usage.get(parent) or {}
            selected[parent] = {
                k: detail.get(k) if numeric(detail.get(k), tokens=True) else None
                for k in fields
            }
        raw = row.get("raw_response") or {}
        raw = raw.get("provider_response", raw)
        choices = raw.get("choices") or []
        request = {
            key: row.get(key)
            for key in "run_id request_number generation_id actual_provider reported_cost_usd effective_cost_usd generation_reported_cost_usd accounting_conflict cost_source tool_format_error".split()
        }
        request.update(
            accounting_scope=row["scope"],
            reported_usage=selected,
            finish_reason=choices[0].get("finish_reason") if choices else None,
            error_type=(
                str(row["error"]).split(":", 1)[0]
                if re.fullmatch(
                    r"[A-Za-z_][A-Za-z_0-9]*", str(row["error"]).split(":", 1)[0]
                )
                else "provider_or_transport_error"
            )
            if row.get("error")
            else None,
            original_response_canonical_sha256=digest(
                canonical(row.get("raw_response"))
            ),
            generation_reconciliation={
                "bundled": False,
                "http_status": (row.get("generation_metadata") or {}).get(
                    "http_status"
                ),
                "provenance": "separately saved Generation API receipt"
                if row.get("generation_metadata")
                else "completion response only",
            },
        )
        requests.append(request)
    output = []
    for record in records:
        richer = {**record, **original.get(record["run_id"], {})}
        item = {key: richer.get(key) for key in ATTEMPT_FIELDS}
        item["native_cost_usd"] = item.pop("cost_usd")
        item["native_known_cost_usd"] = item.pop("known_cost_usd")
        item["calls_complete"] = record.get("calls_complete", True)
        item["split"] = "held_out" if record["run_id"] in original else "development"
        item["scope"] = (
            "held_out_six"
            if record["run_id"] in six_ids
            else "held_out_two"
            if record["run_id"] in two_ids
            else "development"
        )
        attempt_match = re.search(r"attempt-(\d+)", record["run_id"])
        item["attempt_number"] = richer.get("attempt_number") or (
            int(attempt_match.group(1)) if attempt_match else 1
        )
        item["frozen_setting_id"] = richer.get(
            "frozen_setting_id", record.get("declared_frozen_setting_id")
        )
        item["infrastructure_interrupted"] = richer.get(
            "infrastructure_interrupted",
            record.get("classification") == "infrastructure_failure",
        )
        folder = Path(record["artifact_dir"])
        patch = folder / "verifier/submission.patch"
        if patch.is_file():
            lines = patch.read_text().splitlines()
            item["candidate_diff_summary"] = {
                "status": "summary only; source-bearing diff not bundled",
                "original_sha256": digest(patch.read_bytes()),
                "files": [line[6:] for line in lines if line.startswith("+++ b/")],
                "added_lines": sum(
                    line.startswith("+") and not line.startswith("+++")
                    for line in lines
                ),
                "removed_lines": sum(
                    line.startswith("-") and not line.startswith("---")
                    for line in lines
                ),
            }
        else:
            item["candidate_diff_summary"] = {
                "status": "not bundled / no saved diff at declared location"
            }
        trace_events = events(
            folder
            / (
                "trace.jsonl"
                if record["source"] == "legacy"
                else "agent/requests.jsonl"
            )
        )
        tools = [e for e in trace_events if e.get("event") == "tool_result"]
        reproductions = [
            e
            for e in tools
            if (e.get("tool_call") or {}).get("command", "").strip()
            == "python /task/reproduce.py"
        ]
        item["agent_action_summary"] = {
            "recorded_tool_results": len(tools),
            "failed_tool_results": sum(
                (e.get("result") or {}).get(
                    "returncode", (e.get("result") or {}).get("exit_code", 0)
                )
                != 0
                for e in tools
            ),
        }
        item["reproduction_evidence"] = {
            "status": "saved tool result"
            if reproductions
            else "not bundled / not recorded in this trace schema",
            "first_exit": (reproductions[0].get("result") or {}).get(
                "returncode", (reproductions[0].get("result") or {}).get("exit_code")
            )
            if reproductions
            else None,
            "source": "original tool-result events; raw command/output is excluded",
        }
        checks = read(folder / "verifier/checks.json", {})
        native_result = read(folder / "result.json", {})
        policy = (
            read(folder / "config.json", {})
            .get("agent", {})
            .get("kwargs", {})
            .get("policy", {})
        )
        required = policy.get("completion_policy", {}).get("required_checks") or list(
            checks.get("checks", {})
        )
        item["required_checks"] = [
            {
                "check": name,
                "expected_exit": 1 if name == "regression_on_baseline" else 0,
                "observed_exit": checks.get("checks", {})
                .get(name, {})
                .get("exit_code"),
                "status": check_status(
                    checks.get("checks", {}).get(name, {}).get("exit_code"),
                    1 if name == "regression_on_baseline" else 0,
                    present=name in checks.get("checks", {}),
                    ran=bool(checks),
                ),
            }
            for name in required
        ]
        item["check_evidence_status"] = (
            "available"
            if checks
            else "not bundled / not run; inspect original artifact IDs"
        )
        item["original_artifact_id"] = record["run_id"]
        item["original_result_sha256"] = (
            digest((Path(record["result_path"])).read_bytes())
            if Path(record["result_path"]).is_file()
            else None
        )
        item["native_original_verdict"] = {
            "classification": native_result.get("classification"),
            "termination_reason": native_result.get("termination_reason"),
            "controller_metadata": {
                k: (native_result.get("agent_result", {}).get("metadata") or {}).get(k)
                for k in [
                    "termination_reason",
                    "termination_detail",
                    "exit_status",
                    "explicit_submission",
                ]
            },
            "reward": {
                k: v
                for k, v in (
                    (native_result.get("verifier_result") or {}).get("rewards") or {}
                ).items()
                if numeric(v)
                and k
                in [
                    "reward",
                    "patch_acceptance",
                    "patch_accepted",
                    "task_success",
                    "agent_submission",
                    "explicit_submission",
                ]
            },
            "structured_record_classification": record["classification"],
            "task_success": record["task_success"],
            "patch_accepted": record["patch_accepted"],
        }
        item["startup_audit"] = record.get("startup_audit")
        item["reasoning_policy"] = policy.get("model", {}).get(
            "reasoning", "not recorded"
        )
        item["optional_audit"] = (
            "not bundled: source-bearing traces, patches, archives, keys and reference fixtures"
        )
        output.append(item)
    views = read(project / "notebooks/tutorial-study.json")
    contexts = {}
    for name in [
        "held-out-remainder-repeats-v1/context-measurement.json",
        "held-out-two-task-v1/context-measurement.json",
        "context-measurement-v1/measurement.json",
    ]:
        for run in read(project / "reports" / name)["runs"]:
            if run["run_id"] in contexts:
                raise ValueError("Ambiguous context source selection")
            contexts[run["run_id"]] = {
                "requests": [
                    {k: r.get(k) for k in CONTEXT_FIELDS} for r in run["requests"]
                ],
                "milestones": [
                    {
                        k: m.get(k)
                        for k in [
                            "stage",
                            "label",
                            "request",
                            "visible_in_next_request",
                            "command_sha256",
                            "expected_exit_code",
                        ]
                    }
                    for m in run["milestones"]
                ],
                "estimate_provenance": "Saved o200k_base proxy counts; estimates, not provider-native token partitions. Vocabulary and raw source are not bundled.",
            }
    selections = {
        "accepted": views["held_out_repeats"]["walkthroughs"][0]["run_id"],
        "failure": next(
            r["run_id"]
            for r in six["attempts"]
            if r["task_id"] == "LC-INT-TEMPERATURE-001"
            and r["model"].startswith("minimax")
            and r["attempt_number"] == 3
        ),
        "development_bug": views["bug_walkthroughs"][3]["run_id"],
        "feature": views["repaired_calibration_runs"],
    }
    model_preparation = {}
    for model in views["models"]:
        run = next(r for r in records if r["run_id"] in six_ids and r["model"] == model)
        first = next(
            e["body"]
            for e in events(Path(run["artifact_dir"]) / "agent/requests.jsonl")
            if e["event"] == "model_request"
        )
        model_preparation[model] = {
            key: first[key] for key in MODEL_BODY_FIELDS if key in first
        }
    dev_plan = read(project / views["development_plan"])
    study: dict[str, Any] = {
        "schema_version": 1,
        "scope_definition": "Public projections are new derived artifacts, not byte-identical native originals. Manifest checks integrity, not public authentication of private HMAC receipts.",
        "scopes": {
            "held_out_six": {
                "run_ids": sorted(six_ids),
                "setting_id": six["setting_id"],
                "label": "Six held-out tasks, three attempts per model/task",
            },
            "held_out_two": {
                "run_ids": sorted(two_ids),
                "setting_id": two["setting_id"],
                "label": "Earlier two-task held-out study, attempt 001 only",
            },
            "development": {
                "run_ids": sorted(
                    r["run_id"] for r in records if r["run_id"] not in original
                ),
                "label": "Development pilots and calibration: group by original configuration; not a frozen comparison",
            },
        },
        "planned_development": {
            "status": "not run",
            "planned_launches": dev_plan["planned_measured_runs"],
            "schedule": [
                {k: s.get(k) for k in ["job_name", "task_id", "model_id", "attempt"]}
                for s in dev_plan["schedule"]
            ],
        },
        "selections": selections,
        "model_preparation": model_preparation,
        "harness": {
            "controller": "Harbor 0.24.0",
            "agent": "mini-SWE-agent 2.4.6 DefaultAgent",
            "tool": "sandbox-scoped bash",
            "network": "none in task containers; model client on trusted host",
            "budgets": {
                "model_calls": 30,
                "generated_tokens": 10000,
                "response_tokens": 6000,
                "seconds": 1200,
                "tool_seconds": 90,
                "tool_format_repairs": 2,
            },
            "submission": "Optional early termination; final workspace captured and evaluated on submission, budget or error",
            "acceptance": "All frozen required checks pass; checks_passed_pending_review",
            "persistent_agent_memory": False,
        },
        "limitations": [
            "Public projections omit source-bearing traces, patches, candidate archives and private fixtures. Original artifact IDs support owner-mediated audit.",
            "The data are held out from harness tuning, not necessarily from model pretraining.",
            "Three attempts per task/model are descriptive, not a precise reliability estimate.",
            "Cached tokens still occupy context. Proxy category estimates are not native token partitions; context growth is not demonstrated to cause failure.",
            "New inference is not part of Replay; hosted aliases cannot guarantee identical future patches, costs or latency.",
        ],
    }
    study["harness"]["versions"] = frozen["harness"]
    study["harness"]["budgets"] = frozen["budgets"]
    tasks: dict[str, Any] = {}
    for task in sorted({str(r["task_id"]) for r in output}):
        examples = [r for r in output if r["task_id"] == task]
        tasks[task] = {
            "starting_commits": sorted(
                {str(r["commit"]) for r in examples if r.get("commit")}
            ),
            "baseline_checksums": sorted(
                {
                    str(r["baseline_checksum"])
                    for r in examples
                    if r.get("baseline_checksum")
                }
            ),
            "reconstruction": "not independently reconstructed",
        }
    tasks["LC-INT-TEMPERATURE-001"]["portable_recipe"] = read(
        configuration_path(project, "configs/public-temperature-recipe-v1.json")
    )
    reconstructed = read(
        project / "reports/public-review-v1/reconstruction-summary-v2.json"
    )
    tasks["LC-INT-TEMPERATURE-001"]["reconstruction"] = (
        "independent public controls passed; original hidden verifier unavailable"
    )
    tasks["LC-INT-TEMPERATURE-001"]["control_evidence"] = reconstructed
    return output, requests, contexts, study, tasks, index


def export(project: Path, destination: Path) -> dict:
    project = project.resolve()
    if destination.exists():
        raise FileExistsError(
            "Export destination must be new; never overwrite evidence"
        )
    attempts, requests, context, study, tasks, index = projection(project)
    destination.mkdir(parents=True)
    for name in CODE_FILES:
        target = destination / "public_replay" / name
        target.parent.mkdir(exist_ok=True)
        shutil.copyfile(project / "public_replay" / name, target)
    data = destination / "public_replay/data"
    for name, value in [
        ("attempts.json", attempts),
        ("requests.json", requests),
        ("context.json", context),
        ("study.json", study),
        ("tasks.json", tasks),
    ]:
        put(data / name, redact(value))
    for source in PUBLIC_DOCS:
        shutil.copyfile(
            project / "docs/public" / source,
            destination / ("README.md" if source == "README-public.md" else source),
        )
    shutil.copyfile(
        project / "docs/public/pyproject.toml", destination / "pyproject.toml"
    )
    shutil.copyfile(
        project / "docs/public/requirements.lock", destination / "requirements.lock"
    )
    workflows = destination / ".github/workflows"
    workflows.mkdir(parents=True)
    for workflow in ["public-replay.yml", "task-controls.yml"]:
        shutil.copyfile(project / "docs/public" / workflow, workflows / workflow)
    put(
        destination / "projection-policy.json",
        {
            "schema_version": 1,
            "allowlist": CODE_FILES
            + PUBLIC_DOCS
            + [
                "data numeric evidence",
                "new demo",
                "dependency manifests",
                "public tests",
            ],
            "excluded": [
                "runs/",
                ".state/",
                "private resources",
                "source snapshots",
                "candidate tars",
                "raw responses",
                "tool commands/output",
                "patches",
                "keys",
                "tokenizer vocabulary",
                "research notebook",
            ],
            "originals_preserved": True,
        },
    )
    for name in ["test_public_replay.py"]:
        target = destination / "tests_public" / name
        target.parent.mkdir(exist_ok=True)
        shutil.copyfile(project / "tests_public" / name, target)
    shim = destination / "experiments"
    shim.mkdir()
    (shim / "__init__.py").write_text(
        '"""Public Replay shim v1; not the frozen research execution CLI."""\n'
    )
    (shim / "__main__.py").write_text(
        "from public_replay.__main__ import main\n\nif __name__ == '__main__':\n    main()\n"
    )
    entries = {p.name: digest(p.read_bytes()) for p in sorted(data.iterdir())}
    put(
        data / "manifest.json",
        {
            "schema_version": 1,
            "projection_version": "public-evidence-v1",
            "files": entries,
            "content_id": digest(canonical(entries)),
            "identity": "Sanitized derived evidence, mapped to original run IDs; not native byte identity",
        },
    )
    from public_replay.demo import build_demo

    build_demo(data, destination)
    files = {
        str(p.relative_to(destination)): digest(p.read_bytes())
        for p in sorted(destination.rglob("*"))
        if p.is_file()
    }
    put(
        destination / "public-manifest.json",
        {
            "schema_version": 1,
            "files": files,
            "content_id": digest(canonical(files)),
            "release_status": "local review projection; license owner decision pending",
        },
    )
    audit = project / "reports/public-review-v1"
    put(audit / "launch-identities-v1.json", index)
    return {
        "content_id": digest(canonical(files)),
        "files": len(files),
        "attempts": len(attempts),
        "public_data": str(destination / "public_replay/data/manifest.json"),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(export(args.project, args.output), indent=2))


if __name__ == "__main__":
    main()
