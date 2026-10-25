"""Render the task inventory and readiness table from actual public control receipts."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    from experiments.portable_prepare import doctor
    from public_preparation.runtime import task_config
    from public_preparation.wrapper_controls import existing_commands

    status = doctor(ROOT)
    extraction = json.loads(
        (ROOT / "public_preparation/configs/acceptance-extraction.json").read_text()
    )["tasks"]
    inventory = []
    rows = []
    for task, item in status["tasks"].items():
        spec = task_config(task)
        state = ROOT / ".local/public-v1" / task
        dep = (
            json.loads((state / "dependency.json").read_text())
            if (state / "dependency.json").exists()
            else {}
        )
        receipt = (
            json.loads((state / "validated.json").read_text())
            if (state / "validated.json").exists()
            else None
        )
        blocker = item["remaining_blocker"]
        if not item["execute_ready"]:
            known = ROOT / "reports/public-readiness" / (task + "-blocker.json")
            if known.exists():
                blocker = json.loads(known.read_text())["blocker"]
        row = {
            "task_id": task,
            "preparation_command": ".venv/bin/python -m experiments.portable_prepare prepare "
            + task
            + " --allow-setup-network --controls",
            "protocol_id": spec["protocol_id"],
            "ready": item["execute_ready"],
            "control_results": "baseline rejected; reference/submission/budget/error accepted; wrapper and isolation passed"
            if item["execute_ready"]
            else "not all current controls passed",
            "remaining_blocker": None if item["execute_ready"] else blocker,
        }
        rows.append(row)
        try:
            focused = existing_commands(spec)
        except RuntimeError as exc:
            focused = [str(exc)]
        inventory.append(
            {
                **row,
                "repository": spec["repository"],
                "starting_commit": spec["starting_commit"],
                "reference_commit": spec["reference_commit"],
                "baseline_checksum": spec["baseline_checksum"],
                "source_lockfile": spec["package"] + "/uv.lock",
                "source_lock_sha256": dep.get("source_lock_sha256"),
                "resolved_environment_lock": "public_preparation/locks/"
                + task
                + ".requirements.lock",
                "python_image": dep.get("python_image"),
                "uv_image": "ghcr.io/astral-sh/uv@sha256:eb2843a1e56fd9e30c7276ce1a52cba86e64c7b385f5e3279a0e08e02dd058fc",
                "build_inputs": [
                    "pinned source export",
                    "package uv.lock",
                    "retained reproducer",
                    "separate verifier and agent Dockerfiles",
                ]
                + (["build-essential=12.9"] if task == "LC-HF-ACCELERATOR-001" else [])
                + (
                    [
                        "Core-lock-pinned blockbuster and forbiddenfruit with artifact hashes"
                    ]
                    if task == "LC-ASYNC-TOOLS-001"
                    else []
                ),
                "dependency_groups": dep.get("groups"),
                "reproducer": 'task_spec["reproducer"] in public_preparation.catalog; emitted only during preparation',
                "evaluator_resources": spec["task_bundle"] + "/tests",
                "setup_extracted_acceptance": {
                    "recipe": "public_preparation/configs/acceptance-extraction.json",
                    "commit": extraction[task]["commit"],
                    "expected_sha256": extraction[task]["output_sha256"],
                    "generated_path": ".local/public-v1/"
                    + task
                    + "/evaluator/test_acceptance.py",
                    "agent_access": False,
                }
                if task in extraction
                else None,
                "focused_test_commands": focused,
                "required_checks": [
                    "file_policy",
                    "regression_on_baseline",
                    "reproducer",
                    "regression",
                    "existing_tests",
                    "lint",
                    "format",
                    "types",
                    "retained_behavioral_acceptance",
                ],
                "historical_equivalence": "not claimed",
                "fixture_controls": receipt["controls"]
                if receipt and item["execute_ready"]
                else [],
            }
        )
    output = ROOT / "reports/public-readiness"
    output.mkdir(exist_ok=True)
    (output / "inventory.json").write_text(
        json.dumps({"model_requests": 0, "tasks": inventory}, indent=2) + "\n"
    )
    (output / "readiness.json").write_text(
        json.dumps(
            {
                "model_requests": 0,
                "ready_tasks": sum(r["ready"] for r in rows),
                "tasks": rows,
            },
            indent=2,
        )
        + "\n"
    )
    lines = [
        "| Task | Preparation command | Public evaluation protocol | Controls | Remaining blocker |",
        "| --- | --- | --- | --- | --- |",
    ]
    for r in rows:
        controls = "All passed" if r["ready"] else "Incomplete"
        blocker = "None" if r["ready"] else r["remaining_blocker"].replace("|", "/")
        lines.append(
            "| "
            + r["task_id"]
            + " | `prepare "
            + r["task_id"]
            + "` | `"
            + r["protocol_id"]
            + "` | "
            + controls
            + " | "
            + blocker
            + " |"
        )
    path = ROOT / "README.md"
    text = path.read_text()
    start = text.index("<!-- READINESS START -->")
    end = text.index("<!-- READINESS END -->", start)
    text = (
        text[:start]
        + "<!-- READINESS START -->\n"
        + "\n".join(lines)
        + "\n"
        + text[end:]
    )
    path.write_text(text)
    print(
        json.dumps(
            {
                "ready_tasks": sum(r["ready"] for r in rows),
                "total_tasks": len(rows),
                "model_requests": 0,
            }
        )
    )


if __name__ == "__main__":
    main()
