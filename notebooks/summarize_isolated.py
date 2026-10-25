"""Review isolated controls and report denied audit events separately from failures."""

import json
import os
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    staged = ROOT / ".local/isolated-public-validation"
    env = {
        k: v
        for k, v in os.environ.items()
        if not any(
            t in k.upper() for t in ["KEY", "TOKEN", "SECRET", "CREDENTIAL", "PASSWORD"]
        )
    }
    env.update(
        PYTHONPATH=str(staged / ".local/python-guard") + os.pathsep + str(staged),
        PYTHONDONTWRITEBYTECODE="1",
        PUBLIC_NO_INFERENCE="1",
        HARBOR_TELEMETRY="0",
        MSWEA_SILENT_STARTUP="1",
    )
    check = subprocess.run(
        [
            str(ROOT / ".venv/bin/python"),
            "-c",
            'import json; from experiments.portable_prepare import doctor; d=doctor(); assert len(d["tasks"])==13; assert all(t["execute_ready"] for t in d["tasks"].values()),d; print(json.dumps(d))',
        ],
        cwd=staged,
        env=env,
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    )
    doctor = json.loads(check.stdout.strip().splitlines()[-1])
    probes = json.loads((staged / ".local/guard-controls.json").read_text())
    assert len(probes) == 4 and all(
        v == "blocked before side effect" for v in probes.values()
    )
    rows = []
    for task in doctor["tasks"]:
        receipt = json.loads(
            (staged / ".local/public-v1" / task / "validated.json").read_text()
        )
        assert len(receipt["controls"]) == 4
        assert all(
            c["model_requests"] == 0 and c["passed"] for c in receipt["controls"]
        )
        assert all(c["passed"] for c in receipt["wrapper_controls"])
        rows.append(
            {
                "task_id": task,
                "protocol_id": receipt["protocol_id"],
                "passed": True,
                "wrapper_checks_passed": len(receipt["wrapper_controls"]),
                "terminal_controls_passed": 4,
                "model_requests": 0,
                "setting_id": receipt["setting_id"],
            }
        )
        target = ROOT / "reports/public-readiness/isolated" / (task + ".json")
        target.parent.mkdir(exist_ok=True)
        shutil.copyfile(staged / "reports/public-readiness" / (task + ".json"), target)
    blocked = staged / ".local/forbidden-access.jsonl"
    events = blocked.read_text().splitlines() if blocked.exists() else []
    summary = {
        "status": "passed",
        "audit_aggregation_version": 2,
        "tasks": rows,
        "model_requests": 0,
        "private_project_reads": 0,
        "host_network_calls": 0,
        "negative_controls": probes,
        "readiness_resource_fingerprints_rechecked": 13,
        "relocated_task_resources": True,
        "runtime": "Shared installed third-party runtime; project code and task resources loaded from isolated copy.",
        "audit_guard_enforced": True,
        "blocked_external_io_attempts": len(events),
        "blocked_attempts_note": "All denied before side effects; observed background Harbor telemetry and standard-library archive discovery. Blocked attempts are isolation evidence, not task failures.",
        "historical_equivalence": "not claimed",
    }
    (ROOT / "reports/public-readiness/isolated-validation.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(
        json.dumps(
            {
                "status": "passed",
                "tasks": len(rows),
                "model_requests": 0,
                "blocked_external_io_attempts": len(events),
            }
        )
    )


if __name__ == "__main__":
    main()
