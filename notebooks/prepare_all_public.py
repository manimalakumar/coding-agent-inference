"""Prepare all task resources and run fixture controls; never make model requests."""

import concurrent.futures
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from public_preparation.catalog import catalog  # noqa: E402
from public_preparation.runtime import prepare, task_config, validate, write  # noqa: E402


def one(task):
    started = time.time()
    try:
        print(task, "preparing", flush=True)
        prepare(task, True)
        print(task, "validating Harbor fixture controls", flush=True)
        result = validate(task)
        return {
            "task_id": task,
            "status": "ready",
            "protocol_id": result["protocol_id"],
            "controls": result["controls"],
            "model_requests": 0,
            "elapsed_seconds": time.time() - started,
        }
    except Exception as exc:
        print(task, type(exc).__name__, str(exc), flush=True)
        return {
            "task_id": task,
            "status": "blocked",
            "protocol_id": task_config(task)["protocol_id"],
            "blocker": type(exc).__name__ + ": " + str(exc).replace(str(ROOT), "."),
            "model_requests": 0,
            "elapsed_seconds": time.time() - started,
        }


def main():
    tasks = [
        "LC-INT-TEMPERATURE-001",
        *[t for t in catalog() if t != "LC-INT-TEMPERATURE-001"],
    ]
    rows = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures = {pool.submit(one, task): task for task in tasks}
        for future in concurrent.futures.as_completed(futures):
            row = future.result()
            rows[row["task_id"]] = row
            write(
                ROOT / "reports/public-readiness/campaign.json",
                {"model_requests": 0, "tasks": [rows[t] for t in tasks if t in rows]},
            )
            print(row["task_id"], row["status"], flush=True)
    print("Completed zero-inference preparation campaign", flush=True)


if __name__ == "__main__":
    main()
