"""Replay saved records or deliberately start a fresh native Harbor job."""

import argparse
import json
from pathlib import Path

from experiments.launch import launch
from experiments.notebook_workflow import records as load_runs
from experiments.records import aggregate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project", type=Path, default=Path(__file__).resolve().parents[1]
    )
    commands = parser.add_subparsers(dest="mode", required=True)
    commands.add_parser(
        "report", help="Refresh the central result log without inference"
    )
    commands.add_parser(
        "reconcile",
        help="Audit saved Harbor runs and refresh all bookkeeping without inference",
    )
    replay = commands.add_parser("replay")
    replay.add_argument("--task", default="LC-TIMEOUT-001")
    replay.add_argument("--output", type=Path)
    execute = commands.add_parser("execute")
    execute.add_argument("--confirm-paid-run", action="store_true")
    execute.add_argument("--job-name")
    execute.add_argument("--study", help="Study ID from configs/study.json")
    execute.add_argument("--credentials", type=Path)
    args = parser.parse_args()
    if args.mode == "reconcile":
        from experiments.bookkeeping import reconcile

        print(json.dumps(reconcile(args.project), indent=2))
    elif args.mode == "report":
        from experiments.report import write_reports

        data = write_reports(args.project)
        print(
            json.dumps(
                {
                    "records": len(data["records"]),
                    "report": "reports/EXPERIMENT-RESULTS.md",
                }
            )
        )
    elif args.mode == "execute":
        print(
            json.dumps(
                launch(
                    args.project,
                    mode="execute",
                    confirmed=args.confirm_paid_run,
                    job_name=args.job_name,
                    credentials=args.credentials,
                    study_id=args.study,
                ),
                indent=2,
            )
        )
    else:
        records = [
            r
            for r in load_runs(args.project)
            if args.task == "all" or r["task_id"] == args.task
        ]
        data = {
            "schema_version": 1,
            "mode": "replay",
            "records": records,
            "per_task": aggregate(records),
            "across_tasks": aggregate(records, across_tasks=True),
        }
        text = json.dumps(data, indent=2) + "\n"
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(text)
        else:
            print(text)


if __name__ == "__main__":
    main()
