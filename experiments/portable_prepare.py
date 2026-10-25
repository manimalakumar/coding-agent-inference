"""Explicit public source preparation for all 13 tasks; no inference probes."""

import argparse
import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def specs(root=ROOT):
    return json.loads((root / "public_preparation/configs/tasks.json").read_text())


def validate_spec(spec):
    if spec["repository"] != "https://github.com/langchain-ai/langchain.git":
        raise ValueError("Unexpected upstream repository")
    if not re.fullmatch("[0-9a-f]{40}", spec["starting_commit"]):
        raise ValueError("A full pre-fix commit SHA is required")


def doctor(root=ROOT):
    from public_preparation.catalog import catalog
    from public_preparation.launch import readiness

    tasks = {}
    for task_id, spec in catalog().items():
        validate_spec(spec)
        row = {
            "repository": spec["repository"],
            "starting_commit": spec["starting_commit"],
            "protocol_id": spec["protocol_id"],
            "historical_equivalence": "not claimed",
            "preparation_command": "python -m experiments.portable_prepare prepare "
            + task_id
            + " --allow-setup-network --controls",
        }
        try:
            receipt = readiness(root, task_id)
            row.update(
                execute_ready=True,
                controls="passed",
                setting_id=receipt["setting_id"],
                remaining_blocker=None,
            )
        except (OSError, ValueError, RuntimeError) as exc:
            row.update(
                execute_ready=False,
                controls="not all passed",
                remaining_blocker=str(exc),
            )
        tasks[task_id] = row
    return {"model_requests": 0, "tasks": tasks}


def execution_readiness(root, task_ids):
    from public_preparation.launch import readiness

    return {task: readiness(root, task) for task in task_ids}


def obtain_source(task_id, *, allow_network=False, root=ROOT):
    """Fetch a full pinned SHA and produce a safe, history-free source export.

    The new source lives only under ignored .local. Setup network permission is
    explicit and does not authorize paid inference or image building.
    """
    spec = specs(root)[task_id]
    validate_spec(spec)
    if not allow_network:
        raise PermissionError("Source fetch requires --allow-setup-network")
    target = root / ".local/prepared" / task_id
    if target.exists():
        raise FileExistsError(
            "Prepared directory already exists; refusing to overwrite"
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    # Build in a sibling temporary directory so failed fetches do not appear ready.
    with tempfile.TemporaryDirectory(dir=target.parent, prefix=".source-") as tmp:
        stage = Path(tmp)
        cache = stage / "upstream.git"
        env = {
            k: v
            for k, v in os.environ.items()
            if not any(
                t in k.upper()
                for t in ["KEY", "TOKEN", "SECRET", "CREDENTIAL", "PASSWORD"]
            )
        }
        env["GIT_TERMINAL_PROMPT"] = "0"

        def git(*args):
            return subprocess.run(
                ["git", *map(str, args)],
                check=True,
                capture_output=True,
                env=env,
                timeout=300,
            ).stdout

        git("init", "--bare", cache)
        git(
            "--git-dir",
            cache,
            "fetch",
            "--depth=1",
            spec["repository"],
            spec["starting_commit"],
        )
        actual = git("--git-dir", cache, "rev-parse", "FETCH_HEAD").decode().strip()
        if actual != spec["starting_commit"]:
            raise ValueError("Fetched commit identity differs")
        from public_replay.prepare import export_commit

        provenance = export_commit(cache, actual, stage / "baseline")
        if provenance["baseline_checksum"] != spec["baseline_checksum"]:
            raise ValueError(
                "Exported baseline differs from the reviewed historical source checksum"
            )
        record = {
            "task_id": task_id,
            "repository": spec["repository"],
            "starting_commit": actual,
            **provenance,
            "model_requests": 0,
            "status": "source_only; environment and evaluator controls still required",
        }
        (stage / "source.json").write_text(json.dumps(record, indent=2) + "\n")
        shutil.move(str(stage), target)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    sub.add_parser("doctor")
    src = sub.add_parser("source")
    src.add_argument("task_id", choices=sorted(specs()))
    src.add_argument("--allow-setup-network", action="store_true")
    prep = sub.add_parser("prepare")
    prep.add_argument("task_id", choices=sorted(specs()))
    prep.add_argument("--allow-setup-network", action="store_true")
    prep.add_argument("--controls", action="store_true")
    args = parser.parse_args()
    if args.action == "prepare":
        from public_preparation.runtime import prepare, validate

        result = prepare(args.task_id, args.allow_setup_network)
        if args.controls:
            result = validate(args.task_id)
        print(json.dumps(result, indent=2))
        return
    print(
        json.dumps(
            doctor()
            if args.action == "doctor"
            else obtain_source(args.task_id, allow_network=args.allow_setup_network),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
