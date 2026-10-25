"""Clean evaluation. Hidden checks never run in an agent-writable process."""

import difflib
import shlex
import shutil
from pathlib import Path

from sandbox.docker import InfrastructureError, Sandbox, manifest


def collect_changes(box: Sandbox, output: Path) -> list[str]:
    """Reject links and edits to existing tests; export an auditable text patch."""
    changed = []
    patches = []
    for rel in box.config.writable:
        before, after = box.baseline / rel, box.workspace / rel
        old, new = manifest(before), manifest(after)
        for name in sorted(old.keys() | new.keys()):
            if old.get(name) == new.get(name):
                continue
            path = rel + "/" + name
            if not path.endswith(".py"):
                raise InfrastructureError(
                    "Only Python source/test submissions are supported"
                )
            if (
                any(path.startswith(p + "/") for p in box.task.test_paths)
                and name in old
            ):
                raise InfrastructureError("Existing tests must remain unchanged")
            if (
                box.config.test_policy == "read_only_existing_new_test_files"
                and any(path.startswith(p + "/") for p in box.task.test_paths)
                and not (
                    Path(path).name.startswith("test_") and Path(path).suffix == ".py"
                )
            ):
                raise InfrastructureError("Regressions must be new test_*.py files")
            old_text = (before / name).read_text() if name in old else ""
            new_text = (after / name).read_text() if name in new else ""
            changed.append(path)
            patches.extend(
                difflib.unified_diff(
                    old_text.splitlines(keepends=True),
                    new_text.splitlines(keepends=True),
                    fromfile="a/" + path if name in old else "/dev/null",
                    tofile="b/" + path if name in new else "/dev/null",
                )
            )
    output.write_text("".join(patches))
    return changed


def development_checks(box: Sandbox, changes: list[str], remaining: float) -> dict:
    """Used for agent feedback; hidden acceptance is never returned to the model."""
    import time

    deadline = time.monotonic() + remaining
    checks = list(box.task.development_checks)
    new_tests = [
        p for p in changes if any(p.startswith(t + "/") for t in box.task.test_paths)
    ]
    commands = [(c.name, c.command) for c in checks]
    if new_tests:
        commands.insert(
            0,
            (
                "regression",
                "python -m pytest -q --disable-socket --allow-unix-socket "
                + " ".join(shlex.quote("/workspace/" + p) for p in new_tests),
            ),
        )
    if changes:
        files = " ".join(
            shlex.quote("/workspace/" + p)
            for p in changes
            if (box.workspace / p).exists()
        )
        if files:
            commands += [
                ("lint", "ruff check --no-cache " + files),
                ("format", "ruff format --check --no-cache " + files),
            ]
        source = [
            p
            for p in changes
            if any(p.startswith(s + "/") for s in box.task.source_paths)
            and (box.workspace / p).exists()
        ]
        if source:
            commands.append(
                (
                    "types",
                    "mypy --cache-dir=/tmp/mypy "
                    + " ".join(shlex.quote("/workspace/" + p) for p in source),
                )
            )
    results = {}
    for name, command in commands:
        seconds = min(box.config.budgets.tool_seconds, deadline - time.monotonic())
        if seconds <= 0:
            raise TimeoutError("Workflow deadline during development checks")
        results[name] = box.execute(command, seconds)
    return {
        "passed": bool(changes)
        and bool(new_tests)
        and all(r["exit_code"] == 0 and not r["termination"] for r in results.values()),
        "checks": results,
        "regression_test_added": bool(new_tests),
    }


def qualify(box: Sandbox, run: Path, hidden: Path, deadline: float) -> dict:
    """Fresh source/dependencies per command; hidden checks in a distinct container."""
    import time

    changes = collect_changes(box, run / "submission.patch")
    fresh = Sandbox(
        box.config,
        box.task,
        box.baseline,
        run / "evaluation-workspace",
        box.task_file,
        box.image,
        box.resources,
    )
    fresh.prepare()
    for rel in box.config.writable:
        shutil.rmtree(fresh.workspace / rel)
        shutil.copytree(box.workspace / rel, fresh.workspace / rel)
    # New regression tests must actually fail on unchanged source.
    baseline = Sandbox(
        box.config,
        box.task,
        box.baseline,
        run / "regression-workspace",
        box.task_file,
        box.image,
        box.resources,
    )
    baseline.prepare()
    new_tests = []
    for p in changes:
        if any(p.startswith(t + "/") for t in box.task.test_paths):
            target = baseline.workspace / p
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(box.workspace / p, target)
            new_tests.append("/workspace/" + p)
    reproduction = (
        baseline.execute(
            "python -m pytest -q --disable-socket --allow-unix-socket "
            + " ".join(shlex.quote(p) for p in new_tests),
            min(90, deadline - time.monotonic()),
        )
        if new_tests
        else {"exit_code": 0, "output": "No regression test submitted"}
    )
    reproduced = (
        reproduction["exit_code"] == 1
        and box.task.expected_failure in reproduction["output"]
    )
    development = development_checks(fresh, changes, deadline - time.monotonic())
    # Each command gets a new container: no processes/modified dependencies survive.
    acceptance = fresh.execute(
        "python -m pytest -q -c /dev/null --confcutdir=/evaluation "
        "--disable-socket --allow-unix-socket /evaluation/test_acceptance.py",
        min(90, deadline - time.monotonic()),
        extra_mounts=(
            (
                hidden.resolve() / "test_acceptance.py",
                "/evaluation/test_acceptance.py",
                "ro",
            ),
        ),
    )
    passed = reproduced and development["passed"] and acceptance["exit_code"] == 0
    return {
        "status": "checks_passed_pending_review" if passed else "checks_failed",
        "files_changed": changes,
        "regression_on_baseline": reproduction,
        "original_failure_reproduced": reproduced,
        "development": development,
        "hidden_acceptance": acceptance,
        "limitations": "Human review required; no commit, publication or merge authorized.",
    }
