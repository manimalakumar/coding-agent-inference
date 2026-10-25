"""Task-specific checks under Harbor's separate verifier, not a job controller."""

import difflib
import hashlib
import hmac
import json
import os
import shutil
import subprocess
import tarfile
import time
from pathlib import Path

BASE = Path("/baseline")
WORK = Path("/workspace")
LOG = Path("/logs/verifier")
SOURCE = "libs/partners/openai/langchain_openai/chat_models"
TESTS = "libs/partners/openai/tests/unit_tests"
PACKAGE = "libs/partners/openai"
PYTHONPATH = "/workspace/libs/partners/openai:/workspace/libs/core:/workspace/libs/standard-tests"


def manifest(root):
    result = {}
    for p in sorted(root.rglob("*")):
        if p.is_symlink() or (not p.is_file() and not p.is_dir()):
            raise ValueError("Symlinks/special files prohibited")
        if p.is_file():
            if p.stat().st_size > 20_000_000:
                raise ValueError("Oversize candidate file")
            result[str(p.relative_to(root))] = hashlib.sha256(
                p.read_bytes()
            ).hexdigest()
    return result


def restrict(root):
    for p in [*root.rglob("*"), root]:
        p.chmod(0o555 if p.is_dir() else 0o444)


def main():
    LOG.chmod(0o700)  # Candidate subprocesses cannot overwrite Harbor's reward.
    details = {"checks": {}, "status": "checks_failed"}
    rewards = {"reward": 0, "patch_acceptance": 0, "agent_submission": 0}
    try:
        envelope = json.loads(Path("/logs/artifacts/receipt.json").read_text())
        receipt = envelope["receipt"]
        expected = hmac.new(
            Path("/tests/receipt.key").read_bytes(),
            json.dumps(receipt, sort_keys=True).encode(),
            hashlib.sha256,
        ).hexdigest()
        if not hmac.compare_digest(expected, envelope["signature"]):
            raise ValueError("Untrusted or modified controller receipt")
        rewards["agent_submission"] = int(receipt["agent_submission"])
        details["controller_receipt"] = receipt
        deadline = receipt["deadline"]
        export = Path("/tmp/candidate-export")
        export.mkdir()
        with tarfile.open("/logs/artifacts/candidate.tar") as archive:
            if sum(m.size for m in archive.getmembers()) > 50_000_000:
                raise ValueError("Candidate archive exceeds size limit")
            for member in archive.getmembers():
                path = Path(member.name)
                if (
                    path.is_absolute()
                    or ".." in path.parts
                    or not (member.isfile() or member.isdir())
                ):
                    raise ValueError("Unsafe candidate archive entry")
                if not any(
                    str(path) == rel or str(path).startswith(rel + "/")
                    for rel in (SOURCE, TESTS)
                ):
                    raise ValueError("Candidate archive escapes allowed paths")
            archive.extractall(export, filter="data")
        Path("/candidate").mkdir(exist_ok=True)
        for rel, artifact in [(SOURCE, "production"), (TESTS, "development-tests")]:
            manifest(export / rel)
            shutil.copytree(export / rel, Path("/candidate") / artifact)
        changed = []
        new_tests = []
        patch = []
        for rel, artifact in [(SOURCE, "production"), (TESTS, "development-tests")]:
            before, after = (
                manifest(BASE / rel),
                manifest(Path("/candidate") / artifact),
            )
            for name in sorted(before.keys() | after.keys()):
                if before.get(name) == after.get(name):
                    continue
                path = rel + "/" + name
                if not path.endswith(".py"):
                    raise ValueError("Only Python changes permitted")
                if rel == TESTS:
                    if name in before:
                        raise ValueError("Existing test files are read-only")
                    if not Path(name).name.startswith("test_"):
                        raise ValueError("Regressions must be new test_*.py files")
                    new_tests.append(path)
                changed.append(path)
                old = (BASE / path).read_text() if name in before else ""
                new = (
                    (Path("/candidate") / artifact / name).read_text()
                    if name in after
                    else ""
                )
                patch.extend(
                    difflib.unified_diff(
                        old.splitlines(True),
                        new.splitlines(True),
                        fromfile="a/" + path,
                        tofile="b/" + path,
                    )
                )
            target = WORK / rel
            target.chmod(0o755)
            # Copy candidate state without accepting links or changes outside allowed trees.
            shutil.rmtree(target)
            shutil.copytree(Path("/candidate") / artifact, target)
        (LOG / "submission.patch").write_text("".join(patch))
        restrict(WORK)
        details["files_changed"] = changed

        def run(name, args, cwd=WORK / PACKAGE, pythonpath=PYTHONPATH):
            left = deadline - time.time()
            if left <= 0:
                raise TimeoutError("Shared 20-minute workflow deadline exhausted")
            env = {
                "PATH": "/opt/venv/bin:/usr/local/bin:/usr/bin:/bin",
                "HOME": "/tmp",
                "OPENAI_API_KEY": "sk-offline-placeholder",
                "PYTHONDONTWRITEBYTECODE": "1",
                "PYTHONPATH": pythonpath,
                "PYTEST_ADDOPTS": "-p no:cacheprovider --no-cov",
                "TIKTOKEN_CACHE_DIR": "/opt/tiktoken",
            }

            def unprivileged():
                os.setgroups([])
                os.setgid(65532)
                os.setuid(65532)

            result = subprocess.run(
                args,
                cwd=cwd,
                env=env,
                capture_output=True,
                text=True,
                timeout=min(90, left),
                preexec_fn=unprivileged,
            )
            details["checks"][name] = {
                "exit_code": result.returncode,
                "output": result.stdout + result.stderr,
                "command": args,
            }
            return result

        # The candidate's tests must reproduce on fresh unchanged source.
        baseline = Path("/tmp/unchanged-source")
        shutil.copytree(BASE, baseline)
        for directory in baseline.rglob("*"):
            if directory.is_dir():
                directory.chmod(0o755)
        for path in new_tests:
            target = baseline / path
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(WORK / path, target)
        restrict(baseline)
        original_path = ":".join(
            str(baseline / p)
            for p in ["libs/partners/openai", "libs/core", "libs/standard-tests"]
        )
        reg = (
            run(
                "regression_on_baseline",
                [
                    "python",
                    "-m",
                    "pytest",
                    "-q",
                    "--disable-socket",
                    "--allow-unix-socket",
                    *[str(baseline / p) for p in new_tests],
                ],
                cwd=baseline / PACKAGE,
                pythonpath=original_path,
            )
            if new_tests
            else None
        )
        reproduced = bool(reg and reg.returncode == 1)
        run("reproducer", ["python", "/task/reproduce.py"])
        if new_tests:
            run(
                "regression",
                [
                    "python",
                    "-m",
                    "pytest",
                    "-q",
                    "--disable-socket",
                    "--allow-unix-socket",
                    *[str(WORK / p) for p in new_tests],
                ],
            )
        run(
            "existing_tests",
            [
                "python",
                "-m",
                "pytest",
                "tests/unit_tests/chat_models/test_base.py",
                "-q",
                "--disable-socket",
                "--allow-unix-socket",
            ],
        )
        if changed:
            files = [str(WORK / p) for p in changed if (WORK / p).exists()]
            run("lint", ["ruff", "check", "--no-cache", *files])
            run("format", ["ruff", "format", "--check", "--no-cache", *files])
            sources = [
                str(WORK / p)
                for p in changed
                if p.startswith(SOURCE + "/") and (WORK / p).exists()
            ]
            if sources:
                run("types", ["mypy", "--cache-dir=/tmp/mypy", *sources])
        run(
            "hidden_acceptance",
            [
                "python",
                "-m",
                "pytest",
                "-q",
                "-c",
                "/dev/null",
                "--confcutdir=/tests",
                "--disable-socket",
                "--allow-unix-socket",
                "/tests/test_acceptance.py",
            ],
        )
        development_ok = bool(changed and new_tests and reproduced) and all(
            c["exit_code"] == 0
            for n, c in details["checks"].items()
            if n != "regression_on_baseline"
        )
        rewards["patch_acceptance"] = int(development_ok)
        rewards["reward"] = int(development_ok and receipt["agent_submission"])
        details["status"] = (
            "checks_passed_pending_review" if rewards["reward"] else "checks_failed"
        )
    except Exception as exc:
        details["error"] = type(exc).__name__ + ": " + str(exc)
    finally:
        (LOG / "checks.json").write_text(json.dumps(details, indent=2))
        (LOG / "reward.json").write_text(json.dumps(rewards))


if __name__ == "__main__":
    main()
