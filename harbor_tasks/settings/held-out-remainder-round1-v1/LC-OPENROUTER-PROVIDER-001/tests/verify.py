"""Task-specific checks under Harbor's separate verifier, not a job controller."""

import difflib
import hashlib
import hmac
import json
import os
import shutil
import shlex
import subprocess
import tarfile
import time
from pathlib import Path

from workspace_policy import verdict

BASE = Path("/baseline")
WORK = Path("/workspace")
LOG = Path("/logs/verifier")
SOURCES = ['libs/partners/openrouter/langchain_openrouter']
TEST_ROOTS = ['libs/partners/openrouter/tests/unit_tests']
EXISTING = [['libs/partners/openrouter', 'tests/unit_tests/test_chat_models.py']]
PACKAGE = 'libs/partners/openrouter'
PYTHONPATH = '/workspace/libs/partners/openrouter:/workspace/libs/core:/workspace/libs/standard-tests:/workspace/libs/partners/openai:/workspace/libs/langchain_v1'


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
    rewards = {
        "reward": 0,
        "patch_acceptance": 0,
        "agent_submission": 0,
        "patch_accepted": 0,
        "explicit_submission": 0,
        "task_success": 0,
    }
    common = json.loads(Path("/tests/completion-policy.json").read_text())
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
        rewards["agent_submission"] = int(receipt["explicit_submission"])
        rewards["explicit_submission"] = int(receipt["explicit_submission"])
        details["controller_receipt"] = receipt
        if receipt.get("completion_policy") != common["policy_id"] or not receipt.get(
            "workspace_quiesced"
        ):
            raise ValueError(
                "Candidate not frozen under the expected completion policy"
            )
        details["termination_reason"] = receipt["termination_reason"]
        details["explicit_submission"] = receipt["explicit_submission"]
        details["completion_policy"] = common["policy_id"]
        details["evaluation_started_at"] = time.time()
        deadline = details["evaluation_started_at"] + common["evaluation_seconds"]
        details["evaluation_deadline"] = deadline
        details["candidate_sha256"] = hashlib.sha256(
            Path("/logs/artifacts/candidate.tar").read_bytes()
        ).hexdigest()
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
                    for rel in (*SOURCES, *TEST_ROOTS)
                ):
                    raise ValueError("Candidate archive escapes allowed paths")
            archive.extractall(export, filter="data")
        Path("/candidate").mkdir(exist_ok=True)
        for rel in (*SOURCES, *TEST_ROOTS):
            artifact = rel
            manifest(export / rel)
            shutil.copytree(export / rel, Path("/candidate") / artifact)
        changed = []
        new_tests = []
        patch = []
        for rel in (*SOURCES, *TEST_ROOTS):
            artifact = rel
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
                if rel in TEST_ROOTS:
                    if name in before:
                        raise ValueError("Existing test files are read-only")
                    if not Path(name).name.startswith("test_"):
                        raise ValueError("Regressions must be new test_*.py files")
                    if Path(name).parent != Path("."):
                        raise ValueError("Regression files must be directly in tests/unit_tests")
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
        details["checks"]["file_policy"] = {
            "exit_code": 0,
            "output": "Allowed source changes and new regression files only; existing tests preserved.",
        }
        details["files_changed"] = changed

        def run(name, args, cwd=WORK / PACKAGE, pythonpath=PYTHONPATH):
            left = deadline - time.time()
            if left <= 0:
                raise TimeoutError("Trusted five-minute evaluation deadline exhausted")
            env = {
                "PATH": "/opt/venv/bin:/usr/local/bin:/usr/bin:/bin",
                "HOME": "/tmp",
                "OPENAI_API_KEY": "sk-offline-placeholder",
                "PYTHONDONTWRITEBYTECODE": "1",
                "PYTHONPATH": pythonpath,
                "PYTEST_ADDOPTS": "-p no:cacheprovider",
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
            for p in [
                'libs/partners/openrouter',
                "libs/langchain_v1",
                "libs/core",
                "libs/standard-tests",
                "libs/partners/openai",
            ]
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
        details["regression_signal"] = reproduced
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
        existing = []
        for index, (package, module) in enumerate(EXISTING):
            r = run("existing_tests_" + str(index), ["python", "-m", "pytest", *shlex.split(module), "-q", "--disable-socket", "--allow-unix-socket"], cwd=WORK / package)
            existing.append(r)
        details["checks"]["existing_tests"] = {"exit_code": 0 if all(r.returncode == 0 for r in existing) else 1, "output": "\n".join(r.stdout + r.stderr for r in existing)}
        if changed:
            for name in ("lint", "format", "types"):
                results = []
                for package in ['libs/partners/openrouter']:
                    files = [str(WORK / p) for p in changed if p.startswith(package + "/") and (WORK / p).exists() and (name != "types" or any(p.startswith(s + "/") for s in SOURCES))]
                    if not files:
                        continue
                    args = (["ruff", "check", "--no-cache"] if name == "lint" else ["ruff", "format", "--check", "--no-cache"] if name == "format" else ["mypy", "--cache-dir=/dev/null"]) + files
                    results.append(run(name + "_" + package.replace("/", "_"), args, cwd=WORK / package))
                details["checks"][name] = {"exit_code": 0 if results and all(r.returncode == 0 for r in results) else 1, "output": "\n".join(r.stdout + r.stderr for r in results)}
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
                "--asyncio-mode=auto",
                "/tests/test_acceptance.py",
            ],
        )
        rewards = verdict(
            details["checks"],
            required=common["required_checks"],
            changed=changed,
            new_tests=new_tests,
            explicit_submission=receipt["explicit_submission"],
        )
        details["task_success"] = bool(rewards["task_success"])
        details["patch_accepted"] = bool(rewards["patch_accepted"])
        details["status"] = (
            "checks_passed_pending_review"
            if rewards["task_success"]
            else "checks_failed"
        )
    except Exception as exc:
        details["error"] = type(exc).__name__ + ": " + str(exc)
        details["task_success"] = False
        details["patch_accepted"] = False
    finally:
        (LOG / "checks.json").write_text(json.dumps(details, indent=2))
        (LOG / "reward.json").write_text(json.dumps(rewards))


if __name__ == "__main__":
    main()
