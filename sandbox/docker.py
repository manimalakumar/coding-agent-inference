"""All evaluated commands cross a Docker boundary; there is no host fallback."""

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import time
import uuid
from pathlib import Path

from agent.schemas import SandboxConfig, Task


class InfrastructureError(RuntimeError):
    """A run could not be evaluated reliably."""


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def manifest(root: Path) -> dict[str, str]:
    """Never follow submitted symlinks, special files, or Git metadata."""
    entries = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise InfrastructureError(
                f"Symlink is prohibited: {path.relative_to(root)}"
            )
        if ".git" in path.relative_to(root).parts:
            raise InfrastructureError("Git metadata is prohibited")
        if path.is_file():
            if path.stat().st_size > 20_000_000:
                raise InfrastructureError("Resource file exceeds 20 MB")
            entries[path.relative_to(root).as_posix()] = hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
        elif not path.is_dir():
            raise InfrastructureError("Special files are prohibited")
    return entries


def host_env() -> dict[str, str]:
    """The Docker CLI needs connection settings, never inference credentials."""
    keys = ("PATH", "HOME", "DOCKER_HOST", "DOCKER_CONTEXT", "DOCKER_CONFIG")
    return {k: os.environ[k] for k in keys if k in os.environ}


def docker_ready() -> dict:
    if not shutil.which("docker"):
        raise InfrastructureError("Docker CLI is not installed")
    result = subprocess.run(
        ["docker", "version", "--format", "{{json .}}"],
        capture_output=True,
        text=True,
        timeout=15,
        env=host_env(),
    )
    if result.returncode:
        raise InfrastructureError("Docker daemon unavailable: " + result.stderr[-1000:])
    return json.loads(result.stdout)


class Sandbox:
    def __init__(
        self,
        config: SandboxConfig,
        task: Task,
        baseline: Path,
        workspace: Path,
        task_file: Path,
        image: str,
        resources: Path,
    ) -> None:
        self.config, self.task = config, task
        self.baseline = baseline.resolve()
        self.workspace = workspace.resolve()
        self.task_file, self.image = task_file.resolve(), image
        self.resources = resources.resolve()
        allowed = set(task.source_paths + task.test_paths)
        if not set(config.writable) <= allowed:
            raise InfrastructureError("Sandbox writable mounts exceed task permissions")
        if not image.startswith("sha256:"):
            raise InfrastructureError("Execution image must be pinned by immutable ID")

    def prepare(self) -> None:
        if self.workspace.exists():
            raise InfrastructureError("Attempt workspace must be fresh")
        manifest(self.baseline)
        self.workspace.mkdir(parents=True)
        for rel in self.config.writable:
            source, target = self.baseline / rel, self.workspace / rel
            if not source.is_dir():
                raise InfrastructureError(f"Missing allowed source directory: {rel}")
            shutil.copytree(source, target)
            for p in [target, *target.rglob("*")]:
                p.chmod(0o777 if p.is_dir() else 0o666)

    def resource_record(self) -> list[dict]:
        records = []
        for item in self.config.resources:
            source = (self.resources / item.resource).resolve()
            if not source.is_relative_to(self.resources):
                raise InfrastructureError("Resource escaped trusted resource directory")
            if not source.is_dir():
                raise InfrastructureError("Schema v1 resources must be directories")
            records.append({**item.model_dump(), "checksum": digest(manifest(source))})
        return records

    def command(self, name: str, script: str, extra_mounts: tuple = ()) -> list[str]:
        cfg = self.config
        args = [
            "docker",
            "run",
            "--rm",
            "--name",
            name,
            "--pull=never",
            "--init",
            "--network=none",
            "--read-only",
            "--cap-drop=ALL",
            "--security-opt=no-new-privileges",
            "--user=65532:65532",
            f"--cpus={cfg.cpus}",
            f"--memory={cfg.memory_mb}m",
            f"--memory-swap={cfg.memory_mb}m",
            f"--pids-limit={cfg.pids_limit}",
            "--ulimit=nofile=256:256",
            "--ulimit=fsize=16777216:16777216",
            "--tmpfs=/tmp:rw,nosuid,nodev,noexec,size=256m,mode=1777",
            "--workdir=/workspace/" + self.task.package,
        ]
        mounts = [
            (self.baseline, "/workspace", "ro"),
            (self.task_file, "/task/task.json", "ro"),
        ]
        if self.task.reproduction_script:
            reproducer_file = self.task_file.parent / self.task.reproduction_script
            if not reproducer_file.is_file() or reproducer_file.is_symlink():
                raise InfrastructureError(
                    "Missing or unsafe read-only reproduction script"
                )
            mounts.append((reproducer_file, "/task/" + reproducer_file.name, "ro"))
        mounts += [
            (self.workspace / rel, "/workspace/" + rel, "rw") for rel in cfg.writable
        ]
        if cfg.test_policy == "read_only_existing_new_test_files":
            # Pin every pre-existing top-level entry as a read-only mount. Existing
            # subdirectories are entire read-only trees, preventing ancestor renames
            # from hiding protected files. New test files/directories remain writable
            # at the test root, without altering the immutable baseline.
            for rel in self.task.test_paths:
                if rel not in cfg.writable:
                    continue
                for entry in sorted((self.baseline / rel).iterdir()):
                    mounts.append((entry, "/workspace/" + rel + "/" + entry.name, "ro"))
        for item in self.resource_record():
            mounts.append((self.resources / item["resource"], item["target"], "ro"))
        mounts.extend(
            extra_mounts
        )  # Trusted evaluator only; never exposed as tool args.
        for source, target, mode in mounts:
            if any(c in str(source) for c in (",", "\n")):
                raise InfrastructureError("Invalid mount path")
            spec = f"type=bind,source={source},target={target}"
            args += ["--mount", spec + (",readonly" if mode == "ro" else "")]
        args += [
            "--entrypoint=/usr/bin/env",
            self.image,
            "-i",
            "PATH=/opt/venv/bin:/usr/local/bin:/usr/bin:/bin",
            "HOME=/tmp",
            "OPENAI_API_KEY=sk-offline-placeholder",
            "PYTHONDONTWRITEBYTECODE=1",
            "PYTHONPATH=/workspace/libs/partners/openai:/workspace/libs/core:"
            "/workspace/libs/standard-tests",
            "PYTEST_ADDOPTS=-p no:cacheprovider --no-cov",
            "TIKTOKEN_CACHE_DIR=/opt/tiktoken",
            "LC_ALL=C.UTF-8",
            "/bin/sh",
            "-c",
            script,
        ]
        return args

    def execute(self, script: str, timeout: float, extra_mounts: tuple = ()) -> dict:
        """Kill the entire command container on timeout or excessive output."""
        if timeout <= 0:
            raise TimeoutError("Workflow deadline reached")
        name = "economics-" + uuid.uuid4().hex
        started = time.monotonic()
        args = self.command(name, script, extra_mounts)
        with tempfile.TemporaryFile() as output:
            process = subprocess.Popen(
                args, stdout=output, stderr=output, env=host_env()
            )
            reason = None
            try:
                while process.poll() is None:
                    if time.monotonic() - started >= timeout:
                        reason = "tool_timeout"
                        break
                    if os.fstat(output.fileno()).st_size > 2_000_000:
                        reason = "tool_output_limit"
                        break
                    time.sleep(0.05)
            finally:
                if process.poll() is None:
                    subprocess.run(
                        ["docker", "rm", "-f", name],
                        capture_output=True,
                        timeout=10,
                        env=host_env(),
                    )
                    process.kill()
                    process.wait(timeout=10)
            output.seek(0)
            data = output.read(2_000_001).decode(errors="replace")
        if process.returncode == 125:
            raise InfrastructureError("Container launch failed: " + data[-2000:])
        return {
            "command": script,
            "exit_code": process.returncode,
            "elapsed_seconds": time.monotonic() - started,
            "termination": reason,
            "output": data,
            "output_truncated": len(data) > 2_000_000,
        }

    def isolation_checks(self, timeout: float = 30, execute=None) -> dict:
        """Executed before model context exists, against the actual runtime."""
        writable = "/workspace/" + self.config.writable[0]
        # Host sentinels prove absence rather than confusing nonexistent paths with denial.
        with tempfile.TemporaryDirectory(prefix="economics-host-secret-") as tmp:
            secret = Path(tmp) / "secret.txt"
            secret.write_text("host-only-isolation-sentinel")
            script = """python - <<'PY'
import json, os, pathlib, socket, subprocess
results = {}
def check(name, predicate):
    results[name] = bool(predicate())
check('allowed_read', lambda: pathlib.Path('/workspace/libs/partners/openai/langchain_openai/__init__.py').is_file())
p = pathlib.Path(WRITABLE) / 'sandbox_probe.py'
p.write_text('pass\\n'); check('allowed_write', lambda: p.read_text() == 'pass\\n'); p.unlink()
check('credentials_absent', lambda: os.environ.get('OPENAI_API_KEY') == 'sk-offline-placeholder' and not any(k in os.environ for k in ['OPENROUTER_API_KEY','AWS_SECRET_ACCESS_KEY','GITHUB_TOKEN']))
def inaccessible(path):
    try: pathlib.Path(path).read_bytes(); return False
    except (OSError, PermissionError): return True
check('host_file_inaccessible', lambda: inaccessible(HOST_SECRET))
link = pathlib.Path(WRITABLE) / 'sandbox_probe_link'
try:
    link.symlink_to(HOST_SECRET)
    check('host_symlink_inaccessible', lambda: inaccessible(link))
finally: link.unlink(missing_ok=True)
check('hidden_tests_absent', lambda: not pathlib.Path('/evaluation').exists())
check('controller_absent', lambda: not pathlib.Path('/agent/controller.py').exists())
check('git_history_absent', lambda: not list(pathlib.Path('/workspace').rglob('.git')))
check('git_unusable', lambda: subprocess.run(['sh','-c','git log -1'], capture_output=True).returncode != 0)
def read_only():
    try: pathlib.Path('/workspace/pyproject-sandbox-probe').write_text('bad'); return False
    except OSError: return True
check('outside_allowed_paths_read_only', read_only)
def cannot_write(path):
    try:
        with open(path, 'a') as handle: handle.write('bad')
        return False
    except OSError: return True
check('task_instructions_read_only', lambda: cannot_write('/task/task.json'))
check('dependencies_read_only', lambda: cannot_write('/opt/venv/sandbox_probe'))
if pathlib.Path('/task/reproduce.py').exists():
    check('reproduction_script_read_only', lambda: cannot_write('/task/reproduce.py'))
def offline():
    for address in [('1.1.1.1',443),('host.docker.internal',80)]:
        try:
            s=socket.create_connection(address, timeout=1); s.close(); return False
        except OSError: pass
    return True
check('outbound_network_denied', offline)
print(json.dumps(results))
assert all(results.values()), results
PY""".replace("WRITABLE", repr(writable)).replace("HOST_SECRET", repr(str(secret)))
            if self.config.test_policy == "read_only_existing_new_test_files":
                test_probe = """
tr = pathlib.Path(TEST_ROOT)
existing = next(tr.rglob('test_*.py'))
def cannot_open_for_writing():
    try:
        with existing.open('a'): pass
        return False
    except OSError: return True
check('existing_test_file_read_only', cannot_open_for_writing)
def cannot_replace_existing():
    candidate = tr / 'test_permission_probe.py'
    candidate.write_text('pass\\n')
    try:
        candidate.replace(existing)
        return False
    except OSError: return True
    finally: candidate.unlink(missing_ok=True)
check('existing_test_replacement_denied', cannot_replace_existing)
def cannot_rename_existing_directory():
    directory = next(p for p in tr.iterdir() if p.is_dir())
    moved = directory.with_name(directory.name + '_permission_probe')
    try:
        directory.rename(moved)
        moved.rename(directory)
        return False
    except OSError: return True
check('existing_test_directory_rename_denied', cannot_rename_existing_directory)
def new_regression_writable():
    candidate = tr / 'test_permission_probe.py'
    try:
        candidate.write_text('pass\\n')
        return candidate.read_text() == 'pass\\n'
    finally: candidate.unlink(missing_ok=True)
check('new_regression_file_writable', new_regression_writable)
""".replace("TEST_ROOT", repr("/workspace/" + self.task.test_paths[0]))
                script = script.replace(
                    "print(json.dumps(results))",
                    test_probe + "\nprint(json.dumps(results))",
                )
            result = (execute or self.execute)(script, timeout)
            if result["exit_code"]:
                raise InfrastructureError("Isolation probe failed: " + result["output"])
            return result
