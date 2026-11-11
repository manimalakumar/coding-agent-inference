"""Let the existing Harbor launcher use the HTTP callable shown in the notebook.

Only the launcher CLI dispatch and native client's transport are scoped here.
The agent loop, budgets, request trace, capture and evaluator are unchanged.
"""

import os
import re
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from types import SimpleNamespace

from public_preparation import launch as public_launch

_LOCK = threading.Lock()


@contextmanager
def transport_scope(post):
    """Replace the native module's transport only, never global requests.post."""
    from minisweagent.models import openrouter_model

    original = openrouter_model.requests
    original_stats = openrouter_model.GLOBAL_MODEL_STATS
    # Match a fresh CLI process: global native limits/stats start per attempt.
    openrouter_model.GLOBAL_MODEL_STATS = type(original_stats)()
    openrouter_model.requests = SimpleNamespace(
        post=post, exceptions=original.exceptions
    )
    try:
        yield
    finally:
        openrouter_model.requests = original
        openrouter_model.GLOBAL_MODEL_STATS = original_stats


@contextmanager
def kernel_signals():
    """The kernel owns process signals; an embedded CLI cannot replace them."""
    if threading.current_thread() is threading.main_thread():
        yield
        return
    from harbor.cli import jobs

    original = jobs.signal
    jobs.signal = SimpleNamespace(
        SIGTERM=original.SIGTERM,
        signal=lambda signum, handler: original.getsignal(signum),
    )
    try:
        yield
    finally:
        jobs.signal = original


def run_harbor(argv, env, post):
    """Use the same CLI in a worker so Harbor owns its asyncio event loop."""
    previous = dict(os.environ)
    try:
        os.environ.update(env)
        # Keep this local notebook session from emitting Harbor telemetry.
        os.environ["HARBOR_TELEMETRY"] = "0"
        from harbor.cli.main import app

        with transport_scope(post), kernel_signals():
            try:
                returned = app(args=argv[1:], standalone_mode=False)
                code = returned if isinstance(returned, int) else 0
            except SystemExit as exc:
                if exc.code is not None and not isinstance(exc.code, int):
                    raise RuntimeError(
                        "Harbor CLI exited without a numeric status"
                    ) from exc
                code = exc.code or 0
        return subprocess.CompletedProcess(argv, code)
    finally:
        # Harbor loads the host credential file into its process environment.
        # Do not leave that credential behind in the interactive kernel.
        os.environ.clear()
        os.environ.update(previous)


def launch(root, task_id, *, model, confirmed, job_name, post, credentials=None):
    """Call the validated public launcher; bind its model requests to notebook code."""
    if confirmed is not True:
        raise ValueError("Live inference requires explicit confirmation")
    if os.environ.get("PUBLIC_NO_INFERENCE") == "1":
        raise RuntimeError("Model calls are disabled by PUBLIC_NO_INFERENCE")
    if not callable(post):
        raise TypeError("Supply the notebook's OpenAI-compatible HTTP callable")
    try:
        installed = (version("harbor"), version("mini-swe-agent"))
    except PackageNotFoundError as exc:
        raise RuntimeError("Select the live kernel described in setup.md") from exc
    if installed != ("0.24.0", "2.4.6"):
        raise RuntimeError("Live kernel differs from the pinned Harbor/agent versions")
    root = Path(root).resolve()
    if Path.cwd().resolve() != root:
        raise RuntimeError("Launch the notebook from the repository root")
    if not _LOCK.acquire(blocking=False):
        raise RuntimeError(
            "Another notebook launch is active; run attempts sequentially"
        )
    original = public_launch.subprocess
    try:
        with ThreadPoolExecutor(max_workers=1) as worker:

            def dispatch(argv, **kwargs):
                if Path(argv[0]) == root / ".venv/bin/harbor":
                    if argv[1:3] != ["run", "-c"] or kwargs.get("cwd") != root:
                        raise RuntimeError("Unexpected public launcher command")
                    return worker.submit(run_harbor, argv, kwargs["env"], post).result()
                return original.run(argv, **kwargs)

            public_launch.subprocess = SimpleNamespace(run=dispatch)
            return public_launch.launch(
                root,
                task_id,
                model=model,
                confirmed=confirmed,
                job_name=job_name,
                credentials=credentials,
            )
    finally:
        public_launch.subprocess = original
        _LOCK.release()


def check_batch(root, plan):
    """Check every task and attempt ID before spending on the first batch attempt."""
    names = [row["attempt_id"] for row in plan]
    if not names or len(names) != len(set(names)):
        raise ValueError("Choose a nonempty plan with unique attempt IDs")
    for row in plan:
        name = row["attempt_id"]
        # Match the existing launcher's attempt-name contract before any inference.
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,100}", name):
            raise ValueError("Choose a valid explicit batch ID")
        if row["model"] not in {"minimax", "glm5"}:
            raise ValueError("Use a pinned model: minimax or glm5")
        if (root / "runs/public-models" / name).exists() or (
            root / ".local/public-launches" / (name + ".json")
        ).exists():
            raise FileExistsError("Attempt already exists: " + name)
    return {
        task: public_launch.readiness(root, task)
        for task in sorted({row["task_id"] for row in plan})
    }
