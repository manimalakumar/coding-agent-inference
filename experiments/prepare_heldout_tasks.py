"""Trusted preparation of two held-out tasks; never imported by evaluated agents.

Harness policy is inherited byte-for-byte except task resources/provenance. Reference
implementations and added upstream tests are stored outside the experiment project.
"""

import ast
import json
import shutil
import subprocess
import time
from pathlib import Path

from agent.schemas import Task
from experiments.configuration_paths import configuration_path
from sandbox.prepare import export_snapshot

ROOT = Path(__file__).resolve().parents[1]
TRUST = ROOT / ".local/resources/held-out-v1"
REFERENCE = ROOT / ".local/upstream/langchain"
LABEL = "held-out-two-task-v1"
TASKS = {
    "LC-ASYNC-TOOLS-001": {
        "issue": 40205,
        "pr": 40208,
        "family": "feature",
        "package": "libs/partners/openai",
        "sources": [
            "libs/partners/openai/langchain_openai/chat_models",
            "libs/core/langchain_core/messages/block_translators",
        ],
        "tests": [
            "libs/partners/openai/tests/unit_tests",
            "libs/core/tests/unit_tests",
        ],
        "existing": [
            ["libs/partners/openai", "tests/unit_tests/chat_models/test_base.py"],
            ["libs/core", "tests/unit_tests/messages/block_translators/test_openai.py"],
        ],
        "expected": "ASYNC_DECLARATION_MISSING",
        "requirements": """Support OpenAI Responses asynchronous tool declarations and preserve asynchronous call metadata. This is the provider's asynchronous tool-calling feature, not Python async functions or background execution. Forward the known async extra from @tool(extras={\"async\": True}) to Responses declarations. Preserve True, False and omission through non-streamed/streamed native function-call content, conversion to standard v1 content blocks and replay into Responses input. Preserve original call IDs and existing unrelated extras. Handle SDK reserved-word field naming. Keep Chat Completions free of unsupported Responses-only async fields. Preserve existing behavior when no async flag is supplied. No new execution scheduler or dependencies.""",
        "reproduce": '''from unittest.mock import Mock
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
@tool(extras={"async": True})
def lookup(query: str) -> str:
    """Look up a value."""
    return query
model = ChatOpenAI(model="offline-test-model", api_key="offline-placeholder", use_responses_api=True, client=Mock(), async_client=Mock(), root_client=Mock(), root_async_client=Mock())
bound = model.bind_tools([lookup])
assert bound.kwargs["tools"][0].get("async") is True, "ASYNC_DECLARATION_MISSING"
print("Async declaration preserved")
''',
    },
    "LC-SUMMARY-OVERFLOW-001": {
        "issue": 41155,
        "pr": 41159,
        "family": "bug_fix_error_handling",
        "package": "libs/langchain_v1",
        "sources": ["libs/langchain_v1/langchain/agents/middleware"],
        "tests": ["libs/langchain_v1/tests/unit_tests"],
        "existing": [
            [
                "libs/langchain_v1",
                "tests/unit_tests/agents/middleware/implementations/test_summarization.py",
            ]
        ],
        "expected": "IDENTICAL_OVERFLOW_RETRIES",
        "requirements": """Fix SummarizationMiddleware retry behavior. ContextOverflowError must not cause repeated identical requests. Recover with at most one further attempt using a reduced summary input; if that also overflows or no reduced input is possible, propagate the error. Preserve useful beginning and recent conversation context in the reduced input. Apply equivalent synchronous and asynchronous behavior. Do not mutate shared trim_tokens_to_summarize settings or original messages. Preserve normal summaries, empty-input behavior, call metadata and XML serialization. Continue retrying transient failures up to the existing limit; non-retryable ModelError failures must propagate promptly. No dependencies or unrelated API changes.""",
        "reproduce": """from typing import Any
from langchain.agents.middleware.summarization import SummarizationMiddleware
from langchain_core.exceptions import ContextOverflowError
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_core.outputs import ChatResult
class Overflowing(BaseChatModel):
    prompts: list[str] = []
    @property
    def _llm_type(self) -> str:
        return "offline-overflow-probe"
    def _generate(self, messages: Any, stop: Any = None, run_manager: Any = None, **kwargs: Any) -> ChatResult:
        self.prompts.append(messages[-1].text)
        raise ContextOverflowError("prompt is too long")
model = Overflowing()
middleware = SummarizationMiddleware(model=model, trim_tokens_to_summarize=None)
middleware._summary_model.wait_exponential_jitter = False
try:
    middleware._create_summary([HumanMessage(content="first " * 1000), HumanMessage(content="last " * 1000)])
except ContextOverflowError:
    pass
assert len(model.prompts) <= 2 and len(set(model.prompts)) == len(model.prompts), "IDENTICAL_OVERFLOW_RETRIES"
assert len(model.prompts) == 2 and len(model.prompts[1]) < len(model.prompts[0]), "REDUCED_RECOVERY_MISSING"
print("Overflow avoids identical retries and reduces input")
""",
    },
}


def git(*args):
    return subprocess.check_output(["git", "-C", str(REFERENCE), *args], text=True)


def write(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def prepare():
    parent = json.loads(
        (
            configuration_path(
                ROOT,
                "configs/context-counter-repaired-calibration-v1.frozen-setting.json",
            )
        ).read_text()
    )
    for task_id, spec in TASKS.items():
        state = TRUST / task_id
        if (state / "export.json").exists():
            continue
        started = time.time()
        pr = json.loads((TRUST / f"reference-pr-{spec['pr']}.json").read_text())[
            "pull_request"
        ]
        # REST base_sha can predate a required compatibility release. Export the
        # actual first parent of the merged fix, not a stale PR base metadata SHA.
        commit = git("rev-parse", pr["merge_commit_sha"] + "^1").strip()
        task = Task(
            task_id=task_id,
            commit=commit,
            package=spec["package"],
            instructions=spec["requirements"],
            source_paths=spec["sources"],
            test_paths=spec["tests"],
            development_checks=[],
            reproduction_command="python /task/reproduce.py",
            reproduction_script="reproduce.py",
            expected_failure=spec["expected"],
            workspace_imports=["langchain_core", "langchain_openai"]
            if spec["issue"] == 40205
            else ["langchain", "langchain_core"],
        )
        state.mkdir(parents=True, exist_ok=True)
        provenance = export_snapshot(REFERENCE, task, state / "baseline")
        write(state / "provenance.json", json.dumps(provenance, indent=2))
        write(state / "reproduce.py", spec["reproduce"])
        source_changes = {}
        hidden = []
        for name in git(
            "diff", "--name-only", commit, pr["merge_commit_sha"]
        ).splitlines():
            if any(name.startswith(s + "/") for s in spec["sources"]):
                source_changes[name] = git("show", pr["merge_commit_sha"] + ":" + name)
            elif "/tests/" in name and name.endswith(".py"):
                before = ast.parse(git("show", commit + ":" + name))
                previous = {
                    n.name
                    for n in before.body
                    if isinstance(
                        n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
                    )
                }
                text = git("show", pr["merge_commit_sha"] + ":" + name)
                tree = ast.parse(text)
                new = [
                    n
                    for n in tree.body
                    if isinstance(
                        n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
                    )
                    and n.name not in previous
                ]
                for n in new:
                    hidden.append(
                        ast.get_source_segment(text, n)
                        if not getattr(n, "decorator_list", [])
                        else "\n".join(
                            text.splitlines()[
                                min(d.lineno for d in n.decorator_list)
                                - 1 : n.end_lineno
                            ]
                        )
                    )
        write(state / "reference-source.json", json.dumps(source_changes, indent=2))
        write(state / "upstream-added-tests.txt", "\n\n".join(hidden))
        context = state / "dependency-build"
        shutil.copytree(state / "baseline", context / "source")
        recipe = f"""FROM ghcr.io/astral-sh/uv:0.11.29 AS uv
FROM python:3.14.6-slim-bookworm
COPY --from=uv /uv /usr/local/bin/uv
COPY source /workspace
WORKDIR /workspace/{spec["package"]}
ENV UV_PROJECT_ENVIRONMENT=/opt/venv UV_PYTHON_DOWNLOADS=never
RUN uv sync --frozen --group test --group lint --group typing --group test_integration
RUN uv pip freeze --python /opt/venv/bin/python > /opt/requirements.lock && /opt/venv/bin/python -c 'import importlib.metadata as m,json,sys; print(json.dumps({{"python":sys.version,"packages":sorted((d.metadata["Name"],d.version) for d in m.distributions())}},indent=2))' > /opt/environment.json
ENV PATH=/opt/venv/bin:/usr/local/bin:/usr/bin:/bin HOME=/tmp PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/workspace/{spec["package"]}:/workspace/libs/core:/workspace/libs/standard-tests:/workspace/libs/partners/openai
ENV PYTEST_ADDOPTS="-p no:cacheprovider --no-cov"
USER 65532:65532
"""
        write(context / "Dockerfile", recipe)
        record = {
            "task_id": task_id,
            "spec": spec,
            "commit": commit,
            "rest_base_sha": pr["base_sha"],
            "baseline_checksum": provenance["baseline_checksum"],
            "trusted_merge_commit": pr["merge_commit_sha"],
            "setup_started_at": started,
            "model_requests": 0,
            "parent_harness_setting_id": parent["setting_id"],
            "status": "exported_not_validated",
        }
        write(state / "export.json", json.dumps(record, indent=2))
        print(task_id, commit, provenance["baseline_checksum"], flush=True)


if __name__ == "__main__":
    prepare()
