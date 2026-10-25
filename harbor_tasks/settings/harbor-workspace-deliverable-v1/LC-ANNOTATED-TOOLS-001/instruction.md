Task ID: LC-ANNOTATED-TOOLS-001
Title: Preserve Annotated parameter descriptions with postponed annotations

You are working in a supplied, unchanged historical LangChain source snapshot. Use only this workspace.

## Problem

The public @tool interface loses parameter descriptions supplied through typing.Annotated when a function uses postponed (string) annotations, for example with `from __future__ import annotations`.

A function such as:

```python
from __future__ import annotations
from typing import Annotated
from langchain_core.tools import tool

@tool
def choose_fragments(parts: Annotated[list[str], "Fragments to evaluate"]) -> list[str]:
    """Return the requested fragments."""
    return parts
```

should expose a `parts` property with description `Fragments to evaluate`, type `array` and string items. The supplied snapshot omits the description. Tool invocation must keep working.

Preserve descriptions and parameter types for supported eager and postponed annotations, including supported typing.Annotated metadata. Preserve existing tool-schema behavior, including explicit parsed docstring descriptions and parameters without description metadata. Do not mutate the caller's annotations. No model service is involved in this bug.

## Reproduction and environment

Your first sandbox command must be exactly:

`python /task/reproduce.py`

This read-only script exercises the existing tool schema generation offline and asserts the expected metadata. On unchanged source it demonstrates the reported missing parameter description. Add a regression test and observe it fail before changing production code. Preserve the test when implementing the fix.

Use the prepared Python 3.13.16 environment and frozen dependencies. Imports must resolve to the supplied workspace source. Do not install or upgrade packages. No live API request, model download or GPU is required. Use fabricated response data or mocked clients. Do not contact external services.

A missing dependency, import failure or test-collection failure is an environment blocker, not evidence that the issue is invalid. If blocked, distinguish an environment failure, failure to reproduce, or missing task information.

## Allowed changes

Production: `libs/core/langchain_core/tools/`
Development tests: `libs/core/tests/unit_tests/`

Keep changes focused on tool argument descriptions. Existing test files are read-only. Add regressions only in new `test_*.py` files at `tests/unit_tests/` or inside a newly created subdirectory beneath it. Existing test subdirectories are read-only. Do not modify, replace, rename, delete or format existing tests. Do not change dependency manifests, lockfiles, environment settings, unrelated packages or public APIs.

## Access restrictions and workflow

Use only the supplied sandbox tools. Do not access GitHub, Git history, upstream solutions, reference repositories, hidden acceptance tests, evaluator internals or host resources. Sandbox commands remain offline.

The available tool is bash. Create files through bash; use directory creation only for directories. Batch related searches and file reads. Use reproduction evidence and traceback locations to locate relevant code. Once you have enough evidence, make a focused change and test it.

After a failed patch test, inspect the failure and the changed execution path before further broad searches. Track remaining budget and budget calls for implementation and verification.

## Required completion checklist and commands

Run development checks from `/workspace/libs/core`. Choose CHANGED_SOURCE and CHANGED_TESTS from the production files you changed and regression files you added. Shell variables do not persist between tool actions.

1. Execute the supplied reproducer before changing source.
2. Run your new regression tests and observe the reported failure.
3. Implement the fix.
4. Require the supplied reproducer and new regression tests to pass:
   `python /task/reproduce.py`
   `python -m pytest -q --disable-socket --allow-unix-socket $CHANGED_TESTS`
5. Run the existing focused module:
   `python -m pytest tests/unit_tests/test_tools.py tests/unit_tests/utils/test_function_calling.py -q --disable-socket --allow-unix-socket`
6. Run lint:
   `ruff check --no-cache $CHANGED_SOURCE $CHANGED_TESTS`
7. Verify formatting:
   `ruff format --check --no-cache $CHANGED_SOURCE $CHANGED_TESTS`
   If you run a formatter, rerun format --check afterward and require exit code zero.
8. Run type checking on changed production files:
   `mypy --cache-dir=/tmp/mypy $CHANGED_SOURCE`

Run tests directly where practical. For pipelines, use `set -o pipefail` so a failing test command cannot appear successful. Bash pipefail is also enabled by the harness.

## Workspace delivery and stopping

The final workspace is your deliverable. Harbor automatically captures it when you
submit, exhaust a budget, or stop with an execution error. Trusted evaluation then
runs on that frozen candidate in a separate environment. Task success means all
required functional, regression, lint, format, type and file-policy checks pass.
Explicit submission is recorded separately and is not required for patch acceptance.

No model request or output tokens are reserved for submission. The last permitted
request may edit source or execute development checks. Budget exhaustion stops
agent execution; there are no extra model calls or repairs afterward. Tool actions
authorized by the final allowed response may execute within the remaining time.
The agent's limits remain 30 model requests, 10,000 cumulative output tokens and
20 minutes; each response is capped at 1,500 tokens and each command at 90 seconds.
Malformed tool-response recovery is limited to two retries within these same budgets.
Trusted evaluation has a separate five-minute limit after capture and grants no
agent actions, inference or repairs.

Once required checks pass, finish early to save inference. The standalone bash
command `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT` remains available for that
purpose. Do not combine it with other commands. The agent cannot continue after
submission. Passing checks means checks_passed_pending_review, with no authorization
to commit, push, publish or merge.

Run tests directly where practical. For pipelines use `set -o pipefail`; the harness
also enables it. Always run format --check after formatting and inspect exit codes.

