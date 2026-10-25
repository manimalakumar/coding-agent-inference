Task ID: LC-CHROMA-NONE-001
Title: Handle missing document content in Chroma retrieval
Difficulty: Level 2

You are working in a supplied, unchanged historical LangChain source snapshot. Use only this workspace.

## Problem

Chroma-backed retrieval can raise a Pydantic validation error when a stored record has no document content (None). Public retriever invocation and other document-returning retrieval operations should return only valid documents rather than fail on such records.

Skip records whose document content is None. Preserve valid content, including the empty string, result order, metadata (with the existing empty-metadata default), document IDs, associated similarity scores and embedding vectors. Ensure consistent handling for scored search, vector-returning search and ID-based retrieval. Existing normal retrieval behavior must continue to work. Do not mutate the backend result payload.

## Reproduction and environment

Your first sandbox command must be exactly:

`python /task/reproduce.py`

This read-only script exercises retrieval with fabricated backend results offline and demonstrates the validation error. Add a regression test and observe it fail before changing production code. Preserve the regression when fixing the implementation.

Use the prepared Python 3.13.16 environment, Pydantic 2.12.5 and frozen dependencies. Imports must resolve to supplied workspace source. Do not install or upgrade packages. Use fake embeddings and mocked backend results. No model download, live service, network access or GPU is required.

A missing dependency, import failure or test-collection failure is an environment blocker, not evidence that the issue is invalid.

## Allowed changes

Production: `libs/partners/chroma/langchain_chroma/`
Development tests: `libs/partners/chroma/tests/unit_tests/`

Keep changes focused on handling of absent Chroma document content. Existing test files are read-only. Add regressions only in new `test_*.py` files at `tests/unit_tests/` or inside a newly created subdirectory beneath it. Existing test subdirectories are read-only. Do not modify, replace, rename, delete or format existing tests. Do not change dependency manifests, lockfiles, environment settings, unrelated packages or public APIs.

## Access restrictions and workflow

Use only the supplied sandbox tools. Do not access GitHub, Git history, upstream solutions, reference repositories, hidden acceptance tests, evaluator internals or host resources. Sandbox commands remain offline.

The available tool is bash. Create files through bash; use directory creation only for directories. Batch related searches and file reads. Use reproduction evidence and traceback locations to locate relevant code. Once you have enough evidence, make a focused change and test it.

After a failed patch test, inspect the failure and the changed execution path before further broad searches. Track remaining budget and budget calls for implementation and verification.

## Required completion checklist and commands

Run development checks from `/workspace/libs/partners/chroma`. Choose CHANGED_SOURCE and CHANGED_TESTS from the production files you changed and regression files you added. Shell variables do not persist between tool actions.

1. Execute the supplied reproducer before changing source.
2. Run your new regression tests and observe the reported failure.
3. Implement the fix.
4. Require the supplied reproducer and new regression tests to pass:
   `python /task/reproduce.py`
   `python -m pytest -q --disable-socket --allow-unix-socket $CHANGED_TESTS`
5. Run the existing focused module:
   `python -m pytest tests/unit_tests -q --disable-socket --allow-unix-socket`
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

