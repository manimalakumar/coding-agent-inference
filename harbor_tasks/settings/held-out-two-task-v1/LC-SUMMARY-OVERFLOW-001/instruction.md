Task ID: LC-SUMMARY-OVERFLOW-001
Difficulty: L2
Task family: bug_fix_error_handling

Use only the supplied unchanged pre-fix snapshot.

## Required behavior

Fix SummarizationMiddleware retry behavior. ContextOverflowError must not cause repeated identical requests. Recover with at most one further attempt using a reduced summary input; if that also overflows or no reduced input is possible, propagate the error. Preserve useful beginning and recent conversation context in the reduced input. Apply equivalent synchronous and asynchronous behavior. Do not mutate shared trim_tokens_to_summarize settings or original messages. Preserve normal summaries, empty-input behavior, call metadata and XML serialization. Continue retrying transient failures up to the existing limit; non-retryable ModelError failures must propagate promptly. No dependencies or unrelated API changes.

## Reproduction and environment

Your first sandbox command must be exactly: `python /task/reproduce.py`.
This offline probe demonstrates the reported missing behavior. Add new regression tests and observe failure before implementation. Preserve them in the final workspace. Use prepared Python 3.14.6 and exact frozen dependencies. langchain and langchain_core must import from this workspace. Do not install or upgrade packages. Use fake models, deterministic counters and fabricated messages; no live inference or network request is required.

An import, dependency or collection failure is an environment blocker, not evidence against the feature request.

## Allowed changes

Production:
- `libs/langchain_v1/langchain/agents/middleware/`

Development tests:
- `libs/langchain_v1/tests/unit_tests/test_*.py` (new direct-child files only)

Existing tests are read-only. Do not modify, rename, delete or format them. No new test subdirectories or __init__.py files. Do not change dependency manifests, lockfiles, environment configuration or unrelated public APIs.

## Access restrictions and workflow

Use only the supplied sandbox tools. Do not access GitHub, Git history, upstream solutions, reference repositories, hidden acceptance tests, evaluator internals or host resources. Sandbox commands remain offline.

The available tool is bash. Create files through bash; use directory creation only for directories. Batch related searches and file reads. Use reproduction evidence and traceback locations to locate relevant code. Once you have enough evidence, make a focused change and test it.

After a failed patch test, inspect the failure and the changed execution path before further broad searches. Track remaining budget and budget calls for implementation and verification.

## Required completion checklist

Run the public reproducer first: `python /task/reproduce.py`. Add a regression and observe pytest exit 1 before editing source. Run new regressions from their package root with `python -m pytest -q --disable-socket --allow-unix-socket PATH`. Use unique new test filenames across roots.

Existing checks:
- From `/workspace/libs/langchain_v1`: `python -m pytest tests/unit_tests/agents/middleware/implementations/test_summarization.py -q --disable-socket --allow-unix-socket`

From each changed package root, run `ruff check --no-cache CHANGED_FILES`, `ruff format --check --no-cache CHANGED_FILES` and `mypy --cache-dir=/dev/null CHANGED_SOURCE`. Preserve package-specific lint/type configuration. Rerun format --check after any formatting. Run tests directly or use set -o pipefail and inspect exit codes. Shell variables do not persist between tool actions.

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
20 minutes; each response is capped at 6,000 tokens and each command at 90 seconds.
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

