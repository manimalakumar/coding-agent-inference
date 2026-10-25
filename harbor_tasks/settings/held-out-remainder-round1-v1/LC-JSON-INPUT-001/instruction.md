Task ID: LC-JSON-INPUT-001
Task family: bug_fix

Use only the supplied unchanged pre-fix snapshot.

## Required behavior

RecursiveJsonSplitter.split_json must reject unsupported non-dict top-level values with TypeError rather than silently discarding them. Top-level lists require convert_lists=True; the error for a bare list should name its type and explain that option. With conversion enabled, lists must continue to produce the existing converted dictionary chunks. Reject other scalars (str/int/bool etc.) with their type in the message; do not suggest list conversion when it cannot help. Preserve None -> [] with or without conversion, {} -> [], valid dictionary splitting, nested empty values and chunk-boundary behavior. Entry points delegating to split_json must inherit validation. Keep existing signatures and dependencies. Use the prepared package-native strict type checker, ty check (all rules error), for changed production files.

## Reproduction and environment

Your first sandbox command must be exactly: `python /task/reproduce.py`.
This offline probe demonstrates the reported missing behavior. Add new regression tests and observe failure before implementation. Preserve them in the final workspace. Use prepared Python 3.14.6 and exact frozen dependencies. The target package and langchain_core must import from this workspace. Do not install or upgrade packages. Use fake models, deterministic counters and fabricated messages; no live inference or network request is required.

An import, dependency or collection failure is an environment blocker, not evidence against the feature request.

## Allowed changes

Production:
- `libs/text-splitters/langchain_text_splitters/`

Development tests:
- `libs/text-splitters/tests/unit_tests/test_*.py` (new direct-child files only)

Existing tests are read-only. Do not modify, rename, delete or format them. No new test subdirectories or __init__.py files. Do not change dependency manifests, lockfiles, environment configuration or unrelated public APIs.

## Access restrictions and workflow

Use only the supplied sandbox tools. Do not access GitHub, Git history, upstream solutions, reference repositories, hidden acceptance tests, evaluator internals or host resources. Sandbox commands remain offline.

The available tool is bash. Create files through bash; use directory creation only for directories. Batch related searches and file reads. Use reproduction evidence and traceback locations to locate relevant code. Once you have enough evidence, make a focused change and test it.

After a failed patch test, inspect the failure and the changed execution path before further broad searches. Track remaining budget and budget calls for implementation and verification.

## Required completion checklist

Run `python /task/reproduce.py` first. Add a regression test at the permitted root and observe pytest exit 1 before implementing. Preserve it in the final workspace. From the package root, run new tests with `python -m pytest -q --disable-socket --allow-unix-socket CHANGED_TESTS`, then rerun the reproducer and these existing checks:

- From `/workspace/libs/text-splitters`: `python -m pytest tests/unit_tests/test_text_splitters.py -k split_json -q --disable-socket --allow-unix-socket`

From `/workspace/libs/text-splitters`, run `ruff check --no-cache CHANGED_FILES`, `ruff format --check --no-cache CHANGED_FILES` and `ty check CHANGED_SOURCE`. Preserve package-specific lint/type configuration. Always rerun format --check after formatting. Shell variables do not persist between tool actions. Run tests directly or use set -o pipefail; inspect exit codes.

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

