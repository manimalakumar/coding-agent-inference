Task ID: LC-CONTEXT-COUNTER-001
Title: Support a custom token counter in ContextEditingMiddleware
Task type: Feature request

Use only the supplied unchanged source snapshot.

## Feature request and required API

Add this optional keyword-only constructor parameter, using the existing TokenCounter type:

ContextEditingMiddleware(..., token_counter: TokenCounter | None = None)

The middleware currently selects approximate counting or model counting internally. Approximate counting can under-count CJK-heavy conversations, and compatible providers may not implement model token counting. Allow callers to supply a synchronous custom message counter, including functools.partial(count_tokens_approximately, chars_per_token=2.0).

When supplied, token_counter takes precedence over token_count_method for context-edit triggers and subsequent counting performed by edits. Support synchronous and asynchronous model-call wrappers consistently. Preserve the configured counter and honor callable objects regardless of their truth value. When omitted or None, retain the existing approximate and model counting behavior, including handling of system messages and tools in model mode. Preserve empty-conversation behavior, edit ordering, original request immutability, normal handler results and exception propagation. Document the new parameter and precedence. No new dependencies or unrelated API changes.

## Reproduction and environment

Your first sandbox command must be exactly: `python /task/reproduce.py`.
This offline feature probe demonstrates the missing constructor API. Add new regression tests and observe failure before implementation. Preserve them in the final workspace. Use prepared Python 3.14.6 and exact frozen dependencies. langchain and langchain_core must import from this workspace. Do not install or upgrade packages. Use fake models, deterministic counters and fabricated messages; no live inference or network request is required.

An import, dependency or collection failure is an environment blocker, not evidence against the feature request.

## Allowed changes

Production: `libs/langchain_v1/langchain/agents/middleware/`
Development tests: `libs/langchain_v1/tests/unit_tests/`

Keep changes focused on this custom token-counter feature. Existing test files are read-only. Add regressions only in new `test_*.py` files at `tests/unit_tests/` or inside a newly created subdirectory beneath it. Existing test subdirectories are read-only. Do not modify, replace, rename, delete or format existing tests. Do not change dependency manifests, lockfiles, environment settings, unrelated packages or public APIs beyond the requested optional parameter.

## Access restrictions and workflow

Use only the supplied sandbox tools. Do not access GitHub, Git history, upstream solutions, reference repositories, hidden acceptance tests, evaluator internals or host resources. Sandbox commands remain offline.

The available tool is bash. Create files through bash; use directory creation only for directories. Batch related searches and file reads. Use reproduction evidence and traceback locations to locate relevant code. Once you have enough evidence, make a focused change and test it.

After a failed patch test, inspect the failure and the changed execution path before further broad searches. Track remaining budget and budget calls for implementation and verification.

## Required completion checklist and commands

Run development checks from `/workspace/libs/langchain_v1`. Choose CHANGED_SOURCE and CHANGED_TESTS from the production files you changed and regression files you added. Shell variables do not persist between tool actions.

1. Execute the supplied reproducer before changing source.
2. Run your new regression tests and observe the reported failure.
3. Implement the fix.
4. Require the supplied reproducer and new regression tests to pass:
   `python /task/reproduce.py`
   `python -m pytest -q --disable-socket --allow-unix-socket $CHANGED_TESTS`
5. Run the existing focused module:
   `python -m pytest tests/unit_tests/agents/middleware/implementations/test_context_editing.py -q --disable-socket --allow-unix-socket`
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

