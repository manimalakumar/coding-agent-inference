Task ID: LC-INT-TEMPERATURE-001
Title: Preserve integer-temperature tracing metadata

The supplied pre-fix LangChain snapshot drops integer temperature values from
language-model tracing metadata. Preserve supported numeric temperature values
for chat and completion models, including zero, without changing unrelated
metadata behavior. No model invocation or external service is needed.

Your first command must be exactly `python /task/reproduce.py`.
Add a new regression test and observe its failure before changing source.

Production changes are limited to:
- libs/core/langchain_core/language_models/chat_models.py
- libs/core/langchain_core/language_models/llms.py

Existing tests are read-only. Add regressions only as new `test_*.py` files directly
under libs/core/tests/unit_tests. Existing subdirectories are read-only. Do not
modify dependencies, environment settings, public APIs, or unrelated packages.
Use the prepared Python environment; do not install packages.

Use only bash through the sandbox. No internet, Git history, reference code,
hidden checks, host resources, credentials or previous-attempt information is
available. Missing dependencies and test-collection failures are environment
blockers, not evidence that the issue is invalid.

Run checks from /workspace/libs/core. Set CHANGED_TESTS to the newly added test
paths and CHANGED_SOURCE to the changed production paths in each command:

1. python /task/reproduce.py
2. python -m pytest -q --disable-socket --allow-unix-socket $CHANGED_TESTS
3. python -m pytest tests/unit_tests/language_models/chat_models/test_base.py tests/unit_tests/language_models/llms/test_base.py -q --disable-socket --allow-unix-socket
4. ruff check --no-cache $CHANGED_SOURCE $CHANGED_TESTS
5. ruff format --check --no-cache $CHANGED_SOURCE $CHANGED_TESTS
6. mypy --cache-dir=/dev/null $CHANGED_SOURCE

After formatting, rerun format --check. Preserve exit codes with direct execution
or pipefail. Once checks pass, finish early with the standalone submit command.
The final workspace is captured and evaluated even at a budget/error stop;
explicit submission is separate from patch acceptance. There are no extra model
requests or repairs after termination. Acceptance is under the new public-check
protocol; the historical private verifier is unavailable. Passing checks requires
human review and authorizes no commit, push or merge.
