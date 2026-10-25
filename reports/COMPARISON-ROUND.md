# Four-task, two-model comparison — prepared and validated

**24 fresh measured runs are prepared; none have started.** Historical pilots remain exploratory records and are not reused as the three repeats in this round.

Frozen setting: `2a965fa50d5a74974aa5c8ced4009ef3c0e3d996ceb8042428354df7bfa3dfe1`.

The immutable schedule and executable per-run commands are in `configs/comparison-4x2x3-v1.plan.json`. Each command uses the existing Harbor launch CLI, which captures and evaluates automatically and records raw inference usage. Replay does not launch any command.

| Historical task | MiniMax M2.5 | GLM-5 |
| --- | ---: | ---: |
| #31698 — custom timeout objects | 3 fresh attempts | 3 fresh attempts |
| #31756 — Responses API token details | 3 fresh attempts | 3 fresh attempts |
| #31181 — postponed Annotated descriptions | 3 fresh attempts | 3 fresh attempts |
| #31377 — Chroma absent document content | 3 fresh attempts | 3 fresh attempts |

## Common execution and scoring policy

Harbor 0.24.0 + standard mini-SWE-agent 2.4.6, with the same generic workflow, bash tool schema, task-specific instructions, Git-free diff/restore helper, permissions and task bundle for both models. Python/dependencies remain task-specific compatible immutable environments, identical between models on a given task. No dependencies or source starting commits were changed.

Every attempt starts with fresh context, fresh sandbox volumes, the same verified Gitless pre-fix snapshot and the required reproduction script. Existing tests are read-only, including protection against renaming/replacing existing files. Regressions go only in new test files. The agent has no host mounts, real credentials, Git history, external network, hidden tests or evaluator internals. The inference client stays on the trusted host.

Per attempt: **30 model requests; 10,000 cumulative reported output tokens including reasoning; 1,500 tokens per response; 20 minutes agent execution; 90 seconds per tool command; two malformed-tool-response recovery retries.** No transport retries or provider fallback are enabled. Code repairs and Ruff formatting remain distinct from malformed-response recovery. Trusted evaluation has a separate five-minute limit after capture.

The workspace is the deliverable. Capture/evaluation occur after submission, budget exhaustion or error. All required functional, regression, existing-test, lint, format, type, file-policy and hidden checks must pass for task success. Explicit submission is separate telemetry. There is no submission-only final request. Passing means pending human review and gives no publishing authority.

## Setup fixes made before the measured round

- Migrated timeout and Responses-usage tasks from the older submission-required policy to workspace delivery. Removed obsolete stopping instructions and aligned the generic tool instructions and checklist.
- Gave every task the same permissions and Git-free recovery tools. Test/source permissions are enforced inside Docker.
- Preserved all original hidden acceptance files byte-for-byte; the preparation report records matching hashes.
- Added a versioned controller adapter using the earlier of monotonic/civil remaining time. A host pause or forward clock change cannot grant extra tool actions or model requests on resume; a backward clock change cannot extend the monotonic budget.
- Replaced the old clamped stop timestamp with the observed stop timestamp, separate receipt-finalization time, and both elapsed clocks. The standard mini-SWE agent loop, parser and provider client remain upstream implementations.
- Preserved all historical controllers, configurations, candidates, verdicts and costs. This new round has its own hash.

## Models and compatibility

MiniMax: `minimax/minimax-m2.5`, only `minimax/fp8`, fallback disabled, temperature 0; unsupported reasoning settings remain omitted as in its established configuration. The omission is frozen and recorded; it is not a claim that the provider internally performs no reasoning.

GLM: `z-ai/glm-5`, only `z-ai/fp8`, fallback disabled, required-parameter enforcement, temperature 0, explicit `reasoning: {enabled: false}`. Metadata was validated before inference. A real two-step check used the standard mini-SWE loop/client to execute bash `echo` inside a read-only offline Docker container, return its tool output and submit. Both responses came from Z.AI, had valid tool calls and reported zero reasoning tokens.

Compatibility result: **passed; 2 requests; 515 input tokens; 54 output tokens; $0.00068780 reported cost**. It used no bug description, source edit or hidden acceptance test and is setup inference, outside the 24 coding attempts. This confirms that small tool-call round trips work; it does not establish coding quality or provider reliability for longer runs.

Evidence: `reports/compatibility/glm-5-z-ai-fp8-v1/` and `reports/setup-inference-costs.json`. The model configuration and actual request trace are frozen with the round.

## Validation

**16 zero-inference Harbor controls passed:** for each task, unchanged source rejected, reference accepted, reference written on the final allowed call accepted without submission, and reference accepted after a scripted execution error without submission. Source checksums and all isolation probes passed. No task-model inference was used for these controls.

Local controller suite: **101 tests passed**, including forward/backward clock-change checks and permanent budget expiry. Required new Python files pass Ruff lint/format checks.

Validation summary: `reports/comparison-4x2x3-v1-validation-summary.json`. Trusted setup recipe: `harbor_integration/comparison_setting.py`; control/freeze recipes: `reports/task-preparation/comparison-4x2x3-v1/`. Source/permission/evaluator images are immutable and recorded per task.

## Sampling, failures and cost reporting

The schedule alternates model order across tasks/repetitions. Each task/model pair has exactly three independently initialized attempts. No old pilot is selected as a repeat because it happened to succeed. The prepared schedule does not execute itself.

Every scheduled attempt and its inference cost are retained, including failures and malformed responses. Infrastructure failures are distinguished from coding failures. Coding-only exclusions require an explicit reason and visible denominators; excluded inference remains an operational expense. There are no silent replacements or automatic reruns. Any additional attempt uses a separate ID and authorization.

Report per-task/model results and model totals for this frozen round, not mixed historical policies. Include accepted patches, explicit submissions, all attempted requests, tokens, reported API cost, completion time and failure category. Cost per accepted patch includes failed-attempt spend. Unknown costs stay unknown; zero accepted patches makes cost per accepted patch undefined. Show separate operational totals that include setup/compatibility and infrastructure incident costs.

Historical Wi-Fi-affected Chroma attempt 003 stays quarantined, outside this new round. This preparation did not restart it or change its exclusion.
