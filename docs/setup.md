# Setup and fresh execution

[Back to README](../README.md)

This guide contains the full task-preparation and execution instructions. For historical analysis only, use the README quick start; Docker and inference credentials are unnecessary.

## Install

Use Python 3.14 and uv. From the repository root:

```sh
uv sync --frozen --cache-dir .local/uv-cache
uv venv .notebook-venv --python 3.14
uv pip install --python .notebook-venv/bin/python -r notebooks/requirements.txt
```

The execution environment is `.venv`; the notebook environment is `.notebook-venv`. Installation and explicit preparation need network access. The task agent and verifier operate offline after setup.

## Prepare, launch and inspect

Fresh execution additionally needs Docker with Linux ARM64 containers and a host-only OpenRouter credential file. Linux ARM64 is the validated target; other architectures have not been tested. Source downloads, dependency installation and image building require network access during trusted setup. Agent commands and verifier checks run offline.

### 1. Prepare and validate a task

```sh

.venv/bin/python -m experiments.portable_prepare doctor

.venv/bin/python -m experiments.portable_prepare prepare LC-INT-TEMPERATURE-001 --allow-setup-network --controls

```

Replace the task ID to prepare another task. Run preparation commands sequentially within a checkout.

Preparation fetches pinned source, builds the task images and runs **zero-inference controls**. The baseline must reproduce the intended failure, the reference must pass, and regression, quality, isolation and workspace-capture checks must behave as expected. Evaluator resources, reference fixes and credentials stay outside agent access. Paid execution remains blocked until local readiness checks pass.

Current configuration paths additionally passed all 13 [working-folder control sets](../reports/public-readiness/configuration-rename-controls.json) and all 13 [isolated-copy control sets](../reports/public-readiness/isolated-configuration-rename-controls.json), with zero model requests and no environment rebuilds. The [validation report](../reports/portability/configuration-rename-validation.json) preserves the one Hugging Face evaluator-timeout incident and its successful sequential rerun under the unchanged checks and deadline.

**All 13 tasks passed the recorded clean-setup controls on the validated platform.** You must prepare and validate resources on your own machine. See the [task inventory](../reports/public-readiness/inventory.json), [readiness report](../reports/public-readiness/readiness.json) and [clean-setup evidence](../reports/public-readiness/clean-setup-validation.json).

<details>

<summary>All 13 preparation paths and public evaluation protocols</summary>

<!-- READINESS START -->

| Task | Preparation command | Public evaluation protocol | Controls | Remaining blocker |
| --- | --- | --- | --- | --- |
| LC-ASYNC-TOOLS-001 | `prepare LC-ASYNC-TOOLS-001` | `public-lc-async-tools-001-v1` | All passed | None |
| LC-SUMMARY-OVERFLOW-001 | `prepare LC-SUMMARY-OVERFLOW-001` | `public-lc-summary-overflow-001-v1` | All passed | None |
| LC-FIREWORKS-RERANK-001 | `prepare LC-FIREWORKS-RERANK-001` | `public-lc-fireworks-rerank-001-v1` | All passed | None |
| LC-OPENROUTER-PROVIDER-001 | `prepare LC-OPENROUTER-PROVIDER-001` | `public-lc-openrouter-provider-001-v1` | All passed | None |
| LC-HF-ACCELERATOR-001 | `prepare LC-HF-ACCELERATOR-001` | `public-lc-hf-accelerator-001-v1` | All passed | None |
| LC-JSON-INPUT-001 | `prepare LC-JSON-INPUT-001` | `public-lc-json-input-001-v1` | All passed | None |
| LC-INT-TEMPERATURE-001 | `prepare LC-INT-TEMPERATURE-001` | `public-lc-int-temperature-001-v1` | All passed | None |
| LC-PARALLEL-TOOL-MERGE-001 | `prepare LC-PARALLEL-TOOL-MERGE-001` | `public-lc-parallel-tool-merge-001-v1` | All passed | None |
| LC-ANNOTATED-TOOLS-001 | `prepare LC-ANNOTATED-TOOLS-001` | `public-lc-annotated-tools-001-v1` | All passed | None |
| LC-CHROMA-NONE-001 | `prepare LC-CHROMA-NONE-001` | `public-lc-chroma-none-001-v1` | All passed | None |
| LC-CONTEXT-COUNTER-001 | `prepare LC-CONTEXT-COUNTER-001` | `public-lc-context-counter-001-v1` | All passed | None |
| LC-RESPONSES-USAGE-001 | `prepare LC-RESPONSES-USAGE-001` | `public-lc-responses-usage-001-v1` | All passed | None |
| LC-TIMEOUT-001 | `prepare LC-TIMEOUT-001` | `public-lc-timeout-001-v1` | All passed | None |

<!-- READINESS END -->

</details>

### 2. Launch from the notebook

Configure your host-only OpenRouter credential file as required by the launcher. In `main.ipynb`, select:

```python

EXECUTE_TASK_ID = "LC-INT-TEMPERATURE-001"

EXECUTE_MODEL = "minimax"  # or "glm5"

MODE = "execute"

CONFIRM_PAID_EXECUTION = True

FRESH_ATTEMPT_ID = "my-temperature-minimax-001"

```

Use a new attempt ID each time. The launcher rechecks preparation and refuses reused IDs. Installation, Replay and preparation never launch inference; Execute with explicit confirmation does.

Both models use pinned providers, disabled fallback and temperature zero. The public workflow allows **30 requests, 10,000 total output tokens, up to 6,000 per response and 20 agent minutes**. Commands have a 90-second limit; two malformed-tool-response recovery retries share the same budgets. Trusted evaluation has a separate five-minute allowance. There are no automatic HTTP or trial retries.

### 3. Inspect the result

Continue through the notebook to inspect the actual request, captured patch, checks and accounting. The final workspace is captured on submission, budget exhaustion or error. Acceptance requires every required check to pass and means **pending human review**; explicit submission is recorded separately.

Fresh public runs use their own protocol identities and storage. They do not overwrite historical findings or claim equivalence to every historical environment. Generated source, prompts, traces, patches and archives remain in ignored local storage.

## Troubleshooting

- **Preparation is not ready:** run `doctor` and the task's preparation command. Do not bypass failed controls or edit readiness receipts.
- **Architecture differs:** Linux ARM64 is the validated container target; other architectures have not been tested.
- **Download or image build fails:** inspect the preparation logs under ignored local storage. Run preparation sequentially; do not change dependency pins merely to force a pass.
- **Credentials are missing:** supply the host-only OpenRouter credential file through the launcher's supported configuration. Never put the key in the notebook or mount it into a task container.
- **An attempt ID already exists:** choose a new ID; preserve the old attempt.

## Validate preparation and recorded analysis

```sh
.notebook-venv/bin/python notebooks/validate_replay.py
.notebook-venv/bin/python notebooks/check_preparation.py
.venv/bin/python notebooks/run_offline_tests.py
```

These commands do not invoke a model. Review the readiness reports linked above for the recorded controls; local resources must pass their own checks before Execute.
