# Code map and developer checks

[Back to README](../README.md) · [Setup guide](setup.md)

Run commands in this guide from the repository root. This document covers implementation navigation, validation and preparing files for publication.

## Code and configuration map

| Location | Purpose |
| --- | --- |
| `harbor_integration/host_task.py` and `host_reporting.py` | Task-aware host execution and reporting adapters |
| `public_preparation/` and `experiments/portable_prepare.py` | Current public setup, controls and fresh execution |
| `experiments/notebook_workflow.py` and `notebook_presentation.py` | Notebook execution/replay and reader-facing displays |
| `experiments/study_views.py`, `intervention_views.py`, `task_rosters.py`, `notebook_charts.py` | Study analysis, intervention explanations, task tables and charts |
| `experiments/records.py` and `relocated_records.py` | One artifact parser and checked relative configuration references |
| `configs/study.json` and the named study configurations | Historical task/model settings, budgets, source identities and checksum projections |
| `public_preparation/configs/` | Current public source pins, dependencies, preparation recipes and evaluation-protocol identities |
| `evaluator/` | Harness qualification and shared workspace-policy checks; task-specific checks live under `public_preparation/` and `tasks/` |

Configuration filenames describe their conditions: for example, `sandbox-six-calls.json`, `sandbox-twelve-calls.json`, and `mini-swe-explicit-test-policy.yaml`. Original setting IDs and pinned template-recipe IDs remain inside the records. [configuration-paths.json](../configs/configuration-paths.json) maps historical filenames to the descriptive files, preserving old references without distributing duplicate copies. Historical configuration bytes are retained; the study selector and current checksum projections use the new paths.

Harbor adapters also use descriptive filenames; historical import paths resolve during setup. The [Harbor-name validation report](../reports/portability/harbor-names-validation.json) records preservation checks, offline tests and all 13 isolated task controls.

Experiment scripts use descriptive filenames. Historical configuration IDs remain unchanged so saved results retain their provenance. Use `portable_prepare` and the notebook's Execute controls for current execution. Historical acceptance and versioned public evaluation protocols remain distinct.

The public launcher uses the canonical parser in `experiments/records.py`. The [experiment-name validation report](../reports/portability/experiment-names-validation.json) records the module mapping, preservation checks, 199 offline tests, Replay and all 13 [isolated task controls](../reports/public-readiness/isolated-experiment-names-controls.json).

## Offline checks

```sh
.notebook-venv/bin/python notebooks/validate_replay.py
.notebook-venv/bin/python notebooks/check_preparation.py
.venv/bin/python notebooks/run_offline_tests.py
```

Replay validates the tutorial with external access blocked. Preparation controls validate environments separately. Neither is a model-performance observation. Revalidate affected resources when changing dependencies, evaluators, configuration or execution paths.

## Evidence and provenance

Evidence manifests are under `evidence/` and `notebooks/evidence/`. Report and notebook artifact filenames describe their study phase. [artifact-paths.json](../notebooks/artifact-paths.json) resolves historical references while the original run, setting and protocol IDs remain inside the unchanged payloads. The [artifact-name validation report](../reports/portability/artifact-names-validation.json) records payload hashes, the single notebook path update, offline analysis and unchanged execution bindings. Validation reports document [offline tests](../reports/portability/unit-validation.json), [setup extraction integrity](../reports/portability/setup-extraction-integrity.json) and [isolated preparation controls](../reports/public-readiness/isolated-extraction-validation.json). Fixture controls validate setup; they never count as model successes.

## Public files and publication checks

Downloaded upstream source, reconstructed acceptance suites, expanded prompts, environments and generated runs are ignored. Setup obtains upstream material locally and verifies the recorded hashes.

Before committing a notebook used for Execute, restore `MODE="replay"` and `CONFIRM_PAID_EXECUTION=False`, then run:

```sh

.notebook-venv/bin/python notebooks/render_replay.py

.venv/bin/python notebooks/audit_publication.py

```

The publication audit checks for saved upstream prompt and patch passages. Keep full fresh-run artifacts in ignored storage.

Keep raw research artifacts unchanged in private storage. Public summaries must preserve original outcomes, separate historical and public protocol identities, and retain unknown usage or cost as unknown. Keep reporting-test fixtures separate from recorded model runs. Calculate study totals only from saved execution evidence.
