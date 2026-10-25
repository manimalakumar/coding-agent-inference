# Inference economics of coding agents

**What does it cost to produce an accepted repository-maintenance patch?**

This project compares MiniMax M2.5 and GLM-5 on historical LangChain tasks. It connects model requests, context use and inference charges to a concrete outcome: a patch that passes the required checks.

[main.ipynb](main.ipynb) is the complete tutorial: follow one attempt, explore development lessons, compare held-out results, or prepare a task and run a fresh attempt.

## What you will learn

- Prepare pre-fix source and validate repairs using a SWE-bench-style baseline/reference approach.
- Run a coding agent in an isolated workspace with explicit budgets.
- Distinguish implementation failures, unfinished quality checks and infrastructure interruptions.
- Inspect context growth and calculate inference economics, including unsuccessful attempts.

## Study stages and findings

The study progressed in three stages:

1. **Development:** Four bug tasks and one feature request helped refine instructions, budgets and environment setup. Settings changed, so these runs document learning rather than a controlled model comparison.
2. **Initial held-out evaluation:** Two tasks not used for tuning received one attempt per model—four launches—to check how the frozen workflow carried over to new tasks.
3. **Repeated held-out evaluation:** Six additional tasks received three fresh attempts per model/task. These **36 launches form the main comparison**, showing variation between attempts.

| Measure: six-task repeated evaluation | MiniMax M2.5 | GLM-5 |
| --- | ---: | ---: |
| Attempts | 18 | 18 |
| Accepted patches | 2 | 11 |
| Acceptance rate | 11.1% | 61.1% |
| Infrastructure interruptions | 1 | 2 |
| Known inference spend | $0.54 | $1.37 |

Acceptance rates include all attempts; interruptions are shown separately. Spending includes unsuccessful attempts. A few requests returned no cost information, so these are recorded subtotals and exact cost per accepted patch cannot be calculated.

**GLM-5 produced more accepted patches and incurred higher known spending under this workflow.** Acceptance requires every required check to pass; patches still require human review.

Together, the held-out stages contain **40 attempts across eight tasks**. Only the six-task stage received three repeats. Recorded charges across development, evaluation and setup-related inference total **$2.76**; the actual total may be higher because some request charges were unavailable.

## Stack and workflow

**Study:** development → diagnose and tune → freeze settings → held-out evaluation → compare outcomes and costs.

**One attempt:** prepare source → reproduce → inspect and edit → test → capture workspace → evaluate independently → record outcome and cost.

| Component | Role |
| --- | --- |
| Harbor 0.24.0 | Runs jobs and coordinates isolated evaluation |
| mini-SWE-agent 2.4.6 | Runs the coding-agent loop |
| OpenRouter | Provides model access and reported usage/cost |
| MiniMax M2.5 / `minimax/fp8` | Pinned deployment; advertised FP8 |
| GLM-5 / `z-ai/fp8` | Pinned deployment; advertised FP8 |
| Docker | Isolates task commands and verifier checks |
| Project code | Prepares tasks, enforces budgets, captures work and analyzes results |

## Quick start: analyze recorded results

Use Python 3.14 and [uv](https://docs.astral.sh/uv/getting-started/installation/). From the repository root:

```sh
uv sync --frozen --cache-dir .local/uv-cache
uv venv .notebook-venv --python 3.14
uv pip install --python .notebook-venv/bin/python -r notebooks/requirements.txt
.notebook-venv/bin/python -m jupyterlab main.ipynb
```

`.venv` holds execution dependencies; `.notebook-venv` holds tutorial and visualization dependencies.

The notebook defaults to `MODE="replay"`. After installation, analysis requires **no Docker, credentials, network access or model calls**. Tables and charts regenerate from reviewed evidence. Historical request displays are labelled summaries; Replay does not rerun the agents.

## Run a fresh attempt

Fresh execution needs Docker with Linux ARM64 containers—the validated target—and a host-only OpenRouter credential file. Other architectures have not been tested. Trusted preparation needs network access; agent and verifier commands run offline.

Prepare one task with zero model calls:

```sh
.venv/bin/python -m experiments.portable_prepare doctor
.venv/bin/python -m experiments.portable_prepare prepare LC-INT-TEMPERATURE-001 --allow-setup-network --controls
```

Then configure credentials and select the task, model and a unique attempt ID in the notebook. Inference starts only when you explicitly set `MODE="execute"` and `CONFIRM_PAID_EXECUTION=True`.

The workflow permits **30 requests, 10,000 total output tokens, up to 6,000 per response and 20 agent minutes**. The final workspace is evaluated after submission, budget exhaustion or error. Submission and acceptance are separate metrics.

**All 13 tasks passed the recorded clean-setup controls on the validated platform.** Your local resources must pass readiness checks too. Fresh runs use versioned public evaluators and remain separate from historical results.

See [the setup guide](docs/setup.md) for all task preparation commands, execution settings, controls and troubleshooting. **Installation, Replay and preparation make no inference calls; explicitly launching an agent incurs API charges.**

## Evidence and limitations

**Cost per accepted patch = total inference spend / accepted patches**, within the stated comparison group. Failed and malformed responses remain in spending. Missing charges remain unknown; zero accepted patches makes the ratio undefined.

Context charts show input tokens per request, including cached tokens. Cost charts show cumulative reported charges. They do not establish that context size caused an outcome.

This is a small, repository-specific study, not an official SWE-bench evaluation or a general model ranking. Three repeats do not establish precise reliability. Public tasks may have appeared in training data. Passing tests does not establish correctness beyond their coverage.

Inference spend is only part of adoption cost. Environment preparation, evaluator maintenance, operational diagnosis and human review also require effort. Those lessons are documented but not assigned a dollar value.

Explore the [task inventory](reports/public-readiness/inventory.json), [readiness report](reports/public-readiness/readiness.json) and [clean-setup evidence](reports/public-readiness/clean-setup-validation.json). Historical and public evaluation protocols have separate identities; setup controls never count as model successes.

For implementation navigation and offline/publication checks, see [the developer guide](docs/development.md). Downloaded source, reconstructed suites, expanded prompts and raw run artifacts remain ignored.
