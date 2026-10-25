"""Versioned comparison adapter; upstream mini-SWE owns the agent loop.

The historical adapters are untouched. Count host suspension/forward clock changes
conservatively and record actual stop timestamps without clamping them to a budget.
"""

import asyncio
import base64
import hashlib
import hmac
import json
import os
import time
import zlib
from pathlib import Path
from typing import Any

from harbor.agents.installed.mini_swe_agent import convert_and_save_trajectory
from minisweagent.agents.default import DefaultAgent

from agent.budget import BudgetExhausted
from agent.mini_runtime import openrouter_model
from agent.provider import validate_model
from agent.schemas import Budgets, ModelConfig
from harbor_integration.host_response_ceiling import ResponseCeilingAgent
from harbor_integration.host_task import HarborTools
from harbor_integration.host_workspace import (
    POLICY_ID,
    FixtureModel,
    WorkspaceBudget,
    WorkspaceModel,
    termination_reason,
)
from harbor_integration.trace import Trace


class DeadlineBudget(WorkspaceBudget):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.started = time.monotonic()
        self.wall_started = time.time()
        self.expired = False

    def remaining_seconds(self):
        if self.expired:
            return 0.0
        civil = self.limits.seconds - (time.time() - self.wall_started)
        return max(0.0, min(super().remaining_seconds(), civil))

    def expire(self):
        self.expired = True


async def wait_deadline(task, budget):
    while not task.done():
        left = budget.remaining_seconds()
        if left <= 0:
            raise asyncio.TimeoutError
        await asyncio.wait({task}, timeout=min(left, 0.5))
    return task.result()


class ComparisonAgentV2(ResponseCeilingAgent):
    async def run(self, instruction, environment, context):
        limits = Budgets.model_validate(self.policy["budgets"])
        budget = DeadlineBudget(limits)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        trace = Trace(self.logs_dir / "requests.jsonl", "")
        metered = None
        model: Any
        if self.fixture:
            commands = ["python /task/reproduce.py"]
            if self.fixture.startswith("reference"):
                files = json.loads(Path(self.policy["reference_fixture"]).read_text())
                data = base64.b64encode(
                    zlib.compress(json.dumps(files).encode())
                ).decode()
                apply = (
                    "python - <<'PY'\nfrom pathlib import Path\nimport base64,json,zlib\n"
                    f"files=json.loads(zlib.decompress(base64.b64decode({data!r})))\n"
                    "for name,content in files.items():\n    Path('/workspace',name).write_text(content)\nPY"
                )
                if self.fixture == "reference_budget":
                    # The final allowed call writes the qualifying patch, not a submit.
                    commands.extend(["true"] * (limits.model_calls - 2))
                    apply += "\n(sleep 600) >/tmp/fixture-background.log 2>&1 &"
                commands.append(apply)
            if self.fixture in ("baseline", "reference"):
                commands.append("echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT")
            model = FixtureModel(
                commands, budget, error=self.fixture == "reference_error"
            )
        else:
            config = ModelConfig.model_validate(self.policy["model"])
            key = os.environ.get("OPENROUTER_API_KEY")
            if not key:
                raise RuntimeError("Host OPENROUTER_API_KEY required")
            trace.secret = key
            native = openrouter_model(config, validate_model(config), key)
            native.abort_exceptions = [Exception]  # No unmetered transport retries.
            model = metered = WorkspaceModel(native, budget, trace)
        tools = HarborTools(
            environment,
            asyncio.get_running_loop(),
            budget,
            trace,
            self.policy["reproduction"]["expected_failure"],
        )
        from public_preparation.materialize import resolve

        config = resolve(self.policy["mini_config"])["agent"]
        agent = DefaultAgent(
            model,
            tools,
            system_template=config["system_template"],
            instance_template=config["instance_template"],
            cost_limit=0,
            wall_time_limit_seconds=0,  # Controller owns both wall/monotonic deadlines.
            max_consecutive_format_errors=limits.repair_rounds + 1,
            output_path=self.logs_dir / "mini-swe-agent.trajectory.json",
        )
        started = time.time()
        budget.started = time.monotonic()
        budget.wall_started = started
        task = asyncio.create_task(asyncio.to_thread(agent.run, instruction))
        exit_info, execution_error = {}, None
        execution_stopped_at = None
        execution_stopped_monotonic = None
        quiet = None
        try:
            exit_info = await wait_deadline(task, budget)
        except asyncio.TimeoutError, asyncio.CancelledError:
            execution_stopped_at = time.time()
            execution_stopped_monotonic = time.monotonic()
            exit_info = {"exit_status": "wall_clock_budget_exhausted"}
            budget.expire()
            quiet = await self.quiesce(environment)
            # Finalize any in-flight usage; no further tool or model work is permitted.
            try:
                await asyncio.wait_for(asyncio.shield(task), timeout=65)
            except Exception:
                pass
        except BudgetExhausted as exc:
            exit_info = {"exit_status": str(exc)}
        except Exception as exc:
            exit_info = {"exit_status": type(exc).__name__}
            execution_error = type(exc).__name__ + ": " + str(exc)
        finally:
            execution_stopped_at = execution_stopped_at or time.time()
            execution_stopped_monotonic = (
                execution_stopped_monotonic or time.monotonic()
            )
            budget.expire()
            # Kill any agent-started background writers before Harbor collects the tar.
            if quiet is None:
                quiet = await self.quiesce(environment)
            (self.logs_dir / "quiescence.json").write_text(
                json.dumps(
                    {
                        "returncode": quiet.return_code,
                        "stdout": quiet.stdout,
                        "stderr": quiet.stderr,
                    }
                )
            )
            from public_replay.usage import normalize_usage

            values = metered.usage if metered else []
            normalized = normalize_usage(
                [{"response": {"usage": value}} for value in values],
                expected_calls=budget.calls if not self.fixture else 0,
            )
            context.n_input_tokens = normalized["input_tokens"]
            context.n_output_tokens = normalized["output_tokens"]
            context.cost_usd = normalized["cost_usd"]
            status = exit_info.get("exit_status", "interrupted")
            receipt = {
                "completion_policy": POLICY_ID,
                "reporting_revision": "unknown-aware-receipt-v2",
                "usage_summary": normalized,
                "workspace_quiesced": quiet.return_code == 0,
                "started_at": started,
                "agent_deadline": started + limits.seconds,
                "execution_stopped_at": execution_stopped_at,
                "receipt_finalized_at": time.time(),
                "clock_policy": "min_remaining_monotonic_and_civil_v1",
                "monotonic_elapsed_seconds": execution_stopped_monotonic
                - budget.started,
                "finalization_elapsed_seconds": time.monotonic() - budget.started,
                "civil_elapsed_seconds": execution_stopped_at - started,
                "evaluation_seconds": self.policy["completion_policy"][
                    "evaluation_seconds"
                ],
                "termination_reason": termination_reason(status),
                "termination_detail": status,
                "explicit_submission": status == "Submitted",
                "agent_submission": status == "Submitted",
                "exit_status": status,
                "execution_error": execution_error,
                "model_requests": budget.calls if not self.fixture else 0,
                "controller_requests": budget.calls,
                "local_fixture": bool(self.fixture),
                "output_tokens": context.n_output_tokens,
                "repair_rounds": budget.repairs,
                "tool_format_repair_rounds": budget.repairs,
                "generated_tokens_charged": budget.generated,
                "reported_cost": context.cost_usd,
            }
            signature = hmac.new(
                Path(self.policy["receipt_key"]).read_bytes(),
                json.dumps(receipt, sort_keys=True).encode(),
                hashlib.sha256,
            ).hexdigest()
            path = self.logs_dir / "receipt.json"
            path.write_text(json.dumps({"receipt": receipt, "signature": signature}))
            await environment.upload_file(path, "/logs/artifacts/receipt.json")
            context.metadata = receipt
            native = self.logs_dir / "mini-swe-agent.trajectory.json"
            if native.exists():
                convert_and_save_trajectory(
                    native,
                    self.logs_dir / "trajectory.json",
                    self.session_id or "local",
                )
