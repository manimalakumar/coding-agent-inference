"""Standard mini-SWE agent with workspace delivery and independent evaluation.

The historical adapters remain frozen. Harbor still owns container lifecycle,
collection and verification; upstream mini-SWE still owns the entire agent loop.
"""

import asyncio
import base64
import hashlib
import hmac
import json
import os
import shlex
import time
import zlib
from pathlib import Path

from harbor.agents.installed.mini_swe_agent import convert_and_save_trajectory
from minisweagent.agents.default import DefaultAgent
from minisweagent.models.test_models import DeterministicModel, make_output

from agent.budget import Budget, BudgetExhausted
from agent.mini_runtime import MeteredModel, openrouter_model
from agent.provider import validate_model
from agent.schemas import Budgets, ModelConfig
from experiments.configuration_paths import configuration_path
from harbor_integration.host_task import HarborTools, HostMiniSweAgent
from harbor_integration.trace import Trace

POLICY_ID = "workspace-deliverable-v1"


def termination_reason(status):
    if status == "Submitted":
        return "submitted"
    if "budget_exhausted" in status or status in ("TimeExceeded", "LimitsExceeded"):
        return "budget_exhausted"
    return "error"


class WorkspaceBudget(Budget):
    def repair(self):
        # Do not even schedule a format repair after the final allowed response.
        self.check_time()
        if self.calls >= self.limits.model_calls:
            raise BudgetExhausted("model_call_budget_exhausted")
        if self.generated >= self.limits.generated_tokens:
            raise BudgetExhausted("generated_token_budget_exhausted")
        super().repair()


class WorkspaceModel(MeteredModel):
    def __init__(self, model, budget, trace):
        super().__init__(model, budget, trace, submission_priority=False)

    def query(self, messages, **kwargs):
        notice = (
            f"Request {self.budget.calls + 1}/{self.budget.limits.model_calls}; "
            f"{self.budget.limits.generated_tokens - self.budget.generated} output tokens remain. "
            "The final allowed request may perform useful work. No request or tokens are reserved for submission. "
            "Your final workspace is captured and evaluated at termination. "
            "Once required checks pass, finish early with the standalone submit command to save inference."
        )
        return super().query([*messages, {"role": "user", "content": notice}], **kwargs)


class FixtureModel:
    """Exercise the same budget boundary using zero-inference upstream fixtures."""

    def __init__(self, commands, budget, error=False):
        self.budget, self.error = budget, error
        self.length = len(commands)
        self.native = DeterministicModel(
            outputs=[
                make_output("Trusted fixture; no inference.", [{"command": c}], cost=0)
                for c in commands
            ],
            cost_per_call=0,
        )

    def __getattr__(self, name):
        return getattr(self.native, name)

    def query(self, *args, **kwargs):
        self.budget.reserve("trusted_fixture")
        if self.error and self.budget.calls > self.length:
            raise RuntimeError("Scripted controller error after candidate creation")
        return self.native.query(*args, **kwargs)


class WorkspaceAgent(HostMiniSweAgent):
    def __init__(self, *, policy, fixture=None, **kwargs):
        # Parent setup validates the baseline checksum and unchanged isolation probes.
        super().__init__(policy=policy, fixture=None, **kwargs)
        if fixture not in (
            None,
            "baseline",
            "reference",
            "reference_budget",
            "reference_error",
        ):
            raise ValueError("Unsupported trusted fixture")
        self.fixture = fixture
        common = policy["completion_policy"]
        expected = json.loads(
            (
                configuration_path(
                    Path(__file__).resolve().parents[1],
                    "configs/completion-workspace-v1.json",
                )
            ).read_text()
        )
        if (
            common != expected
            or common["policy_id"] != POLICY_ID
            or policy["budgets"] != common["agent_budgets"]
        ):
            raise ValueError(
                "Every compared model must use the exact frozen completion policy"
            )

    async def setup(self, environment):
        await super().setup(environment)
        probe = """import json
from pathlib import Path
print(json.dumps({p.name:p.joinpath('stat').read_text().split(') ')[1].split()[19] for p in Path('/proc').iterdir() if p.name.isdigit() and p.joinpath('stat').exists()}))
"""
        result = await environment.exec(
            command="python -c " + shlex.quote(probe), timeout_sec=10
        )
        if result.return_code:
            raise RuntimeError("Could not identify container keep-alive processes")
        self.protected_processes = json.loads(result.stdout)
        (self.logs_dir / "process-baseline.json").write_text(result.stdout)

    async def quiesce(self, environment):
        # Preserve only pre-agent process identities and the trusted cleanup exec.
        # Reap all unprivileged writers, including descendants/background jobs.
        script = """import json,os,signal,time
from pathlib import Path
protected=json.loads(PROTECTED)
terminated=set()
own=os.getpid()
ancestors={own}
pid=own
while pid>1:
    stat=Path('/proc',str(pid),'stat').read_text().split(') ')[1].split()
    pid=int(stat[1]);ancestors.add(pid)
for attempt in range(30):
    writers=[]
    for path in Path('/proc').iterdir():
        if not path.name.isdigit():continue
        try:
            fields=path.joinpath('stat').read_text().split(') ')[1].split()
            pid=int(path.name)
            if pid in ancestors or fields[0]=='Z' or path.stat().st_uid!=65532:continue
            if protected.get(path.name)==fields[19]:continue
            writers.append(pid)
        except (FileNotFoundError,ProcessLookupError):pass
    if not writers:
        print(json.dumps({'workspace_quiesced':True,'cleanup_passes':attempt+1,'terminated_processes':sorted(terminated)}));break
    for pid in writers:
        try:os.kill(pid,signal.SIGKILL);terminated.add(pid)
        except ProcessLookupError:pass
    time.sleep(0.05)
else:
    raise RuntimeError('Agent writers remained after termination')
""".replace("PROTECTED", repr(json.dumps(self.protected_processes)))
        return await environment.exec(
            command="exec python -c " + shlex.quote(script),
            user="65532:65532",
            timeout_sec=10,
        )

    async def run(self, instruction, environment, context):
        limits = Budgets.model_validate(self.policy["budgets"])
        budget = WorkspaceBudget(limits)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        trace = Trace(self.logs_dir / "requests.jsonl", "")
        metered = None
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
            wall_time_limit_seconds=limits.seconds,
            max_consecutive_format_errors=limits.repair_rounds + 1,
            output_path=self.logs_dir / "mini-swe-agent.trajectory.json",
        )
        started = time.time()
        budget.started = time.monotonic()
        task = asyncio.create_task(asyncio.to_thread(agent.run, instruction))
        exit_info, execution_error = {}, None
        quiet = None
        try:
            exit_info = await asyncio.wait_for(
                asyncio.shield(task), timeout=budget.remaining_seconds()
            )
        except asyncio.TimeoutError, asyncio.CancelledError:
            exit_info = {"exit_status": "wall_clock_budget_exhausted"}
            budget.started = time.monotonic() - limits.seconds
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
            budget.started = time.monotonic() - limits.seconds
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
            usage = metered.usage if metered else []
            context.n_input_tokens = sum(u.get("prompt_tokens", 0) for u in usage)
            context.n_output_tokens = sum(u.get("completion_tokens", 0) for u in usage)
            costs = [u.get("cost") for u in usage]
            context.cost_usd = (
                sum(costs)
                if costs
                and len(usage) == budget.calls
                and all(isinstance(c, (int, float)) for c in costs)
                else (0 if self.fixture else None)
            )
            status = exit_info.get("exit_status", "interrupted")
            receipt = {
                "completion_policy": POLICY_ID,
                "workspace_quiesced": quiet.return_code == 0,
                "started_at": started,
                "agent_deadline": started + limits.seconds,
                "execution_stopped_at": min(time.time(), started + limits.seconds),
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
