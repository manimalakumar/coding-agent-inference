"""Harbor BaseAgent adapter for upstream mini-SWE-agent on the trusted host.

Harbor's installed mini agent runs its client in the task container. This adapter
keeps credentials outside that offline container; it contains no coding-agent loop.
"""

import asyncio
import base64
import hashlib
import hmac
import json
import shlex
import time
import zlib
from pathlib import Path

from harbor.agents.base import BaseAgent
from harbor.agents.capabilities import AgentCapabilities
from harbor.agents.installed.mini_swe_agent import convert_and_save_trajectory
from harbor.models.agent.context import AgentContext
from minisweagent.agents.default import DefaultAgent
from minisweagent.environments.docker import DockerEnvironment
from minisweagent.models.test_models import DeterministicModel, make_output

from agent.budget import Budget, BudgetExhausted
from agent.mini_runtime import MeteredModel, openrouter_model
from agent.provider import validate_model
from agent.schemas import Budgets, ModelConfig
from harbor_integration.trace import Trace

ROOT = Path(__file__).resolve().parents[1]


class HarborTools:
    """Only model tool: delegate commands to Harbor's isolated environment."""

    def __init__(self, environment, loop, budget, trace, expected_failure):
        self.expected_failure = expected_failure
        self.environment, self.loop, self.budget, self.trace = (
            environment,
            loop,
            budget,
            trace,
        )
        self.reproduced = False

    def execute(self, action):
        self.budget.check_time()
        command = action["command"]
        submit = command.strip() == "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
        if self.budget.submission_only and not submit:
            return dict(
                returncode=2,
                output="Final request is reserved for submission.",
                exception_info="",
            )
        if (
            not self.reproduced
            and command.strip() != "python /task/reproduce.py"
            and not (self.budget.submission_only and submit)
        ):
            return dict(
                returncode=2,
                output="First execute exactly: python /task/reproduce.py",
                exception_info="",
            )
        # Capture bounded output inside the container, retaining pipeline failures.
        wrapped = (
            "out=$(mktemp); (ulimit -f 2048; bash -o pipefail -c "
            + shlex.quote(command)
            + ') >"$out" 2>&1; rc=$?; head -c 2000000 "$out"; rm -f "$out"; exit "$rc"'
        )
        timeout = min(90, self.budget.remaining_seconds())
        future = asyncio.run_coroutine_threadsafe(
            self.environment.exec(command=wrapped, timeout_sec=timeout), self.loop
        )
        result = future.result(timeout=timeout + 5)
        output = dict(
            returncode=result.return_code,
            output=(result.stdout or "") + (result.stderr or ""),
            exception_info="",
        )
        self.trace.emit(
            "tool_result",
            tool_call=action,
            result={**output, "exit_code": result.return_code},
        )
        if not self.reproduced:
            if result.return_code != 1 or self.expected_failure not in output["output"]:
                raise RuntimeError(
                    "Prepared reproduction did not reproduce the expected issue"
                )
            self.reproduced = True
        # Reuse upstream's submission protocol, without starting its Docker manager.
        DockerEnvironment._check_finished(None, output)
        return output

    def get_template_vars(self):
        return dict(system="Linux", release="container", version="", machine="")

    def serialize(self):
        return {
            "info": {
                "config": {
                    "environment_type": "harbor.environments.docker.DockerEnvironment"
                }
            }
        }


class HostMiniSweAgent(BaseAgent):
    capabilities = AgentCapabilities(atif=True)

    def __init__(self, *, policy, fixture=None, **kwargs):
        super().__init__(**kwargs)
        self.policy = policy
        self.fixture = fixture
        if fixture not in (None, "baseline", "reference"):
            raise ValueError("Unsupported local fixture")

    @staticmethod
    def name():
        return "host-mini-swe-agent"

    def version(self):
        return "2.4.6"

    async def setup(self, environment):
        for relative, expected in self.policy["code_fingerprints"].items():
            if hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() != expected:
                raise RuntimeError("Frozen Harbor integration changed: " + relative)
        probe = """import hashlib,json
from pathlib import Path
root=Path('/workspace')
paths=list(root.rglob('*'))
assert not any(p.is_symlink() or '.git' in p.relative_to(root).parts for p in paths)
files={p.relative_to(root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in paths if p.is_file()}
print(json.dumps({'baseline_checksum':hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest(),'files':len(files)}))
"""
        snapshot = await environment.exec(
            command="python -c " + shlex.quote(probe), timeout_sec=30
        )
        (self.logs_dir / "source-snapshot.json").write_text(
            snapshot.stdout or snapshot.stderr or ""
        )
        if (
            snapshot.return_code
            or json.loads(snapshot.stdout)["baseline_checksum"]
            != self.policy["source_provenance"]["baseline_checksum"]
        ):
            raise RuntimeError("Agent source differs from immutable pre-fix export")
        result = await environment.exec(
            command="python /task/isolation.py", timeout_sec=30
        )
        (self.logs_dir / "isolation.json").write_text(
            result.stdout or result.stderr or ""
        )
        if result.return_code:
            raise RuntimeError("Harbor isolation preflight failed")

    async def run(self, instruction, environment, context: AgentContext):
        limits = Budgets.model_validate(self.policy["budgets"])
        budget = Budget(limits)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        trace = Trace(self.logs_dir / "requests.jsonl", "")
        model_config = ModelConfig.model_validate(self.policy["model"])
        metered = None
        if self.fixture:
            commands = ["python /task/reproduce.py"]
            if self.fixture == "reference":
                files = json.loads(Path(self.policy["reference_fixture"]).read_text())
                payload = base64.b64encode(
                    zlib.compress(json.dumps(files).encode())
                ).decode()
                commands.append(
                    "python - <<'PY'\nfrom pathlib import Path\nimport base64,json,zlib\nfiles = json.loads(zlib.decompress(base64.b64decode("
                    + repr(payload)
                    + ")))"
                    + "\nfor name, content in files.items():\n    Path('/workspace', name).write_text(content)\nPY"
                )
            commands.append("echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT")
            model = DeterministicModel(
                outputs=[
                    make_output(
                        "Local validation fixture; no inference.",
                        [{"command": c}],
                        cost=0,
                    )
                    for c in commands
                ],
                cost_per_call=0,
            )
        else:
            import os

            key = os.environ.get("OPENROUTER_API_KEY")
            if not key:
                raise RuntimeError("Host OPENROUTER_API_KEY required")
            trace.secret = key
            model = metered = MeteredModel(
                openrouter_model(model_config, validate_model(model_config), key),
                budget,
                trace,
                submission_priority=True,
            )
        from public_preparation.materialize import resolve

        config = resolve(self.policy["mini_config"])["agent"]
        tools = HarborTools(
            environment,
            asyncio.get_running_loop(),
            budget,
            trace,
            self.policy["reproduction"]["expected_failure"],
        )
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
        budget.started = time.monotonic()
        started = time.time()
        exit_info = {}
        try:
            exit_info = await asyncio.to_thread(agent.run, instruction)
        except BudgetExhausted as exc:
            exit_info = {"exit_status": str(exc)}
        finally:
            budget.started = min(budget.started, time.monotonic() - limits.seconds)
            usage = metered.usage if metered else []
            context.n_input_tokens = sum(u.get("prompt_tokens", 0) for u in usage)
            context.n_output_tokens = sum(u.get("completion_tokens", 0) for u in usage)
            costs = [u.get("cost") for u in usage]
            context.cost_usd = (
                sum(costs)
                if costs
                and len(usage) == budget.calls
                and all(isinstance(c, (float, int)) for c in costs)
                else (0 if self.fixture else None)
            )
            receipt = {
                "started_at": started,
                "deadline": started + limits.seconds,
                "agent_submission": exit_info.get("exit_status") == "Submitted",
                "exit_status": exit_info.get("exit_status", "interrupted"),
                "model_requests": budget.calls,
                "local_fixture": bool(self.fixture),
                "output_tokens": context.n_output_tokens,
                "repair_rounds": budget.repairs,
                "generated_tokens_charged": budget.generated,
                "reported_cost": context.cost_usd,
            }
            raw = json.dumps(receipt, sort_keys=True).encode()
            signature = hmac.new(
                Path(self.policy["receipt_key"]).read_bytes(), raw, hashlib.sha256
            ).hexdigest()
            receipt_path = self.logs_dir / "receipt.json"
            receipt_path.write_text(
                json.dumps({"receipt": receipt, "signature": signature})
            )
            # Publish only signed controller metadata; the verifier rejects tampering.
            await environment.upload_file(receipt_path, "/logs/artifacts/receipt.json")
            context.metadata = receipt
            native = self.logs_dir / "mini-swe-agent.trajectory.json"
            if native.exists():
                convert_and_save_trajectory(
                    native,
                    self.logs_dir / "trajectory.json",
                    self.session_id or "local",
                )
