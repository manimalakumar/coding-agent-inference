"""6000-token adapter compatibility check; no maintenance task or prior feedback."""

import argparse
import hashlib
import json
import signal
import subprocess
import time
from pathlib import Path

from dotenv import dotenv_values
from minisweagent.agents.default import DefaultAgent
from minisweagent.environments.docker import DockerEnvironment
from minisweagent.exceptions import Submitted
from minisweagent.models.utils.actions_toolcall import BASH_TOOL

from agent.budget import Budget
from agent.mini_runtime import MeteredModel, openrouter_model
from agent.provider import validate_model
from agent.schemas import Budgets, ModelConfig
from harbor_integration.trace import Trace


def run(config_path, output, image):
    output.mkdir(parents=True, exist_ok=False)
    config = ModelConfig.model_validate_json(config_path.read_text())
    plan = validate_model(config)
    (output / "endpoint-preflight.json").write_text(json.dumps(plan, indent=2))
    assert plan["provider"]["only"] == config.provider["only"]
    assert plan["provider"]["allow_fallbacks"] is False
    assert plan["parameters"].get("reasoning") == config.reasoning
    values = dotenv_values(Path.home() / ".config/openrouter/credentials.env")
    key = values.get("OPENROUTER_API_KEY") or values.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("Host inference key unavailable")
    trace = Trace(output / "requests.jsonl", key)
    budget = Budget(
        Budgets(
            seconds=120,
            model_calls=2,
            generated_tokens=12000,
            response_tokens=6000,
            repair_rounds=0,
            tool_seconds=10,
        )
    )
    native = openrouter_model(config, plan, key)
    native.abort_exceptions = [Exception]
    model = MeteredModel(native, budget, trace, submission_priority=False)
    actions = []

    class TracedDocker(DockerEnvironment):
        def execute(self, action, **kwargs):
            budget.check_time()
            actions.append(action)
            try:
                result = super().execute(action, **kwargs)
            except Submitted:
                trace.emit("tool_submission", action=action)
                raise
            trace.emit("tool_result", tool_call=action, result=result)
            return result

    env = TracedDocker(
        image=image,
        cwd="/tmp",
        timeout=10,
        container_timeout="180",
        interpreter=["bash", "-o", "pipefail", "-c"],
        run_args=[
            "--rm",
            "--pull=never",
            "--network=none",
            "--read-only",
            "--user=65532:65532",
            "--cap-drop=ALL",
            "--security-opt=no-new-privileges",
            "--pids-limit=64",
            "--cpus=1",
            "--memory=512m",
            "--tmpfs=/tmp:rw,nosuid,nodev,noexec,size=32m,mode=1777",
        ],
    )
    agent = DefaultAgent(
        model,
        env,
        system_template="You are a helpful assistant that can interact with a computer.",
        instance_template="{{task}}",
        cost_limit=0,
        wall_time_limit_seconds=120,
        max_consecutive_format_errors=1,
        output_path=output / "mini-swe-agent.trajectory.json",
    )
    started = time.monotonic()
    result = {}
    error = None

    def expire(*_):
        raise TimeoutError("Compatibility wall-time budget exhausted")

    previous = signal.signal(signal.SIGALRM, expire)
    signal.setitimer(signal.ITIMER_REAL, 120)
    try:
        result = agent.run(
            "This is a tool compatibility check, not a coding task. First use the bash tool with exactly `echo GLM5_COMPATIBILITY_OK`. After seeing that successful output, use the bash tool with exactly `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT` as your second response. Make no other commands or requests."
        )
    except Exception as exc:
        error = type(exc).__name__
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
        container = env.container_id
        subprocess.run(
            ["docker", "rm", "-f", container], capture_output=True, timeout=15
        )
        env.container_id = None
        costs = [u.get("cost") for u in model.usage]
        reasoning = [
            u.get("completion_tokens_details", {}).get("reasoning_tokens")
            for u in model.usage
        ]
        success = (
            result.get("exit_status") == "Submitted"
            and len(actions) == 2
            and actions[0]["command"].strip() == "echo GLM5_COMPATIBILITY_OK"
            and actions[1]["command"].strip()
            == "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
            and len(model.usage) == budget.calls == 2
            and all(n == 0 for n in reasoning)
        )
        summary = {
            "kind": "compatibility_preflight",
            "coding_attempt": False,
            "task_ids_used": [],
            "model_config": config.model_dump(),
            "model_config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
            "bash_tool_schema_sha256": hashlib.sha256(
                json.dumps(BASH_TOOL, sort_keys=True).encode()
            ).hexdigest(),
            "status": "passed" if success else "failed",
            "exit_status": result.get("exit_status"),
            "error_type": error,
            "model_requests": budget.calls,
            "input_tokens_reported": sum(
                u.get("prompt_tokens", 0) for u in model.usage
            ),
            "output_tokens_reported": sum(
                u.get("completion_tokens", 0) for u in model.usage
            ),
            "reasoning_tokens_reported": reasoning,
            "actual_providers": model.providers,
            "reported_cost_usd": sum(costs)
            if len(costs) == budget.calls
            and all(isinstance(v, (float, int)) for v in costs)
            else None,
            "elapsed_seconds": time.monotonic() - started,
            "actions": actions,
            "sandbox": {
                "image": image,
                "network": "none",
                "host_mounts": [],
                "credentials_forwarded": False,
            },
            "measured_round_started": False,
            "automatic_retries": 0,
        }
        (output / "result.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--confirm-paid-preflight", action="store_true")
    args = parser.parse_args()
    if not args.confirm_paid_preflight:
        parser.error(
            "Use --confirm-paid-preflight to authorize up to two inference requests"
        )
    print(json.dumps(run(args.model_config, args.output, args.image), indent=2))
