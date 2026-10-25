"""Run the unmodified mini-SWE-agent with isolated tools and experiment accounting."""

import argparse
import importlib.metadata
import json
import os
import signal
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from minisweagent.agents.default import DefaultAgent
from minisweagent.models.utils.actions_toolcall import BASH_TOOL

from agent.budget import Budget, BudgetExhausted
from agent.mini_runtime import (
    MeteredModel,
    ScopedEnvironment,
    openrouter_model,
    stock_config,
)
from agent.provider import ConfigurationError, validate_model
from agent.results import refresh
from agent.schemas import ModelConfig, SandboxConfig, Task
from evaluator.qualify import collect_changes, qualify
from experiments.configuration_paths import configuration_path
from sandbox.docker import InfrastructureError, Sandbox, digest, docker_ready, manifest
from sandbox.prepare import PROJECT, write_json


def tools_for_config(config):
    return [BASH_TOOL]


class Trace:
    def __init__(self, path: Path, secret: str):
        self.path, self.secret = path, secret

    def emit(self, event, **fields):
        text = json.dumps(
            {"at": datetime.now(timezone.utc).isoformat(), "event": event, **fields}
        )
        if self.secret:
            text = text.replace(self.secret, "[REDACTED]")
        with self.path.open("a") as output:
            output.write(text + "\n")


class Controller:
    def __init__(self, box, model, run, key, request_plan):
        self.box, self.model, self.run, self.key = box, model, run, key
        self.request_plan = request_plan

    def execute(self):
        budget = Budget(self.box.config.budgets)
        trace = Trace(self.run / "trace.jsonl", self.key)
        metered = MeteredModel(
            openrouter_model(self.model, self.request_plan, self.key),
            budget,
            trace,
            submission_priority=self.box.config.stopping_protocol
            == "submit_priority_v1",
        )
        environment = None
        evaluation = None
        termination = "infrastructure_failure"
        exit_info = {}
        previous_handler = signal.getsignal(signal.SIGALRM)

        def expired(signum, frame):
            raise BudgetExhausted("wall_clock_budget_exhausted")

        try:
            environment = ScopedEnvironment(self.box, budget, trace, self.run)
            write_json(
                self.run / "native-isolation.json", environment.isolation_checks()
            )
            # Container launch is setup effort, before the evaluated workflow clock.
            initialization_seconds = time.monotonic() - budget.started
            budget.started = time.monotonic()
            signal.signal(signal.SIGALRM, expired)
            signal.setitimer(signal.ITIMER_REAL, budget.limits.seconds)
            standard = stock_config(self.box.config.prompt_config)["agent"]
            agent = DefaultAgent(
                metered,
                environment,
                system_template=standard["system_template"],
                instance_template=standard["instance_template"],
                step_limit=0,
                cost_limit=0,
                wall_time_limit_seconds=budget.limits.seconds,
                max_consecutive_format_errors=budget.limits.repair_rounds + 1,
                output_path=self.run / "trajectory.json",
            )
            instructions = (
                self.box.task.instructions
                + "\nFirst execute exactly: "
                + self.box.task.reproduction_command
            )
            instructions += "\nPreserve existing tests. Add a regression test and run development checks before submission."
            instructions += (
                "\nShared workflow budgets: "
                + self.box.config.budgets.model_dump_json()
            )
            exit_info = agent.run(instructions)
            termination = exit_info.get("exit_status", "unknown_exit")
            if termination == "Submitted":
                # Stop the agent container before hidden tests exist in another environment.
                environment.close()
                evaluation = qualify(
                    self.box,
                    self.run,
                    PROJECT / "evaluator",
                    budget.started + budget.limits.seconds,
                )
                termination = evaluation["status"]
        except BudgetExhausted as exc:
            termination = str(exc)
        except Exception as exc:
            trace.emit(
                "infrastructure_failure",
                error=str(exc),
                exception_type=type(exc).__name__,
            )
            termination = "infrastructure_failure"
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous_handler)
            if environment:
                environment.close()
            try:
                changes = collect_changes(self.box, self.run / "submission.patch")
            except InfrastructureError as exc:
                changes = []
                trace.emit("submission_rejected", error=str(exc))
                termination = "invalid_submission"
            complete = len(metered.usage) == budget.calls and all(
                isinstance(u.get("completion_tokens"), int)
                and u["completion_tokens"] >= 0
                for u in metered.usage
            )
            costs = [u.get("cost") for u in metered.usage]
            generated = sum(
                u["completion_tokens"]
                for u in metered.usage
                if isinstance(u.get("completion_tokens"), int)
            )
            result = {
                "framework": "mini-swe-agent==2.4.6",
                "termination_reason": termination,
                "elapsed_seconds": time.monotonic() - budget.started,
                "container_setup_seconds": locals().get("initialization_seconds"),
                "model_requests": budget.calls,
                "generated_tokens_including_reasoning": generated if complete else None,
                "known_reported_generated_tokens": generated,
                "budget_generated_tokens_charged": budget.generated,
                "input_tokens": sum(u.get("prompt_tokens", 0) for u in metered.usage),
                "reported_reasoning_tokens": [
                    u.get("completion_tokens_details", {}).get("reasoning_tokens")
                    for u in metered.usage
                ],
                "reported_cost": sum(costs)
                if complete
                and costs
                and all(isinstance(c, (int, float)) for c in costs)
                else None,
                "usage_complete": complete,
                "serving_providers": metered.providers,
                "repair_rounds": budget.repairs,
                "roles": budget.roles,
                "submission": exit_info,
                "agent_submission": {
                    "submitted": exit_info.get("exit_status") == "Submitted"
                },
                "patch_acceptance": (
                    "not_evaluated"
                    if evaluation is None
                    else "passed"
                    if evaluation["status"] == "checks_passed_pending_review"
                    else "failed"
                ),
                "termination_policy": {
                    "capture_workspace": "always",
                    "evaluate_workspace": "explicit_submission_only",
                    "automatic_evaluation_at_budget_exhaustion": False,
                },
                "files_changed": changes,
                "evaluation": evaluation,
            }
            write_json(self.run / "result.json", result)
            # Native trajectories can include provider content: redact the known secret too.
            if self.key:
                for name in ("trajectory.json", "result.json"):
                    path = self.run / name
                    if path.exists():
                        path.write_text(
                            path.read_text().replace(self.key, "[REDACTED]")
                        )
            if (self.run / "metadata.json").exists():
                refresh(PROJECT)
        return result


def setting_record(box, setup):
    config, task = box.config, box.task
    return {
        "sandbox": config.model_dump(),
        "task": task.model_dump(),
        "task_assets": {
            task.reproduction_script: digest(
                (box.task_file.parent / task.reproduction_script).read_text()
            )
        }
        if task.reproduction_script
        else {},
        "baseline_checksum": setup["baseline_checksum"],
        "image": box.image,
        "resources": setup["resources"],
        "controller_dependency_lock": digest((PROJECT / "uv.lock").read_text()),
        "tools": tools_for_config(config),
        "framework": "mini-swe-agent==2.4.6",
        "stopping_policy": {
            "protocol": config.stopping_protocol,
            "reserve_final_request": config.stopping_protocol == "submit_priority_v1",
            "submission_token_reserve": 128
            if config.stopping_protocol == "submit_priority_v1"
            else 0,
            "pipefail": config.stopping_protocol == "submit_priority_v1",
            "capture_final_workspace": True,
            "evaluate_on": "explicit_submission_only",
            "automatic_evaluation_at_termination": False,
            "separate_metrics": ["agent_submission", "patch_acceptance"],
        },
        "standard_config": digest(
            (configuration_path(PROJECT, config.prompt_config)).read_text()
        ),
        "evaluator": {
            p.name: digest(p.read_text()) for p in (PROJECT / "evaluator").glob("*.py")
        },
        "controller": {
            p.name: digest(p.read_text()) for p in (PROJECT / "agent").glob("*.py")
        },
        "sandbox_code": {
            p.name: digest(p.read_text()) for p in (PROJECT / "sandbox").glob("*.py")
        },
    }


def launch(
    setup_dir: Path, model_file: Path, expected_setting: str | None = None
) -> dict:
    setup = json.loads((setup_dir / "setup.json").read_text())
    if setup["setup_status"] != "validated":
        raise InfrastructureError("Setup is not validated; no model request was made")
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise InfrastructureError(
            "OPENROUTER_API_KEY missing; no model request was made"
        )
    docker_version = docker_ready()
    config = SandboxConfig.model_validate(setup["sandbox_config"])
    if (
        config.schema_version != 4
        or config.file_writing_tool
        or config.workflow_instructions
    ):
        raise InfrastructureError(
            "Prepare a distinct mini-SWE-agent setting before execution"
        )
    task = Task.model_validate(setup["task"])
    model = ModelConfig.model_validate_json(model_file.read_text())
    if config.schema_version == 4 and not task.reproduction_script:
        raise ConfigurationError(
            "mini-SWE-agent requires a read-only ready-to-run reproduction script"
        )
    if digest(manifest(setup_dir / "baseline")) != setup["baseline_checksum"]:
        raise InfrastructureError("Baseline has changed since setup")
    run = PROJECT / "runs" / (task.task_id + "-" + uuid.uuid4().hex[:12])
    run.mkdir()
    box = Sandbox(
        config,
        task,
        setup_dir / "baseline",
        run / "workspace",
        setup_dir / "task.json",
        setup["image_id"],
        setup_dir.parents[1] / "resources",
    )
    box.prepare()
    if (
        Task.model_validate_json(box.task_file.read_text()).model_dump()
        != task.model_dump()
    ):
        raise InfrastructureError("Task instructions changed since setup")
    for name, expected in setup.get("task_assets", {}).items():
        if digest((box.task_file.parent / name).read_text()) != expected:
            raise InfrastructureError("Task resource changed since setup")
    if box.resource_record() != setup["resources"]:
        raise InfrastructureError("Resource content changed; prepare a new setting")
    # Frozen evaluation plan, controller, prompt, tools and resources identify a setting.
    setting = setting_record(box, setup)
    if expected_setting is not None and digest(setting) != expected_setting:
        raise InfrastructureError(
            "Experimental setting changed; model comparison refused"
        )
    write_json(run / "setting.json", setting)
    write_json(
        run / "controller-environment.json",
        {
            "python": sys.version,
            "packages": sorted(
                (d.metadata["Name"], d.version)
                for d in importlib.metadata.distributions()
            ),
            "uv_lock_checksum": digest((PROJECT / "uv.lock").read_text()),
        },
    )
    write_json(
        run / "metadata.json",
        {
            "task_id": task.task_id,
            "commit": task.commit,
            "baseline_checksum": setup["baseline_checksum"],
            "setting_checksum": digest(setting),
            "model": model.model_dump(),
            "setup_directory": str(setup_dir),
            "setup_elapsed_seconds": setup["setup_elapsed_seconds"],
            "docker_version": docker_version,
            "controller_python": sys.version,
            "workflow_label": config.workflow_label,
        },
    )
    # Preflight and isolation are setup effort, before the measured attempt clock.
    try:
        write_json(run / "isolation.json", box.isolation_checks())
        request_plan = validate_model(model)
        write_json(run / "provider-preflight.json", request_plan)
    except (InfrastructureError, OSError) as exc:
        write_json(
            run / "result.json",
            {
                "termination_reason": "configuration_failure"
                if isinstance(exc, ConfigurationError)
                else "infrastructure_failure",
                "model_requests": 0,
                "reported_cost": 0,
                "usage_complete": True,
                "files_changed": [],
                "error": str(exc),
            },
        )
        refresh(PROJECT)
        raise
    controller = Controller(box, model, run, key, request_plan)
    result = controller.execute()
    return {"run_directory": str(run), **result}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--setup", type=Path, required=True)
    parser.add_argument(
        "--model",
        type=Path,
        default=configuration_path(PROJECT, "configs/minimax-m2.5-mini-v5.json"),
    )
    parser.add_argument(
        "--expected-setting",
        help="Require the setting checksum of a previous comparison run",
    )
    args = parser.parse_args()
    try:
        result = launch(args.setup.resolve(), args.model, args.expected_setting)
        print(json.dumps(result, indent=2))
    except (InfrastructureError, OSError) as exc:
        print(
            json.dumps(
                {"termination_reason": "infrastructure_failure", "error": str(exc)}
            )
        )
        raise SystemExit(2)


if __name__ == "__main__":
    main()
