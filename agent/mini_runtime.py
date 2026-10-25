"""Boundary and accounting adapters around upstream mini-SWE-agent components.

No agent loop, action parser, provider transport, or submission protocol lives here.
"""

import json
import re
import shlex
import subprocess
import time
from pathlib import Path

from minisweagent.environments.docker import DockerEnvironment
from minisweagent.exceptions import FormatError, Submitted
from minisweagent.models.openrouter_model import OpenRouterModel
from minisweagent.models.utils.actions_toolcall import BASH_TOOL

from agent.budget import Budget, BudgetExhausted
from sandbox.docker import InfrastructureError, host_env

PROJECT = Path(__file__).resolve().parents[1]


def stock_config(filename="mini-swe.yaml"):
    from public_preparation.materialize import load_mini

    return load_mini(filename)


class MeteredModel:
    """Delegate all parsing/formatting/HTTP to the standard OpenRouter model."""

    def __init__(self, model, budget: Budget, trace, submission_priority=False):
        self.model, self.budget, self.trace = model, budget, trace
        self.submission_priority = submission_priority
        self.usage = []
        self.providers = []

    def __getattr__(self, key):
        return getattr(self.model, key)

    def query(self, messages, **kwargs):
        cap = self.budget.reserve("coding_agent")
        if self.submission_priority:
            remaining = self.budget.limits.generated_tokens - self.budget.generated
            self.budget.submission_only = (
                self.budget.calls == self.budget.limits.model_calls or remaining <= 128
            )
            if not self.budget.submission_only:
                cap = min(cap, remaining - 128)
            notice = (
                f"Request {self.budget.calls}/{self.budget.limits.model_calls}; "
                f"{remaining} output tokens remain. "
                "Once required checks pass, submit immediately; do not add optional work. "
                "Reserve the final request for submission."
            )
            if self.budget.submission_only:
                notice += (
                    " This is your final request for submission. Issue only the bash command "
                    "`echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`; no edits, searches or tests. "
                    "Submission records your current workspace; it does not claim acceptance."
                )
            messages = [*messages, {"role": "user", "content": notice}]
        self.trace.emit(
            "model_request",
            role="coding_agent",
            number=self.budget.calls,
            body={
                "model": self.model.config.model_name,
                "messages": self.model._prepare_messages_for_api(messages),
                "tools": [BASH_TOOL],
                "usage": {"include": True},
                **self.model.config.model_kwargs,
                **kwargs,
                "max_tokens": cap,
            },
        )
        response = None
        try:
            try:
                message = self.model.query(messages, **(kwargs | {"max_tokens": cap}))
                response = message["extra"]["response"]
            except FormatError as exc:
                response = exc.messages[0]["extra"].get("response")
                if response is not None:
                    self.account(response, cap)
                self.budget.repair()
                raise
            self.account(response, cap)
            self.budget.check_time()
            return message
        except Exception as exc:
            if response is None:
                # Preserve HTTP status and any usage even on a rejected request.
                cause = exc
                http_response = None
                while cause is not None:
                    if getattr(cause, "response", None) is not None:
                        http_response = cause.response
                        break
                    cause = cause.__cause__
                failure = {"error": type(exc).__name__}
                if http_response is not None:
                    failure["status"] = http_response.status_code
                    try:
                        raw = http_response.json()
                        if isinstance(raw, dict):
                            failure["provider_response"] = raw
                            if isinstance(raw.get("usage"), dict):
                                failure["usage"] = raw["usage"]
                    except ValueError, TypeError:
                        pass
                elif match := re.search(r"HTTP (\d{3})", str(exc)):
                    failure["status"] = int(match[1])
                self.trace.emit("model_response", role="coding_agent", response=failure)
                if "usage" in failure:
                    self.usage.append(failure["usage"])
                    self.providers.append(None)
                    self.budget.account("coding_agent", failure["usage"], cap)
                else:
                    # Conservatively reserve unknown output; never invent measured cost.
                    self.budget.generated += cap
            raise

    def account(self, response, cap):
        self.trace.emit("model_response", role="coding_agent", response=response)
        usage = response.get("usage", {})
        self.usage.append(usage)
        self.providers.append(response.get("provider"))
        self.budget.account("coding_agent", usage, cap)


def openrouter_model(config, plan, key):
    import os
    if os.environ.get("PUBLIC_NO_INFERENCE") == "1":
        raise RuntimeError("Inference is disabled during public preparation controls")
    stock = stock_config()["model"]
    model = OpenRouterModel(
        model_name=config.model,
        model_kwargs={
            "temperature": config.temperature,
            "provider": plan["provider"],
            **plan["parameters"],
        },
        observation_template=stock["observation_template"],
        format_error_template=stock["format_error_template"],
        cost_tracking="ignore_errors",
    )
    # Standard client, configured endpoint; credentials never enter Docker/config/trace.
    model._api_url = config.endpoint
    model._api_key = key
    return model


class ScopedEnvironment:
    """Standard DockerEnvironment with trusted mounts, time/output caps and logging."""

    def __init__(self, box, budget, trace, run):
        self.box, self.budget, self.trace, self.run = box, budget, trace, run
        self.reproduction_verified = not bool(box.task.reproduction_script)
        self.submission_priority = box.config.stopping_protocol == "submit_priority_v1"
        # Reuse precisely the security policy used by setup and trusted evaluation.
        command = box.command("unused", "true")
        boundary = command.index("--entrypoint=/usr/bin/env")
        args = command[2:boundary]
        name_at = args.index("--name")
        del args[name_at : name_at + 2]
        args = [a for a in args if not a.startswith("--workdir=")]
        clean_env = command[
            boundary + 2 : -3
        ]  # -i, explicit env, excluding sh -c script
        self.native = DockerEnvironment(
            image=box.image,
            cwd="/workspace/" + box.task.package,
            run_args=args,
            forward_env=[],
            env={},
            interpreter=[
                "/usr/bin/env",
                *clean_env,
                "/bin/bash",
                *(["-o", "pipefail"] if self.submission_priority else []),
                "-c",
            ],
            timeout=box.config.budgets.tool_seconds,
            container_timeout=str(box.config.budgets.seconds + 120),
        )

    def isolation_checks(self):
        def execute(script, timeout):
            result = self.native.execute({"command": script}, timeout=timeout)
            return {**result, "exit_code": result["returncode"]}

        return self.box.isolation_checks(execute=execute)

    def serialize(self):
        return self.native.serialize()

    def get_template_vars(self):
        # Upstream otherwise renders the host's uname into its Docker task prompt.
        return {
            "system": "Linux",
            "release": "container",
            "version": "",
            "machine": "",
            "cwd": self.native.config.cwd,
        }

    def execute(self, action):
        self.budget.check_time()
        command = action.get("command", "")
        if (
            self.submission_priority
            and self.budget.submission_only
            and command.strip() != "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
        ):
            result = {
                "returncode": 2,
                "output": "Final request is reserved for submission; no further development commands are permitted.",
                "exception_info": "",
            }
        elif (
            not self.reproduction_verified
            and not (self.submission_priority and self.budget.submission_only)
            and command.strip() != self.box.task.reproduction_command
        ):
            result = {
                "returncode": 2,
                "output": "First execute exactly: "
                + self.box.task.reproduction_command,
                "exception_info": "",
            }
        else:
            # Bound pipe output before the standard environment captures it in memory.
            wrapped = (
                "out=$(mktemp); (ulimit -f 2048; /bin/bash "
                + ("-o pipefail " if self.submission_priority else "")
                + "-c "
                + shlex.quote(command)
                + ') >"$out" 2>&1; rc=$?; head -c 2000000 "$out"; rm -f "$out"; exit "$rc"'
            )
            started = time.monotonic()
            try:
                result = self.native.execute(
                    dict(action, command=wrapped),
                    timeout=max(
                        0.01,
                        min(
                            self.budget.limits.tool_seconds,
                            self.budget.remaining_seconds(),
                        ),
                    ),
                )
            except Submitted as exc:
                self.trace.emit(
                    "tool_result",
                    tool_call=action,
                    result={
                        "exit_code": 0,
                        "submission": exc.messages[0]["extra"]["submission"],
                    },
                )
                raise
            result["elapsed_seconds"] = time.monotonic() - started
            if result.get("extra", {}).get("exception_type") in {
                "TimeoutExpired",
                "BudgetExhausted",
            }:
                self.close()
                raise BudgetExhausted(
                    "wall_clock_budget_exhausted"
                    if self.budget.remaining_seconds() <= 0
                    else "tool_timeout"
                )
            if not self.reproduction_verified:
                if (
                    result["returncode"] != 1
                    or self.box.task.expected_failure not in result["output"]
                ):
                    raise InfrastructureError(
                        "Validated reproduction failed unexpectedly"
                    )
                self.reproduction_verified = True
                (self.run / "reproduction-evidence.json").write_text(
                    json.dumps(result, indent=2)
                )
        self.trace.emit(
            "tool_result",
            role="coding_agent",
            tool_call=action,
            result={**result, "exit_code": result["returncode"]},
        )
        return result

    def close(self):
        container = self.native.container_id
        if container:
            subprocess.run(
                ["docker", "rm", "-f", container],
                capture_output=True,
                timeout=15,
                env=host_env(),
            )
            self.native.container_id = None
