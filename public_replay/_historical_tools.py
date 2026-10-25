"""Exact historical tool class, extracted for zero-inference reconstruction.

Provenance: harbor_integration/host_task.py, project-authored code.
No model loop or provider client is imported.
"""

import asyncio
import shlex

from minisweagent.environments.docker import DockerEnvironment


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
