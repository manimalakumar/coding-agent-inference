"""Offline integration test: scripted responses, real Docker, no inference requests.

The trusted known-good fixture is deliberately excluded from measured runs and model
contexts. HTTP POST is mocked for the entire controller execution.
"""

import argparse
import json
import time
import uuid
from pathlib import Path
from unittest.mock import Mock, patch

from agent.controller import Controller
from agent.schemas import ModelConfig, SandboxConfig, Task
from evaluator.selftest import selftest
from sandbox.docker import Sandbox, digest, manifest
from sandbox.prepare import PROJECT, write_json


def local_test(setup_dir: Path):
    setup = json.loads((setup_dir / "setup.json").read_text())
    assert setup["setup_status"] == "validated"
    assert setup["sandbox_config"]["schema_version"] == 4
    fixture = selftest(setup_dir)
    assert fixture["status"] == "checks_passed_pending_review", fixture
    fixture_workspace = Path(fixture["selftest_directory"]) / "workspace"
    task = Task.model_validate(setup["task"])
    run = setup_dir / ("mini-local-test-" + uuid.uuid4().hex[:8])
    run.mkdir()
    box = Sandbox(
        SandboxConfig.model_validate(setup["sandbox_config"]),
        task,
        setup_dir / "baseline",
        run / "workspace",
        setup_dir / "task.json",
        setup["image_id"],
        setup_dir.parents[1] / "resources",
    )
    box.prepare()
    fixture_files = {
        task.source_paths[0] + "/_client_utils.py": (
            fixture_workspace / task.source_paths[0] / "_client_utils.py"
        ).read_text(),
        task.test_paths[0] + "/test_timeout_selftest.py": (
            fixture_workspace / task.test_paths[0] / "test_timeout_selftest.py"
        ).read_text(),
    }
    # The fixture edit is performed via the same standard bash tool as an agent edit.
    edit = "python - <<'PY'\nimport json\nfrom pathlib import Path\n"
    edit += "files = json.loads(" + repr(json.dumps(fixture_files)) + ")\n"
    edit += "for name, content in files.items():\n    Path('/workspace', name).write_text(content)\nPY"
    commands = [
        task.reproduction_command,
        edit,
        "python -m pytest tests/unit_tests/test_timeout_selftest.py -q",
        "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT",
    ]
    responses = []
    for i, command in enumerate(commands):
        response = {
            "provider": "scripted-local-fixture",
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "Offline harness test; no model generated this.",
                        "tool_calls": [
                            {
                                "id": f"call{i}",
                                "type": "function",
                                "function": {
                                    "name": "bash",
                                    "arguments": json.dumps({"command": command}),
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "cost": 0},
        }
        responses.append(Mock(json=lambda raw=response: raw))
    started = time.monotonic()
    with patch("requests.post", side_effect=responses) as http:
        result = Controller(
            box,
            ModelConfig(reasoning=None),
            run,
            "",
            {"provider": {}, "parameters": {}},
        ).execute()
        scripted_responses = http.call_count
    report = {
        "kind": "offline_mini_swe_integration_test_not_model_performance",
        "framework": "mini-swe-agent==2.4.6",
        "agent_type": "minisweagent.agents.default.DefaultAgent",
        "environment_type": "minisweagent.environments.docker.DockerEnvironment",
        "setup_directory": str(setup_dir),
        "validation_directory": str(run),
        "model_api_requests": 0,
        "scripted_responses": scripted_responses,
        "inference_cost_usd": 0,
        "elapsed_seconds": time.monotonic() - started,
        "status": result["termination_reason"],
        "baseline_untouched": digest(manifest(box.baseline))
        == setup["baseline_checksum"],
        "fixture_evaluator_selftest": fixture["selftest_directory"],
        "evaluation": result["evaluation"],
    }
    write_json(run / "local-validation.json", report)
    write_json(PROJECT / "reports/mini-swe-local-validation.json", report)
    assert report["status"] == "checks_passed_pending_review", report
    assert report["baseline_untouched"]
    assert scripted_responses == 4
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--setup", type=Path, required=True)
    result = local_test(parser.parse_args().setup.resolve())
    print(
        json.dumps(
            {
                k: result[k]
                for k in (
                    "status",
                    "model_api_requests",
                    "scripted_responses",
                    "baseline_untouched",
                    "validation_directory",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
