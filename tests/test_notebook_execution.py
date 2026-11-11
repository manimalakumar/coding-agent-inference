"""Exercise the notebook HTTP callable with provider fixtures, never live inference."""

import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import requests

from experiments import notebook_execution as execution

ROOT = Path(__file__).resolve().parents[1]
NB = json.loads((ROOT / "main.ipynb").read_text())


def api_callable():
    source = "".join(NB["cells"][11]["source"])
    source = source[: source.index('\nif MODE == "live":')]
    scope = {"json": json, "os": os}
    exec(source, scope)
    return scope["call_model_api"]


def fixture_response(status=200):
    response = requests.Response()
    response.status_code = status
    response._content = json.dumps(
        {
            "id": "fixture-generation",
            "provider": "MiniMax",
            "choices": [
                {
                    "finish_reason": "tool_calls",
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "fixture-tool",
                                "type": "function",
                                "function": {
                                    "name": "bash",
                                    "arguments": '{"command":"echo fixture"}',
                                },
                            }
                        ],
                    },
                }
            ],
            "usage": {"prompt_tokens": 17, "completion_tokens": 6, "cost": 0.002},
        }
    ).encode()
    return response


def test_notebook_transport_is_actual_native_model_request(monkeypatch, tmp_path):
    from agent.budget import Budget
    from agent.controller import Trace
    from agent.mini_runtime import MeteredModel, openrouter_model
    from agent.schemas import Budgets, ModelConfig

    monkeypatch.delenv("PUBLIC_NO_INFERENCE", raising=False)
    post = Mock(return_value=fixture_response())
    monkeypatch.setattr(requests, "post", post)
    plan = {
        "provider": {"only": ["minimax/fp8"], "allow_fallbacks": False},
        "parameters": {},
    }
    native = openrouter_model(
        ModelConfig(model="minimax/minimax-m2.5"), plan, "fixture"
    )
    metered = MeteredModel(
        native, Budget(Budgets()), Trace(tmp_path / "requests.jsonl", "fixture")
    )
    with execution.transport_scope(api_callable()):
        message = metered.query([{"role": "user", "content": "Local fixture"}])
    assert (
        requests.post is post
    )  # The global transport was never replaced by the adapter.
    post.assert_called_once()
    sent = json.loads(post.call_args.kwargs["data"])
    assert sent["model"] == "minimax/minimax-m2.5"
    assert sent["provider"] == plan["provider"]
    assert sent["messages"] == [{"role": "user", "content": "Local fixture"}]
    assert sent["tools"] and sent["usage"] == {"include": True}
    assert message["extra"]["actions"][0]["command"] == "echo fixture"
    assert metered.budget.calls == 1 and metered.budget.generated == 6
    assert metered.usage[0]["cost"] == 0.002


def test_notebook_transport_preserves_failed_http_response(monkeypatch):
    monkeypatch.delenv("PUBLIC_NO_INFERENCE", raising=False)
    response = fixture_response(400)
    post = Mock(return_value=response)
    monkeypatch.setattr(requests, "post", post)
    actual = api_callable()(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={},
        data=json.dumps(
            {"model": "fixture", "provider": {}, "messages": [], "max_tokens": 6}
        ),
        timeout=60,
    )
    assert actual is response
    with pytest.raises(requests.exceptions.HTTPError):
        actual.raise_for_status()
    assert actual.json()["usage"]["cost"] == 0.002
    post.assert_called_once()


def test_native_transport_restored_on_error():
    from minisweagent.models import openrouter_model

    original = openrouter_model.requests
    original_stats = openrouter_model.GLOBAL_MODEL_STATS
    with pytest.raises(RuntimeError, match="fixture failure"):
        with execution.transport_scope(Mock()):
            assert openrouter_model.GLOBAL_MODEL_STATS is not original_stats
            raise RuntimeError("fixture failure")
    assert openrouter_model.requests is original
    assert openrouter_model.GLOBAL_MODEL_STATS is original_stats


def test_worker_preserves_kernel_signals_and_cli_exit_status(monkeypatch):
    import signal

    before = signal.getsignal(signal.SIGTERM)
    fake_jobs = SimpleNamespace(signal=signal)

    def app(**kwargs):
        # The CLI's registration is local; the process keeps the kernel handler.
        fake_jobs.signal.signal(signal.SIGTERM, lambda *args: None)
        assert signal.getsignal(signal.SIGTERM) is before
        return 7

    monkeypatch.setitem(sys.modules, "harbor.cli.main", SimpleNamespace(app=app))
    monkeypatch.setitem(sys.modules, "harbor.cli.jobs", fake_jobs)
    monkeypatch.setitem(sys.modules, "harbor.cli", SimpleNamespace(jobs=fake_jobs))
    with ThreadPoolExecutor(max_workers=1) as worker:
        result = worker.submit(
            execution.run_harbor, ["harbor", "run"], {}, Mock()
        ).result()
    assert result.returncode == 7
    assert fake_jobs.signal is signal
    assert signal.getsignal(signal.SIGTERM) is before


def test_cli_environment_restored_on_error(monkeypatch):
    def failed_app(**kwargs):
        assert os.environ["HARBOR_TELEMETRY"] == "0"
        os.environ["OPENROUTER_API_KEY"] = "fixture"
        raise RuntimeError("fixture CLI failure")

    monkeypatch.setitem(sys.modules, "harbor.cli.main", SimpleNamespace(app=failed_app))
    before = dict(os.environ)
    with pytest.raises(RuntimeError, match="fixture CLI failure"):
        execution.run_harbor(["harbor", "run"], {"NOTEBOOK_FIXTURE": "1"}, Mock())
    assert dict(os.environ) == before


def test_existing_launcher_dispatch_and_cleanup(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("PUBLIC_NO_INFERENCE", raising=False)
    monkeypatch.setattr(
        execution,
        "version",
        lambda pkg: {"harbor": "0.24.0", "mini-swe-agent": "2.4.6"}[pkg],
    )
    original = execution.public_launch.subprocess
    native_post = Mock()
    runner = Mock(return_value=subprocess.CompletedProcess([], 0))
    monkeypatch.setattr(execution, "run_harbor", runner)

    def existing_launch(root, task, **kwargs):
        argv = [str(root / ".venv/bin/harbor"), "run", "-c", "prepared-config"]
        result = execution.public_launch.subprocess.run(
            argv, cwd=root, env={"FIXTURE": "1"}
        )
        assert result.returncode == 0
        return {"record": "fixture"}

    monkeypatch.setattr(execution.public_launch, "launch", existing_launch)
    result = execution.launch(
        tmp_path,
        "task",
        model="minimax",
        confirmed=True,
        job_name="one",
        post=native_post,
    )
    assert result == {"record": "fixture"}
    runner.assert_called_once()
    assert runner.call_args.args[2] is native_post
    assert execution.public_launch.subprocess is original
    assert not execution._LOCK.locked()


@pytest.mark.parametrize("confirmed,disabled", [(False, False), (True, True)])
def test_inference_gates_precede_client_and_resource_access(
    monkeypatch, tmp_path, confirmed, disabled
):
    if disabled:
        monkeypatch.setenv("PUBLIC_NO_INFERENCE", "1")
    else:
        monkeypatch.delenv("PUBLIC_NO_INFERENCE", raising=False)
    launch = Mock()
    monkeypatch.setattr(execution.public_launch, "launch", launch)
    with pytest.raises((ValueError, RuntimeError)):
        execution.launch(
            tmp_path,
            "task",
            model="minimax",
            confirmed=confirmed,
            job_name="one",
            post=Mock(),
        )
    launch.assert_not_called()


def test_batch_checks_all_resources_before_model_calls(monkeypatch, tmp_path):
    readiness = Mock(side_effect=[{}, RuntimeError("missing evaluator")])
    monkeypatch.setattr(execution.public_launch, "readiness", readiness)
    plan = [
        {"attempt_id": str(i), "task_id": task, "model": "minimax"}
        for i, task in enumerate(["task-a", "task-b"])
    ]
    with pytest.raises(RuntimeError, match="missing evaluator"):
        execution.check_batch(tmp_path, plan)
    assert readiness.call_count == 2


@pytest.mark.parametrize(
    "index,flag,batch,expected",
    [
        (21, "RUN_DEVELOPMENT", "DEVELOPMENT_BATCH_ID", 5),
        (27, "RUN_HELD_OUT", "HELD_OUT_BATCH_ID", 40),
    ],
)
def test_each_stage_shows_preparation_and_launches_the_real_callable(
    index, flag, batch, expected
):
    code = "".join(NB["cells"][index]["source"])
    marker = (
        "development_roster, roster_audit ="
        if index == 21
        else "heldout_roster, heldout_audit, heldout_counts ="
    )
    code = code[: code.index(marker)]
    post = Mock()
    launcher = Mock()
    launcher.launch.return_value = {
        "record": {"run_id": "fixture"},
        "public_protocol": "public-fixture-v1",
    }

    scope = {
        "MODE": "live",
        "CONFIRM_PAID_EXECUTION": True,
        "ROOT": ROOT,
        "execution": launcher,
        "display": Mock(),
        "pd": __import__("pandas"),
        "HTML": Mock(),
        "presentation": Mock(),
        "call_model_api": post,
        "workflow": Mock(),
    }
    scope["workflow"].accounting.return_value = (scope["pd"].DataFrame(), {})
    exec(code, scope)
    plan = scope["development_plan" if index == 21 else "heldout_plan"]
    assert len(plan) == expected
    launcher.launch.assert_not_called()
    # Enable the exact notebook plan with fixtures; no provider client is constructed.
    code = code.replace(f"{flag} = False", f"{flag} = True").replace(
        f"{batch} = None", f'{batch} = "fixture-batch"'
    )
    exec(code, scope)
    assert launcher.launch.call_count == expected
    assert all(call.kwargs["post"] is post for call in launcher.launch.call_args_list)
    assert (
        len({call.kwargs["job_name"] for call in launcher.launch.call_args_list})
        == expected
    )
    assert "--allow-setup-network --controls" in code
