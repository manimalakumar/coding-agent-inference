import asyncio
import copy
import json
import time
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest
from minisweagent.agents.default import DefaultAgent

from agent.budget import BudgetExhausted
from agent.mini_runtime import openrouter_model, stock_config
from agent.schemas import Budgets, ModelConfig
from evaluator.workspace_policy import verdict
from experiments.configuration_paths import configuration_path
from experiments.records import patch_economics
from harbor_integration.host_workspace import (
    WorkspaceAgent,
    WorkspaceBudget,
    WorkspaceModel,
    termination_reason,
)
from harbor_integration.trace import Trace
from tests.test_mini import response

ROOT = Path(__file__).resolve().parents[1]
COMMON = json.loads(
    (configuration_path(ROOT, "configs/completion-workspace-v1.json")).read_text()
)


def test_patch_success_does_not_require_submission_and_no_check_can_be_skipped():
    checks = {
        n: {"exit_code": 1 if n == "regression_on_baseline" else 0}
        for n in COMMON["required_checks"]
    }
    for submitted in (False, True):
        result = verdict(
            checks,
            required=COMMON["required_checks"],
            changed=["src.py"],
            new_tests=["test_new.py"],
            explicit_submission=submitted,
        )
        assert (
            result["task_success"] == result["patch_accepted"] == result["reward"] == 1
        )
        assert result["explicit_submission"] == int(submitted)
    for name in checks:
        failed = copy.deepcopy(checks)
        failed[name]["exit_code"] = 0 if name == "regression_on_baseline" else 1
        assert (
            verdict(
                failed,
                required=COMMON["required_checks"],
                changed=["src.py"],
                new_tests=["test_new.py"],
                explicit_submission=True,
            )["reward"]
            == 0
        )
        missing = {k: v for k, v in checks.items() if k != name}
        assert (
            verdict(
                missing,
                required=COMMON["required_checks"],
                changed=["src.py"],
                new_tests=["test_new.py"],
                explicit_submission=True,
            )["reward"]
            == 0
        )


def test_last_call_can_edit_without_reserved_tokens_and_next_call_is_blocked(tmp_path):
    budget = WorkspaceBudget(
        Budgets(model_calls=2, generated_tokens=300, response_tokens=300)
    )
    native = openrouter_model(
        ModelConfig(reasoning=None), {"provider": {}, "parameters": {}}, "mock"
    )
    model = WorkspaceModel(native, budget, Trace(tmp_path / "trace.jsonl", "mock"))
    executed = []

    class Environment:
        def get_template_vars(self):
            return dict(system="Linux", release="", version="", machine="")

        def serialize(self):
            return {}

        def execute(self, action):
            executed.append(action["command"])
            return {"returncode": 0, "output": "done", "exception_info": ""}

    config = stock_config()["agent"]
    agent = DefaultAgent(
        model,
        Environment(),
        system_template=config["system_template"],
        instance_template=config["instance_template"],
        cost_limit=0,
        output_path=tmp_path / "native.json",
    )
    payloads = [
        Mock(json=lambda: response("inspect source", tokens=100)),
        Mock(json=lambda: response("write final patch", tokens=100)),
    ]
    with patch("requests.post", side_effect=payloads) as post:
        with pytest.raises(BudgetExhausted, match="model_call"):
            agent.run("mock issue")
    assert post.call_count == 2 and executed == ["inspect source", "write final patch"]
    caps = [json.loads(c.kwargs["data"])["max_tokens"] for c in post.call_args_list]
    assert caps == [300, 200]  # No 128-token reserve on either request.
    assert not budget.submission_only


@pytest.mark.parametrize("which", ["calls", "tokens", "time"])
def test_exhaustion_forbids_repairs_and_new_requests(tmp_path, which):
    budget = WorkspaceBudget(
        Budgets(model_calls=1, generated_tokens=10, response_tokens=10)
    )
    if which == "calls":
        budget.calls = 1
    elif which == "tokens":
        budget.generated = 10
    else:
        budget.started -= budget.limits.seconds
    with pytest.raises(BudgetExhausted):
        budget.repair()
    assert budget.repairs == 0
    native = openrouter_model(
        ModelConfig(reasoning=None), {"provider": {}, "parameters": {}}, "mock"
    )
    with patch("requests.post") as post:
        with pytest.raises(BudgetExhausted):
            WorkspaceModel(
                native, budget, Trace(tmp_path / "trace.jsonl", "mock")
            ).query([])
        post.assert_not_called()


def test_policy_is_identical_for_every_model_and_drift_is_rejected(tmp_path):
    original = json.loads(
        (
            configuration_path(
                ROOT, "configs/harbor-annotated-tools-focused-editing-v1-model.json"
            )
        ).read_text()
    )["agents"][0]["kwargs"]["policy"]
    for model in ("minimax/minimax-m2.5", "qwen/qwen3-coder", "mistralai/ministral-8b"):
        policy = copy.deepcopy(original)
        policy["completion_policy"] = COMMON
        policy["model"]["model"] = model
        WorkspaceAgent(logs_dir=tmp_path, policy=policy)
        policy["completion_policy"] = {**COMMON, "reserve_final_call": True}
        with pytest.raises(ValueError, match="exact frozen completion policy"):
            WorkspaceAgent(logs_dir=tmp_path, policy=policy)


def test_economics_includes_failed_spend_with_separate_submission_rate():
    entries = [
        dict(patch_accepted=True, explicit_submission=False),
        dict(patch_accepted=False, explicit_submission=True),
    ]
    metrics = patch_economics(entries, Decimal("0.12"), True)
    assert metrics["accepted_patches"] == 1
    assert metrics["cost_per_accepted_patch_usd"] == 0.12
    assert metrics["explicit_workflow_completion_rate"] == 0.5
    assert (
        patch_economics(entries, Decimal("0.12"), False)["cost_per_accepted_patch_usd"]
        is None
    )
    assert (
        patch_economics([{}], Decimal("0.12"), True)["patch_economics_status"]
        == "incomplete_patch_acceptance_records"
    )


@pytest.mark.parametrize(
    "status,reason",
    [
        ("Submitted", "submitted"),
        ("model_call_budget_exhausted", "budget_exhausted"),
        ("TimeExceeded", "budget_exhausted"),
        ("RuntimeError", "error"),
    ],
)
def test_termination_is_separate_from_patch_quality(status, reason):
    assert termination_reason(status) == reason


def test_wall_watchdog_quiesces_before_usage_cleanup_and_never_submits_late(tmp_path):
    from harbor.models.agent.context import AgentContext

    policy = json.loads(
        (
            configuration_path(
                ROOT, "configs/harbor-workspace-deliverable-v1-model.json"
            )
        ).read_text()
    )["agents"][0]["kwargs"]["policy"]
    key = tmp_path / "receipt.key"
    key.write_bytes(b"public-unit-test-signing-key")
    policy["receipt_key"] = str(key)
    agent = WorkspaceAgent(policy=policy, fixture="baseline", logs_dir=tmp_path)
    agent.protected_processes = {}
    timestamps = {}

    async def execute(**kwargs):
        timestamps["quiesced"] = time.monotonic()
        return SimpleNamespace(
            return_code=0, stdout='{"workspace_quiesced":true}', stderr=""
        )

    def finish_late(*args):
        time.sleep(1.1)
        timestamps["thread_finished"] = time.monotonic()
        return {"exit_status": "Submitted"}

    environment = SimpleNamespace(exec=execute, upload_file=AsyncMock())
    controller_agent = Mock(run=finish_late)
    limits = Budgets(**(COMMON["agent_budgets"] | {"seconds": 1}))
    with (
        patch(
            "harbor_integration.host_workspace.Budgets.model_validate",
            return_value=limits,
        ),
        patch(
            "harbor_integration.host_workspace.DefaultAgent",
            return_value=controller_agent,
        ),
    ):
        asyncio.run(agent.run("local watchdog test", environment, AgentContext()))
    receipt = json.loads((tmp_path / "receipt.json").read_text())["receipt"]
    assert receipt["termination_reason"] == "budget_exhausted"
    assert not receipt["explicit_submission"]
    assert timestamps["quiesced"] < timestamps["thread_finished"]
    assert receipt["model_requests"] == 0 and receipt["tool_format_repair_rounds"] == 0


def test_frozen_controls_and_historical_outcomes_remain_separate(tmp_path):
    import hashlib

    from experiments.notebook_workflow import records
    from public_preparation.launch import readiness

    # The reviewed numeric projection replaces private archive and trace reads.
    previous = json.loads((ROOT / "evidence/tutorial.json").read_text())["records"]
    by_id = {r["run_id"]: r for r in records(ROOT)}
    for old in previous:
        current = by_id[old["run_id"]]
        for name in (
            "classification",
            "success",
            "submitted",
            "patch_acceptance",
            "termination_reason",
            "calls",
            "cost_usd",
            "configuration_id",
        ):
            assert current.get(name) == old.get(name), (old["run_id"], name)

    observed = json.loads(
        (ROOT / "reports/public-readiness/LC-INT-TEMPERATURE-001.json").read_text()
    )
    controls = observed["controls"]
    assert len(controls) == 4 and all(c["model_requests"] == 0 for c in controls)
    assert all(c["passed"] for c in controls)
    assert not next(c for c in controls if c["fixture"] == "baseline")["accepted"]
    for stop in ("budget_exhausted", "error"):
        case = next(c for c in controls if c["termination_reason"] == stop)
        assert case["accepted"]
    assert not any(r["run_id"].startswith("public-v1-") for r in by_id.values())

    # A public readiness receipt must bind its actual local resource bytes.
    resource = tmp_path / "resource.txt"
    resource.write_text("trusted public resource")
    observed["files"] = {
        "resource.txt": hashlib.sha256(resource.read_bytes()).hexdigest()
    }
    state = tmp_path / ".local/public-v1/LC-INT-TEMPERATURE-001"
    state.mkdir(parents=True)
    (state / "validated.json").write_text(json.dumps(observed))
    assert readiness(tmp_path, "LC-INT-TEMPERATURE-001")
    resource.write_text("modified resource")
    with pytest.raises(RuntimeError, match="resource changed"):
        readiness(tmp_path, "LC-INT-TEMPERATURE-001")
