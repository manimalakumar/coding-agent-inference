"""Public reporting migrations preserve historical receipts."""

import asyncio
import hashlib
import hmac
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent.usage import usage
from experiments import acceptance_labels, relocated_records
from experiments.configuration_paths import configuration_path


def launch(root):
    (root / "reports/launches").mkdir(parents=True)
    (root / "configs").mkdir()
    (root / "notebooks").mkdir()
    (root / "notebooks/artifact-paths.json").write_text(
        (
            Path(__file__).resolve().parents[1] / "notebooks/artifact-paths.json"
        ).read_text()
    )
    config = configuration_path(root, "configs/job.json")
    config.write_text(
        json.dumps(
            {
                "tasks": [{"path": "/host-only/tasks/LC-CONTEXT-COUNTER-001"}],
                "agents": [
                    {
                        "model_name": "minimax/minimax-m2.5",
                        "kwargs": {
                            "policy": {
                                "model": {
                                    "model": "minimax/minimax-m2.5",
                                    "provider": {"only": ["test-provider"]},
                                },
                                "budgets": {"model_calls": 30},
                                "source_provenance": {
                                    "starting_commit": "a" * 40,
                                    "baseline_checksum": "b" * 64,
                                },
                            }
                        },
                    }
                ],
            }
        )
    )
    job = "context-counter-response-6000-v1-minimax-attempt-001"
    receipt = {"command": ["harbor", "run", "-c", str(config)], "status": "failed"}
    (root / "reports/launches" / (job + ".json")).write_text(json.dumps(receipt))
    (
        root
        / "reports/context-counter-response-6000-initial-environment-startup-incident.json"
    ).write_text(
        json.dumps(
            {
                "classification": "configuration_failure",
                "model_requests": 0,
                "measured_coding_attempt": False,
                "verified_inference_spend_usd": 0,
                "reported_cost_usd": None,
            }
        )
    )


def test_launch_identity_from_two_roots(tmp_path):
    import shutil

    old = tmp_path / "old"
    launch(old)
    index = relocated_records.launch_index(old)
    original = relocated_records.load_runs(old, index)[0]
    new = tmp_path / "new with spaces"
    shutil.copytree(old, new)
    relocated = relocated_records.load_runs(new, index)[0]
    for key in [
        "run_id",
        "model",
        "task_id",
        "provider",
        "configuration_id",
        "inference_id",
        "commit",
        "baseline_checksum",
        "startup_audit",
        "classification",
        "calls",
        "calls_complete",
    ]:
        assert original[key] == relocated[key], key
    assert relocated["startup_audit"]["classification"] == "configuration_failure"
    assert relocated["startup_audit"]["verified_inference_spend_usd"] == 0
    assert relocated["startup_audit"]["reported_cost_usd"] is None
    assert json.loads(next((new / "reports/launches").glob("*.json")).read_text())[
        "command"
    ][-1].startswith(str(old))
    for ref in ["../outside.json", "/outside.json", "configs/nested/job.json"]:
        with pytest.raises(ValueError):
            relocated_records.resolve_config(
                new, {"configuration_ref": ref, "sha256": "x"}
            )
    (configuration_path(new, "configs/job.json")).write_text("{}")
    with pytest.raises(ValueError, match="checksum"):
        relocated_records.resolve_config(
            new, index["launches"][next(iter(index["launches"]))]
        )


def test_external_launch_rejected(tmp_path):
    root = tmp_path / "root"
    launch(root)
    receipt = next((root / "reports/launches").glob("*.json"))
    receipt.write_text(json.dumps({"command": ["harbor", "-c", "/elsewhere/job.json"]}))
    with pytest.raises(ValueError, match="External"):
        relocated_records.launch_index(root)


@pytest.mark.parametrize(
    "split,label",
    [
        ("development", "trusted acceptance (development task)"),
        ("held_out", "trusted acceptance (held-out task)"),
        (None, "trusted acceptance"),
    ],
)
def test_split_labels(monkeypatch, split, label):
    monkeypatch.setattr(
        acceptance_labels,
        "historical_check_rows",
        lambda record: [{"check": "hidden_acceptance", "stage": "wrong"}],
    )
    assert acceptance_labels.check_rows({"split": split})[0]["stage"] == label


def test_safe_reporting_aliases():
    summary = usage(
        [{"response": {"usage": {"prompt_tokens": None, "completion_tokens": 3}}}]
    )
    assert summary["input_tokens"] is None and summary["output_tokens"] == 3
    assert summary["total_cost_usd"] is None
    assert summary["known_reported_cost_usd"] == 0


@pytest.mark.parametrize("ending", ["Submitted", "budget", "error"])
def test_null_usage_still_captures_signed_receipt(tmp_path, monkeypatch, ending):
    import harbor_integration.host_reporting as adapter
    from agent.budget import BudgetExhausted

    monkeypatch.setenv("OPENROUTER_API_KEY", "synthetic-local-control-not-a-real-key")
    monkeypatch.setattr(adapter, "validate_model", lambda config: None)
    monkeypatch.setattr(adapter, "openrouter_model", lambda *args: SimpleNamespace())

    class Model:
        def __init__(self, native, budget, trace):
            self.budget = budget
            self.usage = [{"prompt_tokens": None, "completion_tokens": 3, "cost": None}]

    monkeypatch.setattr(adapter, "WorkspaceModel", Model)

    class Agent:
        def __init__(self, model, *args, **kwargs):
            self.model = model

        def run(self, instruction):
            self.model.budget.calls = 1
            if ending == "budget":
                raise BudgetExhausted("model_call_budget_exhausted")
            if ending == "error":
                raise RuntimeError("synthetic local error")
            return {"exit_status": "Submitted"}

    monkeypatch.setattr(adapter, "DefaultAgent", Agent)
    key = tmp_path / "fresh.key"
    key.write_bytes(b"local-control-key-only")
    policy = {
        "budgets": {
            "model_calls": 30,
            "generated_tokens": 10000,
            "seconds": 1200,
            "repair_rounds": 2,
        },
        "model": {"model": "minimax/minimax-m2.5", "provider": {"only": ["test"]}},
        "reproduction": {"expected_failure": "EXPECTED"},
        "mini_config": {
            "agent": {"system_template": "local", "instance_template": "local"}
        },
        "receipt_key": str(key),
        "completion_policy": {"evaluation_seconds": 300},
    }

    class Environment:
        async def upload_file(self, source, destination):
            self.uploaded = json.loads(Path(source).read_text())

    environment = Environment()

    async def quiesce(environment):
        return SimpleNamespace(return_code=0, stdout="stopped", stderr="")

    agent = object.__new__(adapter.ComparisonAgentV2)
    agent.policy = policy
    agent.fixture = None
    agent.logs_dir = tmp_path / "logs"
    agent.quiesce = quiesce
    context = SimpleNamespace()
    asyncio.run(agent.run("control only", environment, context))
    signed = environment.uploaded
    receipt = signed["receipt"]
    assert receipt["workspace_quiesced"] and receipt["model_requests"] == 1
    assert receipt["reported_cost"] is None and receipt["output_tokens"] == 3
    assert receipt["usage_summary"]["input_tokens"] is None
    signature = hmac.new(
        key.read_bytes(), json.dumps(receipt, sort_keys=True).encode(), hashlib.sha256
    ).hexdigest()
    assert hmac.compare_digest(signature, signed["signature"])
    assert receipt["explicit_submission"] == (ending == "Submitted")
    assert (
        receipt["execution_error"] is not None
        if ending == "error"
        else receipt["execution_error"] is None
    )


def test_exact_historical_tool_class_is_preserved():
    import ast

    from public_replay import _historical_tools

    original = Path(__file__).resolve().parents[1] / "harbor_integration/host_task.py"
    copied = Path(_historical_tools.__file__)

    def class_source(path):
        source = path.read_text()
        node = next(
            n
            for n in ast.parse(source).body
            if isinstance(n, ast.ClassDef) and n.name == "HarborTools"
        )
        return ast.get_source_segment(source, node)

    assert class_source(original) == class_source(copied)


def test_partial_call_stream_stays_unknown_when_aggregated():
    from agent.results import totals

    row = {
        "model": "synthetic",
        "run_id": "control",
        "success": False,
        **usage(
            [
                {
                    "response": {
                        "usage": {
                            "prompt_tokens": 1,
                            "completion_tokens": 2,
                            "cost": 0.5,
                        }
                    }
                }
            ],
            calls_complete=False,
        ),
    }
    result = totals([row])["synthetic"]
    assert result["known_reported_cost_all_attempts_usd"] == 0.5
    assert result["total_cost_all_attempts_usd"] is None
    assert result["cost_complete"] is False


def test_canonical_reporting_keeps_unknown_usage_and_legacy_rows():
    from agent import results

    assert results.usage is usage
    normalized = results.usage([{"response": {"usage": {"cost": None}}}])
    assert normalized["input_tokens_reported"] == 0
    assert normalized["token_usage_complete"] is False
    assert normalized["total_cost_usd"] is None

    legacy = {
        "model": "synthetic",
        "run_id": "legacy",
        "success": True,
        "requests": 1,
        "known_reported_cost_usd": 0.5,
        "requests_without_cost_record": 0,
    }
    complete = results.totals([legacy])["synthetic"]
    assert complete["cost_complete"] is True
    assert complete["cost_per_successful_task_usd"] == 0.5
    incomplete = {**legacy, "run_id": "partial", "cost_complete": False}
    mixed = results.totals([legacy, incomplete])["synthetic"]
    assert mixed["known_reported_cost_all_attempts_usd"] == 1.0
    assert mixed["total_cost_all_attempts_usd"] is None
    assert mixed["cost_per_successful_task_usd"] is None
    assert mixed["metric_status"] == "incomplete_cost_records"


def test_missing_cost_cannot_be_overridden_by_complete_flag():
    from agent.results import totals

    row = {
        "model": "synthetic",
        "run_id": "inconsistent",
        "success": True,
        "requests": 2,
        "known_reported_cost_usd": 0.5,
        "requests_without_cost_record": 1,
        "cost_complete": True,
    }
    result = totals([row])["synthetic"]
    assert result["total_cost_all_attempts_usd"] is None
    assert result["cost_complete"] is False


def test_public_launcher_uses_canonical_parser():
    import ast

    tree = ast.parse(
        (
            Path(__file__).resolve().parents[1] / "public_preparation/launch.py"
        ).read_text()
    )
    assert any(
        isinstance(node, ast.ImportFrom)
        and node.module == "experiments.records"
        and any(item.name == "harbor_record" for item in node.names)
        for node in ast.walk(tree)
    )
