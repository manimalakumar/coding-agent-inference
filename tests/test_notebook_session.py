"""Live notebook routing and failure controls; no provider requests."""

import json
import os
from pathlib import Path
from unittest.mock import Mock

import pytest

from experiments import notebook_session as session

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = json.loads((ROOT / "main.ipynb").read_text())


def launch_cell():
    return "".join(NOTEBOOK["cells"][11]["source"])


def namespace():
    return {
        "MODE": "live",
        "CONFIRM_PAID_EXECUTION": True,
        "FRESH_ATTEMPT_ID": "fresh-control",
        "EXECUTE_TASK_ID": "LC-ANNOTATED-TOOLS-001",
        "EXECUTE_MODEL": "glm5",
        "ROOT": ROOT,
        "workflow": Mock(),
        "execution": Mock(),
        "os": os,
        "json": json,
        "presentation": Mock(),
        "display": Mock(),
        "HTML": Mock(),
    }


def test_default_live_requires_explicit_confirmation():
    setup = "".join(NOTEBOOK["cells"][1]["source"])
    assert 'MODE = "live"' in setup
    assert 'os.environ.get("INFERENCE_ECONOMICS_MODE", MODE)' in setup
    ns = namespace()
    ns["CONFIRM_PAID_EXECUTION"] = False
    with pytest.raises(ValueError, match="explicit confirmation"):
        exec(launch_cell(), ns)
    ns["execution"].launch.assert_not_called()
    ns["workflow"].execute_or_replay.assert_not_called()


def test_live_calls_existing_public_launcher():
    ns = namespace()
    result = {
        "record": {"classification": "patch_accepted", "run_id": "fresh-control/trial"},
        "mode": "execute",
        "launched": True,
        "public_protocol": "public-lc-annotated-tools-001-v1",
    }
    ns["execution"].launch.return_value = result
    exec(launch_cell(), ns)
    ns["execution"].launch.assert_called_once_with(
        ROOT,
        task_id="LC-ANNOTATED-TOOLS-001",
        model="glm5",
        confirmed=True,
        job_name="fresh-control",
        post=ns["call_model_api"],
    )
    assert ns["run"] is result["record"]
    ns["workflow"].execute_or_replay.assert_not_called()


def test_live_failure_never_substitutes_completed_results():
    ns = namespace()
    ns["execution"].launch.side_effect = RuntimeError("provider unavailable")
    with pytest.raises(RuntimeError, match="provider unavailable"):
        exec(launch_cell(), ns)
    ns["workflow"].execute_or_replay.assert_not_called()


def test_walkthrough_task_cannot_change_between_issue_and_call():
    ns = namespace()
    ns["EXECUTE_TASK_ID"] = "LC-INT-TEMPERATURE-001"
    with pytest.raises(ValueError, match="opening walkthrough follows annotated-tools"):
        exec(launch_cell(), ns)
    ns["execution"].launch.assert_not_called()


def test_walkthrough_preparation_rejects_a_different_task_before_loading_resources():
    workflow = Mock()
    session = Mock()
    ns = {
        "MODE": "live",
        "EXECUTE_TASK_ID": "LC-TIMEOUT-001",
        "workflow": workflow,
        "session": session,
    }
    with pytest.raises(ValueError, match="opening walkthrough follows annotated-tools"):
        exec("".join(NOTEBOOK["cells"][3]["source"]), ns)
    workflow.preparation.assert_not_called()
    session.preparation.assert_not_called()


def test_visible_http_excerpt_is_the_operation_the_agent_uses():
    import ast
    import re

    markdown = "".join(NOTEBOOK["cells"][10]["source"])
    excerpt = re.search(r"```python\n(.*?)\n```", markdown, re.S).group(1)
    source = ast.parse("".join(NOTEBOOK["cells"][11]["source"]))
    invocation = next(
        node
        for node in ast.walk(source)
        if isinstance(node, ast.Assign)
        and isinstance(node.value, ast.Call)
        and ast.unparse(node.value.func) == "requests.post"
    )
    assert ast.dump(ast.parse(excerpt).body[0]) == ast.dump(invocation)


def test_missing_public_preparation_stops_without_fallback(tmp_path, monkeypatch):
    load = Mock()
    monkeypatch.setattr(session.workflow, "public_preparation", load)
    with pytest.raises(RuntimeError, match="controls have not passed"):
        session.preparation(tmp_path, "LC-ANNOTATED-TOOLS-001", "minimax")
    load.assert_not_called()


def test_prior_first_response_identity_is_exact():
    proof = json.loads((ROOT / "evidence/first-invocation-provenance.json").read_text())
    identity = json.loads((ROOT / "notebooks/first-exchange-identity.json").read_text())
    assert proof["first_response_sha256"] == identity["first_response_sha256"]
    assert proof["first_response_matches_original_trace"] is True
    assert proof["model_requests_in_original_run"] == 21
    assert proof["model_requests_during_validation"] == 0


def test_transport_display_reads_runtime_package_without_an_extra_probe():
    source = "".join(NOTEBOOK["cells"][7]["source"])
    assert '"public_preparation/launch.py", "launch"' in source
    assert '"OpenRouterModel._query"' in source
    assert "requests.post(" not in source


@pytest.mark.parametrize("launched,mode", [(True, "replay"), (False, "live")])
def test_invocation_preservation_uses_launch_result_not_display_mode(
    launched, mode, capsys
):
    from experiments import notebook_session

    workflow = Mock()
    workflow.first_exchange.return_value = {
        "request": {"model": "fixture"},
        "status": "recorded",
        "response_status": "recorded",
    }
    presentation = Mock()
    presentation.exchange_html.return_value = "fixture response"
    presentation.preserve_invocation.return_value = "archive-path"
    scope = {
        "ROOT": ROOT,
        "MODE": mode,
        "result": {"launched": launched},
        "run": {"run_id": "fixture"},
        "preview": {"model": "fixture"},
        "workflow": workflow,
        "presentation": presentation,
        "session": notebook_session,
        "display": Mock(),
        "HTML": Mock(),
    }
    exec("".join(NOTEBOOK["cells"][12]["source"]), scope)
    if launched:
        presentation.preserve_invocation.assert_called_once_with(
            ROOT, scope["run"], "fixture response", "execute"
        )
    else:
        presentation.preserve_invocation.assert_not_called()
    output = capsys.readouterr().out
    assert "Live call." in output and "archive-path" not in output
