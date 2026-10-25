"""Public boundaries, task coverage and historical preservation; no inference."""

import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_complete_task_source_catalog():
    from experiments.portable_prepare import validate_spec
    from public_preparation.catalog import catalog

    numeric = json.loads((ROOT / "notebooks/evidence/tasks.json").read_text())
    refs = json.loads(
        (ROOT / "public_preparation/configs/reference-pins.json").read_text()
    )
    tasks = catalog()
    assert len(tasks) == 13
    for task, spec in tasks.items():
        validate_spec(spec)
        assert spec["starting_commit"] in numeric[task]["starting_commits"]
        assert spec["baseline_checksum"] in numeric[task]["baseline_checksums"]
        assert len(refs[task]["reference_commit"]) == 40
        assert (ROOT / spec["task_bundle"] / "tests/test_acceptance.py").is_file()
        assert (ROOT / spec["task_bundle"] / "tests/verify.py").is_file()


def test_first_exchange_and_cell_are_preserved():
    saved = json.loads((ROOT / "evidence/tutorial.json").read_text())
    identity = json.loads((ROOT / "notebooks/first-exchange-identity.json").read_text())
    notebook = json.loads((ROOT / "main.ipynb").read_text())
    assert len(notebook["cells"]) == 37
    assert (
        hashlib.sha256(
            json.dumps(saved["first_exchange"], sort_keys=True).encode()
        ).hexdigest()
        == identity["first_exchange_sha256"]
    )
    assert (
        hashlib.sha256(
            json.dumps(notebook["cells"][12], sort_keys=True).encode()
        ).hexdigest()
        == identity["first_code_cell_sha256"]
    )
    assert "patch" not in saved["candidate"]
    assert saved["candidate"]["explanation"]
    assert saved["first_exchange"]["summarized"] is True
    assert "historical_request_summary" in saved["first_exchange"]["request"]
    assert (
        hashlib.sha256(
            json.dumps(saved["first_exchange"]["response"], sort_keys=True).encode()
        ).hexdigest()
        == identity["first_response_sha256"]
    )
    assert (
        identity["historical_first_exchange_sha256"]
        != identity["first_exchange_sha256"]
    )


def test_missing_readiness_stops_before_docker_or_credentials(tmp_path):
    from public_preparation.launch import launch

    with patch("public_preparation.launch.subprocess.run") as process:
        with pytest.raises(RuntimeError, match="preparation controls have not passed"):
            launch(
                tmp_path,
                "LC-INT-TEMPERATURE-001",
                confirmed=True,
                job_name="never-launch",
            )
        process.assert_not_called()


def test_no_confirmation_stops_before_resource_access(tmp_path):
    from public_preparation.launch import launch

    with patch("public_preparation.launch.subprocess.run") as process:
        with pytest.raises(ValueError, match="explicit confirmation"):
            launch(
                tmp_path,
                "LC-INT-TEMPERATURE-001",
                confirmed=False,
                job_name="never-launch",
            )
        process.assert_not_called()


def test_control_mode_blocks_clients_before_any_http(monkeypatch):
    monkeypatch.setenv("PUBLIC_NO_INFERENCE", "1")
    monkeypatch.setenv("MSWEA_SILENT_STARTUP", "1")
    monkeypatch.setenv("MSWEA_GLOBAL_CONFIG_DIR", str(ROOT / ".local/test-mini"))
    from agent.mini_runtime import openrouter_model
    from agent.provider import validate_model

    with patch("httpx.get") as get:
        with pytest.raises(RuntimeError, match="Inference is disabled"):
            validate_model(None)
        with pytest.raises(RuntimeError, match="Inference is disabled"):
            openrouter_model(None, None, "offline-placeholder")
        get.assert_not_called()
