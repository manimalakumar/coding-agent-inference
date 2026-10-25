"""Appended compatibility costs must retain history and count each receipt once."""

import json

import pytest

from experiments.setup_accounting import setup_entries


def test_historical_and_additional_receipts_are_combined_without_modification(tmp_path):
    reports = tmp_path / "reports"
    reports.mkdir()
    old = reports / "setup-inference-costs.json"
    old.write_text(
        json.dumps({"entries": [{"result": "old/result.json", "cost_usd": 0.1}]})
    )
    original = old.read_bytes()
    (reports / "setup-inference-costs-additions.json").write_text(
        json.dumps({"entries": [{"result": "new/result.json", "cost_usd": None}]})
    )
    combined = setup_entries(tmp_path)
    assert [entry["result"] for entry in combined] == [
        "old/result.json",
        "new/result.json",
    ]
    assert combined[1]["cost_usd"] is None
    assert old.read_bytes() == original


def test_duplicate_setup_receipt_is_rejected(tmp_path):
    reports = tmp_path / "reports"
    reports.mkdir()
    for name in ["setup-inference-costs.json", "setup-inference-costs-additions.json"]:
        (reports / name).write_text(
            json.dumps({"entries": [{"result": "same/result.json"}]})
        )
    with pytest.raises(ValueError, match="Duplicate"):
        setup_entries(tmp_path)
