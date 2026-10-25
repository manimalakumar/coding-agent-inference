"""Exact setup identities and tamper rejection; no inference or network."""

import json
from types import SimpleNamespace

import pytest

from public_preparation.materialize import (
    ROOT,
    canonical,
    load_mini,
    materialize_acceptance,
    resolve,
    sha,
)


def test_all_effective_prompt_configurations_are_unchanged():
    recipes = json.loads(
        (ROOT / "public_preparation/configs/template-recipes.json").read_text()
    )
    assert len(recipes["configs"]) == 11
    for filename, recipe in recipes["configs"].items():
        assert sha(canonical(load_mini(filename))) == recipe["effective_sha256"]
    identities = json.loads(
        (
            ROOT / "public_preparation/configs/expanded-config-identities.json"
        ).read_text()
    )["files"]
    assert len(identities) == 146
    for filename, identity in identities.items():
        expanded = resolve(json.loads((ROOT / filename).read_text()))
        assert sha(canonical(expanded)) == identity["effective_sha256"]


def test_all_extracted_acceptance_suites_are_byte_identical(tmp_path):
    recipes = json.loads(
        (ROOT / "public_preparation/configs/acceptance-extraction.json").read_text()
    )["tasks"]
    assert len(recipes) == 5
    for task, recipe in recipes.items():
        destination = tmp_path / task / "test_acceptance.py"
        materialize_acceptance(task, ROOT / ".local/upstream.git", destination)
        assert sha(destination.read_bytes()) == recipe["output_sha256"]


def test_changed_material_or_descriptor_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "public_preparation.materialize.subprocess.run",
        lambda *a, **k: SimpleNamespace(stdout=b"altered downloaded tests"),
    )
    destination = tmp_path / "test_acceptance.py"
    with pytest.raises(ValueError, match="hash mismatch"):
        materialize_acceptance("LC-ASYNC-TOOLS-001", tmp_path / "cache", destination)
    assert not destination.exists()
    with pytest.raises(ValueError, match="hash mismatch"):
        resolve({"$mini_config": "mini-swe.yaml", "effective_sha256": "0" * 64})
    monkeypatch.setattr(
        "public_preparation.materialize.metadata.version",
        lambda name: "unexpected-version",
    )
    with pytest.raises(ValueError, match="package version"):
        load_mini("mini-swe.yaml")
