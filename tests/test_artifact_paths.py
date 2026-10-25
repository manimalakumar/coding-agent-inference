"""Historical artifact paths retain the same evidence under descriptive filenames."""

import hashlib
import json
from pathlib import Path

import pytest

from experiments.artifact_paths import artifact_path, resolve_artifact_path
from public_replay.bundle import Bundle

ROOT = Path(__file__).resolve().parents[1]


def test_historical_payloads_and_paths_are_preserved():
    registry = json.loads((ROOT / "notebooks/artifact-paths.json").read_text())
    for old, new in registry["aliases"].items():
        current = artifact_path(ROOT, old)
        assert current == ROOT / new
        assert resolve_artifact_path(ROOT / old) == current
        assert not (ROOT / old).exists()
        assert hashlib.sha256(current.read_bytes()).hexdigest() == registry.get(
            "adapted_sha256", {}
        ).get(new, registry["historical_sha256"][old])


def test_old_and_current_bundle_names_have_identical_content():
    old = Bundle(ROOT / "notebooks/evidence-v1")
    current = Bundle(ROOT / "notebooks/evidence")
    assert old.root == current.root
    assert old.manifest["content_id"] == current.manifest["content_id"]
    assert old.read("attempts.json") == current.read("attempts.json")


@pytest.mark.parametrize(
    "name", ["../outside", "/outside", "reports/../../outside", "reports\\outside"]
)
def test_artifact_reference_cannot_escape(name):
    with pytest.raises(ValueError, match="Unsafe"):
        artifact_path(ROOT, name)


@pytest.mark.parametrize("target", ["../outside", "/outside", "reports/../../outside"])
def test_artifact_alias_cannot_escape(tmp_path, target):
    (tmp_path / "notebooks").mkdir()
    (tmp_path / "notebooks/artifact-paths.json").write_text(
        json.dumps({"schema_version": 1, "aliases": {"reports/old.json": target}})
    )
    with pytest.raises(ValueError, match="escapes"):
        artifact_path(tmp_path, "reports/old.json")


def test_artifact_symlink_cannot_escape(tmp_path):
    (tmp_path / "notebooks").mkdir()
    (tmp_path / "reports").mkdir()
    (tmp_path / "reports/target.json").symlink_to(tmp_path.parent / "outside")
    (tmp_path / "notebooks/artifact-paths.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "aliases": {"reports/old.json": "reports/target.json"},
            }
        )
    )
    with pytest.raises(ValueError, match="escapes"):
        artifact_path(tmp_path, "reports/old.json")
