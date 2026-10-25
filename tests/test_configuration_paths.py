"""Historical configuration names resolve to the same pinned configuration bytes."""

import hashlib
import json
from pathlib import Path

import pytest

from experiments.configuration_paths import configuration_aliases, configuration_path
from public_preparation.materialize import canonical, load_mini, sha

ROOT = Path(__file__).resolve().parents[1]


def test_all_historical_names_have_one_descriptive_file():
    registry = json.loads((ROOT / "configs/configuration-paths.json").read_text())
    for old, current in registry["aliases"].items():
        path = configuration_path(ROOT, old)
        assert path == configuration_path(ROOT, current)
        assert path.is_file() and not (ROOT / "configs" / old).exists()
        if old in registry["historical_sha256"]:
            assert hashlib.sha256(path.read_bytes()).hexdigest() == registry.get(
                "adapted_sha256", {}
            ).get(current, registry["historical_sha256"][old])


def test_pinned_template_hashes_survive_both_names():
    recipes = json.loads(
        (ROOT / "public_preparation/configs/template-recipes.json").read_text()
    )
    aliases = configuration_aliases(ROOT)
    for old, recipe in recipes["configs"].items():
        current = aliases.get(old, old)
        assert load_mini(old) == load_mini(current)
        assert sha(canonical(load_mini(current))) == recipe["effective_sha256"]


@pytest.mark.parametrize(
    "name",
    ["../secret.json", "/outside.json", "configs/../secret.json", "nested/config.json"],
)
def test_config_references_cannot_escape(name):
    with pytest.raises(ValueError, match="direct configs"):
        configuration_path(ROOT, name)


@pytest.mark.parametrize("target", ["../secret.json", "/outside.json", ".."])
def test_invalid_registry_target_is_rejected(tmp_path, target):
    folder = tmp_path / "configs"
    folder.mkdir()
    (folder / "configuration-paths.json").write_text(
        json.dumps({"schema_version": 1, "aliases": {"old.json": target}})
    )
    with pytest.raises(ValueError, match="escapes configs"):
        configuration_path(tmp_path, "old.json")


def test_symlink_alias_is_rejected(tmp_path):
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs/alias.json").symlink_to(tmp_path / "secret.json")
    with pytest.raises(ValueError, match="escapes configs"):
        configuration_path(tmp_path, "alias.json")


def test_public_policy_binds_registry_and_renamed_prompt():
    from public_preparation.runtime import policy_fingerprints

    values = policy_fingerprints(["configs/mini-swe-v5.yaml"])
    assert set(values) == {
        "configs/mini-swe-explicit-test-policy.yaml",
        "configs/configuration-paths.json",
        "experiments/configuration_paths.py",
    }


def test_historical_harbor_paths_resolve_without_changing_settings():
    from public_preparation.materialize import current_source_path, resolve
    from public_preparation.runtime import policy_fingerprints

    original = {
        "agents": [{"import_path": "harbor_integration.host_mini_v2:HostMiniSweAgent"}]
    }
    prepared = resolve(original)
    assert (
        original["agents"][0]["import_path"]
        == "harbor_integration.host_mini_v2:HostMiniSweAgent"
    )
    assert (
        prepared["agents"][0]["import_path"]
        == "harbor_integration.host_task:HostMiniSweAgent"
    )
    assert resolve(
        {"import_path": "harbor_integration.host_reporting_v2:ReportingAgent"}
    ) == {"import_path": "harbor_integration.host_reporting:ReportingAgent"}
    assert (
        current_source_path("harbor_integration/host_mini_v2.py")
        == "harbor_integration/host_task.py"
    )
    values = policy_fingerprints(["harbor_integration/host_mini_v2.py"])
    assert (
        values["harbor_integration/host_task.py"]
        == hashlib.sha256(
            (ROOT / "harbor_integration/host_task.py").read_bytes()
        ).hexdigest()
    )
    assert "harbor_integration/host_mini_v2.py" not in values
