"""Reconstruct verified upstream material locally; never download during Replay."""

import ast
import copy
import hashlib
import json
import subprocess
from importlib import metadata, resources
from pathlib import Path

from experiments.configuration_paths import (
    configuration_path,
    historical_configuration_name,
)

ROOT = Path(__file__).resolve().parents[1]

# Resolve historical configuration references without rewriting archived settings.
MODULE_RENAMES = {
    "harbor_integration.host_mini_v2": "harbor_integration.host_task",
    "harbor_integration.host_reporting_v2": "harbor_integration.host_reporting",
}


def current_import_path(value):
    module, separator, attribute = value.partition(":")
    return MODULE_RENAMES.get(module, module) + separator + attribute


def current_source_path(value):
    aliases = {
        old.replace(".", "/") + ".py": new.replace(".", "/") + ".py"
        for old, new in MODULE_RENAMES.items()
    }
    return aliases.get(value, value)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def checked(data, expected, label):
    if sha(data) != expected:
        raise ValueError("Prepared content hash mismatch: " + label)
    return data


def load_mini(filename):
    """Load a pinned package resource and apply only project-authored additions."""
    import yaml

    source = configuration_path(ROOT, filename)
    filename = historical_configuration_name(ROOT, source.name)

    recipe = json.loads(
        (ROOT / "public_preparation/configs/template-recipes.json").read_text()
    )
    if metadata.version(recipe["package"]) != recipe["package_version"]:
        raise ValueError("mini-SWE-agent package version differs from pinned templates")
    raw = resources.files("minisweagent").joinpath(recipe["resource"]).read_bytes()
    checked(raw, recipe["resource_sha256"], "installed mini-SWE-agent resource")
    base = yaml.safe_load(raw)
    result = yaml.safe_load(source.read_text())
    config = recipe["configs"][filename]
    for field, entry in config["templates"].items():
        section, key = field.split(".")
        original = base[section][key]
        value = "".join(
            original[slice(*part["copy"])] if "copy" in part else part["project_text"]
            for part in entry["chunks"]
        )
        checked(value.encode(), entry["sha256"], filename + ":" + field)
        result[section][key] = value
    checked(canonical(result), config["effective_sha256"], filename)
    return result


def resolve(value):
    """Expand configuration references in memory; checked-in JSON stays source-free."""
    if isinstance(value, dict):
        if "$mini_config" in value:
            result = load_mini(value["$mini_config"])
            checked(canonical(result), value["effective_sha256"], value["$mini_config"])
            return result
        return {
            key: current_import_path(item)
            if key == "import_path" and isinstance(item, str)
            else resolve(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [resolve(item) for item in value]
    return copy.deepcopy(value)


def materialize_acceptance(task, cache, destination):
    """Extract exact test bodies from already fetched commits into verifier storage."""
    recipes = json.loads(
        (ROOT / "public_preparation/configs/acceptance-extraction.json").read_text()
    )
    if task not in recipes["tasks"]:
        return None
    recipe = recipes["tasks"][task]
    sources = {}
    for name, expected in recipe["source_sha256"].items():
        raw = subprocess.run(
            ["git", "--git-dir", str(cache), "show", recipe["commit"] + ":" + name],
            check=True,
            capture_output=True,
        ).stdout
        checked(raw, expected, name)
        sources[name] = raw.decode()
    chunks = []
    for part in recipe["chunks"]:
        if "project_text" in part:
            chunks.append(part["project_text"])
            continue
        text = sources[part["source"]]
        if not part.get("whole_file"):
            node = next(
                n
                for n in ast.parse(text).body
                if getattr(n, "name", None) == part["symbol"]
            )
            start = min([node.lineno] + [d.lineno for d in node.decorator_list]) - 1
            text = "".join(text.splitlines(keepends=True)[start : node.end_lineno])
        text = part.get("prefix", "") + text + part.get("suffix", "")
        if "normalize_trailing_newlines" in part:
            text = text.rstrip("\n") + "\n" * part["normalize_trailing_newlines"]
        chunks.append(text)
    raw = "".join(chunks).encode()
    checked(raw, recipe["output_sha256"], task + " acceptance")
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(raw)
    license_text = subprocess.run(
        ["git", "--git-dir", str(cache), "show", recipe["commit"] + ":LICENSE"],
        check=True,
        capture_output=True,
    ).stdout
    checked(license_text, recipe["license_sha256"], task + " upstream license")
    destination.with_name("UPSTREAM-LICENSE").write_bytes(license_text)
    return {"sha256": sha(raw), "commit": recipe["commit"], "generated": True}
