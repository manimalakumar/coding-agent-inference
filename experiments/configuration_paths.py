"""Resolve preserved configuration names without duplicating configuration files."""

import json
from pathlib import Path


def configuration_aliases(project):
    registry = Path(project) / "configs/configuration-paths.json"
    if not registry.exists():
        return {}
    data = json.loads(registry.read_text())
    if data.get("schema_version") != 1:
        raise ValueError("Unsupported configuration path registry")
    return data["aliases"]


def configuration_path(project, name):
    """Accept a direct configs reference or basename; reject escaping aliases."""
    relative = Path(name)
    parts = relative.parts
    if parts and parts[0] == "configs":
        parts = parts[1:]
    if relative.is_absolute() or len(parts) != 1 or parts[0] in {".", ".."}:
        raise ValueError("Configuration must be a direct configs reference")
    filename = configuration_aliases(project).get(parts[0], parts[0])
    if (
        not isinstance(filename, str)
        or Path(filename).name != filename
        or filename in {".", ".."}
    ):
        raise ValueError("Configuration alias escapes configs")
    path = Path(project) / "configs" / filename
    if path.is_symlink() or not path.resolve().is_relative_to(
        (Path(project) / "configs").resolve()
    ):
        raise ValueError("Configuration path escapes configs")
    return path


def resolve_configuration_path(path):
    """Resolve saved direct-child paths while leaving other artifact paths intact."""
    path = Path(path)
    return (
        configuration_path(path.parent.parent, path.name)
        if path.parent.name == "configs"
        else path
    )


def historical_configuration_name(project, name):
    """Keep pinned template recipe identities independent of public filenames."""
    return next(
        (
            old
            for old, current in configuration_aliases(project).items()
            if current == name
        ),
        name,
    )
