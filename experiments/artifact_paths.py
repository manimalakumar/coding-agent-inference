"""Resolve historical artifact filenames without rewriting their identities or payloads."""

import json
from pathlib import Path, PurePosixPath


def artifact_path(project, name):
    project = Path(project)
    relative = PurePosixPath(name)
    if relative.is_absolute() or ".." in relative.parts or "\\" in name:
        raise ValueError("Unsafe artifact reference")
    registry = project / "notebooks/artifact-paths.json"
    if not registry.is_file():
        return project / relative
    data = json.loads(registry.read_text())
    if data.get("schema_version") != 1:
        raise ValueError("Unsupported artifact path registry")
    normalized = relative.as_posix()
    target = data["aliases"].get(normalized)
    if target is None:
        target = normalized
        for old, current in sorted(
            data.get("directories", {}).items(), key=lambda item: -len(item[0])
        ):
            if normalized == old or normalized.startswith(old + "/"):
                target = current + normalized[len(old) :]
                break
    resolved = project / target
    if (
        PurePosixPath(target).is_absolute()
        or ".." in PurePosixPath(target).parts
        or "\\" in target
        or resolved.is_symlink()
        or not resolved.resolve().is_relative_to(project.resolve())
    ):
        raise ValueError("Artifact alias escapes project")
    return resolved


def resolve_artifact_path(path):
    path = Path(path)
    for ancestor in (path, *path.parents):
        if ancestor.name in {"reports", "notebooks"}:
            project = ancestor.parent
            return artifact_path(project, path.relative_to(project).as_posix())
    return path
