"""Manifest-based, read-only evidence loading without a research checkout."""

import hashlib
import json
from importlib.resources import files
from pathlib import Path, PurePosixPath
from typing import Any

from experiments.artifact_paths import resolve_artifact_path

REQUIRED = {
    "attempts.json",
    "requests.json",
    "context.json",
    "study.json",
    "tasks.json",
}


def canonical(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False) + "\n"
    ).encode()


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def safe_path(root: Path, name: str) -> Path:
    rel = PurePosixPath(name)
    if rel.is_absolute() or ".." in rel.parts or "\\" in name:
        raise ValueError("Unsafe manifest path")
    path = resolve_artifact_path(root.joinpath(*rel.parts))
    if not path.resolve().is_relative_to(root.resolve()) or path.is_symlink():
        raise ValueError("Manifest path escapes bundle")
    return path


class Bundle:
    def __init__(self, directory: Path | str | None = None):
        self.root = (
            resolve_artifact_path(directory)
            if directory is not None
            else Path(str(files("public_replay").joinpath("data")))
        )
        if directory is None:
            checkout_data = Path(__file__).resolve().parents[1] / "notebooks/evidence"
            if checkout_data.is_dir():
                self.root = checkout_data
        manifest_path = self.root / "manifest.json"
        if not manifest_path.is_file():
            raise ValueError(
                "Required public manifest missing; install the reviewed export or pass --data PATH"
            )
        self.manifest = json.loads(manifest_path.read_text())
        if self.manifest.get("schema_version") != 1:
            raise ValueError("Unsupported public evidence schema")
        entries = self.manifest.get("files", {})
        if not REQUIRED <= set(entries):
            raise ValueError("Required public inputs missing from manifest")
        if digest(canonical(entries)) != self.manifest.get("content_id"):
            raise ValueError("Public manifest content identity mismatch")
        for name, expected in entries.items():
            path = safe_path(self.root, name)
            if not path.is_file() or digest(path.read_bytes()) != expected:
                raise ValueError(
                    f"Public evidence validation failed: {name}; restore the reviewed export"
                )

    def read(self, name: str) -> Any:
        if name not in self.manifest["files"]:
            raise ValueError("Input is not declared by the public manifest")
        data = safe_path(self.root, name).read_bytes()
        if digest(data) != self.manifest["files"][name]:
            raise ValueError(f"Public evidence changed after validation: {name}")
        return json.loads(data)

    def optional_audit(self) -> dict[str, str]:
        return {
            "status": "not bundled",
            "retrieval": "Use the original artifact IDs with the owner. Task source can be fetched by the trusted prepare command; private candidate archives, raw traces and verifier fixtures are not distributed.",
        }
