"""Atomic derived reports; original trial artifacts are never rewritten."""

import os
import tempfile
from pathlib import Path

from experiments.artifact_paths import resolve_artifact_path
from experiments.configuration_paths import resolve_configuration_path


def atomic_write(path, content):
    path = resolve_artifact_path(resolve_configuration_path(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", dir=path.parent, delete=False
        ) as output:
            temporary = Path(output.name)
            output.write(content)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        if temporary and temporary.exists():
            temporary.unlink()
