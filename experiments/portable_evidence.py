"""Verify the complete saved historical inputs before reading tutorial excerpts."""
import hashlib
import json
from pathlib import Path
from public_replay.bundle import safe_path

def tutorial(project):
    root = Path(project)
    manifest = json.loads((root / "evidence/manifest.json").read_text())
    for name, expected in manifest["files"].items():
        if hashlib.sha256(safe_path(root, name).read_bytes()).hexdigest() != expected:
            raise ValueError("Historical evidence changed: " + name)
    return json.loads((root / "evidence/tutorial.json").read_text())
