"""Check retained public identities without private snapshots or raw traces."""

import hashlib
import json


def assert_preserved_first_cell(root):
    notebook = json.loads((root / "main.ipynb").read_text())
    identity = json.loads((root / "notebooks/first-exchange-identity.json").read_text())
    cells = [
        c
        for c in notebook["cells"]
        if "selected-first-invocation" in c.get("metadata", {}).get("tags", [])
    ]
    assert len(notebook["cells"]) == 37
    assert len(cells) == 1
    assert (
        hashlib.sha256(json.dumps(cells[0], sort_keys=True).encode()).hexdigest()
        == identity["first_code_cell_sha256"]
    )
    manifest = json.loads((root / "evidence/manifest.json").read_text())
    for name, expected in manifest["files"].items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == expected
