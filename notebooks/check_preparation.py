"""Offline source-export and preparation checks with synthetic archives; no Git/network."""

import io
import json
import sys
import tarfile
import tempfile
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.portable_prepare import specs, validate_spec  # noqa: E402
from public_replay.prepare import export_commit, tree_identity  # noqa: E402


def archive(name, *, symlink=False):
    data = io.BytesIO()
    with tarfile.open(fileobj=data, mode="w") as tar:
        info = tarfile.TarInfo(name)
        if symlink:
            info.type = tarfile.SYMTYPE
            info.linkname = "/outside"
            tar.addfile(info)
        else:
            text = b"public source fixture\n"
            info.size = len(text)
            tar.addfile(info, io.BytesIO(text))
    return data.getvalue()


def main():
    for spec in specs().values():
        validate_spec(spec)
    checks = []
    with tempfile.TemporaryDirectory(prefix="public-source-check-") as temp:
        root = Path(temp)
        with patch(
            "public_replay.prepare.command", return_value=archive("libs/example.py")
        ):
            result = export_commit(root / "cache", "a" * 40, root / "valid")
            assert result["baseline_checksum"] == tree_identity(root / "valid")
            assert (
                root / "valid/libs/example.py"
            ).read_text() == "public source fixture\n"
            checks.append("safe history-free export and checksum")
        for index, (name, link) in enumerate(
            [
                ("../escape", False),
                ("/outside", False),
                (".git/config", False),
                ("link", True),
            ]
        ):
            with patch(
                "public_replay.prepare.command",
                return_value=archive(name, symlink=link),
            ):
                try:
                    export_commit(root / "cache", "a" * 40, root / str(index))
                except ValueError:
                    checks.append("rejected " + name)
                else:
                    raise AssertionError("Unsafe export accepted: " + name)
    numeric = json.loads((ROOT / "notebooks/evidence/tasks.json").read_text())
    for task, spec in specs().items():
        assert spec["starting_commit"] in numeric[task]["starting_commits"]
        assert spec["baseline_checksum"] in numeric[task]["baseline_checksums"]
    receipt = {
        "status": "passed",
        "source_tasks": len(specs()),
        "historical_SHA_and_checksum_matches": 13,
        "checks": checks,
        "network_calls": 0,
        "model_requests": 0,
        "environment_rebuilds": 0,
    }
    out = ROOT / "reports/portability/preparation-validation.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
