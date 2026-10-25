"""Read immutable historical setup receipts plus separately appended expenses."""

from pathlib import Path

from experiments.records import read


def setup_entries(project):
    project = Path(project)
    entries = list(
        read(project / "reports/setup-inference-costs.json", {}).get("entries", [])
    )
    entries.extend(
        read(project / "reports/setup-inference-costs-additions.json", {}).get(
            "entries", []
        )
    )
    receipts = [e["result"] for e in entries]
    if len(receipts) != len(set(receipts)):
        raise ValueError("Duplicate setup expense receipt")
    return entries
