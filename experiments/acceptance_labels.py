"""Label acceptance checks by study split without changing recorded verdicts."""

from experiments.tutorial import check_rows as historical_check_rows


def acceptance_label(split: str | None) -> str:
    return {
        "development": "trusted acceptance (development task)",
        "held_out": "trusted acceptance (held-out task)",
    }.get(split or "", "trusted acceptance")


def check_rows(record):
    rows = historical_check_rows(record)
    for row in rows:
        if row.get("check") == "hidden_acceptance":
            row["stage"] = acceptance_label(record.get("split"))
    return rows
