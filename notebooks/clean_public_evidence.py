"""Keep historical numbers and first exchange; replace source-bearing artifact excerpts."""

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def reconcile():
    from public_replay.bundle import canonical, digest

    directory = ROOT / "notebooks/evidence"
    manifest = json.loads((directory / "manifest.json").read_text())
    manifest["files"] = {
        name: digest((directory / name).read_bytes()) for name in manifest["files"]
    }
    manifest["content_id"] = digest(canonical(manifest["files"]))
    (directory / "manifest.json").write_bytes(canonical(manifest))
    for f in (ROOT / "notebooks/assets").rglob("manifest.json"):
        m = json.loads(f.read_text())
        m["evidence_content_id"] = manifest["content_id"]
        f.write_text(json.dumps(m, indent=2) + "\n")
    f = ROOT / "evidence/manifest.json"
    m = json.loads(f.read_text())
    m["files"] = {
        name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        for name in m["files"]
        if (ROOT / name).is_file()
    }
    f.write_text(json.dumps(m, indent=2) + "\n")


def main():
    f = ROOT / "evidence/tutorial.json"
    saved = json.loads(f.read_text())
    first = hashlib.sha256(
        json.dumps(saved["first_exchange"], sort_keys=True).encode()
    ).hexdigest()
    candidate = saved["candidate"]
    candidate.pop("patch", None)
    candidate["explanation"] = (
        "The recorded candidate changed tool-schema annotation handling so postponed annotations could retain parameter descriptions. It added a new regression file while preserving existing tests. The independent recorded checks below determine acceptance; source and patch text are not embedded in this distribution."
    )
    for check in candidate["checks"].values():
        if isinstance(check, dict) and "output" in check:
            text = check.pop("output")
            check["historical_output_characters"] = len(text)
            check["explanation"] = (
                "Recorded check completed with exit code "
                + str(check.get("exit_code"))
                + ". Raw source-bearing output is not included."
            )
    descriptions = {
        "source edit": "Edited annotation resolution in tool-schema handling.",
        "test creation": "Added a regression file covering tool parameter metadata.",
        "reproduction": "Ran the unchanged-source reproducer; parameter description was missing.",
        "verification execution": "Executed the candidate regression tests through the command wrapper.",
        "submission, last observed action": "Submitted the final workspace for independent evaluation.",
    }
    for row in saved["milestones"]:
        row["command"] = descriptions.get(
            row["Milestone"], "Recorded sandbox action; source text omitted."
        )
        row["observation (tail)"] = (
            "Recorded command exit: "
            + str(row.get("exit"))
            + ". Numeric check results remain available."
        )
    for messages in [
        saved["context_demo"][1],
        saved["context_demo"][2].get("added_messages", []),
    ]:
        for message in messages:
            if message.get("role") == "tool":
                message["content"] = (
                    "Author explanation: the reproducer returned exit code 1 because the expected parameter description was absent. Its schema retained an array of strings."
                )
    saved["provenance"] = {
        "kind": "historical numeric results and author-written tutorial explanations",
        "first_exchange_sha256": first,
        "acceptance": "Recorded historical outcomes; independent public controls use separate protocol identities.",
    }
    f.write_text(json.dumps(saved, indent=2) + "\n")
    f = (
        ROOT
        / "reports/harbor-validation/workspace-fresh-start-baseline-control-001/LC-ANNOTATED-TOOLS-001__UzeWY8p/verifier/checks.json"
    )
    d = json.loads(f.read_text())
    for c in d.get("checks", {}).values():
        if isinstance(c, dict):
            c.pop("output", None)
    f.write_text(json.dumps(d, indent=2) + "\n")
    n = json.loads((ROOT / "main.ipynb").read_text())
    c = n["cells"][17]
    c["source"] = [
        "candidate = workflow.candidate_evidence(run)\n",
        'display(HTML(presentation.details(ROOT, "Recorded candidate behavior", candidate["explanation"])))\n',
        'display(presentation.checks_frame(run, split="development"))\n',
        "display(pd.DataFrame([{\n",
        '    "Termination reason": str(run.get("termination_reason", "unknown")).replace("_", " "),\n',
        '    "Explicit submission": run.get("explicit_submission"), "Patch accepted": run.get("patch_accepted"),\n',
        '    "Task success": run.get("task_success"), "Original classification": str(run.get("classification", "unknown")).replace("_", " ")}]))\n',
        'display(HTML(presentation.details(ROOT, "Recorded capture receipt", candidate["receipt"])))\n',
    ]
    for index in [15, 17]:
        n["cells"][index]["outputs"] = []
    (ROOT / "main.ipynb").write_text(json.dumps(n, indent=1, ensure_ascii=False) + "\n")
    reconcile()


if __name__ == "__main__":
    main()
