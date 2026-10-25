"""Post-change preservation and allowlist audit. Never invokes inference."""

import argparse
import hashlib
import json
import os
import stat
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from public_replay.bundle import canonical, digest, safe_path
from public_replay.validate import validate_release


def verify_originals(project: Path, resource_root: Path, receipt: Path) -> dict:
    original = json.loads(receipt.read_text())
    roots = {
        "Experimental-Agents": project,
        "Maintenance-Experiment-Resources": resource_root,
    }

    def check(item):
        name, expected = item
        prefix, relative = name.split("/", 1)
        path = roots[prefix] / relative
        if not os.path.lexists(path):
            return name, "missing"
        if "symlink" in expected:
            return name, "unchanged" if path.is_symlink() and os.readlink(
                path
            ) == expected["symlink"] else "changed"
        if "special_mode" in expected:
            return name, "unchanged" if stat.S_IFMT(path.lstat().st_mode) == expected[
                "special_mode"
            ] else "changed"
        hasher = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                hasher.update(block)
        return name, "unchanged" if hasher.hexdigest() == expected[
            "sha256"
        ] else "changed"

    with ThreadPoolExecutor(max_workers=12) as workers:
        results = list(workers.map(check, original["files"].items(), buffersize=96))
    changed = [name for name, status in results if status != "unchanged"]
    # README is the one intended replacement; its full pre-change bytes are preserved.
    allowed = {"Experimental-Agents/README.md"}
    manifests = []
    for manifest in sorted((project / "configs").glob("*.sha256")):
        mismatches = []
        for line in manifest.read_text().splitlines():
            expected, name = line.split("  ", 1)
            path = safe_path(project, name)
            if not path.is_file() or digest(path.read_bytes()) != expected:
                mismatches.append(name)
        manifests.append({"manifest": manifest.name, "mismatches": mismatches})
    return {
        "original_receipt_sha256": digest(receipt.read_bytes()),
        "original_files_checked": len(results),
        "unchanged": sum(status == "unchanged" for _, status in results),
        "intentional_changed_files": sorted(set(changed) & allowed),
        "unexpected_changes_or_missing": sorted(set(changed) - allowed),
        "historical_manifests": manifests,
        "all_historical_manifests_pass": all(not m["mismatches"] for m in manifests),
        "research_notebook": "included in original-file verification",
        "provider_requests": 0,
    }


def audit_export(root: Path):
    manifest = validate_release(root)
    prohibited = {
        ".venv",
        ".state",
        "runs",
        "__pycache__",
        "Maintenance-Experiment-Resources",
    }
    violations = []
    for name in manifest["files"]:
        path = safe_path(root, name)
        if prohibited & set(path.relative_to(root).parts) or path.suffix in {
            ".tar",
            ".tgz",
            ".zlib",
            ".patch",
            ".whl",
            ".zip",
        }:
            violations.append(name)
        data = path.read_bytes()
        if b".local/author-home" in data or b".local/author-home" in data:
            violations.append(name)
    return {
        "manifest_id": manifest["content_id"],
        "reviewed_files": len(manifest["files"]),
        "source_archive_and_author_path_violations": sorted(set(violations)),
        "policy": "only allowlisted projections and project code; no raw source-bearing logs, snapshots, keys or patches",
        "public_authenticity": "not asserted by content hashes",
    }


def git_status(project: Path):
    result = subprocess.run(["git", "ls-files", "-z"], cwd=project, capture_output=True)
    return {
        "git_ls_files_available": result.returncode == 0,
        "tracked_files": len(result.stdout.split(b"\0")) - 1
        if result.returncode == 0
        else None,
        "history_secret_review": "not performed; .git unavailable"
        if result.returncode
        else "owner review required before release",
        "release_gate": "pending",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--resources", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--export", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Audit receipts are append-only; choose a new output")
    result = {
        "preservation": verify_originals(args.project, args.resources, args.receipt),
        "public_export": audit_export(args.export),
        "git": git_status(args.project),
    }
    args.output.write_bytes(canonical(result))
    print(json.dumps(result, indent=2))
    if (
        result["preservation"]["unexpected_changes_or_missing"]
        or not result["preservation"]["all_historical_manifests_pass"]
        or result["public_export"]["source_archive_and_author_path_violations"]
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
