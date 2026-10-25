"""Audit the intended public file set without changing a Git index."""

import ast
import hashlib
import html
import json
import os
import re
import sys
from pathlib import Path

import pathspec

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    spec = pathspec.PathSpec.from_lines(
        "gitwildmatch", (ROOT / ".gitignore").read_text().splitlines()
    )
    files = []
    for base, dirs, names in os.walk(ROOT):
        relative = Path(base).relative_to(ROOT)
        dirs[:] = [
            d
            for d in dirs
            if d != ".git" and not spec.match_file((relative / d).as_posix() + "/")
        ]
        for name in names:
            p = relative / name
            if (
                not spec.match_file(p.as_posix())
                and p.as_posix() != "PUBLIC_FILES.json"
            ):
                files.append(p)
    errors = []
    # Derive signatures from the installed dependency, without bundling its text.
    from importlib import metadata

    import yaml

    native = metadata.distribution("mini-swe-agent").locate_file(
        "minisweagent/config/mini.yaml"
    )
    templates = yaml.safe_load(native.read_text())
    prompt_lines = {
        line.strip()
        for section, key in [
            ("agent", "instance_template"),
            ("model", "format_error_template"),
        ]
        for line in templates[section][key].splitlines()
        if len(line.strip()) >= 100
    }
    for p in files:
        full = ROOT / p
        if full.is_symlink():
            errors.append("public symlink: " + p.as_posix())
        try:
            text = full.read_text()
        except UnicodeError:
            continue
        decoded = html.unescape(text.replace("\\n", "\n"))
        if any(line in decoded for line in prompt_lines):
            errors.append("bundled upstream prompt passage: " + p.as_posix())
        if re.search(
            r"/Us" + r"ers/[A-Za-z0-9_.-]+/|/ho" + r"me/(?:oai|ubuntu)/", text
        ):
            errors.append("private absolute path: " + p.as_posix())
        if re.search(
            r"sk-or-v1-[a-zA-Z0-9]{20,}|AKIA[A-Z0-9]{16}|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----\n",
            text,
        ):
            errors.append("credential pattern: " + p.as_posix())
        if "diff --git " + "a/libs/" in text:
            errors.append("upstream patch: " + p.as_posix())
        if full.suffix == ".py":
            ast.parse(text, filename=p.as_posix())
    nb = json.loads((ROOT / "main.ipynb").read_text())
    identity = json.loads((ROOT / "notebooks/first-exchange-identity.json").read_text())
    # The portability tests independently enforce the protected cell and exchange hashes.
    freeze = json.loads(
        (ROOT / "reports/public-readiness/implementation-freeze.json").read_text()
    )
    changed = [
        name
        for name, expected in freeze["files"].items()
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected
    ]
    assert not changed, changed
    assert not errors, errors
    assert len([p for p in files if p.suffix == ".ipynb"]) == 1
    assert (
        len(
            [
                p
                for p in files
                if p.parent == Path(".") and p.name.lower().startswith("readme")
            ]
        )
        == 1
    )
    assert len(nb["cells"]) == 37 and identity
    report = {
        "status": "passed",
        "public_files_scanned": len(files),
        "private_host_paths": [],
        "credential_patterns": [],
        "raw_patch_or_archive_files": [],
        "python_syntax_errors": [],
        "working_notebooks": 1,
        "root_READMEs": 1,
        "excluded_ignored_local_runtime": True,
        "frozen_execution_files": len(freeze["files"]),
        "changed_frozen_execution_files": [],
        "model_requests": 0,
        "reviewed_test_fixtures": [
            "Synthetic PRIVATE KEY header in redaction test; no credential material"
        ],
        "file_manifest": "PUBLIC_FILES.json",
        "bundled_upstream_prompt_passages": [],
    }
    (ROOT / "reports/portability/publication-scan.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    hashed = {
        p.as_posix(): hashlib.sha256((ROOT / p).read_bytes()).hexdigest()
        for p in sorted(files)
    }
    canonical = json.dumps(hashed, sort_keys=True, separators=(",", ":")).encode()
    (ROOT / "PUBLIC_FILES.json").write_text(
        json.dumps(
            {
                "description": "Intended public files, excluding this manifest and all ignored artifacts; no Git index changes.",
                "content_id": hashlib.sha256(canonical).hexdigest(),
                "files": hashed,
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
