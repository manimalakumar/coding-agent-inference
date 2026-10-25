"""Trusted setup probe baked into the task image; no inference."""

import json
import os
import socket
from pathlib import Path

checks = {}
root = Path("/workspace/libs/partners/chroma")
checks["source_readable"] = (root / "langchain_chroma/vectorstores.py").is_file()
p = root / "langchain_chroma/probe.py"
p.write_text("pass\n")
p.unlink()
checks["source_writable"] = True
p = root / "tests/unit_tests/test_probe.py"
p.write_text("pass\n")
p.unlink()
checks["new_regression_writable"] = True
for label, path in [
    ("existing_test_read_only", root / "tests/unit_tests/test_vectorstores.py"),
    ("dependencies_read_only", Path("/opt/venv/probe")),
    ("outside_source_read_only", Path("/workspace/probe")),
]:
    try:
        with path.open("a"):
            pass
        checks[label] = False
    except OSError:
        checks[label] = True
p = root / "tests/unit_tests/__init__.py"
q = p.with_name("renamed_probe")
try:
    p.rename(q)
    q.rename(p)
    checks["existing_test_rename_denied"] = False
except OSError:
    checks["existing_test_rename_denied"] = True
checks["hidden_tests_absent"] = not Path("/tests").exists()
checks["credentials_absent"] = not any(
    k in os.environ
    for k in ["OPENROUTER_API_KEY", "AWS_SECRET_ACCESS_KEY", "GITHUB_TOKEN"]
)
checks["history_absent"] = not list(Path("/workspace").rglob(".git"))
for label, path in [
    ("host_absent", ".local/credentials.env"),
    (
        "reference_absent",
        ".local/upstream/langchain",
    ),
]:
    checks[label] = not Path(path).exists()
link = root / "langchain_chroma/probe_link"
link.symlink_to(".local/credentials.env")
try:
    checks["host_symlink_inaccessible"] = not link.exists()
finally:
    link.unlink()
try:
    s = socket.create_connection(("1.1.1.1", 443), timeout=1)
    s.close()
    checks["network_disabled"] = False
except OSError:
    checks["network_disabled"] = True
print(json.dumps(checks))
assert all(checks.values()), checks
