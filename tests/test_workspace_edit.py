from pathlib import Path

import pytest

from sandbox.workspace_edit import operate


def test_git_free_restore_diff_and_permissions(tmp_path):
    baseline, workspace = tmp_path / "unchanged", tmp_path / "workspace"
    for root in (baseline, workspace):
        (root / "src").mkdir(parents=True)
        (root / "src/example.py").write_text("original = True\n")
    options = dict(workspace=workspace, baseline=baseline, allowed=["src"])
    (workspace / "src/example.py").write_text("broken = True\n")
    difference = operate("diff", "src/example.py", **options)
    assert "-original = True" in difference and "+broken = True" in difference
    operate("restore", "src/example.py", **options)
    assert operate("diff", "src/example.py", **options) == ""
    assert (baseline / "src/example.py").read_text() == "original = True\n"
    for name in ("/etc/passwd", "../secret", "tests/test_existing.py", "src"):
        with pytest.raises(ValueError):
            operate("restore", name, **options)
    (workspace / "src/example.py").unlink()
    (workspace / "src/example.py").symlink_to(baseline / "src/example.py")
    with pytest.raises(ValueError, match="Symbolic"):
        operate("restore", "src/example.py", **options)
    (workspace / "src/example.py").unlink()
    operate("restore", "src/example.py", **options)
    assert Path(workspace / "src/example.py").read_text() == "original = True\n"
