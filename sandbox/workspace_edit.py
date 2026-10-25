"""Git-free diff and restore against the task's read-only, pre-fix source."""

import argparse
import difflib
import json
from pathlib import Path, PurePosixPath


def source_file(root, name, allowed):
    relative = PurePosixPath(name)
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise ValueError("Use a repository-relative production file path")
    if not any(relative.is_relative_to(PurePosixPath(p)) for p in allowed):
        raise ValueError("Path is outside the allowed production source")
    path = root / relative
    for item in (path, *path.parents):
        if item == root:
            break
        if item.is_symlink():
            raise ValueError("Symbolic links are not supported")
    if path.exists() and not path.is_file():
        raise ValueError("Expected a file")
    return path


def operate(operation, name, *, workspace, baseline, allowed):
    original = source_file(baseline, name, allowed)
    target = source_file(workspace, name, allowed)
    if not original.is_file():
        raise ValueError("No unchanged production file exists at this path")
    before = original.read_text()
    if operation == "restore":
        # Same unprivileged user and writable paths as every other agent command.
        # No host service, Git object store or hidden resource is involved.
        target.write_bytes(original.read_bytes())
        return "Restored unchanged pre-fix source: " + name + "\n"
    after = target.read_text() if target.exists() else ""
    return "".join(
        difflib.unified_diff(
            before.splitlines(keepends=True),
            after.splitlines(keepends=True),
            fromfile="unchanged/" + name,
            tofile="workspace/" + name,
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=["diff", "restore"])
    parser.add_argument("path", help="repository-relative production file")
    args = parser.parse_args()
    try:
        config = json.loads(Path("/task/edit-resources.json").read_text())
        print(
            operate(
                args.operation,
                args.path,
                workspace=Path("/workspace"),
                baseline=Path("/task/unchanged"),
                allowed=config["source_paths"],
            ),
            end="",
        )
    except (ValueError, OSError) as exc:
        parser.exit(2, str(exc) + "\n")


if __name__ == "__main__":
    main()
