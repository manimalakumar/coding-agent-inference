"""Refresh source-free replay displays in a relocated, network-denied public copy."""

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def worker():
    from IPython.core.interactiveshell import InteractiveShell
    from IPython.utils.capture import capture_output

    shell = InteractiveShell.instance()
    denied = []

    def audit(event, args):
        forbidden = event in {
            "subprocess.Popen",
            "os.system",
            "os.exec",
            "os.posix_spawn",
            "socket.connect",
            "socket.getaddrinfo",
            "socket.sendto",
        }
        if event == "open" and args and isinstance(args[0], (str, bytes)):
            name = os.fsdecode(args[0])
            forbidden |= (
                "/Experimental-" + "Agents/" in name
                or name.endswith("credentials.env")
                or "/.config/openrouter/" in name
                or (name.endswith(".zip") and not name.startswith(str(ROOT)))
            )
        if forbidden:
            denied.append(event)
            raise RuntimeError("Offline render denied " + event)

    sys.addaudithook(audit)
    path = ROOT / "main.ipynb"
    nb = json.loads(path.read_text())
    protected = json.loads(json.dumps(nb["cells"][12]))
    for index, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code":
            continue
        with capture_output() as captured:
            result = shell.run_cell("".join(cell["source"]))
        if result.error_in_exec or result.error_before_exec:
            raise RuntimeError(
                f"Cell {index}: {result.error_in_exec or result.error_before_exec}"
            )
        outputs = []
        for name, value in [("stdout", captured.stdout), ("stderr", captured.stderr)]:
            if value:
                outputs.append(
                    {
                        "output_type": "stream",
                        "name": name,
                        "text": value.splitlines(keepends=True),
                    }
                )
        for output in captured.outputs:
            outputs.append(
                {
                    "output_type": "display_data",
                    "data": output.data,
                    "metadata": output.metadata,
                }
            )
        cell["outputs"] = outputs
    assert not denied, denied
    assert nb["cells"][12]["source"] == protected["source"]
    path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")


def main():
    if "--worker" in sys.argv:
        worker()
        return
    original = json.loads((ROOT / "main.ipynb").read_text())
    with tempfile.TemporaryDirectory(prefix="public-render-") as temp:
        staged = Path(temp) / "repository"
        shutil.copytree(
            ROOT,
            staged,
            ignore=shutil.ignore_patterns(
                ".local",
                ".venv",
                ".notebook-venv",
                "runs",
                "__pycache__",
                ".pytest_cache",
                ".ruff_cache",
                "*.pyc",
            ),
        )
        env = {
            k: v
            for k, v in os.environ.items()
            if not any(
                x in k.upper()
                for x in ["KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL"]
            )
        }
        env.update(
            PYTHONPATH=str(staged),
            PYTHONDONTWRITEBYTECODE="1",
            IPYTHONDIR=str(Path(temp) / "ipython"),
            MPLCONFIGDIR=str(Path(temp) / "matplotlib"),
        )
        completed = subprocess.run(
            [sys.executable, str(staged / "notebooks/render_replay.py"), "--worker"],
            cwd=staged,
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
        )
        if completed.returncode:
            raise RuntimeError(completed.stdout + completed.stderr)
        rendered = json.loads((staged / "main.ipynb").read_text())
    assert rendered["cells"][12]["source"] == original["cells"][12]["source"]
    (ROOT / "main.ipynb").write_text(
        json.dumps(rendered, indent=1, ensure_ascii=False) + "\n"
    )
    print(
        "Refreshed replay displays; invocation code retained, historical request summarized; model requests: 0."
    )


if __name__ == "__main__":
    main()
