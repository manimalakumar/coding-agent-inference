"""Fresh-kernel, post-install offline replay with negative I/O controls."""

import json
import os
import sys
import tempfile
import time
from pathlib import Path
from textwrap import dedent

import nbformat
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager
from nbclient import NotebookClient

from public_replay.bundle import Bundle, canonical, digest, safe_path

GUARD = dedent("""
import os as _os, sys as _sys, json as _json
_blocked = []
def _audit(event, args):
    denied = event in {"socket.connect", "socket.getaddrinfo", "socket.sendto", "subprocess.Popen", "os.system", "os.exec", "os.posix_spawn"}
    if event == "open" and args and isinstance(args[0], (str, bytes)):
        name = str(args[0])
        denied = denied or "/.config/openrouter/" in name or name.endswith("credentials.env") or "Maintenance-Experiment-Resources" in name
        denied = denied or any(root and name.startswith(root) for root in _os.environ.get("PUBLIC_FORBIDDEN_ROOTS", "").split(_os.pathsep))
    if denied:
        _blocked.append(event)
        raise PermissionError("Public Replay blocked: " + event)
_sys.addaudithook(_audit)
import socket as _socket, subprocess as _subprocess
from pathlib import Path as _Path
_probes = {}
def _network_probe():
    with _socket.socket() as s: s.connect(("203.0.113.1", 443))
for label, probe in [("network", _network_probe), ("process", lambda: _subprocess.run(["docker", "info"])), ("credentials", lambda: open(_Path.home()/".config/openrouter/credentials.env"))]:
    try: probe()
    except PermissionError as error:
        assert str(error).startswith("Public Replay blocked:")
        _probes[label] = "blocked before side effect"
    else: raise AssertionError("Negative control did not block " + label)
_blocked.clear()
""")
FINAL = 'assert MODE == "replay"\nprint(_json.dumps({"negative_controls": _probes, "blocked_replay_attempts": _blocked, "executed_network_calls": 0, "executed_subprocesses": 0, "paid_calls": 0}))'


def validate_release(root: Path) -> dict:
    manifest = json.loads((root / "public-manifest.json").read_text())
    if digest(canonical(manifest["files"])) != manifest["content_id"]:
        raise ValueError("Public release manifest identity mismatch")
    for name, expected in manifest["files"].items():
        path = safe_path(root, name)
        if not path.is_file() or digest(path.read_bytes()) != expected:
            raise ValueError(f"Public release validation failed: {name}")
    return manifest


def validate_demo(
    data: Path, notebook_path: Path, output: Path, forbid_roots: list[str] | None = None
) -> dict:
    started = time.monotonic()
    bundle = Bundle(data)
    before = {
        name: digest((bundle.root / name).read_bytes())
        for name in bundle.manifest["files"]
    }
    output.mkdir(parents=True, exist_ok=False)
    notebook = nbformat.read(notebook_path, as_version=4)
    for cell in notebook.cells:
        if cell.cell_type == "code":
            cell.outputs = []
            cell.execution_count = None
    notebook.cells.insert(0, nbformat.v4.new_code_cell(GUARD))
    notebook.cells.append(nbformat.v4.new_code_cell(FINAL))
    with tempfile.TemporaryDirectory(prefix="public replay kernel ") as temporary:
        tmp = Path(temporary)
        specs = tmp / "kernels/public-replay"
        specs.mkdir(parents=True)
        (specs / "kernel.json").write_text(
            json.dumps(
                {
                    "argv": [
                        sys.executable,
                        "-m",
                        "ipykernel_launcher",
                        "-f",
                        "{connection_file}",
                    ],
                    "display_name": "Public Replay",
                    "language": "python",
                }
            )
        )
        manager = KernelSpecManager(kernel_dirs=[str(tmp / "kernels")])
        km = KernelManager(
            kernel_name="public-replay",
            kernel_spec_manager=manager,
            transport="ipc",
            connection_file=str(tmp / "connection.json"),
        )
        env = {
            k: v
            for k, v in os.environ.items()
            if not any(
                s in k.upper()
                for s in [
                    "KEY",
                    "TOKEN",
                    "SECRET",
                    "PASSWORD",
                    "CREDENTIAL",
                    "PYTHONPATH",
                ]
            )
        }
        env.update(
            HOME=str(tmp / "home"),
            IPYTHONDIR=str(tmp / "ipython"),
            JUPYTER_CONFIG_DIR=str(tmp / "jupyter"),
            MPLCONFIGDIR=str(tmp / "matplotlib"),
            PUBLIC_EVIDENCE_DIR=str(data.resolve()),
            PUBLIC_FORBIDDEN_ROOTS=os.pathsep.join(forbid_roots or []),
        )
        client = NotebookClient(
            notebook,
            km=km,
            timeout=120,
            resources={"metadata": {"path": str(output.resolve())}},
            allow_errors=False,
        )
        try:
            executed = client.execute(env=env)
        finally:
            if client.kc is not None:
                client.kc.stop_channels()
            if km.has_kernel:
                km.shutdown_kernel(now=True)
    raw = "".join(
        o.get("text", "")
        for o in executed.cells[-1].outputs
        if o.output_type == "stream"
    )
    controls = json.loads(raw)
    executed.cells = executed.cells[1:-1]
    count = 0
    for cell in executed.cells:
        if cell.cell_type == "code":
            count += 1
            cell.execution_count = count
    target = output / "demo.executed.ipynb"
    nbformat.write(executed, target)
    if target.stat().st_size >= 2_000_000:
        raise ValueError("Rendered demo exceeds the 2 MB project budget")
    assert before == {
        name: digest((bundle.root / name).read_bytes()) for name in before
    }
    result = {
        "fresh_kernel": True,
        "public_manifest_id": bundle.manifest["content_id"],
        "code_cells_executed": count,
        "rendered_notebook_bytes": target.stat().st_size,
        "elapsed_seconds": time.monotonic() - started,
        "public_inputs_unchanged": True,
        "cache_directory_was_fresh": True,
        **controls,
    }
    (output / "validation.json").write_bytes(canonical(result))
    return result
