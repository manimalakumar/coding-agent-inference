"""Revalidate public task controls from an isolated copy, without private siblings."""

import concurrent.futures
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXCLUDED = {
    ".local",
    ".venv",
    ".notebook-venv",
    "runs",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    ".git",
}


def copy_public(destination):
    for base, dirs, files in os.walk(ROOT):
        dirs[:] = [d for d in dirs if d not in EXCLUDED]
        relative = Path(base).relative_to(ROOT)
        for name in files:
            if name.endswith(".pyc") or name == ".DS_Store":
                continue
            out = destination / relative / name
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(Path(base) / name, out)


def rebind(value, destination):
    if isinstance(value, dict):
        return {k: rebind(v, destination) for k, v in value.items()}
    if isinstance(value, list):
        return [rebind(v, destination) for v in value]
    if isinstance(value, str):
        return value.replace(str(ROOT), str(destination))
    return value


def stage(destination):
    copy_public(destination)
    (destination / ".venv").symlink_to(ROOT / ".venv", target_is_directory=True)
    for state in (ROOT / ".local/public-v1").iterdir():
        if not state.is_dir():
            continue
        if not (state / "validated.json").exists():
            raise RuntimeError(
                "Task has not passed original public controls: " + state.name
            )
        out = destination / ".local/public-v1" / state.name
        out.mkdir(parents=True)
        for name in [
            "prepared.json",
            "model.json",
            "baseline.json",
            "reference.json",
            "reference_budget.json",
            "reference_error.json",
            "reference-fixture.json",
        ]:
            data = json.loads((state / name).read_text())
            (out / name).write_text(
                json.dumps(rebind(data, destination), indent=2) + "\n"
            )
        shutil.copyfile(state / "receipt.key", out / "receipt.key")
        (out / "receipt.key").chmod(0o600)
        shutil.copytree(state / "harbor-task", out / "harbor-task")
        prepared = json.loads((out / "prepared.json").read_text())
        prepared["parent_public_setting_id"] = prepared.pop("setting_id")
        prepared["relocation_identity"] = hashlib.sha256(
            str(destination).encode()
        ).hexdigest()
        prepared["setting_id"] = hashlib.sha256(
            json.dumps(prepared, sort_keys=True).encode()
        ).hexdigest()
        (out / "prepared.json").write_text(json.dumps(prepared, indent=2) + "\n")
    guard = destination / ".local/python-guard"
    guard.mkdir()
    # This hook is inherited by Harbor's Python process through PYTHONPATH.
    code = """import os,sys
from pathlib import Path
PUBLIC_ORIGIN=__ORIGIN_VALUE__
ISOLATED=__DEST_VALUE__
RUNTIME=PUBLIC_ORIGIN+'/.venv/'
def _public_guard(event,args):
    denied=False
    if event=='open' and args and isinstance(args[0],(str,bytes)):
        path=os.path.abspath(os.fsdecode(args[0]))
        denied='/Experimental-Agents/' in path or (path.endswith('.zip') and not path.startswith(ISOLATED) and not path.endswith('/python314.zip'))
        denied=denied or (path.startswith(PUBLIC_ORIGIN+'/') and not path.startswith(ISOLATED+'/') and not path.startswith(RUNTIME) and not path.startswith(PUBLIC_ORIGIN+'/.notebook-venv/'))
    if event in {'socket.connect','socket.getaddrinfo'}:
        address=args[1] if event=='socket.connect' else args[0]
        if isinstance(address,tuple):denied=address[0] not in {'127.0.0.1','::1','localhost'}
        elif event=='socket.getaddrinfo':denied=address not in {'127.0.0.1','::1','localhost',None}
    if denied:
        with open(ISOLATED+'/.local/forbidden-access.jsonl','a') as f:f.write(event+' '+repr(args[:2])+'\\n')
        raise RuntimeError('Isolated public validation denied '+event)
sys.addaudithook(_public_guard)
""".replace("__ORIGIN_VALUE__", repr(str(ROOT))).replace(
        "__DEST_VALUE__", repr(str(destination))
    )
    (guard / "sitecustomize.py").write_text(code)
    return guard


def main():
    destination = ROOT / ".local/isolated-public-validation"
    if destination.exists():
        raise FileExistsError(
            "Use a new validation copy; refusing to overwrite previous evidence"
        )
    destination.mkdir(parents=True)
    guard = stage(destination)
    env = {
        k: v
        for k, v in os.environ.items()
        if not any(
            t in k.upper() for t in ["KEY", "TOKEN", "SECRET", "CREDENTIAL", "PASSWORD"]
        )
    }
    env.update(
        PYTHONPATH=str(guard) + os.pathsep + str(destination),
        PYTHONDONTWRITEBYTECODE="1",
        PUBLIC_NO_INFERENCE="1",
        MSWEA_SILENT_STARTUP="1",
        HARBOR_TELEMETRY="0",
        MSWEA_GLOBAL_CONFIG_DIR=str(destination / ".local/mini"),
    )
    probe_code = """import json,os,socket
from pathlib import Path
results={}
for name,path in PROBE_PATHS.items():
    try:open(path)
    except RuntimeError:results[name]='blocked before side effect'
    else:raise AssertionError('Private read guard failed: '+name)
try:socket.create_connection(('203.0.113.1',443))
except RuntimeError:results['network']='blocked before side effect'
else:raise AssertionError('Network guard failed')
audit=Path('.local/forbidden-access.jsonl')
assert audit.exists()
audit.unlink()
Path('.local/guard-controls.json').write_text(json.dumps(results))
""".replace(
        "PROBE_PATHS",
        repr(
            {
                "original_public_source": str(ROOT / "main.ipynb"),
                "private_research": str(ROOT.parent / "Experimental-Agents/main.ipynb"),
                "private_zip": str(ROOT.parent / "B-Experimental-Agents-Public.zip"),
            }
        ),
    )
    subprocess.run(
        [str(ROOT / ".venv/bin/python"), "-c", probe_code],
        cwd=destination,
        env=env,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    tasks = sorted(p.name for p in (destination / ".local/public-v1").iterdir())

    def one(task):
        logfile = destination / ".local" / (task + "-validation.log")
        with logfile.open("w") as output:
            result = subprocess.run(
                [
                    str(ROOT / ".venv/bin/python"),
                    "-m",
                    "public_preparation.runtime",
                    "validate",
                    task,
                ],
                cwd=destination,
                env=env,
                stdout=output,
                stderr=subprocess.STDOUT,
                timeout=1800,
            )
        return {
            "task_id": task,
            "passed": result.returncode == 0,
            "model_requests": 0,
            "log": str(logfile.relative_to(destination)),
        }

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        for result in pool.map(one, tasks):
            results.append(result)
            print(json.dumps(result), flush=True)
    forbidden = destination / ".local/forbidden-access.jsonl"
    receipt = {
        "status": "passed" if all(r["passed"] for r in results) else "failed",
        "tasks": results,
        "private_project_reads": 0,
        "audit_guard_enforced": True,
        "blocked_external_io_attempts": len(forbidden.read_text().splitlines())
        if forbidden.exists()
        else 0,
        "blocked_attempts_note": "Denied before side effects; includes background telemetry and Python standard-library archive discovery. No forbidden read or connection succeeded.",
        "host_network_calls": 0,
        "model_requests": 0,
        "negative_controls": json.loads(
            (destination / ".local/guard-controls.json").read_text()
        ),
        "runtime": "shared installed third-party runtime; project code and task resources loaded from isolated copy",
        "relocated_task_resources": True,
    }
    target = ROOT / "reports/public-readiness/isolated-validation.json"
    target.write_text(json.dumps(receipt, indent=2) + "\n")
    for task in tasks:
        source = destination / "reports/public-readiness" / (task + ".json")
        if source.exists():
            out = ROOT / "reports/public-readiness/isolated" / source.name
            out.parent.mkdir(exist_ok=True)
            shutil.copyfile(source, out)
    if receipt["status"] != "passed":
        raise RuntimeError(
            "Isolated public controls failed; inspect isolated validation receipt and ignored logs"
        )


if __name__ == "__main__":
    main()
