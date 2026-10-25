"""Zero-inference validation of the documented development-test locations."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import toml
import yaml

from experiments.configuration_paths import configuration_path
from experiments.launch import verify_freeze

PROBE = r"""
import hashlib, json, os, subprocess
from pathlib import Path
root = Path('/workspace/libs/langchain_v1/tests/unit_tests')
def manifest():
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in root.rglob('*') if p.is_file() and '__pycache__' not in p.parts}
before = manifest()
checks = {}
checks['unprivileged_uid'] = os.getuid() == 65532
root_test = root / 'test_permission_root_probe.py'
child = root / 'new_permission_probe'
child_test = child / 'test_permission_child_probe.py'
code = "def test_permission_probe() -> None:\n    assert 2 + 2 == 4\n"
root_test.write_text(code)
checks['new_test_at_test_root_writable'] = root_test.exists()
child.mkdir()
child_test.write_text(code)
checks['new_directory_and_test_writable'] = child_test.exists()
blocked = root / 'agents/middleware/implementations/test_context_editing_token_counter.py'
try:
    blocked.write_text(code)
    checks['previously_attempted_existing_subdirectory_denied'] = False
except PermissionError:
    checks['previously_attempted_existing_subdirectory_denied'] = True
existing = root / 'agents/middleware/implementations/test_context_editing.py'
try:
    with existing.open('a'):
        pass
    checks['existing_test_append_denied'] = False
except PermissionError:
    checks['existing_test_append_denied'] = True
original_root_file = root / '__init__.py'
replacement = root / 'test_permission_replacement_probe.py'
replacement.write_text(code)
try:
    os.replace(replacement, original_root_file)
    checks['existing_root_test_replace_denied'] = False
except PermissionError:
    checks['existing_root_test_replace_denied'] = True
finally:
    if replacement.exists(): replacement.unlink()
result = subprocess.run(['python','-m','pytest','-q','--disable-socket','--allow-unix-socket',
                         str(root_test),str(child_test)],capture_output=True,text=True,timeout=90)
checks['both_permitted_locations_execute_pytest'] = result.returncode == 0
repro = subprocess.run(['python','/task/reproduce.py'],capture_output=True,text=True,timeout=90)
checks['unchanged_feature_failure_confirmed'] = repro.returncode == 1 and "unexpected keyword argument 'token_counter'" in repro.stderr
isolation = subprocess.run(['python','/task/isolation.py'],capture_output=True,text=True,timeout=90)
checks['frozen_isolation_probe_passed'] = isolation.returncode == 0
root_test.unlink(); child_test.unlink(); child.rmdir()
checks['existing_test_manifest_unchanged'] = before == manifest()
print(json.dumps({'checks':checks,'pytest_exit_code':result.returncode,
                  'pytest_output':result.stdout+result.stderr,'reproducer_exit_code':repro.returncode,
                  'isolation_output':isolation.stdout+isolation.stderr}))
assert all(checks.values()), checks
"""


def run(project):
    project = Path(project)
    prefix = "context-counter-feature-v1"
    verify_freeze(
        project, f"configs/{prefix}.sha256", f"configs/{prefix}.frozen-setting.json"
    )
    job = json.loads(
        (configuration_path(project, f"configs/{prefix}-model.json")).read_text()
    )
    task = Path(job["tasks"][0]["path"])
    instructions = (task / "instruction.md").read_text()
    required = "new `test_*.py` files at `tests/unit_tests/` or inside a newly created subdirectory beneath it. Existing test subdirectories are read-only."
    assert required in instructions
    config = toml.loads((task / "task.toml").read_text())
    compose = yaml.safe_load((task / "environment/docker-compose.yaml").read_text())[
        "services"
    ]["main"]
    assert compose["read_only"] and compose["network_mode"] == "none"
    args = [
        "docker",
        "run",
        "--rm",
        "--interactive",
        "--network=none",
        "--read-only",
        "--user=" + compose["user"],
        "--cap-drop=ALL",
        "--security-opt=no-new-privileges:true",
        "--pids-limit=" + str(compose["pids_limit"]),
        "--cpus=2",
        "--memory=2048m",
    ]
    for value in compose["tmpfs"]:
        args += ["--tmpfs", value]
    for value in compose["volumes"]:
        args += [
            "--volume",
            value.split(":", 1)[1],
        ]  # Fresh anonymous container volumes, no host mounts.
    args += [config["environment"]["docker_image"], "python", "-"]
    result = subprocess.run(
        args, input=PROBE, capture_output=True, text=True, timeout=180
    )
    output = project / "reports/context-counter-test-directories-control-v1.json"
    record = {
        "kind": "trusted_zero_inference_directory_control",
        "model_requests": 0,
        "inference_cost_usd": 0,
        "agent_image": config["environment"]["docker_image"],
        "instruction_sha256": hashlib.sha256(instructions.encode()).hexdigest(),
        "documented_permission_sentence": required,
        "returncode": result.returncode,
        "probe": json.loads(result.stdout.splitlines()[-1])
        if result.stdout.strip()
        else None,
        "stderr": result.stderr,
        "command": args,
        "status": "passed" if result.returncode == 0 else "failed",
    }
    output.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))
    if result.returncode:
        raise RuntimeError("Directory control failed; no measured run allowed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project", type=Path, default=Path(__file__).resolve().parents[1]
    )
    run(parser.parse_args().project)
