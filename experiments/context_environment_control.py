"""Exercise regression placement and strict typing through the frozen tool limits."""

import base64
import json
import subprocess
from pathlib import Path

import toml
import yaml

from experiments.configuration_paths import configuration_path

PROBE = r"""
import base64,hashlib,json,os,shlex,subprocess
from pathlib import Path
fixture=json.loads(base64.b64decode(FIXTURE))
root=Path('/workspace/libs/langchain_v1/tests/unit_tests')
source=Path('/workspace/libs/langchain_v1/langchain/agents/middleware/context_editing.py')
assert os.getuid()==65532
before={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob('*') if p.is_file()}
tests=[p for p in fixture if '/tests/' in p]
assert len(tests)==1 and Path(tests[0]).parent==Path('libs/langchain_v1/tests/unit_tests') and Path(tests[0]).name.startswith('test_')
commands=[]
def run(name,command,expected=0):
    # Same bounded-output wrapper as the frozen HarborTools.execute implementation.
    wrapped='out=$(mktemp); (ulimit -f 2048; bash -o pipefail -c '+shlex.quote(command)+') >"$out" 2>&1; rc=$?; head -c 2000000 "$out"; rm -f "$out"; exit "$rc"'
    r=subprocess.run(['bash','-c',wrapped],capture_output=True,text=True,timeout=90,cwd='/workspace/libs/langchain_v1')
    commands.append({'name':name,'command':command,'exit_code':r.returncode,'expected_exit_code':expected,'output':r.stdout+r.stderr})
    assert r.returncode==expected,commands[-1]
run('baseline_strict_typing','mypy --cache-dir=/dev/null langchain/agents/middleware/context_editing.py')
for p in tests:Path('/workspace',p).write_text(fixture[p])
reg=str(Path('/workspace',tests[0]))
run('new_regression_on_baseline','python -m pytest -q --disable-socket --allow-unix-socket '+shlex.quote(reg),1)
for p,content in fixture.items():
    if p not in tests:Path('/workspace',p).write_text(content)
run('reference_regression','python -m pytest -q --disable-socket --allow-unix-socket '+shlex.quote(reg))
run('reference_reproducer','python /task/reproduce.py')
run('reference_lint','ruff check --no-cache '+shlex.quote(str(source))+' '+shlex.quote(reg))
run('reference_format','ruff format --check --no-cache '+shlex.quote(str(source))+' '+shlex.quote(reg))
run('reference_strict_typing','mypy --cache-dir=/dev/null '+shlex.quote(str(source)))
run('frozen_isolation','python /task/isolation.py')
after={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob('*') if p.is_file()}
assert all(after[k]==v for k,v in before.items())
assert sorted(set(after)-set(before))==[Path(tests[0]).name]
assert not (root/'test_control_directory').exists()
print(json.dumps({'status':'passed','commands':commands,'regression_file':tests[0],'existing_tests_unchanged':True,'no_new_init_files':True,'tool_file_limit':'ulimit -f 2048 (unchanged)','strict_type_analysis':'unchanged, cache writes disabled','fixture_provenance':'trusted pre-existing reference fixture; not a model intermediate workspace'}))
"""


def run(project):
    project = Path(project)
    label = "context-counter-environment-v2"
    preparation = json.loads(
        (project / f"reports/{label}-preparation.json").read_text()
    )
    task = Path(preparation["task_bundle"])
    task_config = toml.loads((task / "task.toml").read_text())
    compose = yaml.safe_load((task / "environment/docker-compose.yaml").read_text())[
        "services"
    ]["main"]
    parent = json.loads(
        (
            configuration_path(
                project, "configs/context-counter-glm5-calibration-v1-model.json"
            )
        ).read_text()
    )
    fixture = Path(parent["agents"][0]["kwargs"]["policy"]["reference_fixture"])
    payload = (
        "FIXTURE="
        + repr(base64.b64encode(fixture.read_bytes()).decode())
        + "\n"
        + PROBE
    )
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
        args += ["--volume", value.split(":", 1)[1]]
    args += [task_config["environment"]["docker_image"], "python", "-"]
    result = subprocess.run(
        args, input=payload, capture_output=True, text=True, timeout=240
    )
    report = {
        "kind": "trusted_zero_inference_environment_control",
        "model_requests": 0,
        "inference_cost_usd": 0,
        "command": args,
        "returncode": result.returncode,
        "probe": json.loads(result.stdout.splitlines()[-1])
        if result.stdout.strip()
        else None,
        "stderr": result.stderr,
    }
    (project / f"reports/{label}-combined-tool-control.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))
    if result.returncode:
        raise RuntimeError("Environment control failed")


if __name__ == "__main__":
    run(Path(__file__).resolve().parents[1])
