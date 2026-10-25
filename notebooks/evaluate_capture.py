"""Evaluate an already captured public workspace offline, without any agent/model."""

import argparse
import json
import subprocess
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    from public_preparation.launch import readiness

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task_id")
    parser.add_argument(
        "capture_directory",
        type=Path,
        help="Directory containing candidate.tar and the signed receipt.json",
    )
    args = parser.parse_args()
    receipt = readiness(ROOT, args.task_id)
    capture = args.capture_directory.resolve(strict=True)
    for name in ["candidate.tar", "receipt.json"]:
        if not (capture / name).is_file():
            raise FileNotFoundError(name + " is missing from the capture")
    prepared = json.loads(
        (ROOT / ".local/public-v1" / args.task_id / "prepared.json").read_text()
    )
    logs = ROOT / ".local/evaluations" / uuid.uuid4().hex
    logs.mkdir(parents=True)
    command = [
        "docker",
        "run",
        "--rm",
        "--network=none",
        "--user=0:0",
        "--cap-drop=ALL",
        "--cap-add=SETUID",
        "--cap-add=SETGID",
        "--cap-add=FOWNER",
        "--cap-add=DAC_OVERRIDE",
        "--security-opt=no-new-privileges",
        "--cpus=2",
        "--memory=2g",
        "--pids-limit=128",
        "--tmpfs",
        "/tmp:rw,nosuid,nodev,size=768m",
        "--mount",
        "type=bind,src=" + str(capture) + ",dst=/logs/artifacts,readonly",
        "--mount",
        "type=bind,src=" + str(logs) + ",dst=/logs/verifier",
        prepared["images"]["verifier"],
        "python",
        "/tests/verify.py",
    ]
    subprocess.run(command, check=True, timeout=360)
    result = json.loads((logs / "checks.json").read_text())
    numeric = {
        "task_id": args.task_id,
        "public_protocol": receipt["protocol_id"],
        "setting_id": receipt["setting_id"],
        "model_requests": 0,
        "patch_accepted": result.get("patch_accepted", result.get("task_success")),
        "checks": {k: v.get("exit_code") for k, v in result.get("checks", {}).items()},
        "error": result.get("error"),
    }
    print(json.dumps(numeric, indent=2))
    if not numeric["patch_accepted"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
