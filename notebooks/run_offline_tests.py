"""Run transport/controller tests with all network, credentials and private reads denied."""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.pop(
    "PUBLIC_NO_INFERENCE", None
)  # Unit tests construct clients with mocked transports.
os.environ.update(
    MSWEA_SILENT_STARTUP="1", MSWEA_GLOBAL_CONFIG_DIR=str(ROOT / ".local/test-mini")
)
for key in list(os.environ):
    if any(
        word in key.upper()
        for word in ["API_KEY", "ACCESS_TOKEN", "SECRET", "PASSWORD", "CREDENTIAL"]
    ):
        os.environ.pop(key)


def guard(event, args):
    denied = event in {
        "socket.connect",
        "socket.getaddrinfo",
        "socket.sendto",
        "subprocess.Popen",
        "os.system",
    }
    if event == "subprocess.Popen":
        executable, argv = args[:2]
        trusted = (
            executable == "/bin/bash"
            and list(argv[:4]) == ["/bin/bash", "-o", "pipefail", "-c"]
            and "false | tail -1" in argv[-1]
        )
        denied = not trusted
    if event == "open" and args and isinstance(args[0], (str, bytes)):
        name = os.fsdecode(args[0])
        denied = (
            denied
            or "/Experimental-Agents/" in name
            or name.endswith("credentials.env")
        )
    if denied:
        raise RuntimeError("Offline test denied external I/O: " + event)


sys.addaudithook(guard)
import pytest  # noqa: E402

raise SystemExit(
    pytest.main(
        [
            "tests/test_public_portability.py",
            "tests/test_controller.py",
            "tests/test_mini.py",
            "tests/test_workspace_policy.py",
            "-q",
            "--tb=short",
            *sys.argv[1:],
        ]
    )
)
