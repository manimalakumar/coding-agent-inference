"""Validation-only I/O guards; never inserted into the saved notebook."""

from textwrap import dedent

GUARD = dedent("""
    import sys as _guard_sys
    from pathlib import Path as _GuardPath
    _forbidden_events = []
    def _replay_audit(event, args):
        denied = event in {
            "subprocess.Popen", "os.system", "os.exec", "os.posix_spawn",
            "socket.connect", "socket.getaddrinfo", "socket.sendto",
        }
        if event == "open" and args and isinstance(args[0], (str, bytes)):
            name = str(args[0])
            denied = denied or "/.config/openrouter/" in name or name.endswith("credentials.env")
        if denied:
            _forbidden_events.append(event)
            raise RuntimeError("Replay blocked: " + event)
    _guard_sys.addaudithook(_replay_audit)
    # Negative controls prove blocking happens before an OS connection/process/read.
    import socket as _guard_socket
    import subprocess as _guard_subprocess
    _probe_results = {}
    def _connection_probe():
        with _guard_socket.socket() as sock:
            sock.connect(("203.0.113.1", 443))
    for _label, _probe in [
        ("outbound_socket", _connection_probe),
        ("docker_process", lambda: _guard_subprocess.run(["docker", "info"], check=True)),
        ("credential_read", lambda: open(str(_GuardPath.home() / ".config/openrouter/credentials.env"))),
    ]:
        try:
            _probe()
        except RuntimeError as error:
            assert str(error).startswith("Replay blocked:")
            _probe_results[_label] = "blocked before side effect"
        else:
            raise AssertionError("Replay tripwire did not block " + _label)
    _forbidden_events.clear()
""")

FINAL_CHECK = dedent("""
    assert MODE == "replay" and CONFIRM_PAID_EXECUTION is False
    assert not _forbidden_events, _forbidden_events
    import json as _guard_json
    print(_guard_json.dumps({"negative_controls": _probe_results,
                             "replay_forbidden_io_attempts": _forbidden_events,
                             "paid_calls": 0, "docker_processes": 0}))
""")
