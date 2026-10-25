"""Redacted request logging alongside Harbor native trajectories."""

import json
from datetime import datetime, timezone
from pathlib import Path


class Trace:
    def __init__(self, path: Path, secret: str):
        self.path, self.secret = path, secret

    def emit(self, event, **fields):
        text = json.dumps(
            {"at": datetime.now(timezone.utc).isoformat(), "event": event, **fields}
        )
        if self.secret:
            text = text.replace(self.secret, "[REDACTED]")
        with self.path.open("a") as output:
            output.write(text + "\n")
