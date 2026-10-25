"""Defense in depth: the exporter also uses strict field allowlists."""

import re
from typing import Any

KEY = re.compile(
    r"api.?key|authorization|password|credential|secret|private.?key|access.?token",
    re.I,
)
PATTERNS = [
    re.compile(r"\b(?:sk|pk)-[A-Za-z0-9_-]{8,}"),
    re.compile(r"(?i)\bBearer\s+[^\s\"'<>]+"),
    re.compile(
        r"(?i)(?:api[_-]?key|password|secret|access_token)\s*[=:]\s*[^\s,;\"']+"
    ),
    re.compile(
        r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z ]*PRIVATE KEY-----"
    ),
    re.compile(r"https?://[^\s/@]+:[^\s/@]+@[^\s]+"),
    re.compile(r"/(?:Users|home)/[^\s\"'<>]+"),
]


def redact(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            k: "[REDACTED]" if KEY.search(k) else redact(v) for k, v in value.items()
        }
    if isinstance(value, list):
        return [redact(v) for v in value]
    if isinstance(value, str):
        for pattern in PATTERNS:
            value = pattern.sub("[REDACTED]", value)
    return value
