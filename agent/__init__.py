"""Trusted host controller. Never mount this package into an agent container."""

import os
from pathlib import Path

os.environ["MSWEA_SILENT_STARTUP"] = "1"
os.environ["MSWEA_GLOBAL_CONFIG_DIR"] = str(
    Path(__file__).resolve().parents[1] / ".state" / "mini"
)
os.environ["MSWEA_MODEL_RETRY_STOP_AFTER_ATTEMPT"] = "1"
