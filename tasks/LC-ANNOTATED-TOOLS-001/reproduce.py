"""Offline reproduction of lost Annotated descriptions with postponed annotations."""

from __future__ import annotations

import json
from typing import Annotated

from langchain_core.tools import tool


@tool
def choose_fragments(parts: Annotated[list[str], "Fragments to evaluate"]) -> list[str]:
    """Return the requested fragments."""
    return parts


actual = choose_fragments.args
print(json.dumps(actual, indent=2))
assert actual["parts"].get("description") == "Fragments to evaluate", (
    "Annotated parameter description missing"
)
assert actual["parts"]["type"] == "array"
assert actual["parts"]["items"]["type"] == "string"
assert choose_fragments.invoke({"parts": ["alpha", "beta"]}) == ["alpha", "beta"]
print("Annotated parameter description and tool behavior preserved")
