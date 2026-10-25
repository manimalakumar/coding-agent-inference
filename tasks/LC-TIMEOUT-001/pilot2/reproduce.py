"""Run offline before any source inspection or edits. Failure is expected."""

import httpx

from langchain_openai import ChatOpenAI

ChatOpenAI(
    api_key="sk-offline-placeholder",
    model="gpt-4.1-mini",
    request_timeout=httpx.Timeout(timeout=60.0, connect=5.0),
)

print("Initialization succeeded")
