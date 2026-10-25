"""Trusted harness smoke test with a disposable known-good patch, never a model run."""

import argparse
import json
import time
import uuid
from pathlib import Path

from agent.schemas import SandboxConfig, Task
from evaluator.qualify import qualify
from sandbox.docker import InfrastructureError, Sandbox, digest, manifest
from sandbox.prepare import PROJECT, write_json


def selftest(setup_dir: Path) -> dict:
    started = time.monotonic()
    setup = json.loads((setup_dir / "setup.json").read_text())
    if setup["setup_status"] != "validated":
        raise InfrastructureError("Validate setup before evaluator smoke tests")
    root = setup_dir / ("evaluator-selftest-" + uuid.uuid4().hex[:8])
    root.mkdir()
    task = Task.model_validate(setup["task"])
    box = Sandbox(
        SandboxConfig.model_validate(setup["sandbox_config"]),
        task,
        setup_dir / "baseline",
        root / "workspace",
        setup_dir / "task.json",
        setup["image_id"],
        setup_dir.parents[1] / "resources",
    )
    box.prepare()
    negative_root = root / "unchanged-baseline-evaluation"
    negative_root.mkdir()
    negative = qualify(
        box, negative_root, PROJECT / "evaluator", time.monotonic() + 300
    )
    write_json(negative_root / "result.json", negative)
    if (
        negative["status"] != "checks_failed"
        or negative["hidden_acceptance"]["exit_code"] == 0
    ):
        raise InfrastructureError("Evaluator failed to reject the unchanged baseline")
    # Fixture remains in trusted setup resources, never in baseline or model context.
    source = box.workspace / task.source_paths[0] / "_client_utils.py"
    text = source.read_text()
    text = text.replace(
        "from functools import lru_cache", "from functools import lru_cache, wraps"
    )
    text = text.replace(
        "from typing import Any, Optional", "from typing import Any, Callable, Optional"
    )
    helper = """
def _cache_client(func: Callable[..., Any]) -> Callable[..., Any]:
    cached = lru_cache(func)

    @wraps(func)
    def wrapped(base_url: Optional[str], timeout: Any) -> Any:
        try:
            hash((base_url, timeout))
        except TypeError:
            return func(base_url, timeout)
        return cached(base_url, timeout)

    return wrapped


"""
    text = text.replace("@lru_cache", "@_cache_client")
    text = text.replace(
        "class _SyncHttpxClientWrapper", helper + "class _SyncHttpxClientWrapper"
    )
    source.write_text(text)
    regression = box.workspace / task.test_paths[0] / "test_timeout_selftest.py"
    regression.write_text("""import httpx

from langchain_openai import ChatOpenAI


def test_custom_timeout() -> None:
    timeout = httpx.Timeout(60.0, connect=5.0)
    model = ChatOpenAI(api_key="sk-offline-placeholder", request_timeout=timeout)
    assert model.root_client._client.timeout.as_dict() == timeout.as_dict()
    assert model.root_async_client._client.timeout.as_dict() == timeout.as_dict()
""")
    formatting = box.execute(
        "ruff check --no-cache --select I --fix langchain_openai/chat_models/_client_utils.py "
        "tests/unit_tests/test_timeout_selftest.py && ruff format --no-cache "
        "langchain_openai/chat_models/_client_utils.py tests/unit_tests/test_timeout_selftest.py",
        30,
    )
    write_json(root / "formatting.json", formatting)
    if formatting["exit_code"]:
        raise InfrastructureError("Smoke-test fixture formatting failed")
    result = qualify(box, root, PROJECT / "evaluator", time.monotonic() + 300)
    result["kind"] = "trusted_evaluator_selftest_not_model_result"
    result["unchanged_baseline_rejected"] = True
    result["negative_control"] = negative
    result["reference_implementation"] = (
        "trusted known-good fixture, not an upstream patch"
    )
    result["baseline_untouched"] = (
        digest(manifest(box.baseline)) == setup["baseline_checksum"]
    )
    result["setup_elapsed_seconds"] = time.monotonic() - started
    result["model_calls"] = 0
    write_json(root / "result.json", result)
    return {"selftest_directory": str(root), **result}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--setup", type=Path, required=True)
    result = selftest(parser.parse_args().setup.resolve())
    print(
        json.dumps(
            {
                "selftest_directory": result["selftest_directory"],
                "status": result["status"],
                "baseline_untouched": result["baseline_untouched"],
            },
            indent=2,
        )
    )
    raise SystemExit(0 if result["status"] == "checks_passed_pending_review" else 2)


if __name__ == "__main__":
    main()
