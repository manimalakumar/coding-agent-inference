"""Build public setup contracts from retained task definitions, without importing clients."""

import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def literal_tasks(file):
    tree = ast.parse((ROOT / file).read_text())
    return next(
        ast.literal_eval(n.value)
        for n in tree.body
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "TASKS" for t in n.targets)
    )


def catalog():
    pins = json.loads((ROOT / "public_preparation/configs/tasks.json").read_text())
    result = {}
    held = {
        **literal_tasks("experiments/prepare_heldout_tasks.py"),
        **literal_tasks("experiments/prepare_heldout_remainder.py"),
    }
    development = {
        "LC-TIMEOUT-001": (31698, "comparison-4x2x3-v1"),
        "LC-RESPONSES-USAGE-001": (31756, "comparison-4x2x3-v1"),
        "LC-ANNOTATED-TOOLS-001": (31181, "harbor-workspace-deliverable-v2"),
        "LC-CHROMA-NONE-001": (31377, "harbor-chroma-none-workspace-v1"),
        "LC-CONTEXT-COUNTER-001": (39754, "context-counter-environment-v2"),
    }
    for tid, pin in pins.items():
        spec = dict(pin)
        spec["task_id"] = tid
        if tid in held:
            data = held[tid]
            group = (
                "held-out-two-task-v1"
                if tid in literal_tasks("experiments/prepare_heldout_tasks.py")
                else "held-out-remainder-round1-v1"
            )
            spec.update(
                package=data["package"],
                source_paths=data["sources"],
                test_paths=data["tests"],
                reproducer=data["reproduce"],
                expected_failure=data["expected"],
                pull_request=data["pr"],
                requirements=data["requirements"],
            )
        else:
            data = json.loads((ROOT / "tasks" / tid / "task.json").read_text())
            pr, group = development[tid]
            spec.update(
                package=data["package"],
                source_paths=data["source_paths"],
                test_paths=data["test_paths"],
                pull_request=pr,
                requirements=data["instructions"],
                expected_failure=data["expected_failure"],
            )
            f = ROOT / "tasks" / tid / "reproduce.py"
            spec["reproducer"] = f.read_text() if f.exists() else ""
        spec["task_bundle"] = f"harbor_tasks/settings/{group}/{tid}"
        bundle = ROOT / spec["task_bundle"]
        spec["protocol_id"] = "public-" + tid.lower() + "-v1"
        spec["historical_equivalence"] = "not claimed"
        spec["retained_evaluator"] = str(bundle / "tests/verify.py")
        spec["retained_acceptance"] = str(bundle / "tests/test_acceptance.py")
        result[tid] = spec
    return result
