"""Task packaging, preserving the frozen agent loop, instructions and limits."""

import copy
import json
import secrets
import shutil
import subprocess

import toml
import yaml
from harbor.models.job.config import JobConfig
from harbor.models.task.config import TaskConfig

from experiments.configuration_paths import configuration_path
from experiments.prepare_heldout_tasks import LABEL, ROOT, TASKS, TRUST, write

PARENT = (
    ROOT / "harbor_tasks/settings/context-counter-environment-v2/LC-CONTEXT-COUNTER-001"
)


def image(tag):
    return subprocess.check_output(
        ["docker", "image", "inspect", tag, "--format", "{{.Id}}"], text=True
    ).strip()


def prepare(task_id):
    spec = TASKS[task_id]
    state = TRUST / task_id
    saved = json.loads((state / "export.json").read_text())
    h = ROOT / "harbor_tasks/settings" / LABEL / task_id
    if (h / "task.toml").exists():
        raise FileExistsError("Never overwrite a packaged held-out task")
    package, sources, tests = spec["package"], spec["sources"], spec["tests"]
    dependency_tag = (
        "heldout-async-deps:v1"
        if spec["issue"] == 40205
        else "heldout-overflow-deps:v1"
    )
    dependency_id = image(dependency_tag)
    for name in ("test.sh", "workspace_policy.py", "docker-compose.yaml"):
        write(h / "tests" / name, (PARENT / "tests" / name).read_text())
    completion = json.loads(
        (
            configuration_path(ROOT, "configs/completion-response-6000-v1.json")
        ).read_text()
    )
    write(h / "tests/completion-policy.json", json.dumps(completion, indent=2))
    # Generalize only the verifier's task resource mapping, not its success rule.
    v = (PARENT / "tests/verify.py").read_text()
    v = v.replace(
        'SOURCE = "libs/langchain_v1/langchain/agents/middleware"',
        "SOURCES = " + repr(sources),
    )
    v = v.replace(
        'TESTS = "libs/langchain_v1/tests/unit_tests"', "TEST_ROOTS = " + repr(tests)
    )
    v = v.replace('PACKAGE = "libs/langchain_v1"', "PACKAGE = " + repr(package))
    v = v.replace("for rel in (SOURCE, TESTS)", "for rel in (*SOURCES, *TEST_ROOTS)")
    v = v.replace(
        'for rel, artifact in [(SOURCE, "production"), (TESTS, "development-tests")]:',
        "for rel in (*SOURCES, *TEST_ROOTS):\n            artifact = rel",
    )
    v = v.replace("if rel == TESTS:", "if rel in TEST_ROOTS:")
    v = v.replace(
        'p.startswith(SOURCE + "/")', 'any(p.startswith(s + "/") for s in SOURCES)'
    )
    start = v.index('        run(\n            "existing_tests",')
    end = v.index("        if changed:", start)
    v = (
        v[:start]
        + """        existing = []
        for index, (package, module) in enumerate(EXISTING):
            r = run("existing_tests_" + str(index), ["python", "-m", "pytest", module, "-q", "--disable-socket", "--allow-unix-socket"], cwd=WORK / package)
            existing.append(r)
        details["checks"]["existing_tests"] = {"exit_code": 0 if all(r.returncode == 0 for r in existing) else 1, "output": "\\n".join(r.stdout + r.stderr for r in existing)}
"""
        + v[end:]
    )
    v = v.replace(
        "PACKAGE = " + repr(package),
        "EXISTING = " + repr(spec["existing"]) + "\nPACKAGE = " + repr(package),
    )
    # Several package roots can have a top-level test_*.py: use distinct filenames.
    compile(v, "heldout_verify.py", "exec")
    write(h / "tests/verify.py", v)
    upstream = (state / "upstream-added-tests.txt").read_text()
    if spec["issue"] == 40205:
        imports = """from typing import Any, cast
import pytest
from langchain_core.messages import AIMessage, HumanMessage, AIMessageChunk
from langchain_core.messages import content as types
from langchain_openai import ChatOpenAI
from langchain_openai.chat_models.base import _construct_lc_result_from_responses_api, _convert_responses_chunk_to_generation_chunk
from openai.types.responses.response import Response
from openai.types.responses.response_function_tool_call import ResponseFunctionToolCall
OPENAI_TEST_MODEL = "offline-test-model"
"""
    else:
        imports = """from typing import Any
import pytest
from typing_extensions import override
from pydantic import Field
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.exceptions import ContextOverflowError, ModelAuthenticationError
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage, BaseMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain.agents.middleware.summarization import SummarizationMiddleware
"""
    # Acceptance controls are finalized and hashed before any evaluated inference.
    write(h / "tests/test_acceptance.py", imports + "\n" + upstream + "\n")
    new_test = tests[0] + "/test_heldout_reference_regression.py"
    reg = '''"""Check the public reproduction under the permitted test-root policy."""

import runpy


def test_public_reproduction() -> None:
    """The supplied offline probe succeeds only after implementation."""
    runpy.run_path("/task/reproduce.py")
'''
    fixture = json.loads((state / "reference-source.json").read_text())
    fixture[new_test] = reg
    write(state / "reference-fixture.json", json.dumps(fixture, indent=2))
    key = state / "receipt.key"
    if not key.exists():
        key.write_bytes(secrets.token_bytes(32))
        key.chmod(0o600)
    # Preserve all shared workflow instructions verbatim; substitute task resources.
    parent_instruction = (PARENT / "instruction.md").read_text()
    instruction = f"Task ID: {task_id}\nDifficulty: L2\nTask family: {spec['family']}\n\nUse only the supplied unchanged pre-fix snapshot.\n\n## Required behavior\n\n{spec['requirements']}\n\n"
    instruction += parent_instruction[
        parent_instruction.index(
            "## Reproduction and environment"
        ) : parent_instruction.index("## Allowed changes")
    ].replace(
        "This offline feature probe demonstrates the missing constructor API.",
        "This offline probe demonstrates the reported missing behavior.",
    )
    instruction += (
        "## Allowed changes\n\nProduction:\n"
        + "\n".join("- `" + p + "/`" for p in sources)
        + "\n\nDevelopment tests:\n"
        + "\n".join(
            "- `" + p + "/test_*.py` (new direct-child files only)" for p in tests
        )
    )
    instruction += "\n\nExisting tests are read-only. Do not modify, rename, delete or format them. No new test subdirectories or __init__.py files. Do not change dependency manifests, lockfiles, environment configuration or unrelated public APIs.\n\n"
    instruction += parent_instruction[
        parent_instruction.index("## Access restrictions") : parent_instruction.index(
            "## Required completion checklist"
        )
    ]
    instruction += "## Required completion checklist\n\nRun the public reproducer first: `python /task/reproduce.py`. Add a regression and observe pytest exit 1 before editing source. Run new regressions from their package root with `python -m pytest -q --disable-socket --allow-unix-socket PATH`. Use unique new test filenames across roots.\n\nExisting checks:\n"
    instruction += "\n".join(
        f"- From `/workspace/{p}`: `python -m pytest {m} -q --disable-socket --allow-unix-socket`"
        for p, m in spec["existing"]
    )
    instruction += "\n\nFrom each changed package root, run `ruff check --no-cache CHANGED_FILES`, `ruff format --check --no-cache CHANGED_FILES` and `mypy --cache-dir=/dev/null CHANGED_SOURCE`. Preserve package-specific lint/type configuration. Rerun format --check after any formatting. Run tests directly or use set -o pipefail and inspect exit codes. Shell variables do not persist between tool actions.\n\n"
    instruction += parent_instruction[
        parent_instruction.index("## Workspace delivery and stopping") :
    ]
    write(h / "instruction.md", instruction)
    iso = (ROOT / "tasks/LC-CONTEXT-COUNTER-001/isolation.py").read_text()
    root = "/workspace/" + package
    first_source = sources[0].removeprefix(package + "/")
    existing_rel = spec["existing"][0][1]
    # Probe concrete task paths, with all other isolation assertions unchanged.
    read_file = next(
        p
        for p in (state / "baseline" / sources[0]).glob("*.py")
        if p.name != "__init__.py"
    )
    rel_file = str(read_file.relative_to(state / "baseline" / package))
    iso = (
        iso.replace("/workspace/libs/langchain_v1", root)
        .replace("langchain/agents/middleware/context_editing.py", rel_file)
        .replace("langchain/agents/middleware/", first_source + "/")
        .replace(
            "tests/unit_tests/agents/middleware/implementations/test_context_editing.py",
            existing_rel,
        )
    )
    images = {}
    for role in ("agent", "verifier"):
        context = state / role
        context.mkdir(exist_ok=True)
        shutil.copytree(state / "baseline", context / "source")
        write(context / "reproduce.py", spec["reproduce"])
        lines = [
            "FROM " + dependency_tag,
            "USER root",
            "COPY source /workspace",
            "COPY reproduce.py /task/reproduce.py",
        ]
        if role == "agent":
            write(context / "isolation.py", iso)
            shutil.copyfile(
                ROOT / "sandbox/workspace_edit.py", context / "workspace_edit.py"
            )
            write(
                context / "edit-resources.json", json.dumps({"source_paths": sources})
            )
            for s in sources:
                shutil.copytree(state / "baseline" / s, context / "unchanged" / s)
            lines += [
                "COPY isolation.py /task/isolation.py",
                "COPY workspace_edit.py /task/workspace_edit.py",
                "COPY edit-resources.json /task/edit-resources.json",
                "COPY unchanged /task/unchanged",
            ]
        else:
            shutil.copytree(state / "baseline", context / "baseline")
            shutil.copytree(h / "tests", context / "tests")
            shutil.copyfile(key, context / "tests/receipt.key")
            lines += [
                "COPY baseline /baseline",
                "COPY tests /tests",
                "RUN chmod -R a-w /baseline /tests && chmod 400 /tests/receipt.key",
            ]
        lines += [
            "RUN chmod -R a-w /workspace /task && mkdir -p /logs/agent /logs/verifier /logs/artifacts && chmod 777 /logs/agent /logs/verifier /logs/artifacts"
        ]
        if role == "agent":
            lines += [
                "RUN "
                + " && ".join(
                    f"find /workspace/{s} -type d -exec chmod 777 {{}} + && find /workspace/{s} -type f -exec chmod 666 {{}} +"
                    for s in sources
                )
                + " && "
                + " && ".join("chmod 1777 /workspace/" + t for t in tests)
            ]
        lines += [
            "WORKDIR /workspace/" + package,
            "USER " + ("65532:65532" if role == "agent" else "root"),
        ]
        write(context / "Dockerfile", "\n".join(lines) + "\n")
        images[role] = "heldout-" + task_id.lower() + "-" + role + ":v1"
    cfg = toml.loads((PARENT / "task.toml").read_text())
    cfg["metadata"].update(
        task_id=task_id,
        starting_commit=saved["commit"],
        baseline_checksum=saved["baseline_checksum"],
        experimental_setting=LABEL,
        task_type=spec["family"],
        difficulty="L2",
        split="held_out",
    )
    cfg["environment"].update(
        docker_image=images["agent"], workdir="/workspace/" + package
    )
    cfg["verifier"]["environment"]["docker_image"] = images["verifier"]
    cfg["verifier"]["collect"][0]["command"] = (
        "rm -f /logs/artifacts/candidate.tar && tar --mode='u+rwX,go+rX' -cf /logs/artifacts/candidate.tar -C /workspace "
        + " ".join(sources + tests)
    )
    TaskConfig.model_validate(cfg)
    write(h / "task.toml", toml.dumps(cfg))
    compose = yaml.safe_load((PARENT / "environment/docker-compose.yaml").read_text())
    volumes = {"source" + str(i): {} for i in range(len(sources))} | {
        "tests" + str(i): {} for i in range(len(tests))
    }
    compose["volumes"] = volumes
    compose["services"]["main"]["volumes"] = [
        f"source{i}:/workspace/{p}" for i, p in enumerate(sources)
    ] + [f"tests{i}:/workspace/{p}" for i, p in enumerate(tests)]
    write(
        h / "environment/docker-compose.yaml", yaml.safe_dump(compose, sort_keys=False)
    )
    for model_key in ("minimax", "glm5"):
        original = json.loads(
            (
                configuration_path(
                    ROOT,
                    f"configs/context-counter-repaired-calibration-v1-{model_key}-model.json",
                )
            ).read_text()
        )
        for mode in ("model", "baseline", "reference", "reference_budget"):
            job = copy.deepcopy(original)
            job["job_name"] = f"{LABEL}-{task_id}-{model_key}-{mode}-001"
            job["tasks"] = [{"path": str(h)}]
            policy = job["agents"][0]["kwargs"]["policy"]
            policy.update(
                receipt_key=str(key),
                reference_fixture=str(state / "reference-fixture.json"),
                source_provenance={
                    "starting_commit": saved["commit"],
                    "baseline_checksum": saved["baseline_checksum"],
                },
                reproduction={"expected_failure": spec["expected"]},
            )
            policy["environment_revision"] = {
                "version": LABEL,
                "regression_location": "new direct-child test_*.py at each permitted test root",
                "type_command": "mypy --cache-dir=/dev/null $CHANGED_SOURCE",
                "authorized_scope": "one held-out attempt per task per model, no automatic retries",
            }
            if mode != "model":
                job["agents"][0]["kwargs"]["fixture"] = mode
                job["jobs_dir"] = str(ROOT / "reports/harbor-validation")
            JobConfig.model_validate(job)
            write(
                configuration_path(
                    ROOT, f"configs/{LABEL}-{task_id}-{model_key}-{mode}.json"
                ),
                json.dumps(job, indent=2) + "\n",
            )
    write(
        state / "bundle.json",
        json.dumps(
            {
                "task_id": task_id,
                "bundle": str(h),
                "images": images,
                "dependency_id": dependency_id,
            },
            indent=2,
        ),
    )
    print(task_id, "packaged; images not yet validated")


if __name__ == "__main__":
    import sys

    prepare(sys.argv[1])
