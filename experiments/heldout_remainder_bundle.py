"""Package validated historical resources; inherit the frozen agent policy unchanged."""

import copy
import json
import secrets
import shutil

import toml
import yaml
from harbor.models.job.config import JobConfig
from harbor.models.task.config import TaskConfig

from experiments.configuration_paths import configuration_path
from experiments.heldout_bundle import image
from experiments.prepare_heldout_remainder import (
    LABEL,
    ROOT,
    TASKS,
    TRUST,
    segment,
    write,
)

PARENT = ROOT / "harbor_tasks/settings/held-out-two-task-v1/LC-ASYNC-TOOLS-001"


def acceptance(tid):
    spec = TASKS[tid]
    state = TRUST / tid
    files = json.loads((state / "reference-tests.json").read_text())
    if spec["issue"] == 39340:
        text = next(v for k, v in files.items() if k.endswith("test_rerank.py"))
        text = "from __future__ import annotations\n" + text.replace(
            "from langchain_fireworks import FireworksRerank",
            "try:\n    from langchain_fireworks import FireworksRerank\nexcept ImportError:\n    FireworksRerank = None",
        )
        return (
            text
            + """\n\ndef test_previous_exports_retained():
    from langchain_fireworks import __all__
    assert {"__version__", "ChatFireworks", "Fireworks", "FireworksEmbeddings", "FireworksRerank"} <= set(__all__)
"""
        )
    if spec["issue"] == 38855:
        return next(
            v for k, v in files.items() if k.endswith("test_huggingface_pipeline.py")
        )
    if spec["issue"] == 39192:
        import ast

        text = next(iter(files.values()))
        nodes = [
            n
            for n in ast.parse(text).body
            if isinstance(n, ast.FunctionDef)
            and n.name.startswith(
                (
                    "test_split_json_non_dict",
                    "test_split_json_list_",
                    "test_split_json_none_",
                    "test_split_json_convert_lists_true_non_list",
                )
            )
        ]
        return (
            "import pytest\nfrom langchain_text_splitters import RecursiveJsonSplitter\n\n"
            + "\n\n".join(segment(text, n) for n in nodes)
        )
    if spec["issue"] == 35300:
        return """from typing import Any
import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.language_models.fake import FakeListLLM

class Chat(FakeListChatModel):
    temperature: Any = None

class Completion(FakeListLLM):
    temperature: Any = None

@pytest.mark.parametrize("kind", [Chat, Completion])
@pytest.mark.parametrize("value", [0, 1, -1, 0.0, 0.25])
def test_numeric_keyword_temperature(kind,value):
    model=kind(responses=["unused"], temperature=0.7)
    result=model._get_ls_params(temperature=value,max_tokens=2048)
    assert result["ls_temperature"]==value
    assert result["ls_max_tokens"]==2048

@pytest.mark.parametrize("kind", [Chat, Completion])
@pytest.mark.parametrize("value", [0, 1, 0.25])
def test_numeric_configured_temperature(kind,value):
    assert kind(responses=["unused"],temperature=value)._get_ls_params()["ls_temperature"]==value

@pytest.mark.parametrize("kind", [Chat, Completion])
def test_invalid_keyword_falls_back_without_losing_numeric_attribute(kind):
    assert kind(responses=["unused"],temperature=1)._get_ls_params(temperature="invalid")["ls_temperature"]==1
    assert "ls_temperature" not in kind(responses=["unused"],temperature=None)._get_ls_params(temperature="invalid")
"""
    if spec["issue"] == 34807:
        return """import pytest
from langchain_core.utils._merge import merge_lists, merge_dicts
from langchain_core.messages import AIMessageChunk

@pytest.mark.parametrize("ids,expected", [(("a","b"),2),(("a","a"),1),(("a",None),1),((None,"b"),1),((None,None),1)])
def test_identity_controls_merging(ids,expected):
    left={"index":0,"id":ids[0],"args":"A"}
    right={"index":0,"id":ids[1],"args":"B"}
    result=merge_lists([left],[right])
    assert len(result)==expected
    if expected==1:assert result[0]["args"]=="AB"
    else:assert [v["args"] for v in result]==["A","B"]

def test_many_parallel_calls_and_message_chunks():
    calls=[{"name":n,"id":n,"index":0,"args":"{}"} for n in ["a","b","c"]]
    assert [v["id"] for v in merge_lists(*[[c] for c in calls])]==["a","b","c"]
    combined=AIMessageChunk(content="",tool_call_chunks=[calls[0]])+AIMessageChunk(content="",tool_call_chunks=[calls[1]])
    assert [c["id"] for c in combined.tool_calls]==["a","b"]

def test_generic_fragment_concatenation_is_preserved():
    assert merge_dicts({"name":"read_","args":"{"},{"name":"file","args":"}"})=={"name":"read_file","args":"{}"}
    assert merge_lists([{"index":0,"args":"a"}],[{"index":0,"args":"b"}])==[{"index":0,"args":"ab"}]
"""
    return """import pytest
from langchain_openrouter import ChatOpenRouter
from langchain_openrouter.chat_models import _create_stream_generation_info

@pytest.mark.parametrize("provider", ["Backend-A", "Backend-B", None, ""])
def test_nonstream_provider_is_actual_and_optional(provider):
    model=ChatOpenRouter(model="offline-model", api_key="offline-placeholder", openrouter_provider={"only":["Requested-Backend"]})
    response={"id":"offline", "model":"offline-model", "provider":provider, "choices":[{"message":{"role":"assistant","content":"hello"},"finish_reason":"stop"}]}
    msg=model._create_chat_result(response).generations[0].message
    assert msg.content=="hello" and msg.response_metadata["model_provider"]=="openrouter"
    if provider:assert msg.response_metadata["provider"]==provider
    else:assert "provider" not in msg.response_metadata

@pytest.mark.parametrize("provider", ["Backend-A", "Backend-B", None, ""])
def test_stream_generation_provider_is_actual_and_optional(provider):
    info=_create_stream_generation_info({"id":"offline","model":"offline-model","provider":provider},{"finish_reason":"stop"},"offline-model")
    assert info["finish_reason"]=="stop"
    if provider:assert info["provider"]==provider
    else:assert "provider" not in info

def test_absent_provider_does_not_create_metadata():
    model=ChatOpenRouter(model="offline-model",api_key="offline-placeholder")
    msg=model._create_chat_result({"choices":[{"message":{"role":"assistant","content":"hello"},"finish_reason":"stop"}]}).generations[0].message
    assert "provider" not in msg.response_metadata
"""


def prepare(tid):
    spec = TASKS[tid]
    state = TRUST / tid
    saved = json.loads((state / "export.json").read_text())
    h = ROOT / "harbor_tasks/settings" / LABEL / tid
    if (h / "task.toml").exists():
        raise FileExistsError("Preserve packaged task revisions")
    package = spec["package"]
    sources = spec["sources"]
    tests = spec["tests"]
    dependency_tag = tid.lower() + "-deps:r1"
    dependency_id = image(dependency_tag)
    for name in [
        "test.sh",
        "workspace_policy.py",
        "docker-compose.yaml",
        "completion-policy.json",
    ]:
        write(h / "tests" / name, (PARENT / "tests" / name).read_text())
    v = (PARENT / "tests/verify.py").read_text()
    lines = v.splitlines()
    overrides = {
        "SOURCES =": "SOURCES = " + repr(sources),
        "TEST_ROOTS =": "TEST_ROOTS = " + repr(tests),
        "PACKAGE =": "PACKAGE = " + repr(package),
        "EXISTING =": "EXISTING = " + repr(spec["existing"]),
        "PYTHONPATH =": "PYTHONPATH = "
        + repr(
            ":".join(
                "/workspace/" + p
                for p in dict.fromkeys(
                    [
                        package,
                        "libs/core",
                        "libs/standard-tests",
                        "libs/partners/openai",
                        "libs/langchain_v1",
                    ]
                )
            )
        ),
    }
    for i, line in enumerate(lines):
        for key, value in overrides.items():
            if line.startswith(key):
                lines[i] = value
                break
    v = "\n".join(lines) + "\n"
    v = v.replace("import shutil\n", "import shutil\nimport shlex\n")
    v = v.replace('"pytest", module,', '"pytest", *shlex.split(module),')
    v = v.replace("sorted(['libs/core', 'libs/partners/openai'])", repr([package]))
    v = v.replace(
        'for p in [\n                "libs/langchain_v1",',
        "for p in [\n                "
        + repr(package)
        + ',\n                "libs/langchain_v1",',
    )
    if spec.get("type_command"):
        v = v.replace('["mypy", "--cache-dir=/dev/null"]', repr(spec["type_command"]))
    v = v.replace(
        '"/tests/test_acceptance.py",',
        '"--asyncio-mode=auto",\n                "/tests/test_acceptance.py",',
    )
    v = v.replace("-p no:cacheprovider --no-cov", "-p no:cacheprovider")
    compile(v, "verify.py", "exec")
    write(h / "tests/verify.py", v)
    write(h / "tests/test_acceptance.py", acceptance(tid))
    fixture = json.loads((state / "reference-source.json").read_text())
    fixture[
        tests[0] + "/test_heldout_reference_regression.py"
    ] = '''"""Check public behavior under the permitted regression-file arrangement."""

import runpy


def test_public_reproduction() -> None:
    """The offline probe succeeds only after the implementation change."""
    runpy.run_path("/task/reproduce.py")
'''
    write(state / "reference-fixture.json", json.dumps(fixture, indent=2))
    key = state / "receipt.key"
    key.write_bytes(secrets.token_bytes(32))
    key.chmod(0o600)
    template = (PARENT / "instruction.md").read_text()
    a = template.index("## Reproduction and environment")
    b = template.index("## Allowed changes")
    instruction = (
        f"Task ID: {tid}\nTask family: {spec['family']}\n\nUse only the supplied unchanged pre-fix snapshot.\n\n## Required behavior\n\n{spec['requirements']}\n\n"
        + template[a:b]
    )
    instruction = instruction.replace(
        "langchain_openai and langchain_core must import from this workspace.",
        "The target package and langchain_core must import from this workspace.",
    )
    instruction += (
        "## Allowed changes\n\nProduction:\n"
        + "\n".join("- `" + s + "/`" for s in sources)
        + "\n\nDevelopment tests:\n"
        + "\n".join(
            "- `" + t + "/test_*.py` (new direct-child files only)" for t in tests
        )
    )
    instruction += (
        "\n\n"
        + template[
            template.index("Existing tests are read-only.") : template.index(
                "## Access restrictions"
            )
        ]
    )
    instruction += template[
        template.index("## Access restrictions") : template.index(
            "## Required completion checklist"
        )
    ]
    instruction += "## Required completion checklist\n\nRun `python /task/reproduce.py` first. Add a regression test at the permitted root and observe pytest exit 1 before implementing. Preserve it in the final workspace. From the package root, run new tests with `python -m pytest -q --disable-socket --allow-unix-socket CHANGED_TESTS`, then rerun the reproducer and these existing checks:\n\n"
    instruction += "\n".join(
        f"- From `/workspace/{p}`: `python -m pytest {m} -q --disable-socket --allow-unix-socket`"
        for p, m in spec["existing"]
    )
    type_command = " ".join(spec.get("type_command", ["mypy", "--cache-dir=/dev/null"]))
    instruction += (
        f"\n\nFrom `/workspace/{package}`, run `ruff check --no-cache CHANGED_FILES`, `ruff format --check --no-cache CHANGED_FILES` and `{type_command} CHANGED_SOURCE`. Preserve package-specific lint/type configuration. Always rerun format --check after formatting. Shell variables do not persist between tool actions. Run tests directly or use set -o pipefail; inspect exit codes.\n\n"
        + template[template.index("## Workspace delivery and stopping") :]
    )
    write(h / "instruction.md", instruction)
    iso = (ROOT / "tasks/LC-CONTEXT-COUNTER-001/isolation.py").read_text()
    first_source = sources[0].removeprefix(package + "/")
    read_file = next(
        p
        for p in (state / "baseline" / sources[0]).glob("*.py")
        if p.name != "__init__.py"
    )
    iso = (
        iso.replace("/workspace/libs/langchain_v1", "/workspace/" + package)
        .replace(
            "langchain/agents/middleware/context_editing.py",
            str(read_file.relative_to(state / "baseline" / package)),
        )
        .replace("langchain/agents/middleware/", first_source + "/")
        .replace(
            "tests/unit_tests/agents/middleware/implementations/test_context_editing.py",
            spec["existing"][0][1].split()[0],
        )
    )
    images = {}
    for role in ["agent", "verifier"]:
        c = state / role
        c.mkdir(exist_ok=True)
        shutil.copytree(state / "baseline", c / "source")
        write(c / "reproduce.py", spec["reproduce"])
        lines = [
            "FROM " + dependency_tag,
            "USER root",
            "COPY source /workspace",
            "COPY reproduce.py /task/reproduce.py",
        ]
        if role == "agent":
            write(c / "isolation.py", iso)
            shutil.copyfile(ROOT / "sandbox/workspace_edit.py", c / "workspace_edit.py")
            write(c / "edit-resources.json", json.dumps({"source_paths": sources}))
            for source in sources:
                shutil.copytree(state / "baseline" / source, c / "unchanged" / source)
            lines += [
                "COPY isolation.py /task/isolation.py",
                "COPY workspace_edit.py /task/workspace_edit.py",
                "COPY edit-resources.json /task/edit-resources.json",
                "COPY unchanged /task/unchanged",
            ]
        else:
            shutil.copytree(state / "baseline", c / "baseline")
            shutil.copytree(h / "tests", c / "tests")
            shutil.copyfile(key, c / "tests/receipt.key")
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
            'ENV PYTEST_ADDOPTS="-p no:cacheprovider"',
            "WORKDIR /workspace/" + package,
            "USER " + ("65532:65532" if role == "agent" else "root"),
        ]
        write(c / "Dockerfile", "\n".join(lines) + "\n")
        images[role] = tid.lower() + "-" + role + ":r1"
    cfg = toml.loads((PARENT / "task.toml").read_text())
    cfg["metadata"].update(
        task_id=tid,
        starting_commit=saved["commit"],
        baseline_checksum=saved["baseline_checksum"],
        experimental_setting=LABEL,
        task_type=spec["family"],
        split="held_out",
        round=1,
    )
    cfg["metadata"].pop("difficulty", None)
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
    compose["volumes"] = {"source0": {}, "tests0": {}}
    compose["services"]["main"]["volumes"] = [
        "source0:/workspace/" + sources[0],
        "tests0:/workspace/" + tests[0],
    ]
    write(
        h / "environment/docker-compose.yaml", yaml.safe_dump(compose, sort_keys=False)
    )
    for model in ["minimax", "glm5"]:
        original = json.loads(
            (
                configuration_path(
                    ROOT,
                    f"configs/context-counter-repaired-calibration-v1-{model}-model.json",
                )
            ).read_text()
        )
        for mode in ["model", "baseline", "reference", "reference_budget"]:
            job = copy.deepcopy(original)
            job["job_name"] = f"{LABEL}-{tid}-{model}-{mode}-001"
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
                "regression_location": "new direct-child test_*.py at permitted test root",
                "type_command": type_command + " $CHANGED_SOURCE",
                "authorized_scope": "round 1: one fresh attempt per task per model; no automatic retries",
            }
            if mode != "model":
                job["agents"][0]["kwargs"]["fixture"] = mode
                job["jobs_dir"] = str(ROOT / "reports/harbor-validation")
            JobConfig.model_validate(job)
            write(
                configuration_path(ROOT, f"configs/{LABEL}-{tid}-{model}-{mode}.json"),
                json.dumps(job, indent=2) + "\n",
            )
    write(
        state / "bundle.json",
        json.dumps(
            {
                "task_id": tid,
                "bundle": str(h),
                "images": images,
                "dependency_id": dependency_id,
            },
            indent=2,
        ),
    )
    print(tid, "packaged", flush=True)


if __name__ == "__main__":
    import sys

    prepare(sys.argv[1])
