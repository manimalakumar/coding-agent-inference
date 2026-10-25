"""Trusted preparation of versioned, identical task bundles for both models."""

import copy
import hashlib
import json
import secrets
import shutil
import subprocess
from pathlib import Path

import toml
from harbor.models.job.config import JobConfig
from harbor.models.task.config import TaskConfig

from experiments.configuration_paths import configuration_path
from harbor_integration.prepare_resources import build
from sandbox.docker import digest, manifest


def prepare():
    R = Path.cwd()
    LABEL = "comparison-4x2x3-v1"
    if (configuration_path(R, f"configs/{LABEL}.frozen-setting.json")).exists():
        raise FileExistsError("Never overwrite a frozen comparison setting")
    catalog = json.loads((configuration_path(R, "configs/study.json")).read_text())
    common = json.loads(
        (configuration_path(R, "configs/completion-workspace-v1.json")).read_text()
    )
    basejob = json.loads(
        (
            configuration_path(R, "configs/harbor-chroma-none-workspace-v1-model.json")
        ).read_text()
    )
    generic = basejob["agents"][0]["kwargs"]["policy"]["mini_config"]
    TEMPLATE = (
        R
        / "harbor_tasks/settings/harbor-workspace-deliverable-v2/LC-ANNOTATED-TOOLS-001"
    )
    STUDIES = [
        "timeout",
        "responses-usage",
        "annotated-tools-workspace-v2",
        "chroma-none-workspace-v1",
    ]
    records = []
    for study in STUDIES:
        oldjob = json.loads((R / catalog["studies"][study]["job_config"]).read_text())
        oldpolicy = oldjob["agents"][0]["kwargs"]["policy"]
        oldtask = Path(oldjob["tasks"][0]["path"])
        ID = oldtask.name
        H = R / "harbor_tasks/settings" / LABEL / ID
        state = R / ".local/resources" / LABEL / ID
        if (H / "task.toml").exists():
            records.append(json.loads((state / "preparation.json").read_text()))
            continue
        H.mkdir(parents=True, exist_ok=True)
        state.mkdir(parents=True, exist_ok=True)
        prior = json.loads(
            (R / catalog["studies"][study]["frozen_setting"]).read_text()
        )
        oldpolicy.setdefault(
            "source_provenance",
            {
                "starting_commit": prior["starting_commit"],
                "baseline_checksum": prior["baseline_checksum"],
            },
        )
        oldpolicy.setdefault(
            "reproduction",
            {
                "expected_failure": json.loads(
                    (R / "tasks" / ID / "task.json").read_text()
                )["expected_failure"]
            },
        )
        source_config = toml.loads((oldtask / "task.toml").read_text())
        source = (
            source_config["verifier"]["collect"][0]["command"]
            .split(" -C /workspace ")[1]
            .split()[0]
        )
        tests = (
            source.replace("/langchain_chroma", "/tests/unit_tests")
            if "chroma" in source
            else "libs/core/tests/unit_tests"
            if source.startswith("libs/core")
            else "libs/partners/openai/tests/unit_tests"
        )
        package = source_config["environment"]["workdir"].removeprefix("/workspace/")
        baseline = next(
            p / "baseline"
            for p in [
                Path(oldpolicy["receipt_key"]).parent,
                *Path(oldpolicy["receipt_key"]).parents,
            ]
            if (p / "baseline").is_dir()
        )
        assert (
            digest(manifest(baseline))
            == oldpolicy["source_provenance"]["baseline_checksum"]
        )
        existing = (
            ["tests/unit_tests"]
            if study == "chroma-none-workspace-v1"
            else [
                "tests/unit_tests/test_tools.py",
                "tests/unit_tests/utils/test_function_calling.py",
            ]
            if study == "annotated-tools-workspace-v2"
            else ["tests/unit_tests/chat_models/test_base.py"]
        )
        pythonpaths = list(
            dict.fromkeys(
                [package, "libs/core", "libs/standard-tests", "libs/partners/openai"]
            )
        )
        for d in ["tests", "environment"]:
            (H / d).mkdir(exist_ok=True)
        for name in [
            "test.sh",
            "docker-compose.yaml",
            "workspace_policy.py",
            "completion-policy.json",
        ]:
            shutil.copyfile(TEMPLATE / "tests" / name, H / "tests" / name)
        shutil.copyfile(
            oldtask / "tests/test_acceptance.py", H / "tests/test_acceptance.py"
        )
        verify = (
            (TEMPLATE / "tests/verify.py")
            .read_text()
            .replace(
                'SOURCE = "libs/core/langchain_core/tools"', f'SOURCE = "{source}"'
            )
            .replace('TESTS = "libs/core/tests/unit_tests"', f'TESTS = "{tests}"')
            .replace('PACKAGE = "libs/core"', f'PACKAGE = "{package}"')
        )
        verify = verify.replace(
            '"/workspace/libs/core:/workspace/libs/standard-tests:/workspace/libs/partners/openai"',
            repr(":".join("/workspace/" + p for p in pythonpaths)),
        ).replace(
            '["libs/core", "libs/standard-tests", "libs/partners/openai"]',
            repr(pythonpaths),
        )
        verify = verify.replace(
            '"tests/unit_tests/test_tools.py",\n                "tests/unit_tests/utils/test_function_calling.py",',
            "\n                ".join(repr(p) + "," for p in existing),
        )
        if existing == ["tests/unit_tests"]:
            verify = verify.replace(
                "'tests/unit_tests',",
                '*[str(WORK / TESTS / p) for p in manifest(BASE / TESTS) if Path(p).name.startswith("test_") and p.endswith(".py")],',
            )
        (H / "tests/verify.py").write_text(verify)
        instruction = (oldtask / "instruction.md").read_text()
        if study == "responses-usage":
            instruction = instruction[
                : instruction.index("## Budget and submission")
            ] + (TEMPLATE / "instruction.md").read_text().split(
                "## Workspace delivery and stopping", 1
            )[1].join(["## Workspace delivery and stopping", ""])
        if study == "timeout":
            sibling = (
                R / "harbor_tasks/LC-RESPONSES-USAGE-001/instruction.md"
            ).read_text()
            rest = sibling[
                sibling.index("## Allowed changes") : sibling.index(
                    "## Budget and submission"
                )
            ]
            rest = rest.replace(
                "Responses API usage metadata",
                "timeout handling and client initialization",
            )
            intro = """Task ID: LC-TIMEOUT-001
    Title: Support custom timeout objects during ChatOpenAI initialization
    Difficulty: Level 2
    
    Use only the supplied unchanged historical LangChain snapshot.
    
    ## Problem
    
    Initializing ChatOpenAI(api_key="sk-offline-placeholder", model="gpt-4.1-mini", request_timeout=httpx.Timeout(timeout=60.0, connect=5.0)) raises TypeError: unhashable type: 'Timeout' before any model request.
    Supported timeout configurations must initialize successfully and preserve their configured values on the underlying synchronous and asynchronous HTTP clients, including every configured timeout component. Checking only the model's configuration field is insufficient. Equivalent defaults must still reuse HTTP clients. Distinct base URLs or numeric timeout settings must not alias. Do not replace configured values with defaults or disable reuse for all configurations. Do not invoke, stream or contact any remote service.
    
    ## Reproduction and environment
    
    Your first sandbox command must be exactly: `python /task/reproduce.py`.
    Run it before any inspection or edit. Add a new regression file and observe the issue before changing implementation. Use the prepared Python 3.14.6 environment and frozen dependencies, with workspace imports, httpx 0.28.1, OpenAI 1.90.0 and Pydantic 2.12.5. Do not install or upgrade packages. Use offline placeholder credentials and mocked/fabricated data where needed. Missing dependencies or collection failures are environment blockers, not invalid issues.
    
    """
            instruction = (
                intro
                + rest
                + "## Workspace delivery and stopping"
                + (TEMPLATE / "instruction.md")
                .read_text()
                .split("## Workspace delivery and stopping", 1)[1]
            )
        instruction = instruction.replace(
            "reserve calls for implementation, verification and final submission",
            "budget calls for implementation and verification",
        )
        (H / "instruction.md").write_text(instruction)
        shutil.copyfile(
            oldtask / "environment/docker-compose.yaml",
            H / "environment/docker-compose.yaml",
        )
        subprocess.run(
            [str(R / ".venv/bin/ruff"), "check", "--fix", str(H / "tests/verify.py")],
            check=True,
            capture_output=True,
        )
        subprocess.run(
            [str(R / ".venv/bin/ruff"), "format", str(H / "tests/verify.py")],
            check=True,
            capture_output=True,
        )
        for name in ["agent", "verifier"]:
            context = state / name
            context.mkdir(exist_ok=True)
            oldimage = (
                source_config["environment"]["docker_image"]
                if name == "agent"
                else source_config["verifier"]["environment"]["docker_image"]
            )
            tag = "maintenance-comparison-parent:" + name + "-" + ID.lower()
            subprocess.run(["docker", "tag", oldimage, tag], check=True)
            lines = [f"FROM {tag}", "USER root"]
            if name == "agent":
                assets = context / "assets"
                assets.mkdir(exist_ok=True)
                shutil.copyfile(
                    R / "sandbox/workspace_edit.py", assets / "workspace_edit.py"
                )
                (assets / "edit-resources.json").write_text(
                    json.dumps({"source_paths": [source]})
                )
                destination = assets / "unchanged" / source
                destination.parent.mkdir(parents=True, exist_ok=True)
                if not destination.exists():
                    shutil.copytree(baseline / source, destination)
                lines += [
                    "COPY assets /task",
                    f"RUN chmod -R a-w /task && find /workspace/{source} -type d -exec chmod 777 {{}} + && find /workspace/{source} -type f -exec chmod 666 {{}} + && find /workspace/{tests} -type d -exec chmod 555 {{}} + && find /workspace/{tests} -type f -exec chmod 444 {{}} + && chmod 1777 /workspace/{tests}",
                    "USER 65532:65532",
                ]
            else:
                shutil.copytree(H / "tests", context / "tests", dirs_exist_ok=True)
                key = state / "receipt.key"
                if not key.exists():
                    key.write_bytes(secrets.token_bytes(32))
                    key.chmod(0o600)
                shutil.copyfile(key, context / "tests/receipt.key")
                lines += [
                    "COPY tests /tests",
                    "RUN chmod -R a-w /tests && chmod 400 /tests/receipt.key",
                    "USER root",
                ]
            (context / "Dockerfile").write_text("\n".join(lines) + "\n")
        agentimage = build(
            state / "agent", "maintenance-comparison-agent:" + ID.lower()
        )
        verifierimage = build(
            state / "verifier", "maintenance-comparison-verifier:" + ID.lower()
        )
        cfg = copy.deepcopy(source_config)
        cfg["metadata"].pop("shared_workflow_seconds", None)
        cfg["metadata"].update(
            experimental_setting=LABEL,
            completion_policy=common["policy_id"],
            agent_budget_seconds=1200,
            evaluation_budget_seconds=300,
        )
        cfg["agent"]["timeout_sec"] = 1280
        cfg["verifier"]["timeout_sec"] = 300
        cfg["environment"]["docker_image"] = agentimage
        cfg["verifier"]["environment"]["docker_image"] = verifierimage
        TaskConfig.model_validate(cfg)
        (H / "task.toml").write_text(toml.dumps(cfg))
        fixture = state / "reference-fixture.json"
        shutil.copyfile(oldpolicy["reference_fixture"], fixture)
        policy = copy.deepcopy(basejob["agents"][0]["kwargs"]["policy"])
        policy.update(
            source_provenance=oldpolicy["source_provenance"],
            reproduction=oldpolicy["reproduction"],
            receipt_key=str(state / "receipt.key"),
            reference_fixture=str(fixture),
            mini_config=generic,
            completion_policy=common,
        )
        policy["code_fingerprints"]["harbor_integration/host_comparison.py"] = (
            hashlib.sha256(
                (R / "harbor_integration/host_comparison.py").read_bytes()
            ).hexdigest()
        )
        policy["runtime_revision"] = "comparison-civil-deadline-v1"
        for modelkey, modelfile in [
            ("minimax", "configs/minimax-m2.5-explicit-test-policy.json"),
            ("glm5", "configs/glm-5-comparison.json"),
        ]:
            modelpolicy = copy.deepcopy(policy)
            modelpolicy["model"] = json.loads((R / modelfile).read_text())
            for mode in (
                [
                    "model",
                    "baseline",
                    "reference",
                    "reference_budget",
                    "reference_error",
                ]
                if modelkey == "minimax"
                else ["model"]
            ):
                job = copy.deepcopy(basejob)
                job["job_name"] = f"{LABEL}-{ID}-{modelkey}-{mode}"
                job["tasks"] = [{"path": str(H)}]
                job["agents"][0]["import_path"] = (
                    "harbor_integration.host_comparison:ComparisonAgent"
                )
                job["agents"][0]["model_name"] = (
                    "openrouter/" + modelpolicy["model"]["model"]
                )
                job["agents"][0]["kwargs"] = {"policy": modelpolicy}
                if mode != "model":
                    job["agents"][0]["kwargs"]["fixture"] = mode
                    job["jobs_dir"] = str(R / "reports/harbor-validation")
                JobConfig.model_validate(job)
                (
                    configuration_path(
                        R, f"configs/{LABEL}-{ID}-{modelkey}-{mode}.json"
                    )
                ).write_text(json.dumps(job, indent=2) + "\n")
        record = {
            "task_id": ID,
            "parent_study": study,
            "task_directory": str(H),
            "agent_image": agentimage,
            "verifier_image": verifierimage,
            "starting_commit": policy["source_provenance"]["starting_commit"],
            "baseline_checksum": policy["source_provenance"]["baseline_checksum"],
            "state_directory": str(state),
            "hidden_acceptance_sha256": hashlib.sha256(
                (H / "tests/test_acceptance.py").read_bytes()
            ).hexdigest(),
            "original_hidden_acceptance_sha256": hashlib.sha256(
                (oldtask / "tests/test_acceptance.py").read_bytes()
            ).hexdigest(),
            "setup_model_requests": 0,
        }
        (state / "preparation.json").write_text(json.dumps(record, indent=2))
        records.append(record)
        print(ID, "prepared", flush=True)
    (R / f"reports/{LABEL}-preparation.json").write_text(
        json.dumps(
            {
                "task_bundles": records,
                "setup_model_requests": 0,
                "measured_round_started": False,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    prepare()
