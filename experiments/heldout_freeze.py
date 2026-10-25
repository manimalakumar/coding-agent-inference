"""Freeze held-out task environments only after all zero-inference gates pass."""

import copy
import hashlib
import json
import subprocess
import time
from pathlib import Path

from experiments.configuration_paths import configuration_path
from experiments.launch import verify_freeze
from experiments.prepare_heldout_tasks import LABEL, ROOT, TASKS, TRUST, write
from harbor_integration.host_response_ceiling import ResponseCeilingAgent
from sandbox.docker import digest, manifest


def freeze():
    destination = configuration_path(ROOT, f"configs/{LABEL}.frozen-setting.json")
    if destination.exists():
        raise FileExistsError("Never replace a frozen setting")
    parent_name = "context-counter-repaired-calibration-v1"
    parent_id = verify_freeze(
        ROOT,
        f"configs/{parent_name}.sha256",
        f"configs/{parent_name}.frozen-setting.json",
    )
    parent = json.loads(
        (
            configuration_path(ROOT, f"configs/{parent_name}.frozen-setting.json")
        ).read_text()
    )
    paths = set(parent["source_and_config_sha256"])
    paths |= {
        "experiments/prepare_heldout_tasks.py",
        "experiments/heldout_bundle.py",
        "experiments/heldout_controls.py",
        "experiments/heldout_freeze.py",
    }
    tasks, jobs = {}, {}
    for tid, spec in TASKS.items():
        revision = "controls-003" if tid == "LC-ASYNC-TOOLS-001" else "controls-002"
        state = TRUST / tid
        export = json.loads((state / "export.json").read_text())
        bundle = json.loads((state / "bundle.json").read_text())
        baseline = state / "baseline"
        assert digest(manifest(baseline)) == export["baseline_checksum"]
        assert not list(baseline.rglob(".git"))
        readiness = {}
        for mode in ("wrapper", "baseline", "reference", "reference_budget"):
            # The overflow agent image/tool checks were unchanged by the evaluator
            # async-marker repair, so their exact controls-001 wrapper is reused.
            rev = (
                "controls-001"
                if tid == "LC-SUMMARY-OVERFLOW-001" and mode == "wrapper"
                else revision
            )
            rel = f"reports/{LABEL}/{rev}/{tid}-{mode}.json"
            record = json.loads((ROOT / rel).read_text())
            assert record["status"] == "passed", (tid, mode, record["status"])
            assert record["model_requests"] == 0
            if mode == "baseline":
                assert (
                    record["checks"]["reproducer"]
                    == record["checks"]["hidden_acceptance"]
                    == 1
                )
                assert record["checks"]["existing_tests"] == 0
            elif mode != "wrapper":
                assert record["rewards"]["task_success"] == 1
                assert record["rewards"]["explicit_submission"] == int(
                    mode == "reference"
                )
                assert record["receipt"]["termination_reason"] == (
                    "submitted" if mode == "reference" else "budget_exhausted"
                )
            paths.add(rel)
            readiness[mode] = rel
        env_dir = ROOT / "reports" / LABEL / "environments" / tid
        for name in ("requirements.lock", "environment.json"):
            content = subprocess.check_output(
                [
                    "docker",
                    "run",
                    "--rm",
                    "--network=none",
                    bundle["dependency_id"],
                    "cat",
                    "/opt/" + name,
                ],
                text=True,
            )
            write(env_dir / name, content)
            paths.add(str((env_dir / name).relative_to(ROOT)))
        json.loads((env_dir / "environment.json").read_text())
        task_bundle = Path(bundle["bundle"])
        for p in task_bundle.rglob("*"):
            if p.is_file():
                paths.add(str(p.relative_to(ROOT)))
        common = []
        for model in ("minimax", "glm5"):
            config = f"configs/{LABEL}-{tid}-{model}-model.json"
            job = json.loads((ROOT / config).read_text())
            p = job["agents"][0]["kwargs"]["policy"]
            assert p["budgets"] == parent["policy"]["budgets"]
            assert p["mini_config"] == parent["policy"]["mini_config"]
            assert p["completion_policy"] == parent["policy"]["completion_policy"]
            for k in (
                "code_fingerprints",
                "editing_strategy",
                "repair_semantics",
                "runtime_revision",
            ):
                assert p[k] == parent["policy"][k], k
            ResponseCeilingAgent(
                policy=p, logs_dir=Path("/tmp/heldout-constructor-" + model)
            )
            q = copy.deepcopy(p)
            q.pop("model")
            common.append(q)
            paths.add(config)
            study_id = f"{LABEL}-{tid}-{model}"
            jobs[study_id] = {
                "task_id": tid,
                "model": model,
                "job_config": config,
                "job_name": f"{LABEL}-{tid}-{model}-attempt-001",
            }
        assert common[0] == common[1], (
            "Only model/provider settings may differ within a task"
        )
        tasks[tid] = {
            "issue_url": f"https://github.com/langchain-ai/langchain/issues/{spec['issue']}",
            "family": spec["family"],
            "difficulty": "L2",
            "starting_commit": export["commit"],
            "baseline_checksum": export["baseline_checksum"],
            "task_bundle": str(task_bundle.relative_to(ROOT)),
            "agent_image": bundle["image_ids"]["agent"],
            "verifier_image": bundle["image_ids"]["verifier"],
            "dependency_image": bundle["dependency_id"],
            "readiness": readiness,
            "source_paths": spec["sources"],
            "test_paths": spec["tests"],
            "environment_manifest": str(
                (env_dir / "environment.json").relative_to(ROOT)
            ),
            "setup_seconds_to_freeze": time.time() - export["setup_started_at"],
            "trusted_reference_fixture_sha256": hashlib.sha256(
                (state / "reference-fixture.json").read_bytes()
            ).hexdigest(),
        }
    run_order = [
        f"{LABEL}-LC-ASYNC-TOOLS-001-minimax",
        f"{LABEL}-LC-ASYNC-TOOLS-001-glm5",
        f"{LABEL}-LC-SUMMARY-OVERFLOW-001-glm5",
        f"{LABEL}-LC-SUMMARY-OVERFLOW-001-minimax",
    ]
    setting = {
        "schema_version": 1,
        "experimental_setting": LABEL,
        "split": "held_out",
        "parent_harness_setting_id": parent_id,
        "frozen_at": time.time(),
        "harness": {
            "harbor": "0.24.0",
            "mini-swe-agent": "2.4.6",
            "adapter": "harbor_integration.host_response_ceiling:ResponseCeilingAgent",
        },
        "budgets": parent["policy"]["budgets"],
        "completion_policy": parent["policy"]["completion_policy"],
        "models": parent["models"],
        "tasks": tasks,
        "jobs": jobs,
        "run_order": run_order,
        "authorized_measured_attempts": 4,
        "automatic_retries": 0,
        "setup_model_requests": 0,
        "new_setup_inference_cost_usd": 0,
        "prior_patches_or_hidden_feedback_supplied": False,
        "independent_contexts": True,
        "persistent_agent_memory": False,
        "context_policy": "unchanged full-history mini-SWE; existing bounded tool observation rendering",
        "context_measurement": "native OpenRouter input/cached/output/reasoning/cost on every request; local tokenizer category estimates after execution",
        "heldout_rule": "Do not tune prompts, tools, budgets, checks or environments in response to measured results. Preserve all unsuccessful attempts and costs.",
        "task_environment_scope": "Task-specific dependencies/resources/checks are separately validated and frozen; controller, workflow instructions, provider settings and effective limits inherited unchanged.",
        "compatibility_evidence": "Reuse previously validated model adapters with identical effective request parameters; read-only endpoint metadata preflight occurs before inference.",
        "source_and_config_sha256": {
            p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest()
            for p in sorted(paths)
        },
    }
    setting["setting_id"] = digest(setting)
    write(destination, json.dumps(setting, indent=2) + "\n")
    paths_receipt = "".join(
        f"{sha}  {p}\n"
        for p, sha in sorted(setting["source_and_config_sha256"].items())
    )
    paths_receipt += (
        hashlib.sha256(destination.read_bytes()).hexdigest()
        + "  "
        + str(destination.relative_to(ROOT))
        + "\n"
    )
    write(configuration_path(ROOT, f"configs/{LABEL}.sha256"), paths_receipt)
    catalog = json.loads((configuration_path(ROOT, "configs/study.json")).read_text())
    for study, job in jobs.items():
        catalog["studies"][study] = {
            "task_ids": [job["task_id"]],
            "job_config": job["job_config"],
            "freeze_manifest": f"configs/{LABEL}.sha256",
            "frozen_setting": str(destination.relative_to(ROOT)),
            "split": "held_out",
        }
    write(
        configuration_path(ROOT, "configs/study.json"),
        json.dumps(catalog, indent=2) + "\n",
    )
    assert (
        verify_freeze(
            ROOT, f"configs/{LABEL}.sha256", str(destination.relative_to(ROOT))
        )
        == setting["setting_id"]
    )
    print(
        json.dumps(
            {
                "setting_id": setting["setting_id"],
                "budgets": setting["budgets"],
                "readiness": "all_passed_zero_inference",
                "run_order": run_order,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    freeze()
