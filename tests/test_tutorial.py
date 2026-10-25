"""Reporting safeguards: no invented results, cross-setting pooling or free unknowns."""

import pytest

from experiments.tutorial import (
    compact_request,
    development_slots,
    economics,
    exact_run,
    operational_rows,
    redacted,
    selected_records,
    usage_requests,
)


def record(**kwargs):
    return {
        "kind": "model_attempt",
        "run_id": "job/trial",
        "model": "m",
        "task_id": "t",
        "declared_frozen_setting_id": "freeze",
        "known_cost_usd": 0.2,
        "cost_usd": 0.2,
        "task_success": False,
        "patch_accepted": False,
        "explicit_submission": False,
        **kwargs,
    }


def test_missing_and_duplicate_explicit_ids():
    assert exact_run([], "job/trial") is None
    with pytest.raises(ValueError):
        exact_run([record(), record()], "job/trial")


def test_unrun_slot_does_not_become_a_failed_record():
    plan = {"schedule": [{"job_name": "job", "task_id": "t", "model_id": "m"}]}
    row = development_slots(plan, [], "freeze")[0]
    assert row["status"] == "not run" and row["record"] is None
    assert economics([])["status"] == "not run"
    assert economics([])["total_spend_usd"] is None


def test_wrong_freeze_does_not_enter_comparison():
    plan = {"schedule": [{"job_name": "job", "task_id": "t", "model_id": "m"}]}
    row = development_slots(
        plan, [record(declared_frozen_setting_id="other")], "freeze"
    )[0]
    assert row["status"] == "configuration mismatch" and row["record"] is None


def test_failure_spend_included_and_submission_is_separate():
    e = economics(
        [record(), record(cost_usd=0.3, known_cost_usd=0.3, patch_accepted=True)]
    )
    assert e["total_spend_usd"] == 0.5 and e["cost_per_accepted_patch_usd"] == 0.5
    assert e["successful_attempts"] == 0 and e["explicit_completion_rate"] == 0
    assert e["cost_per_successful_task_usd"] is None


def test_unknown_charge_prevents_full_cost_ratio():
    e = economics(
        [record(patch_accepted=True, task_success=True), record(cost_usd=None)]
    )
    assert e["known_spend_usd"] == 0.4
    assert e["total_spend_usd"] is None and e["cost_per_accepted_patch_usd"] is None


def test_unknown_acceptance_not_counted_as_rejection():
    e = economics([record(patch_accepted=None)])
    assert e["accepted_patches"] is None and e["patch_status"] == "unknown acceptance"


def test_explicit_zero_usage_differs_from_missing_and_no_double_counting():
    r = [
        {
            "response": {
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 20,
                    "prompt_tokens_details": {"cached_tokens": 0},
                    "completion_tokens_details": {"reasoning_tokens": 4},
                }
            }
        }
    ]
    e = usage_requests(r)
    assert e["cached_input_subset"] == 0 and e["cache_write_input"] is None
    assert e["output_including_reasoning"] == 20 and e["reasoning_subset"] == 4
    incomplete = usage_requests(r, expected=2)
    assert incomplete["input"] is None and incomplete["input_known"] == 100


def test_nested_credentials_redacted_and_request_preview_keeps_controls():
    d = {
        "headers": {"Authorization": "secret"},
        "api_key": "secret",
        "model": "m",
        "messages": [{"role": "user", "content": "task"}],
        "max_tokens": 1500,
    }
    assert "secret" not in str(redacted(d))
    assert compact_request(d)["max_tokens"] == 1500
    assert compact_request(None) == {"status": "not run"}


def test_view_filters_models_without_mutating_archives():
    rows = [record(), record(model="other"), record(kind="local_control")]
    assert selected_records(rows, ["m"]) == rows[:1]
    assert len(rows) == 3


def test_operational_bucket_retains_other_archived_expenses(tmp_path):
    rows = operational_rows(
        tmp_path, [record(), record(model="other", cost_usd=None)], ["m"]
    )
    assert rows[-1]["known_spend_usd"] == 0.4
    assert rows[-1]["total_spend_usd"] is None
    assert rows[1]["entries"] == 1
