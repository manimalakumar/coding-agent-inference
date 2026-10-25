"""Model-independent verdict contract for final-workspace delivery."""


def task_success(checks, *, required, changed, new_tests):
    if not changed or not new_tests or not set(required).issubset(checks):
        return False
    return all(
        checks[name]["exit_code"] == (1 if name == "regression_on_baseline" else 0)
        for name in required
    )


def verdict(checks, *, required, changed, new_tests, explicit_submission):
    accepted = task_success(
        checks, required=required, changed=changed, new_tests=new_tests
    )
    return {
        "reward": int(accepted),
        "patch_acceptance": int(accepted),
        "patch_accepted": int(accepted),
        "task_success": int(accepted),
        "agent_submission": int(explicit_submission),
        "explicit_submission": int(explicit_submission),
    }
