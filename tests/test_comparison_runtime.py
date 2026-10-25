"""Clock changes must not grant extra agent work or inference."""

import pytest

from agent.budget import BudgetExhausted
from agent.schemas import Budgets
from harbor_integration.host_comparison import DeadlineBudget


def test_forward_civil_jump_expires_even_if_monotonic_is_unchanged(monkeypatch):
    monkeypatch.setattr("time.monotonic", lambda: 100.0)
    monkeypatch.setattr("time.time", lambda: 1000.0)
    budget = DeadlineBudget(Budgets(seconds=20))
    monkeypatch.setattr("time.time", lambda: 1030.0)
    assert budget.remaining_seconds() == 0
    with pytest.raises(BudgetExhausted):
        budget.reserve("agent")
    with pytest.raises(BudgetExhausted):
        budget.repair()


def test_backward_civil_jump_cannot_extend_monotonic_budget(monkeypatch):
    monkeypatch.setattr("time.monotonic", lambda: 100.0)
    monkeypatch.setattr("time.time", lambda: 1000.0)
    budget = DeadlineBudget(Budgets(seconds=20))
    monkeypatch.setattr("time.time", lambda: 500.0)
    monkeypatch.setattr("time.monotonic", lambda: 125.0)
    assert budget.remaining_seconds() == 0


def test_expired_budget_stays_expired(monkeypatch):
    budget = DeadlineBudget(Budgets(seconds=20))
    budget.expire()
    monkeypatch.setattr("time.time", lambda: 0)
    assert budget.remaining_seconds() == 0
