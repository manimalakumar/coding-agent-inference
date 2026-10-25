"""Shared controller-enforced budgets, with per-role accounting."""

import time
from dataclasses import dataclass, field

from agent.schemas import Budgets


class BudgetExhausted(RuntimeError):
    pass


@dataclass
class Budget:
    limits: Budgets
    started: float = field(default_factory=time.monotonic)
    calls: int = 0
    generated: int = 0
    repairs: int = 0
    submission_only: bool = False
    roles: dict = field(default_factory=dict)

    def remaining_seconds(self) -> float:
        return max(0.0, self.limits.seconds - (time.monotonic() - self.started))

    def check_time(self) -> None:
        if self.remaining_seconds() <= 0:
            raise BudgetExhausted("wall_clock_budget_exhausted")

    def reserve(self, role: str) -> int:
        self.check_time()
        if self.calls >= self.limits.model_calls:
            raise BudgetExhausted("model_call_budget_exhausted")
        remaining = self.limits.generated_tokens - self.generated
        if remaining <= 0:
            raise BudgetExhausted("generated_token_budget_exhausted")
        self.calls += 1  # Failed and interrupted requests still count.
        counters = self.roles.setdefault(
            role, {"model_calls": 0, "generated_tokens": 0}
        )
        counters["model_calls"] += 1
        return min(remaining, self.limits.response_tokens)

    def account(self, role: str, usage: dict, reservation: int) -> None:
        total = usage.get("completion_tokens")
        if not isinstance(total, int) or total < 0:
            # Conservatively consume the reservation; do not report it as measured usage.
            self.generated += reservation
            raise ValueError(
                "Provider omitted completion usage; attempt cannot continue"
            )
        self.generated += total
        self.roles[role]["generated_tokens"] += total
        if total > reservation:
            raise ValueError("Provider exceeded the requested generation cap")

    def repair(self) -> None:
        self.check_time()
        if self.repairs >= self.limits.repair_rounds:
            raise BudgetExhausted("repair_round_budget_exhausted")
        self.repairs += 1
