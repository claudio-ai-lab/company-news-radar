"""Token and cost accounting with a daily budget guard."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Protocol

from newsradar.config import Price

logger = logging.getLogger(__name__)


class BudgetExceededError(Exception):
    """Raised before an LLM call once the daily budget has been spent."""


class CostLedger(Protocol):
    def record_llm_call(
        self,
        run_id: str,
        stage: str,
        provider: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cost_usd: float,
        ok: bool,
        error: str | None,
    ) -> None: ...

    def llm_cost_since(self, since: datetime) -> float: ...


class CostTracker:
    """Prices each call, persists it to the ledger (if any) and enforces the budget.

    The budget is a soft limit: calls already in flight when it is reached still finish.
    """

    def __init__(
        self,
        pricing: dict[str, Price],
        daily_budget_usd: float,
        run_id: str,
        ledger: CostLedger | None = None,
    ) -> None:
        self.pricing = pricing
        self.daily_budget_usd = daily_budget_usd
        self.run_id = run_id
        self.ledger = ledger
        self.run_cost_usd = 0.0
        self.run_calls = 0
        self.run_failed_calls = 0
        self.run_input_tokens = 0
        self.run_output_tokens = 0
        self._unpriced: set[str] = set()
        self._spent_before_run = (
            ledger.llm_cost_since(_start_of_today()) if ledger is not None else 0.0
        )

    def price(self, model: str, input_tokens: int, output_tokens: int) -> float:
        price = self.pricing.get(model)
        if price is None:
            if model not in self._unpriced and (input_tokens or output_tokens):
                self._unpriced.add(model)
                logger.warning("no pricing configured for model %r; counting cost as 0", model)
            return 0.0
        return (
            input_tokens * price.input_per_mtok + output_tokens * price.output_per_mtok
        ) / 1_000_000

    def record(
        self,
        stage: str,
        provider: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        *,
        ok: bool,
        error: str | None = None,
    ) -> float:
        cost = self.price(model, input_tokens, output_tokens)
        self.run_cost_usd += cost
        self.run_calls += 1
        self.run_failed_calls += 0 if ok else 1
        self.run_input_tokens += input_tokens
        self.run_output_tokens += output_tokens
        if self.ledger is not None:
            self.ledger.record_llm_call(
                self.run_id,
                stage,
                provider,
                model,
                input_tokens,
                output_tokens,
                cost,
                ok,
                (error or "")[:500] or None,
            )
        return cost

    @property
    def spent_today_usd(self) -> float:
        return self._spent_before_run + self.run_cost_usd

    def ensure_budget(self) -> None:
        if self.spent_today_usd >= self.daily_budget_usd:
            raise BudgetExceededError(
                f"daily LLM budget of ${self.daily_budget_usd:.2f} reached "
                f"(spent ${self.spent_today_usd:.4f})"
            )


def _start_of_today() -> datetime:
    now = datetime.now(UTC)
    return now.replace(hour=0, minute=0, second=0, microsecond=0)
