"""Token/cost budget: per-agent accounting, global enforcement, graceful degradation."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from pydantic import BaseModel

from crew.llm import DEFAULT_PRICE, LLM, LLMResult, Usage
from crew.orchestrator import NonRetryable
from crew.tracing import Tracer

EST_RESEARCHER_TOKENS = 8000  # planning figure for one researcher (queries + extraction)


class BudgetExceeded(NonRetryable):
    pass


class BudgetManager:
    """Researchers may only use `researcher_cap` of the budget; the rest is reserved for the
    writer and verifier so a greedy research phase can never starve the final report."""

    def __init__(
        self,
        max_tokens: int,
        max_usd: float | None = None,
        price: tuple[float, float] = DEFAULT_PRICE,
        researcher_cap: float = 0.65,
    ) -> None:
        self.max_tokens, self.max_usd, self.price = max_tokens, max_usd, price
        self.researcher_cap = researcher_cap
        self.usage: dict[str, Usage] = defaultdict(Usage)
        self._inflight = 0  # tokens reserved by calls that have not finished yet

    @property
    def spent_tokens(self) -> int:
        return sum(u.total for u in self.usage.values())

    @property
    def cost_usd(self) -> float:
        pin, pout = self.price
        return sum(u.input_tokens * pin + u.output_tokens * pout for u in self.usage.values()) / 1e6

    def remaining_frac(self) -> float:
        return max(0.0, 1 - self.spent_tokens / self.max_tokens)

    def context_chars(self) -> int:
        """Per-source context window handed to LLM calls; shrinks as the budget drains."""
        f = self.remaining_frac()
        return 6000 if f > 0.5 else 2500 if f > 0.25 else 1000

    def plan_researchers(self, requested: int, est_each: int = EST_RESEARCHER_TOKENS) -> int:
        room = self.max_tokens * self.researcher_cap - self.spent_tokens - self._inflight
        affordable = int(room // est_each)
        if affordable < 1:
            raise BudgetExceeded("budget too small for even one researcher")
        return min(requested, affordable)

    def authorize(self, agent: str, est_tokens: int) -> int:
        group = agent.split(":")[0]
        cap = self.max_tokens * (self.researcher_cap if group == "researcher" else 1.0)
        if self.spent_tokens + self._inflight + est_tokens > cap:
            raise BudgetExceeded(
                f"{agent}: token budget exhausted ({self.spent_tokens}/{int(cap)} used)"
            )
        if self.max_usd is not None:
            worst = self.cost_usd + (self._inflight + est_tokens) * self.price[1] / 1e6
            if worst > self.max_usd:
                raise BudgetExceeded(f"{agent}: USD budget exhausted (${self.cost_usd:.4f})")
        self._inflight += est_tokens
        return est_tokens

    def settle(self, agent: str, reserved: int, usage: Usage) -> None:
        self._inflight -= reserved
        self.usage[agent] += usage

    def snapshot(self) -> dict[str, Any]:
        return {
            "tokens": self.spent_tokens,
            "max_tokens": self.max_tokens,
            "cost_usd": round(self.cost_usd, 4),
            "by_agent": {a: u.total for a, u in sorted(self.usage.items())},
        }


class BudgetedLLM:
    """Wraps an LLM for one agent: pre-authorizes worst-case tokens, then records actual usage."""

    def __init__(self, inner: LLM, budget: BudgetManager, agent: str, tracer: Tracer | None = None):
        self.inner, self.budget, self.agent, self.tracer = inner, budget, agent, tracer
        self.model = inner.model

    async def complete(
        self,
        *,
        role: str,
        system: str,
        prompt: str,
        schema: type[BaseModel] | None = None,
        max_tokens: int = 1500,
    ) -> LLMResult:
        reserved = self.budget.authorize(self.agent, (len(system) + len(prompt)) // 4 + max_tokens)
        usage = Usage()
        try:
            res = await self.inner.complete(
                role=role, system=system, prompt=prompt, schema=schema, max_tokens=max_tokens
            )
            usage = res.usage
        finally:
            self.budget.settle(self.agent, reserved, usage)
        if self.tracer:
            self.tracer.emit(
                "llm_call",
                self.agent,
                role=role,
                tokens_in=usage.input_tokens,
                tokens_out=usage.output_tokens,
                spent=self.budget.spent_tokens,
            )
        return res
