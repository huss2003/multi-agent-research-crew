import pytest

from crew.budget import BudgetedLLM, BudgetExceeded, BudgetManager
from crew.llm import LLMResult, Usage


class Fixed:
    model = "fixed"

    def __init__(self, used=100):
        self.used = used

    async def complete(self, **kw):
        return LLMResult("{}", Usage(self.used, 0))


async def test_usage_tracked_per_agent_and_cost():
    b = BudgetManager(10_000, price=(3.0, 15.0))
    await BudgetedLLM(Fixed(200), b, "planner").complete(role="r", system="", prompt="x")
    await BudgetedLLM(Fixed(300), b, "writer").complete(role="r", system="", prompt="x")
    assert b.snapshot()["by_agent"] == {"planner": 200, "writer": 300}
    assert b.cost_usd == pytest.approx(500 * 3 / 1e6)


async def test_global_budget_enforced_before_call():
    b = BudgetManager(1_000)
    llm = BudgetedLLM(Fixed(10), b, "writer")
    with pytest.raises(BudgetExceeded):  # worst case (max_tokens) does not fit
        await llm.complete(role="r", system="", prompt="x", max_tokens=5_000)
    assert b.spent_tokens == 0 and b._inflight == 0  # reservation released


async def test_budget_exhausts_across_calls():
    b = BudgetManager(1_000)
    llm = BudgetedLLM(Fixed(400), b, "writer")
    await llm.complete(role="r", system="", prompt="x", max_tokens=500)
    await llm.complete(role="r", system="", prompt="x", max_tokens=500)
    with pytest.raises(BudgetExceeded):
        await llm.complete(role="r", system="", prompt="x", max_tokens=500)


def test_usd_budget_enforced():
    b = BudgetManager(1_000_000, max_usd=0.001, price=(3.0, 15.0))
    with pytest.raises(BudgetExceeded, match="USD"):
        b.authorize("writer", 1_000)  # 1000 tokens * $15/M = $0.015 > $0.001


def test_researchers_cannot_consume_writer_reserve():
    b = BudgetManager(10_000, researcher_cap=0.6)
    b.authorize("researcher:s1", 5_000)
    with pytest.raises(BudgetExceeded):
        b.authorize("researcher:s2", 2_000)  # would cross the 60% researcher cap
    b.authorize("writer", 2_000)  # the reserved slice is still available


def test_degrades_to_fewer_researchers():
    b = BudgetManager(30_000)  # 65% cap = 19.5k -> 2 researchers at 8k each
    assert b.plan_researchers(4) == 2
    assert BudgetManager(1_000_000).plan_researchers(4) == 4
    with pytest.raises(BudgetExceeded):
        BudgetManager(5_000).plan_researchers(4)


def test_context_shrinks_as_budget_drains():
    b = BudgetManager(1_000)
    sizes = []
    for used in (0, 600, 800):
        b.usage["x"] = Usage(used, 0)
        sizes.append(b.context_chars())
    assert sizes == [6000, 2500, 1000]
