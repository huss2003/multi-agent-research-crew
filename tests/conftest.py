import pytest

from crew.budget import BudgetManager
from crew.config import Settings, build_runtime
from crew.pipeline import run_crew
from crew.tracing import Tracer


@pytest.fixture
def demo_run(tmp_path):
    """Factory: run the full offline pipeline and return (result, tracer)."""

    async def go(question="How do RAG and long-context models compare?", **kw):
        llm, search, fetch = build_runtime(Settings(demo=True))
        budget = kw.pop("budget", None) or BudgetManager(200_000)
        tracer = Tracer(tmp_path / "t.jsonl", "test")
        res = await run_crew(
            question, llm=llm, search=search, fetch=fetch, budget=budget, tracer=tracer, **kw
        )
        return res, tracer

    return go
