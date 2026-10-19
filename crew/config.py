"""Environment-driven settings and runtime wiring."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from crew.llm import DEFAULT_PRICE, LLM, PRICES_PER_MTOK
from crew.tools import (
    BraveSearch,
    FetchTool,
    HttpFetch,
    SearchTool,
    SerpApiSearch,
    TavilySearch,
)


def load_dotenv(path: Path = Path(".env")) -> None:
    """Minimal .env reader; real environment variables win."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            k, v = line.split("=", 1)
            if v.strip():
                os.environ.setdefault(k.strip(), v.strip().strip("'\""))


@dataclass(frozen=True)
class Settings:
    demo: bool = False
    model: str = "claude-sonnet-5-5"
    search_provider: str = "tavily"
    budget_tokens: int = 200_000
    budget_usd: float | None = None
    trace_dir: Path = Path(".crew/runs")

    @classmethod
    def from_env(cls) -> Settings:
        e = os.environ
        return cls(
            demo=e.get("DEMO_MODE", "").lower() in ("1", "true", "yes"),
            model=e.get("CREW_MODEL", cls.model),
            search_provider=e.get("CREW_SEARCH_PROVIDER", cls.search_provider).lower(),
            budget_tokens=int(e.get("CREW_BUDGET_TOKENS") or cls.budget_tokens),
            budget_usd=float(e["CREW_BUDGET_USD"]) if e.get("CREW_BUDGET_USD") else None,
            trace_dir=Path(e.get("CREW_TRACE_DIR", ".crew/runs")),
        )

    @property
    def price(self) -> tuple[float, float]:
        return PRICES_PER_MTOK.get("mock-1" if self.demo else self.model, DEFAULT_PRICE)


def build_runtime(s: Settings) -> tuple[LLM, SearchTool, FetchTool]:
    if s.demo:
        from crew.demo import MockLLM
        from crew.tools import FixtureCorpus, FixtureFetch, FixtureSearch

        corpus = FixtureCorpus()
        return MockLLM(), FixtureSearch(corpus), FixtureFetch(corpus)
    from crew.llm import AnthropicLLM

    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("ANTHROPIC_API_KEY is not set (or use --demo / DEMO_MODE=1).")
    providers = {
        "tavily": (TavilySearch, "TAVILY_API_KEY"),
        "brave": (BraveSearch, "BRAVE_API_KEY"),
        "serpapi": (SerpApiSearch, "SERPAPI_API_KEY"),
    }
    if s.search_provider not in providers:
        raise SystemExit(f"unknown CREW_SEARCH_PROVIDER {s.search_provider!r}")
    cls, key_env = providers[s.search_provider]
    if not os.environ.get(key_env):
        raise SystemExit(f"{key_env} is not set for provider {s.search_provider!r}.")
    return AnthropicLLM(s.model), cls(os.environ[key_env]), HttpFetch()
