"""Planner, researcher and writer agents. Each is a plain async function over a shared Ctx."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field

from crew.budget import BudgetedLLM, BudgetManager
from crew.llm import LLM
from crew.models import Draft, Findings, Plan, Queries
from crew.safety import UNTRUSTED_NOTICE, is_suspicious, sanitize, wrap_untrusted
from crew.store import EvidenceStore, Source, canonical_url
from crew.structured import generate
from crew.tools import FetchTool, SearchTool
from crew.tracing import Tracer

MAX_SOURCE_CHARS = 30_000


@dataclass
class Ctx:
    llm: LLM
    search: SearchTool
    fetch: FetchTool
    store: EvidenceStore
    tracer: Tracer
    budget: BudgetManager
    fetch_per_task: int = 3

    def llm_for(self, agent: str) -> BudgetedLLM:
        return BudgetedLLM(self.llm, self.budget, agent, self.tracer)


@dataclass
class ResearchResult:
    subtask_id: str
    evidence_ids: list[str] = field(default_factory=list)
    claims: list[str] = field(default_factory=list)


# ------------------------------------------------------------------ planner
async def plan(ctx: Ctx, question: str, max_subtasks: int) -> Plan:
    p = await generate(
        ctx.llm_for("planner"),
        Plan,
        role="planner",
        agent="planner",
        tracer=ctx.tracer,
        system="You are a research planner. Decompose the question into independent, "
        "searchable sub-questions. Use depends_on only when a sub-question needs another's findings.",
        prompt=f"QUESTION: {question}\nMAX_SUBTASKS: {max_subtasks}",
        max_tokens=800,
    )
    return p


def prune_plan(p: Plan, keep: int) -> Plan:
    """Keep the `keep` highest-priority subtasks (stable); drop edges to removed ones."""
    ranked = sorted(range(len(p.subtasks)), key=lambda i: (p.subtasks[i].priority, i))[:keep]
    kept = {p.subtasks[i].id for i in ranked}
    subs = [
        s.model_copy(update={"depends_on": [d for d in s.depends_on if d in kept]})
        for s in p.subtasks
        if s.id in kept
    ]
    return Plan(subtasks=subs)


# ------------------------------------------------------------------ researcher
async def research(
    ctx: Ctx, sub_id: str, question: str, deps: dict[str, ResearchResult]
) -> ResearchResult:
    agent = f"researcher:{sub_id}"
    llm = ctx.llm_for(agent)
    known = "\n".join(f"- {c}" for d in deps.values() for c in d.claims[:4]) or "(none)"
    qs = await generate(
        llm,
        Queries,
        role="queries",
        agent=agent,
        tracer=ctx.tracer,
        max_tokens=300,
        system="You write web search queries.",
        prompt=f"SUBTASK: {question}\nALREADY KNOWN:\n{known}\nReturn up to 3 distinct queries.",
    )
    batches = await asyncio.gather(
        *(ctx.search.search(q, 4) for q in qs.queries), return_exceptions=True
    )
    hits = [h for b in batches if not isinstance(b, BaseException) for h in b]
    if not hits and any(isinstance(b, BaseException) for b in batches):
        raise RuntimeError(
            f"all searches failed: {next(b for b in batches if isinstance(b, BaseException))}"
        )
    ctx.tracer.emit("search", agent, queries=len(qs.queries), hits=len(hits))

    urls: dict[str, str] = {}
    for h in hits:  # dedupe by canonical URL, keep rank order
        urls.setdefault(canonical_url(h.url), h.url)
    pages = await asyncio.gather(
        *(ctx.fetch.fetch(u) for u in list(urls.values())[: ctx.fetch_per_task]),
        return_exceptions=True,
    )
    fetched: dict[str, Source] = {}
    for url, page in zip(list(urls.values()), pages, strict=False):
        if isinstance(page, BaseException):
            ctx.tracer.emit("fetch_failed", agent, url=url, error=str(page)[:100])
            continue
        clean, flagged = sanitize(page.text)
        if flagged:
            ctx.tracer.emit(
                "injection_flagged", agent, url=url, removed=len(flagged), sample=flagged[0]
            )
        src = ctx.store.add_source(url, page.title, clean[:MAX_SOURCE_CHARS])
        fetched[canonical_url(url)] = src
    if not fetched:
        raise RuntimeError("no sources could be fetched")

    limit = ctx.budget.context_chars()
    blocks = "\n\n".join(wrap_untrusted(s.url, s.text[:limit]) for s in fetched.values())
    found = await generate(
        llm,
        Findings,
        role="extract",
        agent=agent,
        tracer=ctx.tracer,
        max_tokens=1200,
        system="You extract atomic factual claims, each with a VERBATIM supporting quote copied "
        "exactly from one source. " + UNTRUSTED_NOTICE,
        prompt=f"SUBTASK: {question}\n\n{blocks}\n\nReturn at most 6 findings.",
    )
    res = ResearchResult(sub_id)
    for f in found.findings[:6]:
        src = fetched.get(canonical_url(f.source_url))
        if src is None:
            ctx.tracer.emit("evidence_rejected", agent, reason="unknown_source", url=f.source_url)
        elif is_suspicious(f.claim) or is_suspicious(f.quote):
            ctx.tracer.emit("evidence_rejected", agent, reason="suspicious_text", url=src.url)
        elif (ev := ctx.store.add_evidence(src.id, f.claim, f.quote, sub_id)) is None:
            ctx.tracer.emit("evidence_rejected", agent, reason="quote_not_verbatim", url=src.url)
        else:
            res.evidence_ids.append(ev.id)
            res.claims.append(ev.claim)
            ctx.tracer.emit("evidence_added", agent, id=ev.id, source=src.id)
    return res


# ------------------------------------------------------------------ writer
async def write(ctx: Ctx, question: str, p: Plan) -> Draft:
    topics = {s.id: s.question for s in p.subtasks}
    per_task: dict[str, list] = {}
    for ev in ctx.store.evidence.values():
        per_task.setdefault(ev.subtask_id, []).append(ev)
    cap = 40 if ctx.budget.remaining_frac() > 0.4 else 14  # shorter context when budget is tight
    chosen = []
    for i in range(cap):  # round-robin across subtasks so every topic stays represented
        for evs in per_task.values():
            if i < len(evs) and len(chosen) < cap:
                chosen.append(evs[i])
    payload = [
        {
            "id": e.id,
            "topic": topics.get(e.subtask_id, ""),
            "claim": e.claim,
            "source": ctx.store.sources[e.source_id].title,
        }
        for e in chosen
    ]
    return await generate(
        ctx.llm_for("writer"),
        Draft,
        role="writer",
        agent="writer",
        tracer=ctx.tracer,
        system="You write concise, well-structured research reports using ONLY the supplied "
        "evidence. Every statement must list the evidence_ids that support it. Do not add facts.",
        prompt=f"QUESTION: {question}\nEVIDENCE_JSON: {json.dumps(payload)}\nGroup statements into sections.",
        max_tokens=2500,
    )
