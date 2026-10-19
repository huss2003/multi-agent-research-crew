"""End-to-end run: plan -> parallel research (DAG) -> write -> verify -> render."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from crew.agents import Ctx, ResearchResult, plan, prune_plan, research, write
from crew.budget import BudgetManager
from crew.llm import LLM
from crew.models import Draft, Plan, VerificationRow
from crew.orchestrator import Node, NonRetryable, Orchestrator, Status
from crew.report import render_report
from crew.store import EvidenceStore
from crew.tools import FetchTool, SearchTool
from crew.tracing import Tracer
from crew.verifier import verify


class CrewError(NonRetryable):
    pass


@dataclass
class RunResult:
    run_id: str
    report: str
    rows: list[VerificationRow]
    budget: dict[str, Any]
    trace_path: Path | None
    degraded: list[str] = field(default_factory=list)


async def run_crew(
    question: str,
    *,
    llm: LLM,
    search: SearchTool,
    fetch: FetchTool,
    budget: BudgetManager,
    researchers: int = 4,
    tracer: Tracer | None = None,
    policy: str = "remove",
    timeouts: tuple[float, float, float] = (90, 120, 120),  # researcher, writer, verifier
    orchestrator: Orchestrator | None = None,
) -> RunResult:
    tracer = tracer or Tracer(run_id=uuid.uuid4().hex[:8])
    store = EvidenceStore()
    ctx = Ctx(llm, search, fetch, store, tracer, budget)
    degraded: list[str] = []
    tracer.emit(
        "run_start", question=question, researchers=researchers, max_tokens=budget.max_tokens
    )

    # Graceful degradation: fewer researchers if the budget cannot fund the requested number.
    n = budget.plan_researchers(researchers)
    if n < researchers:
        degraded.append(f"researchers reduced {researchers} -> {n} (budget)")
        tracer.emit("budget_degrade", action="fewer_researchers", requested=researchers, granted=n)
    orch = orchestrator or Orchestrator(max_concurrency=n, tracer=tracer)

    async def plan_fn(_: dict) -> Plan:
        return await plan(ctx, question, n)

    res = await orch.run([Node("planner", plan_fn, timeout=timeouts[0], retries=2)])
    if res["planner"].status is not Status.OK:
        raise CrewError(f"planning failed: {res['planner'].error}")
    full_plan = prune_plan(res["planner"].value, n)
    if len(full_plan.subtasks) < len(res["planner"].value.subtasks):
        degraded.append(f"subtasks trimmed to {len(full_plan.subtasks)} (budget)")

    def researcher_node(s: Any) -> Node:
        async def fn(deps: dict[str, ResearchResult]) -> ResearchResult:
            return await research(
                ctx, s.id, s.question, {k.split(":")[1]: v for k, v in deps.items()}
            )

        return Node(
            f"researcher:{s.id}",
            fn,
            deps=[f"researcher:{d}" for d in s.depends_on],
            timeout=timeouts[0],
            retries=2,
            optional=True,
        )

    nodes = [researcher_node(s) for s in full_plan.subtasks]

    async def write_fn(_: dict) -> Draft:
        if not store.evidence:
            raise CrewError("no evidence was collected")
        return await write(ctx, question, full_plan)

    async def verify_fn(deps: dict) -> list[VerificationRow]:
        return await verify(deps["writer"], store, ctx.llm_for("verifier"), tracer)

    nodes += [
        Node("writer", write_fn, deps=[x.id for x in nodes], timeout=timeouts[1], retries=1),
        Node("verifier", verify_fn, deps=["writer"], timeout=timeouts[2], retries=1),
    ]
    results = await orch.run(nodes)
    for nid, r in results.items():
        if nid.startswith("researcher:") and r.status is not Status.OK:
            degraded.append(f"{nid} {r.status.value}: {r.error}")
    for stage in ("writer", "verifier"):
        if results[stage].status is not Status.OK:
            raise CrewError(f"{stage} {results[stage].status.value}: {results[stage].error}")

    draft, rows = results["writer"].value, results["verifier"].value
    snap = budget.snapshot()
    summary = (
        f"run {tracer.run_id} | {len(store.sources)} sources, {len(store.evidence)} quotes | "
        f"{snap['tokens']} tokens, ${snap['cost_usd']}"
        + (f" | degraded: {'; '.join(degraded)}" if degraded else "")
    )
    md = render_report(question, draft, rows, store, {"summary": summary}, policy)
    tracer.emit(
        "run_done", tokens=snap["tokens"], cost_usd=snap["cost_usd"], degraded=len(degraded)
    )
    return RunResult(tracer.run_id, md, rows, snap, tracer.path, degraded)
