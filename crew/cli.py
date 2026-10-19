"""`crew run | trace | serve`."""

from __future__ import annotations

import asyncio
import uuid
from pathlib import Path
from typing import Annotated

import typer

from crew.budget import BudgetExceeded, BudgetManager
from crew.config import Settings, build_runtime, load_dotenv
from crew.pipeline import CrewError, run_crew
from crew.tracing import Tracer, load_events, render_timeline

app = typer.Typer(no_args_is_help=True, help="Multi-agent research crew.")
LIVE = {
    "run_start",
    "node_start",
    "node_retry",
    "node_failed",
    "node_skipped",
    "repair",
    "budget_degrade",
    "injection_flagged",
    "evidence_rejected",
    "verification",
}


def _progress(ev: dict) -> None:
    if ev["kind"] in LIVE:
        detail = " ".join(
            f"{k}={v}"
            for k, v in ev.items()
            if k not in ("seq", "ts", "run", "kind", "agent", "question", "sample")
        )
        typer.echo(f"  [{ev['kind']}] {ev['agent']} {detail}"[:150], err=True)


@app.command()
def run(
    question: Annotated[str, typer.Argument(help="Research question")],
    researchers: Annotated[int, typer.Option(min=1, max=8, help="Max parallel researchers")] = 4,
    out: Annotated[Path, typer.Option(help="Where to write the markdown report")] = Path(
        "report.md"
    ),
    demo: Annotated[
        bool, typer.Option("--demo", envvar="DEMO_MODE", help="Offline mock LLM + fixtures")
    ] = False,
    model: Annotated[
        str | None, typer.Option(envvar="CREW_MODEL", help="Anthropic model id")
    ] = None,
    budget_tokens: Annotated[int | None, typer.Option(help="Global token budget")] = None,
    budget_usd: Annotated[float | None, typer.Option(help="Global USD budget")] = None,
    unverified: Annotated[
        str, typer.Option(help="remove | flag unsupported statements")
    ] = "remove",
) -> None:
    load_dotenv()
    s = Settings.from_env()
    s = Settings(**{**s.__dict__, "demo": demo or s.demo, "model": model or s.model})
    llm, search, fetch = build_runtime(s)
    budget = BudgetManager(budget_tokens or s.budget_tokens, budget_usd or s.budget_usd, s.price)
    run_id = uuid.uuid4().hex[:8]
    tracer = Tracer(s.trace_dir / f"{run_id}.jsonl", run_id)
    tracer.subscribe(_progress)
    try:
        res = asyncio.run(
            run_crew(
                question,
                llm=llm,
                search=search,
                fetch=fetch,
                budget=budget,
                researchers=researchers,
                tracer=tracer,
                policy=unverified,
            )
        )
    except (CrewError, BudgetExceeded) as e:
        typer.echo(f"run failed: {e}", err=True)
        raise typer.Exit(1) from e
    out.write_text(res.report, encoding="utf-8")
    ok = sum(r.status == "supported" for r in res.rows)
    typer.echo(
        f"report: {out} | verified {ok}/{len(res.rows)} | tokens {res.budget['tokens']} "
        f"| ${res.budget['cost_usd']} | trace: {res.trace_path}"
    )
    for d in res.degraded:
        typer.echo(f"  degraded: {d}", err=True)


@app.command()
def trace(
    run: Annotated[str | None, typer.Argument(help="Trace file or run id; default latest")] = None,
) -> None:
    """Print a timeline of a run from its JSONL trace."""
    load_dotenv()
    d = Settings.from_env().trace_dir
    if run and Path(run).is_file():
        path = Path(run)
    elif run:
        path = d / f"{run}.jsonl"
    else:
        files = sorted(d.glob("*.jsonl"), key=lambda p: p.stat().st_mtime)
        if not files:
            typer.echo(f"no traces in {d}", err=True)
            raise typer.Exit(1)
        path = files[-1]
    if not path.exists():
        typer.echo(f"trace not found: {path}", err=True)
        raise typer.Exit(1)
    typer.echo(render_timeline(load_events(path)))


@app.command()
def serve(host: str = "127.0.0.1", port: int = 8000) -> None:
    """Serve the SSE API (needs the `api` extra)."""
    try:
        import uvicorn

        from crew.api import create_app
    except ImportError as e:  # pragma: no cover
        typer.echo("install the api extra: pip install 'multi-agent-research-crew[api]'", err=True)
        raise typer.Exit(1) from e
    load_dotenv()
    uvicorn.run(create_app(), host=host, port=port)
