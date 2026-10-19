import asyncio

from typer.testing import CliRunner

from crew.budget import BudgetManager
from crew.cli import app
from crew.tracing import load_events, render_timeline


async def test_full_pipeline_offline(demo_run):
    res, tracer = await demo_run()
    md = res.report
    # structure
    assert md.startswith("# ") and "## Sources" in md and "## Verification" in md
    # verifier caught both injected writer faults and removed them from the body
    body, table = md.split("## Verification")
    assert "outperforms long-context models on every benchmark" not in body
    assert "halve operating costs" not in body
    assert sum(r.status == "unsupported" for r in res.rows) == 2
    # every surviving statement is cited
    assert "[1]" in body
    # prompt injection never reaches the report; it was flagged in the trace
    assert "evil.example" not in md and "API key" not in md
    kinds = [e["kind"] for e in tracer.events]
    assert "injection_flagged" in kinds and "repair" in kinds  # planner glitched once, repaired
    # sources deduped: the utm/www mirror of the overview page collapsed into one source
    assert md.count("rag.example.org/overview") == 1


async def test_researchers_run_in_parallel_and_respect_dag(demo_run):
    res, tracer = await demo_run(researchers=4)
    starts = {e["agent"]: e["seq"] for e in tracer.events if e["kind"] == "node_start"}
    dones = {e["agent"]: e["seq"] for e in tracer.events if e["kind"] == "node_done"}
    assert starts["researcher:s1"] < dones["researcher:s2"]  # overlapped
    assert starts["researcher:s2"] < dones["researcher:s1"]
    assert starts["researcher:s4"] > max(dones["researcher:s1"], dones["researcher:s2"])
    assert starts["writer"] > max(v for k, v in dones.items() if k.startswith("researcher"))


async def test_small_budget_degrades_gracefully(demo_run):
    res, tracer = await demo_run(budget=BudgetManager(30_000), researchers=4)
    assert any("researchers reduced 4 -> 2" in d for d in res.degraded)
    assert res.report and res.budget["tokens"] <= 30_000
    assert any(e["kind"] == "budget_degrade" for e in tracer.events)


async def test_trace_jsonl_and_timeline(demo_run, tmp_path):
    _, tracer = await demo_run()
    events = load_events(tmp_path / "t.jsonl")
    assert len(events) == len(tracer.events) and events[0]["kind"] == "run_start"
    out = render_timeline(events)
    assert "researcher:s1" in out and "agent spans" in out and "llm_call" in out


def test_cli_run_then_trace(tmp_path, monkeypatch):
    monkeypatch.setenv("CREW_TRACE_DIR", str(tmp_path / "runs"))
    monkeypatch.chdir(tmp_path)
    r = CliRunner()
    out = tmp_path / "report.md"
    res = r.invoke(
        app,
        [
            "run",
            "How do RAG and long context compare?",
            "--researchers",
            "4",
            "--out",
            str(out),
            "--demo",
        ],
    )
    assert res.exit_code == 0, res.output
    assert out.exists() and "## Verification" in out.read_text()
    tr = r.invoke(app, ["trace"])
    assert tr.exit_code == 0 and "researcher:s1" in tr.output


async def test_sse_endpoint_streams_events_then_report(tmp_path):
    import httpx

    from crew.api import create_app
    from crew.config import Settings

    app_ = create_app(Settings(demo=True, trace_dir=tmp_path))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app_), base_url="http://t") as c:
        r = await asyncio.wait_for(
            c.get("/research/stream", params={"question": "How do RAG and long context compare?"}),
            30,
        )
    assert r.headers["content-type"].startswith("text/event-stream")
    assert "event: trace" in r.text and "event: report" in r.text and "## Verification" in r.text
