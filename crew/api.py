"""Optional FastAPI app: GET /research/stream streams trace events as Server-Sent Events."""

from __future__ import annotations

import asyncio
import json
import uuid
from collections.abc import AsyncIterator

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import StreamingResponse

from crew.budget import BudgetManager
from crew.config import Settings, build_runtime
from crew.pipeline import run_crew
from crew.tracing import Tracer


def _sse(event: str, data: object) -> str:
    return f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n"


def create_app(settings: Settings | None = None) -> FastAPI:
    app = FastAPI(title="Multi-Agent Research Crew")

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/research/stream")
    async def stream(
        question: str = Query(min_length=8, max_length=500),
        researchers: int = Query(4, ge=1, le=8),
    ) -> StreamingResponse:
        s = settings or Settings.from_env()
        try:
            llm, search, fetch = build_runtime(s)
        except SystemExit as e:  # config error must not kill the server process
            raise HTTPException(503, str(e.code)) from None
        run_id = uuid.uuid4().hex[:8]
        queue: asyncio.Queue[dict | None] = asyncio.Queue()
        tracer = Tracer(s.trace_dir / f"{run_id}.jsonl", run_id)
        tracer.subscribe(queue.put_nowait)
        budget = BudgetManager(s.budget_tokens, s.budget_usd, s.price)
        task = asyncio.create_task(
            run_crew(
                question,
                llm=llm,
                search=search,
                fetch=fetch,
                budget=budget,
                researchers=researchers,
                tracer=tracer,
            )
        )
        task.add_done_callback(lambda _: queue.put_nowait(None))

        async def events() -> AsyncIterator[str]:
            try:
                while (ev := await queue.get()) is not None:
                    yield _sse("trace", ev)
                try:
                    res = task.result()
                    yield _sse(
                        "report",
                        {
                            "run_id": res.run_id,
                            "markdown": res.report,
                            "budget": res.budget,
                            "degraded": res.degraded,
                        },
                    )
                except Exception as e:
                    yield _sse("error", {"message": str(e)})
            finally:  # client disconnected -> cancel the whole pipeline
                if not task.done():
                    task.cancel()

        return StreamingResponse(
            events(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"}
        )

    return app
