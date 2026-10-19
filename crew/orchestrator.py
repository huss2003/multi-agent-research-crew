"""A small DAG orchestrator: bounded concurrency, retries with backoff, timeouts, cancellation."""

from __future__ import annotations

import asyncio
import random
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from crew.tracing import Tracer


class CycleError(ValueError):
    pass


class NonRetryable(Exception):  # noqa: N818 - marker base class
    """Raise from a node to fail it immediately without retries (e.g. budget exhausted)."""


class Status(StrEnum):
    OK = "ok"
    FAILED = "failed"
    SKIPPED = "skipped"
    CANCELLED = "cancelled"


def topo_sort(deps: dict[str, list[str]]) -> list[str]:
    """Return ids in dependency order; raises CycleError / ValueError on bad graphs."""
    order: list[str] = []
    state: dict[str, int] = {}

    def visit(n: str, stack: list[str]) -> None:
        if state.get(n) == 2:
            return
        if state.get(n) == 1:
            raise CycleError(" -> ".join([*stack, n]))
        state[n] = 1
        for d in deps[n]:
            if d not in deps:
                raise ValueError(f"{n!r} depends on unknown node {d!r}")
            visit(d, [*stack, n])
        state[n] = 2
        order.append(n)

    for n in deps:
        visit(n, [])
    return order


@dataclass
class Node:
    id: str
    fn: Callable[[dict[str, Any]], Awaitable[Any]]
    deps: list[str] = field(default_factory=list)
    timeout: float = 60.0
    retries: int = 2
    optional: bool = False  # dependents still run if this node fails


@dataclass
class NodeResult:
    status: Status
    value: Any = None
    error: str | None = None
    attempts: int = 0
    duration: float = 0.0


class Orchestrator:
    def __init__(
        self,
        *,
        max_concurrency: int = 4,
        tracer: Tracer | None = None,
        backoff_base: float = 0.5,
        backoff_max: float = 8.0,
        jitter: float = 0.25,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self.sem = asyncio.Semaphore(max_concurrency)
        self.tracer = tracer or Tracer()
        self.backoff_base, self.backoff_max, self.jitter = backoff_base, backoff_max, jitter
        self._sleep = sleep
        self._tasks: dict[str, asyncio.Task[None]] = {}

    def cancel(self) -> None:
        """Cancel every in-flight and pending node."""
        for t in self._tasks.values():
            t.cancel()

    async def run(self, nodes: list[Node]) -> dict[str, NodeResult]:
        by_id = {n.id: n for n in nodes}
        order = topo_sort({n.id: n.deps for n in nodes})
        results: dict[str, NodeResult] = {}
        # Created in topological order, so a node always finds its dependencies' tasks.
        self._tasks = {
            nid: asyncio.create_task(self._run_node(by_id[nid], by_id, results), name=nid)
            for nid in order
        }
        try:
            await asyncio.gather(*self._tasks.values(), return_exceptions=True)
        finally:  # outer cancellation propagates to every child
            for t in self._tasks.values():
                t.cancel()
        for nid in order:
            results.setdefault(nid, NodeResult(Status.CANCELLED))
        return results

    async def _run_node(
        self, node: Node, by_id: dict[str, Node], results: dict[str, NodeResult]
    ) -> None:
        try:
            values: dict[str, Any] = {}
            blocked: list[str] = []
            for d in node.deps:
                await asyncio.wait([self._tasks[d]])  # never raises the dep's exception
                r = results.get(d)
                if r and r.status is Status.OK:
                    values[d] = r.value
                elif not by_id[d].optional:
                    blocked.append(d)
            if blocked:
                results[node.id] = NodeResult(Status.SKIPPED, error=f"blocked by {blocked}")
                self.tracer.emit("node_skipped", node.id, blocked_by=blocked)
                return
            results[node.id] = await self._attempts(node, values)
        except asyncio.CancelledError:
            results[node.id] = NodeResult(Status.CANCELLED)
            self.tracer.emit("node_cancelled", node.id)
            raise

    async def _attempts(self, node: Node, values: dict[str, Any]) -> NodeResult:
        err: BaseException | None = None
        attempt = 0
        t_first = time.monotonic()
        for attempt in range(1, node.retries + 2):
            try:
                async with self.sem:  # slot is released while backing off
                    self.tracer.emit("node_start", node.id, attempt=attempt)
                    t0 = time.monotonic()
                    async with asyncio.timeout(node.timeout):
                        value = await node.fn(values)
                dur = time.monotonic() - t0
                self.tracer.emit("node_done", node.id, attempt=attempt, duration=round(dur, 3))
                return NodeResult(Status.OK, value, attempts=attempt, duration=dur)
            except NonRetryable as e:
                err = e
                break
            except Exception as e:  # includes TimeoutError from asyncio.timeout
                err = e
                kind = "timeout" if isinstance(e, TimeoutError) else type(e).__name__
                if attempt <= node.retries:
                    delay = min(self.backoff_max, self.backoff_base * 2 ** (attempt - 1))
                    delay *= 1 + self.jitter * random.random()
                    self.tracer.emit(
                        "node_retry",
                        node.id,
                        attempt=attempt,
                        error=f"{kind}: {e}",
                        delay=round(delay, 3),
                    )
                    await self._sleep(delay)
        msg = f"{type(err).__name__}: {err}"
        self.tracer.emit("node_failed", node.id, attempts=attempt, error=msg)
        return NodeResult(
            Status.FAILED, error=msg, attempts=attempt, duration=time.monotonic() - t_first
        )
