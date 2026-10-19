import asyncio

import pytest

from crew.orchestrator import CycleError, Node, NonRetryable, Orchestrator, Status, topo_sort


async def test_concurrency_is_bounded_by_semaphore():
    running = peak = 0

    async def work(_):
        nonlocal running, peak
        running += 1
        peak = max(peak, running)
        await asyncio.sleep(0.02)
        running -= 1

    orch = Orchestrator(max_concurrency=3)
    res = await orch.run([Node(f"n{i}", work) for i in range(9)])
    assert all(r.status is Status.OK for r in res.values())
    assert peak == 3


async def test_dag_order_and_dependency_values():
    order = []

    def mk(name, val):
        async def fn(deps):
            order.append(name)
            return {"val": val, "saw": sorted(deps)}

        return fn

    nodes = [
        Node("c", mk("c", 3), deps=["a", "b"]),
        Node("a", mk("a", 1)),
        Node("b", mk("b", 2), deps=["a"]),
    ]
    res = await Orchestrator().run(nodes)
    assert order == ["a", "b", "c"]
    assert res["c"].value["saw"] == ["a", "b"]


async def test_retry_with_exponential_backoff():
    delays, calls = [], 0

    async def fake_sleep(d):
        delays.append(d)

    async def flaky(_):
        nonlocal calls
        calls += 1
        if calls < 4:
            raise RuntimeError("boom")
        return "ok"

    orch = Orchestrator(backoff_base=0.1, jitter=0, sleep=fake_sleep)
    res = await orch.run([Node("n", flaky, retries=3)])
    assert res["n"].status is Status.OK and res["n"].attempts == 4
    assert delays == pytest.approx([0.1, 0.2, 0.4])


async def test_retries_exhausted_fails_and_skips_dependents():
    async def bad(_):
        raise ValueError("nope")

    async def after(_):
        return 1

    orch = Orchestrator(backoff_base=0, jitter=0)
    res = await orch.run([Node("bad", bad, retries=1), Node("after", after, deps=["bad"])])
    assert res["bad"].status is Status.FAILED and res["bad"].attempts == 2
    assert "ValueError" in res["bad"].error
    assert res["after"].status is Status.SKIPPED


async def test_optional_dependency_failure_does_not_block():
    async def bad(_):
        raise ValueError("x")

    async def after(deps):
        return sorted(deps)

    orch = Orchestrator(backoff_base=0, jitter=0)
    res = await orch.run(
        [Node("bad", bad, retries=0, optional=True), Node("after", after, deps=["bad"])]
    )
    assert res["after"].status is Status.OK and res["after"].value == []


async def test_per_node_timeout_then_retry():
    attempts = 0

    async def slow_then_fast(_):
        nonlocal attempts
        attempts += 1
        await asyncio.sleep(1 if attempts == 1 else 0)
        return "done"

    orch = Orchestrator(backoff_base=0, jitter=0)
    res = await orch.run([Node("n", slow_then_fast, timeout=0.05, retries=1)])
    assert res["n"].status is Status.OK and res["n"].attempts == 2


async def test_timeout_failure_is_reported():
    async def hang(_):
        await asyncio.sleep(5)

    orch = Orchestrator(backoff_base=0, jitter=0)
    res = await orch.run([Node("n", hang, timeout=0.05, retries=0)])
    assert res["n"].status is Status.FAILED and "TimeoutError" in res["n"].error


async def test_non_retryable_skips_retries():
    calls = 0

    async def fn(_):
        nonlocal calls
        calls += 1
        raise NonRetryable("stop")

    res = await Orchestrator().run([Node("n", fn, retries=5)])
    assert calls == 1 and res["n"].status is Status.FAILED


async def test_cancel_stops_running_and_pending_nodes():
    started = []

    async def long(_):
        started.append(1)
        await asyncio.sleep(10)

    async def never(_):
        started.append("never")

    orch = Orchestrator(max_concurrency=2)
    runner = asyncio.create_task(
        orch.run([Node("a", long), Node("b", long), Node("c", never, deps=["a"])])
    )
    await asyncio.sleep(0.05)
    orch.cancel()
    res = await asyncio.wait_for(runner, 1)
    assert {r.status for r in res.values()} == {Status.CANCELLED}
    assert started == [1, 1]


async def test_outer_task_cancellation_propagates_to_nodes():
    cancelled = asyncio.Event()

    async def long(_):
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            cancelled.set()
            raise

    runner = asyncio.create_task(Orchestrator().run([Node("a", long)]))
    await asyncio.sleep(0.05)
    runner.cancel()
    with pytest.raises(asyncio.CancelledError):
        await runner
    assert cancelled.is_set()


def test_cycle_and_unknown_dependency_detected():
    with pytest.raises(CycleError):
        topo_sort({"a": ["b"], "b": ["a"]})
    with pytest.raises(ValueError, match="unknown"):
        topo_sort({"a": ["zzz"]})
