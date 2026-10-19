"""JSONL event log + timeline renderer."""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

CORE = {"seq", "ts", "run", "kind", "agent"}
TERMINAL = {"node_done", "node_failed", "node_skipped", "node_cancelled"}


class Tracer:
    def __init__(self, path: Path | None = None, run_id: str = "adhoc") -> None:
        self.path, self.run_id = path, run_id
        self.events: list[dict[str, Any]] = []
        self._subs: list[Callable[[dict[str, Any]], None]] = []
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)

    def subscribe(self, cb: Callable[[dict[str, Any]], None]) -> None:
        self._subs.append(cb)

    def emit(self, kind: str, agent: str = "", **data: Any) -> dict[str, Any]:
        ev = {
            "seq": len(self.events),
            "ts": time.time(),
            "run": self.run_id,
            "kind": kind,
            "agent": agent,
            **data,
        }
        self.events.append(ev)
        if self.path:
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(ev, default=str) + "\n")
        for cb in self._subs:
            cb(ev)
        return ev


def load_events(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _fmt(v: Any) -> str:
    if isinstance(v, list):
        return f"[{len(v)}]"
    s = str(v)
    return s if len(s) <= 60 else s[:57] + "..."


def render_timeline(events: list[dict[str, Any]], width: int = 40) -> str:
    if not events:
        return "(empty trace)"
    t0 = events[0]["ts"]
    total = max(events[-1]["ts"] - t0, 1e-6)
    lines = [f"run {events[0]['run']} | {len(events)} events | {total:.2f}s", ""]
    for e in events:
        extra = " ".join(f"{k}={_fmt(v)}" for k, v in e.items() if k not in CORE)
        lines.append(f"+{e['ts'] - t0:7.3f}s  {e['agent'] or '-':<16} {e['kind']:<18} {extra}")
    spans: dict[str, list[float]] = {}
    for e in events:
        a = e["agent"]
        if e["kind"] == "node_start":
            spans.setdefault(a, [e["ts"] - t0, e["ts"] - t0])
        elif e["kind"] in TERMINAL and a in spans:
            spans[a][1] = e["ts"] - t0
    lines += ["", "agent spans"]
    for a, (s, f) in spans.items():
        lo = int(s / total * width)
        n = max(1, int((f - s) / total * width))
        lines.append(f"  {a:<16} |{' ' * lo}{'#' * n}{' ' * (width - lo - n)}| {f - s:.2f}s")
    return "\n".join(lines)
