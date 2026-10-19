"""Render the final markdown report: kept statements, numbered citations, verification table."""

from __future__ import annotations

from typing import Any

from crew.models import Draft, VerificationRow
from crew.store import EvidenceStore


def _cell(s: str, n: int = 90) -> str:
    s = s.replace("|", "\\|").replace("\n", " ")
    return s if len(s) <= n else s[: n - 1] + "…"


def render_report(
    question: str,
    draft: Draft,
    rows: list[VerificationRow],
    store: EvidenceStore,
    stats: dict[str, Any],
    policy: str = "remove",
) -> str:
    status = {r.id: r for r in rows}
    numbers: dict[str, int] = {}  # source id -> [n], by first citation in kept statements
    cited: dict[str, list[str]] = {}  # source id -> quotes
    out = [f"# {draft.title}", "", f"> **Question:** {question}", ""]
    idx = 0
    for sec in draft.sections:
        lines: list[str] = []
        for st in sec.statements:
            idx += 1
            r = status[idx]
            if r.status != "supported" and policy == "remove":
                continue
            refs = ""
            for eid in st.evidence_ids:
                ev = store.evidence.get(eid)
                if not ev:
                    continue
                n = numbers.setdefault(ev.source_id, len(numbers) + 1)
                quotes = cited.setdefault(ev.source_id, [])
                if ev.quote not in quotes:
                    quotes.append(ev.quote)
                refs += f"[{n}]"
            flag = " **[UNVERIFIED]**" if r.status != "supported" else ""
            lines.append(f"{st.text.rstrip('.')}{refs}.{flag}")
        if lines:
            out += [f"## {sec.heading}", "", " ".join(lines), ""]
    out += ["## Sources", ""]
    for sid, n in sorted(numbers.items(), key=lambda kv: kv[1]):
        src = store.sources[sid]
        out.append(f"{n}. [{src.title}]({src.url})")
        out += [f"   - > {_cell(q, 160)}" for q in cited[sid]]
    ok = sum(r.status == "supported" for r in rows)
    out += [
        "",
        "## Verification",
        "",
        f"{ok}/{len(rows)} statements supported by a stored quote; "
        f"{len(rows) - ok} {'removed' if policy == 'remove' else 'flagged'}.",
        "",
        "| # | Statement | Evidence | Verdict | Note |",
        "|---|---|---|---|---|",
    ]
    for r in rows:
        out.append(
            f"| {r.id} | {_cell(r.text)} | {', '.join(r.evidence_ids) or '-'} | "
            f"{r.status} | {_cell(r.reason, 60)} |"
        )
    out += ["", "---", f"*{stats.get('summary', '')}*", ""]
    return "\n".join(out)
