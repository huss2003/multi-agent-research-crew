"""Claim verifier: every statement must map to stored, intact quotes that actually support it."""

from __future__ import annotations

import json
import re
from collections.abc import Iterator

from crew.budget import BudgetExceeded
from crew.llm import LLM
from crew.models import Draft, Section, Statement, Verdicts, VerificationRow
from crew.safety import UNTRUSTED_NOTICE
from crew.store import EvidenceStore
from crew.structured import generate
from crew.tracing import Tracer

_STOP = set(
    [
        "the",
        "and",
        "for",
        "that",
        "with",
        "this",
        "from",
        "are",
        "was",
        "were",
        "have",
        "has",
        "been",
        "their",
        "they",
        "which",
        "also",
        "into",
        "more",
        "than",
        "can",
        "not",
        "but",
        "its",
        "all",
    ]
)

SYSTEM = (
    "You are a strict fact-checker. For each statement decide whether the quoted passages "
    "directly state it or logically entail it. Extra claims, stronger wording (always, every, "
    "never), or numbers absent from the quotes mean NOT supported. " + UNTRUSTED_NOTICE
)


def iter_statements(draft: Draft) -> Iterator[tuple[int, Section, Statement]]:
    i = 0
    for sec in draft.sections:
        for st in sec.statements:
            i += 1
            yield i, sec, st


def _words(s: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]{4,}", s.lower()) if w not in _STOP}


def lexical_support(statement: str, quotes: list[str], threshold: float = 0.6) -> bool:
    """Cheap offline fallback: share of the statement's content words found in the quotes."""
    need = _words(statement)
    return bool(need) and len(need & _words(" ".join(quotes))) / len(need) >= threshold


async def verify(
    draft: Draft, store: EvidenceStore, llm: LLM, tracer: Tracer | None = None
) -> list[VerificationRow]:
    rows: dict[int, VerificationRow] = {}
    pending: list[dict] = []

    def row(i: int, sec: Section, st: Statement, status: str, reason: str) -> None:
        rows[i] = VerificationRow(
            id=i,
            section=sec.heading,
            text=st.text,
            evidence_ids=st.evidence_ids,
            status=status,
            reason=reason,
        )

    for i, sec, st in iter_statements(draft):
        evs = [store.evidence.get(e) for e in st.evidence_ids]
        if not st.evidence_ids:
            row(i, sec, st, "uncited", "no evidence cited")
        elif any(e is None for e in evs):
            bad = [eid for eid, e in zip(st.evidence_ids, evs, strict=True) if e is None]
            row(i, sec, st, "unsupported", f"cites unknown evidence {bad}")
        elif not all(store.quote_intact(e) for e in evs if e):
            row(i, sec, st, "unsupported", "stored quote no longer matches its source")
        else:
            pending.append({"id": i, "text": st.text, "quotes": [e.quote for e in evs if e]})
    by_id = {i: (sec, st) for i, sec, st in iter_statements(draft)}

    if pending:
        try:
            out = await generate(
                llm,
                Verdicts,
                role="verifier",
                system=SYSTEM,
                agent="verifier",
                tracer=tracer,
                prompt="STATEMENTS_JSON: "
                + json.dumps(pending)
                + "\nReturn one verdict per statement id.",
                max_tokens=min(4000, 120 * len(pending) + 200),
            )
            verdicts = {v.statement_id: v for v in out.verdicts}
            for p in pending:
                v = verdicts.get(p["id"])
                sec, st = by_id[p["id"]]
                if v is None:
                    row(p["id"], sec, st, "unsupported", "verifier returned no verdict")
                else:
                    row(p["id"], sec, st, "supported" if v.supported else "unsupported", v.reason)
        except BudgetExceeded:  # degrade: lexical overlap instead of an LLM judge
            if tracer:
                tracer.emit("budget_degrade", "verifier", action="lexical_verification")
            for p in pending:
                sec, st = by_id[p["id"]]
                ok = lexical_support(p["text"], p["quotes"])
                row(
                    p["id"],
                    sec,
                    st,
                    "supported" if ok else "unsupported",
                    "lexical check only (budget)",
                )
    result = [rows[i] for i in sorted(rows)]
    if tracer:
        tracer.emit(
            "verification",
            "verifier",
            total=len(result),
            supported=sum(r.status == "supported" for r in result),
        )
    return result
