import json

from crew.budget import BudgetExceeded
from crew.llm import LLMResult, Usage
from crew.models import Draft, Section, Statement
from crew.report import render_report
from crew.store import EvidenceStore
from crew.verifier import lexical_support, verify

TEXT = "Prompt caching cuts the cost of repeated long prompts. RAG adds a small search latency."


class Judge:
    """Verdict = statement id is in `ok`."""

    model = "j"

    def __init__(self, ok):
        self.ok, self.calls = set(ok), 0

    async def complete(self, *, prompt, **kw):
        self.calls += 1
        items = json.loads(prompt.split("STATEMENTS_JSON: ")[1].split("\nReturn")[0])
        v = [
            {"statement_id": i["id"], "supported": i["id"] in self.ok, "reason": "r"} for i in items
        ]
        return LLMResult(json.dumps({"verdicts": v}), Usage(1, 1))


class Broke:
    model = "b"

    async def complete(self, **kw):
        raise BudgetExceeded("out")


def setup():
    store = EvidenceStore()
    src = store.add_source("https://ex.org/a", "A", TEXT)
    e1 = store.add_evidence(
        src.id, "c", "Prompt caching cuts the cost of repeated long prompts.", "s1"
    )
    e2 = store.add_evidence(src.id, "c", "RAG adds a small search latency.", "s1")
    return store, e1, e2


def draft(*stmts):
    return Draft(
        title="T",
        sections=[
            Section(heading="H", statements=[Statement(text=t, evidence_ids=e) for t, e in stmts])
        ],
    )


async def test_each_failure_mode_is_classified():
    store, e1, e2 = setup()
    d = draft(
        ("Caching reduces cost for repeated prompts.", [e1.id]),  # 1 supported by judge
        ("An uncited claim about everything.", []),  # 2 uncited
        ("A claim with a fabricated citation.", ["E42"]),  # 3 unknown evidence
        ("Latency is always zero for RAG systems.", [e2.id]),  # 4 judge says no
    )
    rows = await verify(d, store, Judge(ok={1}))
    assert [r.status for r in rows] == ["supported", "uncited", "unsupported", "unsupported"]
    assert "E42" in rows[2].reason


async def test_tampered_quote_is_rejected_without_calling_judge():
    store, e1, _ = setup()
    store.sources[e1.source_id].text = "Totally different text now, nothing matches here at all."
    judge = Judge(ok={1})
    rows = await verify(draft(("Caching reduces cost.", [e1.id])), store, judge)
    assert rows[0].status == "unsupported" and judge.calls == 0


async def test_budget_exhaustion_falls_back_to_lexical_check():
    store, e1, e2 = setup()
    d = draft(
        ("Prompt caching cuts the cost of repeated long prompts.", [e1.id]),
        ("Quantum databases replace every index structure.", [e2.id]),
    )
    rows = await verify(d, store, Broke())
    assert [r.status for r in rows] == ["supported", "unsupported"]
    assert "lexical" in rows[0].reason


def test_lexical_support_threshold():
    assert lexical_support("RAG adds search latency", ["RAG adds a small search latency."])
    assert not lexical_support(
        "RAG eliminates all hallucinations", ["RAG adds a small search latency."]
    )


async def test_report_removes_unsupported_and_renders_table():
    store, e1, e2 = setup()
    d = draft(("Caching reduces cost.", [e1.id]), ("Bogus claim here.", ["E9"]))
    rows = await verify(d, store, Judge(ok={1}))
    md = render_report("Q?", d, rows, store, {"summary": "s"})
    assert "Caching reduces cost[1]." in md and "Bogus claim" in md.split("## Verification")[1]
    assert "Bogus claim" not in md.split("## Verification")[0]
    assert "| 2 | Bogus claim here. | E9 | unsupported |" in md
    flagged = render_report("Q?", d, rows, store, {"summary": "s"}, policy="flag")
    assert "[UNVERIFIED]" in flagged.split("## Verification")[0]
