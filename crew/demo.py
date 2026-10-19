"""Deterministic offline LLM for DEMO_MODE and tests. It reads the same prompts a real model
would and returns schema-shaped JSON, so the whole pipeline (including repair loop and
verifier) runs without network access."""

from __future__ import annotations

import asyncio
import json
import re

from pydantic import BaseModel

from crew.llm import LLMResult, Usage
from crew.verifier import lexical_support

DEMO_QUESTION = (
    "How do retrieval-augmented generation and long-context models compare for question "
    "answering over large document collections?"
)
_STOP = set(
    [
        "what",
        "does",
        "how",
        "the",
        "and",
        "for",
        "are",
        "with",
        "this",
        "that",
        "from",
        "have",
        "about",
        "into",
        "over",
        "their",
    ]
)


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s.strip()) > 40]


def _kw(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z]{4,}", text.lower()) if w not in _STOP}


def _kw_ordered(text: str) -> list[str]:
    return [w for w in re.findall(r"[a-z]{4,}", text.lower()) if w not in _STOP]


class MockLLM:
    model = "mock-1"

    def __init__(self, glitch_roles: tuple[str, ...] = ("planner",), inject_faults: bool = True):
        """glitch_roles: emit malformed JSON once for these roles (exercises the repair loop).
        inject_faults: the writer 'hallucinates' two statements for the verifier to catch."""
        self.glitch, self.inject_faults, self._glitched = set(glitch_roles), inject_faults, set()

    async def complete(
        self,
        *,
        role: str,
        system: str,
        prompt: str,
        schema: type[BaseModel] | None = None,
        max_tokens: int = 1500,
    ) -> LLMResult:
        await asyncio.sleep(0.01)  # simulated latency so parallelism is visible in traces
        text = json.dumps(getattr(self, f"_{role}")(prompt))
        if role in self.glitch and role not in self._glitched:
            self._glitched.add(role)
            text = "Sure! Here is the JSON you asked for:\n" + text[:-1] + ",}"
        return LLMResult(text, Usage((len(system) + len(prompt)) // 4, len(text) // 4))

    # ---- roles
    def _planner(self, prompt: str) -> dict:
        n = int(re.search(r"MAX_SUBTASKS: (\d+)", prompt).group(1))  # type: ignore[union-attr]
        subs = [
            (
                "s1",
                "How does retrieval-augmented generation work, and what are its strengths?",
                [],
                1,
            ),
            ("s2", "What are the limits and failure modes of long-context models?", [], 1),
            ("s3", "How do cost and latency compare for retrieval and long context?", [], 2),
            ("s4", "What hybrid designs and practical recommendations exist?", ["s1", "s2"], 3),
        ]
        return {
            "subtasks": [
                {"id": i, "question": t, "depends_on": d, "priority": p}
                for i, t, d, p in subs[: max(n, 1)]
            ]
        }

    def _queries(self, prompt: str) -> dict:
        sub = re.search(r"SUBTASK: (.*)", prompt).group(1)  # type: ignore[union-attr]
        words = list(dict.fromkeys(_kw_ordered(sub)))
        return {"queries": [" ".join(words[:5]), " ".join(words[-4:])]}

    def _extract(self, prompt: str) -> dict:
        sub = re.search(r"SUBTASK: (.*)", prompt).group(1)  # type: ignore[union-attr]
        want = _kw(sub)
        out = []
        for url, body in re.findall(
            r'<untrusted_source url="(.*?)" nonce="\w+">\n(.*?)\n</untrusted', prompt, re.S
        ):
            ranked = sorted(_sentences(body), key=lambda s: -len(want & _kw(s)))
            for s in ranked[:2]:
                if "removed:" not in s:
                    out.append({"claim": s, "source_url": url, "quote": s})
        return {"findings": out[:6]}

    def _writer(self, prompt: str) -> dict:
        ev = json.loads(re.search(r"EVIDENCE_JSON: (\[.*\])", prompt).group(1))  # type: ignore[union-attr]
        groups: dict[str, list[dict]] = {}
        for e in ev:
            groups.setdefault(e["topic"].rstrip("?"), []).append(e)
        sections = [
            {
                "heading": h,
                "statements": [{"text": e["claim"], "evidence_ids": [e["id"]]} for e in es],
            }
            for h, es in groups.items()
        ]
        if self.inject_faults and ev:
            sections[-1]["statements"] += [
                {
                    "text": "RAG outperforms long-context models on every benchmark.",
                    "evidence_ids": [ev[0]["id"]],
                },
                {
                    "text": "Hybrid pipelines halve operating costs in all deployments.",
                    "evidence_ids": ["E99"],
                },
            ]
        return {"title": "RAG versus Long-Context Models", "sections": sections}

    def _verifier(self, prompt: str) -> dict:
        items = json.loads(re.search(r"STATEMENTS_JSON: (\[.*\])", prompt).group(1))  # type: ignore[union-attr]
        verdicts = []
        for it in items:
            ok = lexical_support(it["text"], it["quotes"], 0.7)
            verdicts.append(
                {
                    "statement_id": it["id"],
                    "supported": ok,
                    "reason": "quote states the claim"
                    if ok
                    else "quote does not support this claim",
                }
            )
        return {"verdicts": verdicts}
