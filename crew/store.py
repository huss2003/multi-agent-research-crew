"""Shared evidence store: deduplicated sources + claims anchored to verbatim quote spans."""

from __future__ import annotations

import hashlib
import re
from urllib.parse import parse_qsl, urlencode, urlsplit

from pydantic import BaseModel

_TRACKING = re.compile(r"^(utm_.*|gclid|fbclid|ref|ref_src|mc_cid|mc_eid)$")


def canonical_url(url: str) -> str:
    p = urlsplit(url.strip())
    host = (p.hostname or "").lower().removeprefix("www.")
    port = f":{p.port}" if p.port and p.port not in (80, 443) else ""
    q = urlencode(sorted((k, v) for k, v in parse_qsl(p.query) if not _TRACKING.match(k)))
    path = p.path.rstrip("/") or "/"
    return f"{p.scheme.lower()}://{host}{port}{path}" + (f"?{q}" if q else "")


class Source(BaseModel):
    id: str
    url: str
    title: str
    text: str
    content_hash: str


class Evidence(BaseModel):
    id: str
    source_id: str
    claim: str
    quote: str
    start: int
    end: int
    subtask_id: str


class EvidenceStore:
    def __init__(self) -> None:
        self.sources: dict[str, Source] = {}
        self.evidence: dict[str, Evidence] = {}
        self._by_url: dict[str, str] = {}
        self._by_hash: dict[str, str] = {}

    def add_source(self, url: str, title: str, text: str) -> Source:
        """Dedupe by canonical URL, then by content hash (mirrors / tracking-param variants)."""
        key = canonical_url(url)
        digest = hashlib.sha256(" ".join(text.split()).lower().encode()).hexdigest()[:16]
        sid = self._by_url.get(key) or self._by_hash.get(digest)
        if sid:
            self._by_url[key] = sid
            return self.sources[sid]
        src = Source(
            id=f"S{len(self.sources) + 1}",
            url=canonical_url(url),
            title=title or url,
            text=text,
            content_hash=digest,
        )
        self.sources[src.id] = src
        self._by_url[key] = self._by_hash[digest] = src.id
        return src

    def source_for_url(self, url: str) -> Source | None:
        sid = self._by_url.get(canonical_url(url))
        return self.sources[sid] if sid else None

    def add_evidence(
        self, source_id: str, claim: str, quote: str, subtask_id: str
    ) -> Evidence | None:
        """Store a claim only if `quote` occurs verbatim (whitespace-insensitive) in the source."""
        words = quote.split()
        m = (
            re.search(r"\s+".join(map(re.escape, words)), self.sources[source_id].text)
            if words
            else None
        )
        if not m:
            return None
        for ev in self.evidence.values():  # dedupe identical spans
            if ev.source_id == source_id and (ev.start, ev.end) == m.span():
                return ev
        ev = Evidence(
            id=f"E{len(self.evidence) + 1}",
            source_id=source_id,
            claim=claim.strip(),
            quote=m.group(0),
            start=m.start(),
            end=m.end(),
            subtask_id=subtask_id,
        )
        self.evidence[ev.id] = ev
        return ev

    def quote_intact(self, ev: Evidence) -> bool:
        src = self.sources.get(ev.source_id)
        return bool(src) and src.text[ev.start : ev.end] == ev.quote
