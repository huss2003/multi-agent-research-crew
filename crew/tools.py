"""Search + fetch tool interfaces with real (Tavily/Brave/SerpAPI, httpx+trafilatura) and
fixture implementations."""

from __future__ import annotations

import asyncio
import ipaddress
import json
import re
import socket
from pathlib import Path
from typing import Protocol
from urllib.parse import urljoin, urlsplit

import httpx
import trafilatura
from pydantic import BaseModel

from crew.store import canonical_url


class SearchHit(BaseModel):
    url: str
    title: str = ""
    snippet: str = ""


class Page(BaseModel):
    url: str
    title: str = ""
    text: str


class FetchError(RuntimeError):
    pass


class SearchTool(Protocol):
    async def search(self, query: str, k: int = 5) -> list[SearchHit]: ...


class FetchTool(Protocol):
    async def fetch(self, url: str) -> Page: ...


# ---------------------------------------------------------------- real search providers
class _HttpSearch:
    def __init__(self, api_key: str, client: httpx.AsyncClient | None = None) -> None:
        self.key = api_key
        self.client = client or httpx.AsyncClient(timeout=15)


class TavilySearch(_HttpSearch):
    async def search(self, query: str, k: int = 5) -> list[SearchHit]:
        r = await self.client.post(
            "https://api.tavily.com/search",
            headers={"Authorization": f"Bearer {self.key}"},
            json={"query": query, "max_results": k},
        )
        r.raise_for_status()
        return [
            SearchHit(url=x["url"], title=x.get("title", ""), snippet=x.get("content", ""))
            for x in r.json().get("results", [])
        ]


class BraveSearch(_HttpSearch):
    async def search(self, query: str, k: int = 5) -> list[SearchHit]:
        r = await self.client.get(
            "https://api.search.brave.com/res/v1/web/search",
            headers={"X-Subscription-Token": self.key, "Accept": "application/json"},
            params={"q": query, "count": k},
        )
        r.raise_for_status()
        return [
            SearchHit(url=x["url"], title=x.get("title", ""), snippet=x.get("description", ""))
            for x in r.json().get("web", {}).get("results", [])
        ]


class SerpApiSearch(_HttpSearch):
    async def search(self, query: str, k: int = 5) -> list[SearchHit]:
        r = await self.client.get(
            "https://serpapi.com/search.json",
            params={"q": query, "num": k, "engine": "google", "api_key": self.key},
        )
        r.raise_for_status()
        return [
            SearchHit(url=x["link"], title=x.get("title", ""), snippet=x.get("snippet", ""))
            for x in r.json().get("organic_results", [])[:k]
        ]


# ---------------------------------------------------------------- real fetch
async def assert_public(url: str) -> None:
    """SSRF guard: http(s) only, and the host must not resolve to a private/loopback address."""
    p = urlsplit(url)
    if p.scheme not in ("http", "https") or not p.hostname:
        raise FetchError(f"blocked scheme/host: {url}")
    infos = await asyncio.to_thread(socket.getaddrinfo, p.hostname, p.port or 443)
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if not ip.is_global:
            raise FetchError(f"blocked non-public address {ip} for {p.hostname}")


class HttpFetch:
    def __init__(
        self,
        max_bytes: int = 1_500_000,
        timeout: float = 15.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.max_bytes = max_bytes
        self.client = client or httpx.AsyncClient(
            timeout=timeout, headers={"User-Agent": "research-crew/0.1 (+portfolio project)"}
        )

    async def fetch(self, url: str) -> Page:
        cur = url
        for _ in range(4):  # manual redirects so every hop passes the SSRF check
            await assert_public(cur)
            async with self.client.stream("GET", cur) as r:
                if r.is_redirect:
                    cur = urljoin(cur, r.headers.get("location", ""))
                    continue
                r.raise_for_status()
                if not re.search(r"html|text|xml", r.headers.get("content-type", "text/html")):
                    raise FetchError(f"unsupported content-type for {cur}")
                body = bytearray()
                async for chunk in r.aiter_bytes():
                    body += chunk
                    if len(body) > self.max_bytes:
                        break
                html = bytes(body).decode(r.encoding or "utf-8", errors="replace")
            text = await asyncio.to_thread(trafilatura.extract, html, include_comments=False)
            meta = await asyncio.to_thread(trafilatura.extract_metadata, html)
            if not text:
                raise FetchError(f"no readable text at {cur}")
            return Page(url=cur, title=(meta.title if meta and meta.title else ""), text=text)
        raise FetchError(f"too many redirects: {url}")


# ---------------------------------------------------------------- fixtures (DEMO_MODE / tests)
class FixtureCorpus:
    def __init__(self, path: Path | None = None) -> None:
        path = path or Path(__file__).parent / "fixtures" / "corpus.json"
        self.docs = [Page(**d) for d in json.loads(path.read_text(encoding="utf-8"))]


def _tok(s: str) -> set[str]:
    return {w for w in re.findall(r"[a-z]{3,}", s.lower())}


class FixtureSearch:
    def __init__(self, corpus: FixtureCorpus | None = None) -> None:
        self.corpus = corpus or FixtureCorpus()

    async def search(self, query: str, k: int = 5) -> list[SearchHit]:
        await asyncio.sleep(0.005)
        q = _tok(query)
        scored = [(len(q & _tok(d.title)) * 2 + len(q & _tok(d.text)), d) for d in self.corpus.docs]
        top = sorted((s for s in scored if s[0] > 0), key=lambda s: -s[0])[:k]
        return [SearchHit(url=d.url, title=d.title, snippet=d.text[:120]) for _, d in top]


class FixtureFetch:
    def __init__(self, corpus: FixtureCorpus | None = None) -> None:
        self.by_url = {canonical_url(d.url): d for d in (corpus or FixtureCorpus()).docs}

    async def fetch(self, url: str) -> Page:
        await asyncio.sleep(0.005)
        try:
            return self.by_url[canonical_url(url)]
        except KeyError:
            raise FetchError(f"404 {url}") from None
