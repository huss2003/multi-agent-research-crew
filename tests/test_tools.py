import httpx
import pytest

from crew.tools import BraveSearch, FetchError, HttpFetch, TavilySearch, assert_public


async def test_ssrf_guard_blocks_private_and_non_http():
    for bad in (
        "http://127.0.0.1/",
        "http://169.254.169.254/latest",
        "file:///etc/passwd",
        "http://10.0.0.5/",
    ):
        with pytest.raises(FetchError):
            await assert_public(bad)
    await assert_public("https://93.184.216.34/")


async def test_http_fetch_extracts_readable_text():
    html = (
        "<html><head><title>Doc</title></head><body><article><p>"
        + "Readable sentence about retrieval. " * 20
        + "</p></article></body></html>"
    )
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda r: httpx.Response(200, html=html))
    )
    page = await HttpFetch(client=client).fetch("https://93.184.216.34/x")
    assert "Readable sentence" in page.text


async def test_http_fetch_rechecks_redirect_targets():
    def handler(r):
        return httpx.Response(302, headers={"location": "http://127.0.0.1/admin"})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    with pytest.raises(FetchError):
        await HttpFetch(client=client).fetch("https://93.184.216.34/x")


async def test_search_providers_parse_responses():
    tav = httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda r: httpx.Response(
                200, json={"results": [{"url": "https://a", "title": "A", "content": "c"}]}
            )
        )
    )
    brave = httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda r: httpx.Response(
                200,
                json={"web": {"results": [{"url": "https://b", "title": "B", "description": "d"}]}},
            )
        )
    )
    assert (await TavilySearch("k", tav).search("q"))[0].url == "https://a"
    assert (await BraveSearch("k", brave).search("q"))[0].snippet == "d"
