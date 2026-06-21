from datetime import datetime, timedelta, timezone
from email.utils import format_datetime

import httpx

from kami.crawler import _retry_after_seconds, _retry_delay_seconds
from kami.crawler import KamihimeCrawler


def test_retry_after_accepts_delta_seconds():
    assert _retry_after_seconds("12") == 12


def test_retry_after_accepts_http_date():
    future = datetime.now(timezone.utc) + timedelta(seconds=30)

    assert 0 < _retry_after_seconds(format_datetime(future)) <= 30


def test_429_backoff_respects_minimum_cooldown(monkeypatch):
    monkeypatch.setenv("KAMI_HTTP_429_COOLDOWN", "45")
    monkeypatch.setenv("KAMI_HTTP_BACKOFF_JITTER", "0")
    response = httpx.Response(429, request=httpx.Request("GET", "https://example.test"))

    assert _retry_delay_seconds(response, 0) == 45


def test_retry_after_header_takes_precedence(monkeypatch):
    monkeypatch.setenv("KAMI_HTTP_BACKOFF_JITTER", "0")
    response = httpx.Response(
        429,
        headers={"Retry-After": "7"},
        request=httpx.Request("GET", "https://example.test"),
    )

    assert _retry_delay_seconds(response, 0) == 7


def test_crawl_reports_detail_progress(monkeypatch):
    crawler = KamihimeCrawler([])
    links = [
        {"name": "A", "link": "https://example.test/a", "list_image": "", "release_date": "-", "acquisition_method": "-"},
        {"name": "B", "link": "https://example.test/b", "list_image": "", "release_date": "-", "acquisition_method": "-"},
    ]

    def fake_crawl_character(character):
        return {"info": {"name": character["name"], "source_url": character["link"]}, "skill": []}

    monkeypatch.setattr(crawler, "crawl_character", fake_crawl_character)
    events = []

    try:
        records = crawler.crawl(links, events.append)
    finally:
        crawler.close()

    assert [event["processed"] for event in events] == [0, 1, 2]
    assert all(event["total"] == 2 for event in events)
    assert [record["info"]["name"] for record in records] == ["A", "B"]
