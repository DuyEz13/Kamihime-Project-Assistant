from datetime import datetime, timedelta, timezone
from email.utils import format_datetime

import httpx
from bs4 import BeautifulSoup

from kami.crawler import (
    EidolonCrawler,
    KamihimeCrawler,
    SOURCE_URLS,
    WeaponCrawler,
    _retry_after_seconds,
    _retry_delay_seconds,
    configured_elements,
)


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


def test_parse_character_preserves_skill_icons():
    html = """
    <table>
      <tr><th>バースト</th></tr>
      <tr>
        <td rowspan="2"><img src="/icons/burst.png"></td>
        <td>Queen Fire</td>
        <td>-</td>
        <td>Fire damage</td>
      </tr>
      <tr>
        <td>Queen Fire+</td>
        <td>Limit break</td>
        <td>Fire damage up</td>
      </tr>
    </table>
    """

    record = KamihimeCrawler.parse_character(
        html,
        "Test",
        "https://example.test/characters/test",
    )

    assert record["skill"][0]["icon"] == "https://example.test/icons/burst.png"
    assert record["skill"][1]["icon"] == "https://example.test/icons/burst.png"


def test_kamihime_parser_captures_unlock_weapon_url():
    html = """
    <table>
      <tr>
        <th>解放武器</th>
        <td><a href="/武器/SSR/解放剣">解放剣</a></td>
      </tr>
    </table>
    """

    record = KamihimeCrawler.parse_character(
        html,
        "Test",
        "https://example.test/神姫/test",
    )

    assert record["info"]["解放武器"] == "解放剣"
    assert (
        record["info"]["unlock_weapon_url"]
        == "https://example.test/武器/SSR/解放剣"
    )


def test_weapon_parser_captures_unlock_kamihime_url():
    html = """
    <div class="h-scrollable">
      <table>
        <tr><th>基本性能</th></tr>
        <tr>
          <th>解放神姫</th>
          <td><a href="/神姫/SSR/解放姫">解放姫</a></td>
        </tr>
      </table>
    </div>
    """

    record = WeaponCrawler.parse_character(
        html,
        "Test Weapon",
        "https://example.test/武器/test",
    )

    assert record["info"]["解放神姫"] == "解放姫"
    assert (
        record["info"]["unlock_kamihime_url"]
        == "https://example.test/神姫/SSR/解放姫"
    )


def test_weapon_list_skips_wiki_edit_red_links(monkeypatch):
    html = """
    <table>
      <thead>
        <tr><th>武器名</th></tr>
      </thead>
      <tbody>
        <tr>
          <td><img src="/missing.png"></td>
          <td><a href="/kamiprodb/::cmd/edit?page=GA_Mk1Rance">?</a></td>
          <td>火</td><td>槍</td><td>125</td><td>-</td><td>20/01/01</td><td>-</td>
        </tr>
        <tr>
          <td><img src="/valid.png"></td>
          <td><a href="/kamiprodb/武器/SSR/ValidWeapon">Valid Weapon</a></td>
          <td>火</td><td>剣</td><td>200</td><td>-</td><td>26/01/01</td><td>ガチャ</td>
        </tr>
      </tbody>
    </table>
    """
    crawler = WeaponCrawler(
        ["https://wikiwiki.jp/kamiprodb/武器/SSR/火"]
    )
    monkeypatch.setattr(
        crawler,
        "_get_soup",
        lambda _url: BeautifulSoup(html, "html.parser"),
    )

    try:
        links = crawler.extract_character_links()
    finally:
        crawler.close()

    assert [item["name"] for item in links] == ["Valid Weapon"]
    assert links[0]["link"] == (
        "https://wikiwiki.jp/kamiprodb/武器/SSR/ValidWeapon"
    )


def test_weapon_parser_extracts_stats_bursts_and_rowspan_skills():
    html = """
    <div class="h-scrollable">
      <table>
        <tr><th colspan="4">基本性能</th><td rowspan="8"><img src="/weapon.png"></td></tr>
        <tr><th>レアリティ</th><td colspan="3">SSR</td></tr>
        <tr><th>属性</th><td colspan="3">火</td></tr>
        <tr><th>能力値</th><th>最大Lv</th><th>HP</th><th>攻撃力</th></tr>
        <tr><th>～3</th><td>125</td><td>18～108</td><td>480～2,880</td></tr>
        <tr><td colspan="5">Weapon flavor text.</td></tr>
        <tr><th>バースト</th><th colspan="4">効果</th></tr>
        <tr><th>4～5</th><td colspan="4">火属性ダメージ(極大)</td></tr>
        <tr><th>スキル</th><th>最大Lv</th><th colspan="2">スキル名</th><th>効果</th></tr>
        <tr><th>4</th><td>Lv.30</td><td colspan="2" rowspan="2">インフェルノスティンガー</td><td rowspan="2">急所攻撃確率UP</td></tr>
        <tr><th>5</th><td>Lv.40</td></tr>
      </table>
    </div>
    """

    record = WeaponCrawler.parse_character(
        html,
        "Test Weapon",
        "https://example.test/weapons/test",
    )

    assert record["info"]["object_type"] == "weapon"
    assert record["info"]["img"] == "https://example.test/weapon.png"
    assert record["stats"][0]["最大Lv"] == "125"
    assert record["bursts"][0]["effect"] == "火属性ダメージ(極大)"
    assert record["weapon_skills"][1] == {
        "limit_break": "5",
        "max_level": "Lv.40",
        "name": "インフェルノスティンガー",
        "effect": "急所攻撃確率UP",
    }


def test_eidolon_parser_extracts_summon_main_and_sub_effects():
    html = """
    <div class="h-scrollable">
      <table>
        <tr><th colspan="4">基本性能</th><td colspan="3" rowspan="4"><img src="/eidolon.png"></td></tr>
        <tr><th>レアリティ</th><td colspan="3">SSR</td></tr>
        <tr><th>能力値</th><th>最大Lv</th><th>HP</th><th>攻撃力</th></tr>
        <tr><th>～4</th><td>100</td><td>190～1,140</td><td>532～3,192</td></tr>
        <tr><td colspan="7">Eidolon flavor text.</td></tr>
        <tr><th colspan="3">召喚効果</th><th>習得条件</th><th>効果</th><th>使用間隔</th><th>効果時間</th></tr>
        <tr><td colspan="3">焔舞</td><td></td><td>敵全体に火属性ダメージ</td><td>0T⇒3T</td><td>7T</td></tr>
        <tr><th colspan="3">メイン効果</th><th>習得条件</th><th colspan="3">効果</th></tr>
        <tr><td colspan="3" rowspan="2">火の加護</td><td></td><td colspan="3">火属性攻撃100%UP</td></tr>
        <tr><td>1</td><td colspan="3">火属性攻撃105%UP</td></tr>
        <tr><th colspan="3">サブ効果</th><th>習得条件</th><th colspan="3">効果</th></tr>
        <tr><td colspan="3">守護</td><td></td><td colspan="3">防御20%UP</td></tr>
      </table>
    </div>
    """

    record = EidolonCrawler.parse_character(
        html,
        "Test Eidolon",
        "https://example.test/eidolons/test",
    )

    assert record["info"]["object_type"] == "eidolon"
    assert record["stats"][0]["攻撃力"] == "532～3,192"
    assert [effect["type"] for effect in record["eidolon_effects"]] == [
        "召喚効果",
        "メイン効果",
        "メイン効果",
        "サブ効果",
    ]
    assert record["eidolon_effects"][2]["name"] == "火の加護"
    assert record["eidolon_effects"][2]["requirements"] == "1"


def test_non_kamihime_sources_and_elements_include_phantom(monkeypatch):
    monkeypatch.delenv("KAMI_EIDOLON_ELEMENTS", raising=False)
    monkeypatch.delenv("KAMI_WEAPON_ELEMENTS", raising=False)

    assert configured_elements("eidolon")[-1] == "phantom"
    assert configured_elements("weapon")[-1] == "phantom"
    assert "phantom" in SOURCE_URLS["eidolon"]
    assert "phantom" in SOURCE_URLS["weapon"]
    assert "phantom" not in SOURCE_URLS["kamihime"]
