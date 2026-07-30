from fastapi.testclient import TestClient

from app import main


def _catalog_item(object_type: str):
    base = {
        "slug": f"test-{object_type}",
        "object_type": object_type,
        "name": f"Test {object_type.title()}",
        "element": "phantom",
        "image": "https://example.test/full.png",
        "list_image": "https://example.test/list.png",
        "release_date": "2026/07/30",
        "acquisition_method": "Test source",
        "info": {
            "source_url": "https://example.test/source",
            "Rarity": "SSR",
        },
        "display_info": {"Rarity": "SSR"},
        "stats": [
            {
                "limit_break": "4",
                "Max Level": "100",
                "HP": "1000",
                "Attack": "3000",
            }
        ],
        "bursts": [],
        "weapon_skills": [],
        "eidolon_effects": [],
        "flavor": "Test flavor.",
    }
    if object_type == "weapon":
        base["bursts"] = [
            {"limit_break": "4", "effect": "Extreme Phantom damage"}
        ]
        base["weapon_skills"] = [
            {
                "limit_break": "4",
                "max_level": "Lv.30",
                "name": "Phantom Assault",
                "effect": "Increases Attack",
            }
        ]
    if object_type == "eidolon":
        summon_effect = {
            "type": "Summon Effect",
            "name": "Phantom Call",
            "requirements": "-",
            "interval": "8T",
            "duration": "3T",
            "effect": "Extreme Phantom damage",
        }
        passive_effects = [
            {
                "type": "Main Effect",
                "name": "Phantom Power",
                "requirements": "-",
                "effect": "Increases Phantom Attack",
                "show_type": True,
                "type_rowspan": 2,
                "show_name": True,
                "name_rowspan": 2,
            },
            {
                "type": "Main Effect",
                "name": "Phantom Power",
                "requirements": "1",
                "effect": "Further increases Phantom Attack",
                "show_type": False,
                "type_rowspan": 0,
                "show_name": False,
                "name_rowspan": 0,
            },
        ]
        base["eidolon_effects"] = [summon_effect, *passive_effects]
        base["eidolon_summon_effects"] = [summon_effect]
        base["eidolon_passive_effects"] = passive_effects
    return base


def test_catalog_page_renders_dropdowns_and_object_links(monkeypatch):
    monkeypatch.setattr(
        main,
        "load_catalog_items",
        lambda object_type, element: [_catalog_item(object_type)],
    )
    client = TestClient(main.app)

    response = client.get("/catalog/weapon/phantom")

    assert response.status_code == 200
    assert response.text.count("data-catalog-group=") == 3
    assert 'data-catalog-group="weapon"' in response.text
    assert 'aria-expanded="true"' in response.text
    assert 'data-object-type="weapon"' in response.text
    assert '<option value="deepl" selected>DeepL</option>' in response.text
    assert "/objects/weapon/test-weapon" in response.text
    assert "Phantom Weapons" in response.text


def test_catalog_rejects_phantom_for_kamihime():
    response = TestClient(main.app).get("/catalog/kamihime/phantom")

    assert response.status_code == 404


def test_weapon_detail_renders_weapon_specific_sections(monkeypatch):
    item = _catalog_item("weapon")
    item["info"].update(
        {
            "Unlock Kamihime": "Test Kamihime",
            "unlock_kamihime_url": (
                "https://example.test/%E7%A5%9E%E5%A7%AB/test"
            ),
        }
    )
    item["display_info"]["Unlock Kamihime"] = "Test Kamihime"
    kamihime = _catalog_item("kamihime")
    kamihime["info"]["source_url"] = (
        "https://example.test/神姫/test?source=wiki"
    )
    monkeypatch.setattr(
        main,
        "get_catalog_item",
        lambda object_type, slug: (
            item
            if (object_type, slug) == ("weapon", "test-weapon")
            else None
        ),
    )
    monkeypatch.setattr(
        main,
        "load_catalog_items",
        lambda object_type: (
            [kamihime] if object_type == "kamihime" else []
        ),
    )

    response = TestClient(main.app).get("/objects/weapon/test-weapon")

    assert response.status_code == 200
    assert "Burst Effects" in response.text
    assert "Weapon Skills" in response.text
    assert "Phantom Assault" in response.text
    assert 'class="data-table object-data-table weapon-burst-table"' in response.text
    assert 'class="data-table object-data-table weapon-skill-table"' in response.text
    assert 'class="col-skill-name"' in response.text
    assert (
        'class="object-reference-link" '
        'href="/objects/kamihime/test-kamihime"'
    ) in response.text


def test_kamihime_detail_links_unlock_weapon_by_source_url(monkeypatch):
    item = _catalog_item("kamihime")
    item["info"].update(
        {
            "Unlock Weapon": "Test Weapon",
            "unlock_weapon_url": (
                "https://example.test/%E6%AD%A6%E5%99%A8/test"
            ),
        }
    )
    item["display_info"]["Unlock Weapon"] = "Test Weapon"
    weapon = _catalog_item("weapon")
    weapon["info"]["source_url"] = "https://example.test/武器/test"
    monkeypatch.setattr(
        main,
        "get_catalog_item",
        lambda object_type, slug: (
            item
            if (object_type, slug) == ("kamihime", "test-kamihime")
            else None
        ),
    )
    monkeypatch.setattr(
        main,
        "load_catalog_items",
        lambda object_type: (
            [weapon] if object_type == "weapon" else []
        ),
    )

    response = TestClient(main.app).get(
        "/objects/kamihime/test-kamihime"
    )

    assert response.status_code == 200
    assert (
        'class="object-reference-link" '
        'href="/objects/weapon/test-weapon"'
    ) in response.text


def test_eidolon_detail_renders_eidolon_effects(monkeypatch):
    item = _catalog_item("eidolon")
    monkeypatch.setattr(
        main,
        "get_catalog_item",
        lambda object_type, slug: item,
    )

    response = TestClient(main.app).get("/objects/eidolon/test-eidolon")

    assert response.status_code == 200
    assert "Eidolon Effects" in response.text
    assert "Phantom Call" in response.text
    assert "Phantom Power" in response.text
    assert "/catalog/eidolon/phantom" in response.text
    assert 'class="data-table object-data-table eidolon-effect-table"' in response.text
    assert response.text.count('class="eidolon-section-heading"') == 2
    assert 'class="col-effect"' in response.text
    assert 'rowspan="2"' in response.text
    assert response.text.count("Phantom Power") == 1
    assert "Usage interval" in response.text


def test_legacy_element_route_uses_new_kamihime_catalog(monkeypatch):
    monkeypatch.setattr(
        main,
        "load_catalog_items",
        lambda object_type, element: [],
    )

    response = TestClient(main.app).get("/elements/fire")

    assert response.status_code == 200
    assert 'data-object-type="kamihime"' in response.text
    assert "Fire Kamihime" in response.text
