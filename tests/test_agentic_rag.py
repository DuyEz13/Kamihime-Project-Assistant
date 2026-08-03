import json

import pytest
import torch

from kami.agent import memory
from kami.agent.documents import object_documents
from kami.agent.graph import _ground_query_constraints, deterministic_plan, run_agent
from kami.agent.language import detect_response_language
from kami.agent.providers import available_models
from kami.agent.retrieval import (
    OBJECT_CANDIDATES,
    OBJECT_CANDIDATES_BY_TYPE,
    resolve_object_variants,
    resolve_embedding_device,
    retrieve_entity,
)
from kami.agent.schemas import EntityQuery, QueryPlan
from kami.series import (
    detect_series_candidates,
    enrich_info_series,
    reconcile_series_data,
    series_metadata,
)


def test_vietnamese_latest_query_has_exact_catalog_constraints():
    plan = deterministic_plan(
        "t\u00ecm th\u00f4ng tin v\u1ec1 kamihime m\u1edbi nh\u1ea5t h\u1ec7 n\u01b0\u1edbc",
        {},
        lambda _object_type: [],
    )

    assert plan.in_domain is True
    assert plan.intent == "filter"
    assert plan.target_types == ["kamihime"]
    assert plan.elements == ["water"]
    assert plan.sort_by == "release_date"
    assert plan.sort_order == "desc"
    assert plan.result_limit == 1
    assert plan.include_ties is True


@pytest.mark.parametrize("phrase", ["latest", "newest", "most recent"])
def test_english_latest_phrases_are_grounded(phrase):
    plan = deterministic_plan(
        f"Find the {phrase} Water Eidolon",
        {},
        lambda _object_type: [],
    )

    assert plan.target_types == ["eidolon"]
    assert plan.elements == ["water"]
    assert plan.sort_by == "release_date"
    assert plan.include_ties is True


def test_grounding_corrects_model_plan_that_dropped_latest_and_element():
    model_plan = QueryPlan(
        in_domain=True,
        standalone_question="Find Water Kamihime",
        target_types=["kamihime"],
    )

    grounded = _ground_query_constraints(
        model_plan,
        "t\u00ecm th\u00f4ng tin v\u1ec1 kamihime m\u1edbi nh\u1ea5t h\u1ec7 n\u01b0\u1edbc",
    )

    assert grounded.elements == ["water"]
    assert grounded.sort_by == "release_date"
    assert grounded.result_limit == 1
    assert grounded.include_ties is True


def test_latest_query_without_object_type_requests_clarification():
    plan = deterministic_plan(
        "d\u1eef li\u1ec7u h\u1ec7 n\u01b0\u1edbc m\u1edbi nh\u1ea5t",
        {},
        lambda _object_type: [],
    )

    assert plan.needs_clarification is True
    assert plan.clarification_question


def _item(name: str, slug: str, element: str = "fire", object_type: str = "kamihime"):
    item = {
        "name": name,
        "slug": slug,
        "element": element,
        "object_type": object_type,
        "release_date": "2026/01/01",
        "acquisition_method": "Test",
        "display_info": {"Rarity": "SSR"},
        "info": {"source_url": f"https://example.test/{object_type}/{slug}"},
        "flavor": "Test object.",
    }
    if object_type == "kamihime":
        item["skill_sections"] = [
            {
                "type": "Ability",
                "rows": [
                    {
                        "name": "Test Skill",
                        "effect": "Restores HP to all allies.",
                        "interval": "6T",
                    }
                ],
            }
        ]
    elif object_type == "weapon":
        item.update({"stats": [], "bursts": [], "weapon_skills": []})
    else:
        item.update({"stats": [], "eidolon_effects": []})
    return item


def _complete_item(
    name: str,
    slug: str,
    element: str = "fire",
    object_type: str = "kamihime",
):
    item = _item(name, slug, element, object_type)
    if object_type == "kamihime":
        item["skill_sections"] = [
            {"type": "Burst", "rows": [{"name": "Test Burst", "effect": "Burst damage."}]},
            {
                "type": "Ability",
                "rows": [
                    {"name": "First Skill", "effect": "First effect."},
                    {"name": "Second Skill", "effect": "Second effect."},
                ],
            },
            {"type": "Assist", "rows": [{"name": "Test Assist", "effect": "Passive effect."}]},
        ]
    elif object_type == "weapon":
        item.update(
            {
                "stats": [
                    {"limit_break": "~3", "max_level": "125"},
                    {"limit_break": "4", "max_level": "150"},
                ],
                "bursts": [
                    {"limit_break": "~3", "effect": "Massive damage."},
                    {"limit_break": "4", "effect": "Massive damage plus."},
                ],
                "weapon_skills": [
                    {"name": "Assault", "effect": "Increases Attack."},
                    {"name": "Defender", "effect": "Increases HP."},
                ],
            }
        )
    else:
        item.update(
            {
                "stats": [
                    {"limit_break": "~4", "max_level": "100"},
                    {"limit_break": "5", "max_level": "150"},
                ],
                "eidolon_effects": [
                    {"type": "Summon Effect", "name": "Summon", "effect": "Summon damage."},
                    {"type": "Summon Effect", "name": "Summon+", "effect": "Summon damage plus."},
                    {"type": "Main Effect", "name": "Main", "requirements": "", "effect": "Main 20%."},
                    {"type": "Main Effect", "name": "Main", "requirements": "5", "effect": "Main 100%."},
                    {"type": "Sub Effect", "name": "Sub", "requirements": "5", "effect": "Sub 30%."},
                ],
            }
        )
    return item


def test_document_builder_chunks_skill_rows_with_routing_metadata():
    documents = object_documents(_item("Alpha", "alpha"))

    assert len(documents) == 2
    assert documents[1].metadata["object_type"] == "kamihime"
    assert documents[1].metadata["slug"] == "alpha"
    assert documents[1].metadata["section"] == "Ability"
    assert "Restores HP" in documents[1].page_content


def test_document_builder_groups_complete_sections_for_each_object_type():
    kamihime_docs = object_documents(_complete_item("Alpha", "alpha"))
    weapon_docs = object_documents(
        _complete_item("Blade", "blade", object_type="weapon")
    )
    eidolon_docs = object_documents(
        _complete_item("Phoenix", "phoenix", object_type="eidolon")
    )

    assert [doc.metadata["section"] for doc in kamihime_docs] == [
        "basic",
        "Burst",
        "Ability",
        "Assist",
    ]
    assert "Row 2:" in kamihime_docs[2].page_content
    assert [doc.metadata["section"] for doc in weapon_docs] == [
        "basic",
        "Stats",
        "Burst Effects",
        "Weapon Skills",
    ]
    assert "Massive damage plus" in weapon_docs[2].page_content
    assert [doc.metadata["section"] for doc in eidolon_docs] == [
        "basic",
        "Stats",
        "Summon Effect",
        "Main Effect",
        "Sub Effect",
    ]
    assert "Main 20%" in eidolon_docs[3].page_content
    assert "Main 100%" in eidolon_docs[3].page_content


def test_broad_eidolon_series_hydrates_every_available_section(monkeypatch):
    monkeypatch.setenv("KAMI_RAG_RERANK", "0")
    elements = ("fire", "water", "wind", "thunder", "light", "dark")
    items = [
        _complete_item(
            f"{element.title()} Catastrophe",
            f"{element}-catastrophe",
            element,
            "eidolon",
        )
        for element in elements
    ]
    loader = lambda object_type: items if object_type == "eidolon" else []

    evidence = retrieve_entity(
        EntityQuery(
            mention="Catastrophe",
            object_type="eidolon",
            retrieval_query="Find information about the Catastrophe Eidolon series",
        ),
        ["eidolon"],
        loader,
    )

    grouped = {}
    for item in evidence:
        grouped.setdefault(item["slug"], []).append(item)
    assert len(grouped) == 6
    expected = {"basic", "Stats", "Summon Effect", "Main Effect", "Sub Effect"}
    for values in grouped.values():
        assert {item["section"] for item in values} == expected
        assert values[0]["coverage_complete"] is True
        assert set(values[0]["available_sections"]) == expected


def _series_item(name, original_name, slug, element, object_type):
    item = _complete_item(name, slug, element, object_type)
    enrich_info_series(item["info"], object_type, original_name)
    item.update(
        {
            "original_name": item["info"]["original_name"],
            "series_key": item["info"].get("series_key", ""),
            "series_name": item["info"].get("series_name", ""),
            "series_aliases": item["info"].get("series_aliases", []),
            "series_expected_elements": item["info"].get(
                "series_expected_elements", []
            ),
        }
    )
    return item


def test_weapon_series_alias_resolves_all_six_elements_and_reports_coverage(monkeypatch):
    monkeypatch.setenv("KAMI_RAG_RERANK", "0")
    elements = ("fire", "water", "wind", "thunder", "light", "dark")
    names = (
        "Apocalypse Polemophylos",
        "Apocalypse Adikhydros",
        "Apocalypse Thanatoanemos",
        "Apocalyptic Leucobrontes",
        "Apocalypse Aster",
        "Apocalypse Emadikeis",
    )
    originals = (
        "黙示烙ポレモファイロス",
        "黙示烙アディクヒュドルス",
        "黙示烙サナトアネモス",
        "黙示烙レウコブロンテス",
        "黙示烙アステルセン",
        "黙示烙エマディケイス",
    )
    items = [
        _series_item(name, original, f"apocalypse-{element}", element, "weapon")
        for name, original, element in zip(names, originals, elements)
    ]
    loader = lambda object_type: items if object_type == "weapon" else []

    evidence = retrieve_entity(
        EntityQuery(
            mention="Apocalypse",
            object_type="weapon",
            retrieval_query="List the Apocalypse weapon series",
        ),
        ["weapon"],
        loader,
    )

    assert {item["element"] for item in evidence} == set(elements)
    assert all(item["series_key"] == "weapon:apocalypse" for item in evidence)
    assert all(item["series_coverage_complete"] for item in evidence)
    assert all(item["series_missing_elements"] == [] for item in evidence)


def test_deterministic_planner_collapses_series_to_one_entity():
    elements = ("fire", "water", "wind", "thunder", "light", "dark")
    originals = (
        "黙示烙ポレモファイロス",
        "黙示烙アディクヒュドルス",
        "黙示烙サナトアネモス",
        "黙示烙レウコブロンテス",
        "黙示烙アステルセン",
        "黙示烙エマディケイス",
    )
    items = [
        _series_item(
            f"Apocalypse {element.title()}",
            original,
            f"apocalypse-{element}",
            element,
            "weapon",
        )
        for original, element in zip(originals, elements)
    ]
    loader = lambda object_type: items if object_type == "weapon" else []

    plan = deterministic_plan(
        "Tell me about the Apocalypse weapon series",
        {},
        loader,
    )

    assert len(plan.entities) == 1
    assert plan.entities[0].mention == "Apocalypse"
    assert plan.entities[0].object_type == "weapon"


def test_series_coverage_exposes_missing_element(monkeypatch):
    monkeypatch.setenv("KAMI_RAG_RERANK", "0")
    elements = ("fire", "water", "wind", "light", "dark")
    originals = (
        "六欲天アーケーンルスト",
        "六欲天ディープグラトニー",
        "六欲天ブリーズエンヴィー",
        "六欲天ホーリーグリード",
        "六欲天ダークプライド",
    )
    items = [
        _series_item(
            f"Six Desires Heaven {element.title()}",
            original,
            f"six-desires-{element}",
            element,
            "weapon",
        )
        for original, element in zip(originals, elements)
    ]
    loader = lambda object_type: items if object_type == "weapon" else []

    evidence = retrieve_entity(
        EntityQuery(
            mention="Six Desires Heaven",
            object_type="weapon",
            retrieval_query="List the Six Desires Heaven weapon series",
        ),
        ["weapon"],
        loader,
    )

    assert {item["element"] for item in evidence} == set(elements)
    assert all(not item["series_coverage_complete"] for item in evidence)
    assert all(item["series_missing_elements"] == ["thunder"] for item in evidence)


def test_eidolon_series_is_derived_from_japanese_source_suffix():
    metadata = series_metadata("eidolon", "炎天獄カタストロフィア")

    assert metadata["series_key"] == "eidolon:catastrophe"
    assert metadata["series_name"] == "Catastrophe"


def test_broad_weapon_and_kamihime_profiles_keep_type_specific_sections(monkeypatch):
    monkeypatch.setenv("KAMI_RAG_RERANK", "0")
    items = {
        "kamihime": [_complete_item("Alpha", "alpha")],
        "weapon": [_complete_item("Test Blade", "test-blade", object_type="weapon")],
    }
    loader = lambda object_type: items.get(object_type, [])

    kamihime = retrieve_entity(
        EntityQuery(mention="Alpha", object_type="kamihime", retrieval_query="Tell me about Alpha"),
        ["kamihime"],
        loader,
    )
    weapon = retrieve_entity(
        EntityQuery(mention="Test Blade", object_type="weapon", retrieval_query="Tell me about Test Blade"),
        ["weapon"],
        loader,
    )

    assert {item["section"] for item in kamihime} == {
        "basic",
        "Burst",
        "Ability",
        "Assist",
    }
    assert {item["section"] for item in weapon} == {
        "basic",
        "Stats",
        "Burst Effects",
        "Weapon Skills",
    }
    assert all(item["coverage_complete"] for item in kamihime + weapon)


def test_specific_field_query_is_partial_without_claiming_database_sections_missing(monkeypatch):
    monkeypatch.setenv("KAMI_RAG_RERANK", "0")
    item = _complete_item("Phoenix", "phoenix", object_type="eidolon")
    loader = lambda object_type: [item] if object_type == "eidolon" else []

    evidence = retrieve_entity(
        EntityQuery(
            mention="Phoenix",
            object_type="eidolon",
            retrieval_query="What are Phoenix stats?",
        ),
        ["eidolon"],
        loader,
    )

    assert {item["section"] for item in evidence} == {"basic", "Stats"}
    assert evidence[0]["coverage_complete"] is False
    assert set(evidence[0]["available_sections"]) == {
        "basic",
        "Stats",
        "Summon Effect",
        "Main Effect",
        "Sub Effect",
    }


def test_retrieval_keeps_seven_same_name_object_variants(monkeypatch):
    items = [
        _item(
            "Nike" if index == 0 else f"[Variant {index}] Nike",
            f"nike-{index}",
            ("fire", "water", "wind", "thunder", "light", "dark")[index % 6],
        )
        for index in range(9)
    ]
    monkeypatch.setenv("KAMI_RAG_RERANK", "0")
    loader = lambda object_type: items if object_type == "kamihime" else []

    evidence = retrieve_entity(
        EntityQuery(
            mention="Nike",
            name="Nike",
            object_type="kamihime",
            retrieval_query="Nike healing skill",
        ),
        ["kamihime"],
        loader,
    )

    variants = {(item["object_type"], item["slug"]) for item in evidence}
    assert len(variants) == OBJECT_CANDIDATES == 7


def test_deterministic_planner_routes_eidolon_database():
    loader = lambda object_type: (
        [_item("Phoenix", "phoenix", object_type="eidolon")]
        if object_type == "eidolon"
        else []
    )

    plan = deterministic_plan("What does the Eidolon Phoenix do?", {}, loader)

    assert plan.in_domain is True
    assert plan.target_types == ["eidolon"]
    assert plan.entities[0].object_type == "eidolon"


def test_agent_retrieves_each_entity_before_synthesis(monkeypatch):
    monkeypatch.setenv("KAMI_RAG_RERANK", "0")
    items = [_item("Alpha", "alpha"), _item("Beta", "beta", "water")]
    loader = lambda object_type: items if object_type == "kamihime" else []
    contexts = []

    def answer(_provider, _model, _session_id, _message, context):
        contexts.append(context)
        return "Comparison complete."

    result = run_agent(
        session_id="thread-1",
        client_id="client-1",
        provider="gpt",
        model="test-model",
        message="Compare Alpha and Beta characters",
        history=[],
        memory_state={},
        answer_callback=answer,
        loader=loader,
    )

    assert result["answer"] == "Comparison complete."
    assert {source["name"] for source in result["sources"]} == {"Alpha", "Beta"}
    assert "Name: Alpha" in contexts[0]
    assert "Name: Beta" in contexts[0]


def test_sqlite_memory_persists_focus_and_long_term_state(tmp_path):
    path = tmp_path / "chat_sessions.json"
    state = {
        "focus_entities": [{"name": "Nike", "object_type": "kamihime"}],
        "long_term": {"language": "vi", "last_provider": "deepseek"},
    }
    memory.append_exchange(
        path,
        "thread-1",
        "client-1",
        {"role": "user", "content": "Nike là ai?", "created_at": memory.now_iso()},
        {
            "role": "assistant",
            "content": "Nike là một Kamihime.",
            "created_at": memory.now_iso(),
            "sources": [],
        },
        state,
    )

    assert memory.get_state(path, "thread-1")["focus_entities"][0]["name"] == "Nike"
    assert memory.get_client_state(path, "client-1")["language"] == "vi"
    assert path.exists()
    assert memory.database_path(path).exists()


def test_deepseek_is_exposed_as_chat_provider():
    providers = {item.provider: item for item in available_models()}

    assert providers["deepseek"].model == "deepseek-chat"


def test_embedding_device_auto_falls_back_to_cpu_without_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    assert resolve_embedding_device("auto") == "cpu"
    assert resolve_embedding_device("cpu") == "cpu"
    with pytest.raises(RuntimeError, match="CUDA-enabled PyTorch"):
        resolve_embedding_device("cuda")


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("tìm thông tin về Oni Gen 2", "vi"),
        ("tim thong tin ve Oni Gen 2", "vi"),
        ("Find information about Oni Gen 2", "en"),
    ],
)
def test_response_language_detection(message, expected):
    assert detect_response_language(message) == expected


def test_vietnamese_language_guard_retries_wrong_english_draft(monkeypatch):
    item = _complete_item("Nike", "nike")
    loader = lambda object_type: [item] if object_type == "kamihime" else []
    calls = []
    monkeypatch.setenv("KAMI_AGENT_DISABLE_LLM_PLANNER", "1")

    def answer(_provider, _model, _session, question, _context):
        calls.append(question)
        if len(calls) == 1:
            return "There are six members in this series."
        return "Đây là câu trả lời bằng tiếng Việt."

    result = run_agent(
        session_id="language-guard",
        client_id="test-client",
        provider="gpt",
        model="test-model",
        message="Hãy cho tôi biết thông tin về Kamihime Nike",
        history=[],
        memory_state={},
        answer_callback=answer,
        loader=loader,
    )

    assert result["answer"] == "Đây là câu trả lời bằng tiếng Việt."
    assert len(calls) == 2
    assert "Required response language: Vietnamese (vi)" in calls[0]
    assert "wrong language" in calls[1]


def test_deterministic_refusal_uses_original_user_language(monkeypatch):
    monkeypatch.setenv("KAMI_AGENT_DISABLE_LLM_PLANNER", "1")
    result = run_agent(
        session_id="language-refusal",
        client_id="test-client",
        provider="gpt",
        model="test-model",
        message="Thời tiết hôm nay như thế nào?",
        history=[],
        memory_state={},
        answer_callback=lambda *_args: pytest.fail("model must not be called"),
        loader=lambda _object_type: [],
    )

    assert result["answer"].startswith("Tôi chỉ có thể trả lời")


def test_series_alias_typo_resolves_all_members():
    items = []
    for element in ("fire", "water", "wind", "thunder", "light", "dark"):
        item = _item(f"Oni {element}", f"oni-{element}", element, "eidolon")
        item.update(
            series_key="eidolon:oni-gen-2",
            series_name="Oni Gen 2",
            series_aliases=["Demon Armor", "Oni Outfit"],
        )
        items.append(item)
    loader = lambda object_type: items if object_type == "eidolon" else []

    variants = resolve_object_variants("Demon Amor", ["eidolon"], loader=loader)
    plan = deterministic_plan("Tell me about Demon Amor", {}, loader)

    assert len(variants) == 6
    assert plan.entities[0].name == "Oni Gen 2"


def _series_items(series_key: str, series_name: str, alias: str):
    items = []
    for element in ("fire", "water", "wind", "thunder", "light", "dark"):
        item = _complete_item(
            f"{series_name} {element}",
            f"{series_key.rsplit(':', 1)[-1]}-{element}",
            element,
            "eidolon",
        )
        item.update(
            series_key=series_key,
            series_name=series_name,
            series_aliases=[series_name, alias, "Oni"],
            series_expected_elements=["fire", "water", "wind", "thunder", "light", "dark"],
            series_lifecycle="complete",
            series_catalog_elements=["fire", "water", "wind", "thunder", "light", "dark"],
            series_catalog_member_count=6,
        )
        items.append(item)
    return items


def test_exact_generation_suppresses_fuzzy_neighbor_series():
    items = _series_items("eidolon:oni-gen-1", "Oni Gen 1", "Oni base")
    items += _series_items("eidolon:oni-gen-2", "Oni Gen 2", "Demon Armor")
    loader = lambda object_type: items if object_type == "eidolon" else []

    variants = resolve_object_variants("Oni Gen 1", ["eidolon"], loader=loader)
    plan = deterministic_plan("Tell me about Eidolon Oni Gen 1", {}, loader)

    assert {item["series_key"] for item, _score in variants} == {"eidolon:oni-gen-1"}
    assert len(variants) == 6
    assert [(entity.name, entity.object_type) for entity in plan.entities] == [
        ("Oni Gen 1", "eidolon")
    ]


def test_exact_series_grounding_corrects_wrong_llm_planner(monkeypatch):
    items = _series_items("eidolon:oni-gen-1", "Oni Gen 1", "Oni base")
    items += _series_items("eidolon:oni-gen-2", "Oni Gen 2", "Demon Armor")
    loader = lambda object_type: items if object_type == "eidolon" else []
    wrong_plan = QueryPlan(
        in_domain=True,
        standalone_question="Find information about the Oni Gen 2 eidolon series",
        target_types=["eidolon"],
        entities=[
            EntityQuery(
                mention="Oni Gen 1",
                name="Oni Gen 1",
                object_type="eidolon",
                retrieval_query="Oni Gen 1 series",
            )
        ],
    )
    monkeypatch.setattr(
        "kami.agent.graph.model_info",
        lambda _provider: type("Info", (), {"configured": True})(),
    )
    monkeypatch.setattr(
        "kami.agent.graph.plan_with_model",
        lambda *_args, **_kwargs: wrong_plan.model_copy(deep=True),
    )

    result = run_agent(
        session_id="planner-grounding",
        client_id="test-client",
        provider="deepseek",
        model="deepseek-chat",
        message="Find information about the Eidolon series Oni Gen 2",
        history=[],
        memory_state={},
        answer_callback=lambda *_args: "done",
        loader=loader,
    )

    assert [(entity.name, entity.series_key) for entity in result["plan"].entities] == [
        ("Oni Gen 2", "eidolon:oni-gen-2")
    ]
    assert {source["series_key"] for source in result["sources"]} == {
        "eidolon:oni-gen-2"
    }
    assert len(result["sources"]) == 6


def test_generic_series_hydrates_all_sections_and_marks_shared_effects():
    items = _series_items("eidolon:oni-gen-1", "Oni Gen 1", "Oni base")
    loader = lambda object_type: items if object_type == "eidolon" else []

    evidence = retrieve_entity(
        EntityQuery(
            mention="Oni Gen 1",
            name="Oni Gen 1",
            object_type="eidolon",
            retrieval_query="Find information about Eidolon Oni Gen 1",
        ),
        ["eidolon"],
        loader,
    )

    assert len({item["slug"] for item in evidence}) == 6
    assert {item["section"] for item in evidence} == {
        "basic",
        "Stats",
        "Summon Effect",
        "Main Effect",
        "Sub Effect",
    }
    assert {item["selection_mode"] for item in evidence} == {"full_series"}
    assert all(item["coverage_complete"] for item in evidence)
    assert any(item["effect_is_shared"] for item in evidence)


def test_large_series_uses_fair_budgeted_overview(monkeypatch):
    items = _series_items("eidolon:oni-gen-1", "Oni Gen 1", "Oni base")
    for item in items:
        for row in item["eidolon_effects"]:
            row["effect"] = str(row.get("effect") or "") + " detailed" * 250
    loader = lambda object_type: items if object_type == "eidolon" else []
    monkeypatch.setenv("KAMI_RAG_SERIES_CONTEXT_CHARS", "4000")

    evidence = retrieve_entity(
        EntityQuery(
            mention="Oni Gen 1",
            name="Oni Gen 1",
            object_type="eidolon",
            retrieval_query="Overview of Oni Gen 1",
        ),
        ["eidolon"],
        loader,
    )

    assert {item["selection_mode"] for item in evidence} == {"budgeted_overview"}
    assert {item["section"] for item in evidence} >= {"basic", "Stats"}
    assert all(item["omitted_sections"] for item in evidence)
    assert all("context budget" in item["omission_reason"] for item in evidence)


def test_series_context_references_repeated_shared_effects(monkeypatch):
    items = _series_items("eidolon:oni-gen-1", "Oni Gen 1", "Oni base")
    loader = lambda object_type: items if object_type == "eidolon" else []
    contexts = []
    monkeypatch.setenv("KAMI_AGENT_DISABLE_LLM_PLANNER", "1")

    run_agent(
        session_id="series-summary",
        client_id="test-client",
        provider="gpt",
        model="test-model",
        message="Find information about Eidolon Oni Gen 1",
        history=[],
        memory_state={},
        answer_callback=lambda _p, _m, _s, _q, context: (
            contexts.append(context) or "done"
        ),
        loader=loader,
    )

    assert "summarize the shared mechanic once" in contexts[0]
    assert "shared mechanic already shown" in contexts[0]


def test_member_followup_prefers_the_focused_series():
    gen1 = _complete_item("Ibaraki Doji", "ibaraki-doji", object_type="eidolon")
    gen1.update(
        series_key="eidolon:oni-gen-1",
        series_name="Oni Gen 1",
        series_aliases=["Oni Gen 1", "Oni"],
    )
    gen2 = _complete_item(
        "[Demon Armor] Ibaraki Doji",
        "demon-armor-ibaraki-doji",
        object_type="eidolon",
    )
    gen2.update(
        series_key="eidolon:oni-gen-2",
        series_name="Oni Gen 2",
        series_aliases=["Oni Gen 2", "Demon Armor", "Oni"],
    )
    items = [gen1, gen2]
    loader = lambda object_type: items if object_type == "eidolon" else []
    memory_state = {
        "focus_entities": [
            {
                "name": "Ibaraki Doji",
                "object_type": "eidolon",
                "series_key": "eidolon:oni-gen-1",
            }
        ]
    }

    plan = deterministic_plan(
        "Show the main effect of Ibaraki Doji",
        memory_state,
        loader,
    )
    evidence = retrieve_entity(plan.entities[0], ["eidolon"], loader)

    assert [(entity.name, entity.object_type) for entity in plan.entities] == [
        ("Ibaraki Doji", "eidolon")
    ]
    assert {(item["name"], item["series_key"]) for item in evidence} == {
        ("Ibaraki Doji", "eidolon:oni-gen-1")
    }
    assert {item["section"] for item in evidence} == {"basic", "Main Effect"}


def test_explicit_member_name_wins_over_series_alias_in_its_title(monkeypatch):
    gen1 = _complete_item("Ibaraki Doji", "ibaraki-doji", object_type="eidolon")
    gen1.update(
        series_key="eidolon:oni-gen-1",
        series_name="Oni Gen 1",
        series_aliases=["Oni Gen 1", "Oni"],
    )
    gen2 = _complete_item(
        "[Demon Armor: Flame Dance] Ibaraki Doji",
        "demon-armor-flame-dance-ibaraki-doji",
        object_type="eidolon",
    )
    gen2.update(
        series_key="eidolon:oni-gen-2",
        series_name="Oni Gen 2",
        series_aliases=["Oni Gen 2", "Demon Armor", "Oni"],
    )
    loader = lambda object_type: [gen1, gen2] if object_type == "eidolon" else []
    wrong_plan = QueryPlan(
        in_domain=True,
        standalone_question="Show the main effect of Demon Armor Ibaraki Doji",
        target_types=["eidolon"],
        entities=[EntityQuery(mention="Oni Gen 2", object_type="eidolon")],
    )
    monkeypatch.setattr(
        "kami.agent.graph.model_info",
        lambda _provider: type("Info", (), {"configured": True})(),
    )
    monkeypatch.setattr(
        "kami.agent.graph.plan_with_model",
        lambda *_args, **_kwargs: wrong_plan.model_copy(deep=True),
    )

    result = run_agent(
        session_id="member-grounding",
        client_id="test-client",
        provider="deepseek",
        model="deepseek-chat",
        message="Show the main effect of [Demon Armor: Flame Dance] Ibaraki Doji",
        history=[],
        memory_state={},
        answer_callback=lambda *_args: "done",
        loader=loader,
    )

    assert [(entity.name, entity.series_key) for entity in result["plan"].entities] == [
        ("[Demon Armor: Flame Dance] Ibaraki Doji", "eidolon:oni-gen-2")
    ]
    assert {(source["slug"], source["series_key"]) for source in result["sources"]} == {
        ("demon-armor-flame-dance-ibaraki-doji", "eidolon:oni-gen-2")
    }


def test_candidate_routing_uses_weapon_limit_24():
    items = []
    for index in range(30):
        item = _item(f"Future Weapon {index}", f"future-{index}", object_type="weapon")
        item.update(
            series_key="weapon:future",
            series_name="Future Arsenal",
            series_aliases=["Future Arsenal"],
        )
        items.append(item)
    loader = lambda object_type: items if object_type == "weapon" else []

    variants = resolve_object_variants("Future Arsenal", ["weapon"], loader=loader)

    assert len(variants) == OBJECT_CANDIDATES_BY_TYPE["weapon"] == 24


def test_reconcile_only_auto_attaches_high_confidence_changed_series(tmp_path):
    records = []
    for element in ("fire", "water", "wind", "thunder"):
        records.append(
            {
                "info": {
                    "name": f"新系・{element}",
                    "source_url": f"https://example.test/{element}",
                    "element": element,
                }
            }
        )
    candidates = detect_series_candidates(records, "weapon")
    assert candidates[0]["confidence"] == "high"

    path = tmp_path / "weapon" / "fire" / "raw.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )
    result = reconcile_series_data(
        tmp_path,
        "weapon",
        changed_source_urls=["https://example.test/fire"],
        allow_auto_attach=True,
    )
    updated = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

    assert result["auto_attached"] == 4
    assert {record["info"]["series_detection"] for record in updated} == {"auto-high"}
