from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable

from langgraph.graph import END, START, StateGraph

from ..data_store import load_catalog_items
from .catalog_query import CatalogSelection, select_latest_catalog_items
from .documents import CatalogLoader, OBJECT_TYPES
from .language import detect_response_language, guarded_question, language_name
from .providers import model_info, plan_with_model
from .retrieval import (
    exact_series_matches,
    contains_normalized_phrase,
    hydrate_catalog_items,
    normalize_text,
    retrieve_entity,
    series_alias_score,
)
from .schemas import AgentState, EntityQuery, QueryPlan
from .section_scope import detect_requested_sections


DOMAIN_WORDS = {
    "kamihime",
    "kami",
    "eidolon",
    "eidolons",
    "weapon",
    "weapons",
    "character",
    "characters",
    "skill",
    "skills",
    "ability",
    "abilities",
    "burst",
    "assist",
    "element",
    "fire",
    "water",
    "wind",
    "thunder",
    "light",
    "dark",
    "phantom",
    "rarity",
    "attack",
    "damage",
    "healer",
    "gacha",
    "than thu",
    "vu khi",
    "nhan vat",
    "ky nang",
    "nguyen to",
    "he lua",
    "he nuoc",
}
FOLLOWUP_WORDS = {
    "she",
    "her",
    "he",
    "him",
    "it",
    "that character",
    "that weapon",
    "that eidolon",
    "co ay",
    "nhan vat do",
    "vu khi do",
    "than thu do",
    "no",
}


def _target_types(message: str) -> list[str]:
    normalized = normalize_text(message)
    values: list[str] = []
    if any(word in normalized for word in ("eidolon", "eidolons", "than thu")):
        values.append("eidolon")
    if any(word in normalized for word in ("weapon", "weapons", "vu khi")):
        values.append("weapon")
    if any(word in normalized for word in ("kamihime", "character", "characters", "nhan vat", "kami")):
        values.append("kamihime")
    return list(dict.fromkeys(values))


ELEMENT_ALIASES = {
    "fire": ("fire", "he lua"),
    "water": ("water", "he nuoc"),
    "wind": ("wind", "he gio"),
    "thunder": ("thunder", "lightning", "he set"),
    "light": ("light", "he anh sang"),
    "dark": ("dark", "he bong toi"),
    "phantom": ("phantom", "he ao"),
}
LATEST_PHRASES = ("moi nhat", "latest", "newest", "most recent")


def _deterministic_elements(message: str) -> list[str]:
    normalized = normalize_text(message)
    return [
        element
        for element, aliases in ELEMENT_ALIASES.items()
        if any(contains_normalized_phrase(normalized, alias) for alias in aliases)
    ]


def _latest_requested(message: str) -> bool:
    normalized = normalize_text(message)
    return any(
        contains_normalized_phrase(normalized, phrase)
        for phrase in LATEST_PHRASES
    )


def _ground_query_constraints(plan: QueryPlan, message: str) -> QueryPlan:
    detected_types = _target_types(message)
    detected_elements = _deterministic_elements(message)
    if detected_types:
        plan.target_types = detected_types
    if detected_elements:
        plan.elements = detected_elements
    if _latest_requested(message):
        plan.intent = "filter"
        plan.sort_by = "release_date"
        plan.sort_order = "desc"
        plan.result_limit = 1
        plan.include_ties = True
        if not detected_types and not plan.entities:
            plan.target_types = []
            plan.needs_clarification = True
            plan.clarification_question = (
                "Please specify Kamihime, Eidolon, or Weapon."
            )
        else:
            plan.needs_clarification = False
            plan.clarification_question = None
    plan.requested_sections = detect_requested_sections(
        message,
        plan.target_types or detected_types,
    )
    return plan


def _deterministic_entities(
    message: str,
    target_types: list[str],
    memory_state: dict,
    loader: CatalogLoader,
) -> list[EntityQuery]:
    normalized_message = normalize_text(message)
    matches: dict[str, tuple[str, str | None, str | None]] = {}
    types = target_types or list(OBJECT_TYPES)
    matched_series: set[str] = set()
    focus_series_keys = {
        str(item.get("series_key") or "")
        for item in memory_state.get("focus_entities") or []
        if isinstance(item, dict) and item.get("series_key")
    }

    exact_matches = exact_series_matches(message, types, loader)
    for series_key, (series_name, object_type) in exact_matches.items():
        matched_series.add(series_key)
        matches[f"series:{series_key}"] = (
            series_name,
            object_type,
            series_key,
        )

    if not exact_matches:
        for object_type in types:
            for item in loader(object_type):
                series_key = str(item.get("series_key") or "")
                series_name = str(item.get("series_name") or "")
                aliases = list(item.get("series_aliases") or [])
                if series_name and series_name not in aliases:
                    aliases.append(series_name)
                if not series_key or series_key in matched_series:
                    continue
                if series_alias_score(message, aliases) >= 0.88:
                    matched_series.add(series_key)
                    matches[f"series:{series_key}"] = (
                        series_name or str(aliases[0]),
                        object_type,
                        series_key,
                    )

    for object_type in types:
        for item in loader(object_type):
            if str(item.get("series_key") or "") in matched_series:
                continue
            name = str(item.get("name") or "")
            normalized_name = normalize_text(name)
            base_name = normalize_text(name.split("]", 1)[-1])
            selected = ""
            if contains_normalized_phrase(normalized_message, normalized_name):
                selected = name
            elif contains_normalized_phrase(normalized_message, base_name):
                if (
                    focus_series_keys
                    and item.get("series_key")
                    and str(item.get("series_key")) not in focus_series_keys
                ):
                    continue
                selected = name.split("]", 1)[-1].strip()
            if selected:
                key = normalize_text(selected)
                current = matches.get(key)
                if current is None:
                    matches[key] = (
                        selected,
                        object_type,
                        str(item.get("series_key") or "") or None,
                    )
                elif current[1] != object_type and current[1] != "kamihime":
                    matches[key] = (
                        selected,
                        object_type,
                        str(item.get("series_key") or "") or None,
                    )

    if not matches and any(word in normalized_message for word in FOLLOWUP_WORDS):
        for focus in memory_state.get("focus_entities") or []:
            if not isinstance(focus, dict):
                continue
            name = str(focus.get("name") or "")
            if name:
                matches[normalize_text(name)] = (
                    name,
                    str(focus.get("object_type") or "") or None,
                    str(focus.get("series_key") or "") or None,
                )

    return [
        EntityQuery(
            mention=name,
            name=name,
            object_type=(object_type if object_type in OBJECT_TYPES else None),
            series_key=series_key,
            retrieval_query=f"{name}: {message}",
        )
        for name, object_type, series_key in matches.values()
    ]


def deterministic_plan(
    message: str,
    memory_state: dict,
    loader: CatalogLoader = load_catalog_items,
) -> QueryPlan:
    target_types = _target_types(message)
    entities = _deterministic_entities(message, target_types, memory_state, loader)
    if not target_types:
        target_types = list(
            dict.fromkeys(
                entity.object_type
                for entity in entities
                if entity.object_type is not None
            )
        )
    normalized = normalize_text(message)
    domain = bool(
        entities
        or any(word in normalized for word in DOMAIN_WORDS)
        or (
            memory_state.get("focus_entities")
            and any(word in normalized for word in FOLLOWUP_WORDS)
        )
    )
    focus_names = [
        str(item.get("name") or "")
        for item in memory_state.get("focus_entities") or []
        if isinstance(item, dict) and item.get("name")
    ]
    standalone = message
    if not entities and focus_names and any(
        word in normalized for word in FOLLOWUP_WORDS
    ):
        standalone = f"{message} Context entities: {', '.join(focus_names)}"
    intent = "compare" if any(word in normalized for word in ("compare", "versus", " vs ", "so sanh")) else "lookup"
    plan = QueryPlan(
        in_domain=domain,
        standalone_question=standalone,
        intent=intent,
        target_types=target_types,
        entities=entities,
    )
    return _ground_query_constraints(plan, message)


def _explicit_object_matches(
    message: str,
    object_types: list[str],
    loader: CatalogLoader,
) -> list[dict[str, Any]]:
    normalized_message = normalize_text(message)
    matches: list[dict[str, Any]] = []
    for object_type in object_types:
        for item in loader(object_type):
            normalized_name = normalize_text(str(item.get("name") or ""))
            if not normalized_name:
                continue
            exact_message = normalized_name == normalized_message
            descriptive_name = len(normalized_name.split()) >= 2
            if exact_message or (
                descriptive_name
                and contains_normalized_phrase(normalized_message, normalized_name)
            ):
                matches.append(item)
    normalized_names = {
        id(item): normalize_text(str(item.get("name") or ""))
        for item in matches
    }
    return [
        item
        for item in matches
        if not any(
            normalized_names[id(item)] != normalized_names[id(other)]
            and contains_normalized_phrase(
                normalized_names[id(other)],
                normalized_names[id(item)],
            )
            for other in matches
        )
    ]


def _ground_model_entities(
    plan: QueryPlan,
    message: str,
    loader: CatalogLoader,
) -> QueryPlan:
    detected_types = _target_types(message)
    object_types = list(plan.target_types) or detected_types or list(OBJECT_TYPES)
    explicit_objects = _explicit_object_matches(message, object_types, loader)
    if explicit_objects:
        plan.entities = [
            EntityQuery(
                mention=str(item.get("name") or ""),
                name=str(item.get("name") or ""),
                object_type=str(item.get("object_type") or ""),
                series_key=str(item.get("series_key") or "") or None,
                retrieval_query=(
                    f"{item.get('name')}: {plan.standalone_question}"
                ),
            )
            for item in explicit_objects
        ]
        plan.target_types = list(
            dict.fromkeys(
                item.object_type
                for item in plan.entities
                if item.object_type is not None
            )
        )
        plan.in_domain = True
        plan.needs_clarification = False
        plan.clarification_question = None
        return plan

    exact_series = exact_series_matches(message, object_types, loader)
    if not exact_series:
        if _latest_requested(message):
            plan.entities = _deterministic_entities(
                message,
                detected_types,
                {},
                loader,
            )
            if plan.entities:
                plan.target_types = list(
                    dict.fromkeys(
                        entity.object_type
                        for entity in plan.entities
                        if entity.object_type is not None
                    )
                )
                plan.in_domain = True
                plan.needs_clarification = False
                plan.clarification_question = None
        return plan
    plan.entities = [
        EntityQuery(
            mention=series_name,
            name=series_name,
            object_type=object_type,
            series_key=series_key,
            retrieval_query=f"{series_name}: {plan.standalone_question}",
        )
        for series_key, (series_name, object_type) in exact_series.items()
    ]
    plan.target_types = list(
        dict.fromkeys(
            entity.object_type
            for entity in plan.entities
            if entity.object_type is not None
        )
    )
    plan.in_domain = True
    plan.needs_clarification = False
    plan.clarification_question = None
    return plan


def _plan_node(state: AgentState, loader: CatalogLoader) -> dict[str, Any]:
    info = model_info(state["provider"])
    if info.configured and os.getenv("KAMI_AGENT_DISABLE_LLM_PLANNER", "0") != "1":
        plan = plan_with_model(
            state["provider"],
            state["model"],
            state["message"],
            state.get("history", []),
            state.get("memory_state", {}),
        )
        for entity in plan.entities:
            if not entity.retrieval_query:
                entity.retrieval_query = (
                    f"{entity.name or entity.mention}: {plan.standalone_question}"
                )
        plan = _ground_model_entities(plan, state["message"], loader)
        plan = _ground_query_constraints(plan, state["message"])
    else:
        plan = deterministic_plan(
            state["message"],
            state.get("memory_state", {}),
            loader,
        )
    return {"plan": plan}


def _after_plan(state: AgentState) -> str:
    plan = state["plan"]
    if not plan.in_domain:
        return "refuse"
    if plan.needs_clarification and plan.clarification_question:
        return "clarify"
    return "retrieve"


def _state_language(state: AgentState) -> str:
    fallback = str(
        (state.get("memory_state", {}).get("long_term") or {}).get("language")
        or "en"
    )
    return detect_response_language(str(state.get("message") or ""), fallback)


def _refuse_node(state: AgentState) -> dict[str, Any]:
    answer = (
        "Tôi chỉ có thể trả lời các câu hỏi về nhân vật, Eidolon, Weapon và "
        "dữ liệu game của Kamihime Project."
        if _state_language(state) == "vi"
        else (
            "I can only answer questions about Kamihime Project characters, "
            "Eidolons, Weapons, and their game data."
        )
    )
    return {
        "answer": answer,
        "evidence": [],
        "sources": [],
    }


def _clarify_node(state: AgentState) -> dict[str, Any]:
    answer = (
        "Vui lòng nói rõ bạn đang muốn hỏi đối tượng Kamihime Project nào."
        if _state_language(state) == "vi"
        else state["plan"].clarification_question
        or "Please clarify which Kamihime Project object you mean."
    )
    return {
        "answer": answer,
        "evidence": [],
        "sources": [],
    }


def _catalog_selection_note(
    selection: CatalogSelection,
    object_types: list[str],
) -> str:
    winner_counts: dict[str, int] = {}
    for item in selection.items:
        object_type = str(item.get("object_type") or "")
        winner_counts[object_type] = winner_counts.get(object_type, 0) + 1

    lines = ["Selected by maximum release date per object type:"]
    for object_type in dict.fromkeys(object_types):
        label = object_type.title()
        latest = selection.latest_dates.get(object_type)
        matching_count = selection.matching_counts.get(object_type, 0)
        valid_date_count = selection.valid_date_counts.get(object_type, 0)
        if latest is not None:
            winner_count = winner_counts.get(object_type, 0)
            count_text = (
                f"{winner_count} tied newest records"
                if winner_count != 1
                else "1 newest record"
            )
            lines.append(
                f"{label}: maximum release date {latest.isoformat()} "
                f"({count_text})."
            )
        elif matching_count and not valid_date_count:
            lines.append(
                f"{label}: matching records have no valid release dates."
            )
        else:
            lines.append(f"{label}: no matching catalog records.")
    return "\n".join(lines) + "\n"


def _retrieve_node(state: AgentState, loader: CatalogLoader) -> dict[str, Any]:
    plan = state["plan"]
    selection_note = ""
    retrieval_issue = ""
    if plan.sort_by == "release_date" and not plan.entities:
        selection = select_latest_catalog_items(
            list(plan.target_types),
            list(plan.elements),
            loader,
        )
        if not selection.items:
            retrieval_issue = (
                "release_date_unavailable"
                if selection.matching_count and not selection.valid_date_count
                else "no_matching_catalog_records"
            )
            return {
                "evidence": [],
                "sources": [],
                "context": "",
                "retrieval_issue": retrieval_issue,
            }
        evidence = hydrate_catalog_items(
            selection.items,
            plan.standalone_question,
        )
        selection_note = _catalog_selection_note(
            selection,
            list(plan.target_types),
        )
    else:
        entities = plan.entities or [
            EntityQuery(
                mention=plan.standalone_question,
                retrieval_query=plan.standalone_question,
            )
        ]
        workers = min(len(entities), max(1, int(os.getenv("KAMI_RAG_ENTITY_WORKERS", "4"))))
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(
                    retrieve_entity,
                    entity,
                    list(plan.target_types),
                    loader,
                )
                for entity in entities
            ]
            evidence = [item for future in futures for item in future.result()]

    unique: list[dict] = []
    seen_docs: set[tuple[str, str, str, str]] = set()
    for item in evidence:
        key = (
            item["object_type"],
            item["slug"],
            item["section"],
            item["content"],
        )
        if key not in seen_docs:
            seen_docs.add(key)
            unique.append(item)

    source_map: dict[tuple[str, str], dict] = {}
    for item in unique:
        key = (item["object_type"], item["slug"])
        source = source_map.setdefault(
            key,
            {
                "name": item["name"],
                "object_type": item["object_type"],
                "slug": item["slug"],
                "element": item["element"],
                "section": item["section"],
                "sections": [],
                "available_sections": item.get("available_sections", []),
                "coverage_complete": item.get("coverage_complete", False),
                "retrieval_mode": item.get("retrieval_mode", "semantic"),
                "selection_mode": item.get("selection_mode", "full_entity"),
                "omitted_sections": item.get("omitted_sections", []),
                "omission_reason": item.get("omission_reason", ""),
                "series_key": item.get("series_key", ""),
                "series_name": item.get("series_name", ""),
                "series_elements": item.get("series_elements", []),
                "series_expected_elements": item.get(
                    "series_expected_elements", []
                ),
                "series_lifecycle": item.get("series_lifecycle", "complete"),
                "series_catalog_elements": item.get("series_catalog_elements", []),
                "series_catalog_member_count": item.get("series_catalog_member_count", 0),
                "series_retrieved_member_count": item.get("series_retrieved_member_count", 0),
                "series_unreleased_elements": item.get("series_unreleased_elements", []),
                "series_missing_elements": item.get(
                    "series_missing_elements", []
                ),
                "series_coverage_complete": item.get(
                    "series_coverage_complete", False
                ),
                "local_url": item["local_url"],
                "score": item["score"],
            },
        )
        if item["section"] not in source["sections"]:
            source["sections"].append(item["section"])
        if item["score"] > source["score"]:
            source["score"] = item["score"]
            source["section"] = item["section"]
    sources = sorted(source_map.values(), key=lambda item: item["score"], reverse=True)
    evidence_groups: dict[tuple[str, str], list[dict]] = {}
    for item in unique:
        evidence_groups.setdefault(
            (item["object_type"], item["slug"]), []
        ).append(item)
    context_blocks: list[str] = []
    shared_effect_owner: dict[str, int] = {}
    for index, items in enumerate(evidence_groups.values(), start=1):
        first = items[0]
        available = first.get("available_sections") or [
            item["section"] for item in items
        ]
        included = first.get("included_sections") or [
            item["section"] for item in items
        ]
        coverage = "complete" if first.get("coverage_complete") else "partial"
        selection_lines = (
            f"Context selection mode: {first.get('selection_mode', 'full_entity')}\n"
            f"Intentionally omitted sections: "
            f"{', '.join(first.get('omitted_sections') or []) or 'none'}\n"
            f"Omission reason: {first.get('omission_reason') or 'none'}\n"
        )
        series_lines = ""
        if (
            first.get("series_key")
            and first.get("selection_mode") != "catalog_latest"
        ):
            series_coverage = (
                "complete"
                if first.get("series_coverage_complete")
                else "partial"
            )
            series_lines = (
                f"Series: {first.get('series_name')} "
                f"({first.get('series_key')})\n"
                "Observed series elements: "
                f"{', '.join(first.get('series_elements') or [])}\n"
                "Expected series elements: "
                f"{', '.join(first.get('series_expected_elements') or [])}\n"
                f"Series lifecycle: {first.get('series_lifecycle') or 'complete'}\n"
                "Catalog series elements: "
                f"{', '.join(first.get('series_catalog_elements') or [])}\n"
                "Unreleased series elements: "
                f"{', '.join(first.get('series_unreleased_elements') or []) or 'none'}\n"
                "Missing series elements: "
                f"{', '.join(first.get('series_missing_elements') or []) or 'none'}\n"
                f"Series coverage: {series_coverage}\n"
                "Series answer policy: Compare effects across members. When the "
                "mechanics are the same apart from element, names, wording, or "
                "translation variation, summarize the shared mechanic once and "
                "list only meaningful differences. Give full per-member effect "
                "progression only when the user explicitly asks for those members.\n"
            )
        section_blocks: list[str] = []
        for item in items:
            effect_group_id = str(item.get("effect_group_id") or "")
            if item.get("effect_is_shared") and effect_group_id:
                owner = shared_effect_owner.get(effect_group_id)
                if owner is not None:
                    section_blocks.append(
                        f"{item['section']}: shared mechanic already shown in S{owner}; "
                        "only element, effect name, or translation wording differs."
                    )
                    continue
                shared_effect_owner[effect_group_id] = index
            section_blocks.append(item["content"])
        sections = "\n\n".join(section_blocks)
        context_blocks.append(
            f"[S{index}] {first['object_type'].title()} / {first['name']} / "
            f"{first['element']}\n"
            f"Available database sections: {', '.join(available)}\n"
            f"Included sections: {', '.join(included)}\n"
            f"Evidence coverage: {coverage}\n"
            f"{selection_lines}"
            f"{series_lines}"
            f"Retrieval mode: {first.get('retrieval_mode', 'semantic')}\n\n"
            f"{sections}"
        )
    context = "\n\n".join(context_blocks)
    if selection_note:
        context = f"{selection_note}\n{context}"
    summary = str(state.get("memory_state", {}).get("summary") or "")
    if summary:
        context = f"Conversation summary:\n{summary}\n\n{context}"
    return {
        "evidence": unique,
        "sources": sources,
        "context": context,
        "retrieval_issue": retrieval_issue,
    }


def _answer_node(
    state: AgentState,
    answer_callback: Callable[[str, str, str, str, str], str],
) -> dict[str, Any]:
    language = _state_language(state)
    question = guarded_question(
        str(state.get("message") or ""),
        state["plan"].standalone_question,
        language,
    )
    answer = answer_callback(
        state["provider"],
        state["model"],
        state["session_id"],
        question,
        state.get("context", ""),
    )
    if detect_response_language(answer, language) != language:
        selected = language_name(language)
        answer = answer_callback(
            state["provider"],
            state["model"],
            state["session_id"],
            (
                "The previous draft used the wrong language. Rewrite it "
                f"entirely in {selected} ({language}) without changing facts.\n\n"
                f"Previous draft:\n{answer}\n\n{question}"
            ),
            state.get("context", ""),
        )
    return {"answer": answer}


def _after_retrieve(state: AgentState) -> str:
    return "answer" if state.get("evidence") else "missing"


def _missing_node(state: AgentState) -> dict[str, Any]:
    if state.get("retrieval_issue") == "release_date_unavailable":
        answer = (
            "Các bản ghi phù hợp chưa có ngày phát hành hợp lệ để xác định đối tượng mới nhất."
            if _state_language(state) == "vi"
            else "The matching records do not have valid release dates, so the newest object cannot be determined."
        )
    else:
        answer = (
            "Cơ sở dữ liệu Kamihime Project cục bộ không có đủ thông tin để trả lời "
            "câu hỏi này."
            if _state_language(state) == "vi"
            else (
                "The local Kamihime Project database does not contain enough "
                "information to answer that question."
            )
        )
    return {
        "answer": answer
    }


def run_agent(
    *,
    session_id: str,
    client_id: str,
    provider: str,
    model: str,
    message: str,
    history: list[dict],
    memory_state: dict,
    answer_callback: Callable[[str, str, str, str, str], str],
    loader: CatalogLoader = load_catalog_items,
) -> dict[str, Any]:
    builder = StateGraph(AgentState)
    builder.add_node("plan", lambda state: _plan_node(state, loader))
    builder.add_node("refuse", _refuse_node)
    builder.add_node("clarify", _clarify_node)
    builder.add_node("retrieve", lambda state: _retrieve_node(state, loader))
    builder.add_node(
        "answer",
        lambda state: _answer_node(state, answer_callback),
    )
    builder.add_node("missing", _missing_node)
    builder.add_edge(START, "plan")
    builder.add_conditional_edges(
        "plan",
        _after_plan,
        {"refuse": "refuse", "clarify": "clarify", "retrieve": "retrieve"},
    )
    builder.add_conditional_edges(
        "retrieve",
        _after_retrieve,
        {"answer": "answer", "missing": "missing"},
    )
    for node in ("refuse", "clarify", "answer", "missing"):
        builder.add_edge(node, END)
    graph = builder.compile()
    return graph.invoke(
        {
            "session_id": session_id,
            "client_id": client_id,
            "provider": provider,
            "model": model,
            "message": message,
            "history": history,
            "memory_state": memory_state,
        }
    )
