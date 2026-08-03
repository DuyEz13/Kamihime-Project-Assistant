# Latest Catalog Query Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make latest/newest queries select the exact newest catalog records for the requested object type and element, including every same-date tie.

**Architecture:** Extend `QueryPlan` with explicit ordering constraints and deterministically ground latest/element phrases from the original message. Route release-date ordering to a new cached-catalog selector instead of vector RAG, then hydrate only the selected records through the existing Evidence pipeline.

**Tech Stack:** Python 3.11, Pydantic, LangGraph, pytest, existing JSONL catalog loader and agentic RAG document builders.

## Global Constraints

- Latest selection applies independently to Kamihime, Eidolon, and Weapon.
- Every object on the maximum valid release date is returned.
- Named-object and series candidate limits remain 7/7/24.
- Vector search is not called for latest catalog queries.
- Supported date formats are `YY/MM/DD`, `YYYY/MM/DD`, `YY-MM-DD`, and `YYYY-MM-DD`.
- Two-digit release years are interpreted as `2000 + year`.
- Invalid dates are ignored; the system must not guess through semantic fallback.
- No RAG index schema change or index rebuild is required.

---

### Task 1: Represent and ground latest query constraints

**Files:**
- Modify: `kami/agent/schemas.py`
- Modify: `kami/agent/graph.py`
- Modify: `kami/agent/providers.py`
- Test: `tests/test_agentic_rag.py`

**Interfaces:**
- Consumes: `normalize_text(text: str) -> str` and `contains_normalized_phrase(text: str, phrase: str) -> bool` from `kami.agent.retrieval`.
- Produces: `QueryPlan.sort_by`, `QueryPlan.sort_order`, `QueryPlan.result_limit`, `QueryPlan.include_ties`, `_deterministic_elements(message: str) -> list[str]`, and `_ground_query_constraints(plan: QueryPlan, message: str) -> QueryPlan`.

- [ ] **Step 1: Write failing planner tests**

Add these imports and tests to `tests/test_agentic_rag.py`:

```python
from kami.agent.graph import _ground_query_constraints


def test_vietnamese_latest_query_has_exact_catalog_constraints():
    plan = deterministic_plan(
        "tìm thông tin về kamihime mới nhất hệ nước",
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
        "tìm thông tin về kamihime mới nhất hệ nước",
    )

    assert grounded.elements == ["water"]
    assert grounded.sort_by == "release_date"
    assert grounded.result_limit == 1
    assert grounded.include_ties is True


def test_latest_query_without_object_type_requests_clarification():
    plan = deterministic_plan(
        "dữ liệu hệ nước mới nhất",
        {},
        lambda _object_type: [],
    )

    assert plan.needs_clarification is True
    assert plan.clarification_question
```

- [ ] **Step 2: Run planner tests and verify the new behavior is absent**

Run:

```powershell
uv run pytest tests/test_agentic_rag.py -k "latest_query_has_exact or latest_phrases_are_grounded or grounding_corrects_model or latest_query_without" -v
```

Expected: failures because the ordering fields and grounding functions do not exist.

- [ ] **Step 3: Extend `QueryPlan` with validated selection fields**

Add to `QueryPlan` in `kami/agent/schemas.py`:

```python
    sort_by: Literal["relevance", "release_date"] = "relevance"
    sort_order: Literal["asc", "desc"] = "desc"
    result_limit: int | None = Field(default=None, ge=1, le=100)
    include_ties: bool = False
```

- [ ] **Step 4: Add deterministic element/latest detection and grounding**

Add constants and helpers near `_target_types` in `kami/agent/graph.py`:

```python
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
        if not plan.target_types:
            plan.needs_clarification = True
            plan.clarification_question = (
                "Please specify Kamihime, Eidolon, or Weapon."
            )
    return plan
```

Build the deterministic plan into a local variable and pass it through `_ground_query_constraints`. In `_plan_node`, call the same grounding helper after the LLM/entity-grounding branch so explicit user constraints override malformed model output.

- [ ] **Step 5: Document the structured fields in the planner prompt**

Append this rule to `PLANNER_SYSTEM_PROMPT` in `kami/agent/providers.py`:

```text
For latest, newest, most recent, or equivalent Vietnamese requests, set
sort_by=release_date, sort_order=desc, result_limit=1, and include_ties=true.
Extract the requested element even when no named entity is present.
```

- [ ] **Step 6: Run planner tests and the existing planner subset**

Run:

```powershell
uv run pytest tests/test_agentic_rag.py -k "planner or latest or grounding" -v
```

Expected: all selected tests pass.

- [ ] **Step 7: Commit the planner deliverable**

```powershell
git add kami/agent/schemas.py kami/agent/graph.py kami/agent/providers.py tests/test_agentic_rag.py
git commit -m "feat: plan exact latest catalog queries"
```

---

### Task 2: Select maximum catalog dates with stable tie handling

**Files:**
- Create: `kami/agent/catalog_query.py`
- Test: `tests/test_catalog_query.py`

**Interfaces:**
- Consumes: `CatalogLoader = Callable[[str], list[dict[str, Any]]]` from `kami.agent.documents`.
- Produces: `CatalogSelection`, `parse_release_date(value: object) -> date | None`, and `select_latest_catalog_items(object_types: Sequence[str], elements: Sequence[str], loader: CatalogLoader) -> CatalogSelection`.

- [ ] **Step 1: Write date parser and catalog selection tests**

Create `tests/test_catalog_query.py`:

```python
from datetime import date

import pytest

from kami.agent.catalog_query import (
    parse_release_date,
    select_latest_catalog_items,
)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("26/07/30", date(2026, 7, 30)),
        ("2026/07/30", date(2026, 7, 30)),
        ("26-07-30", date(2026, 7, 30)),
        ("2026-07-30", date(2026, 7, 30)),
        ("-", None),
        ("2026/13/40", None),
    ],
)
def test_parse_release_date(value, expected):
    assert parse_release_date(value) == expected


def test_select_latest_filters_type_and_element_and_keeps_all_ties():
    records = {
        "kamihime": [
            {"name": "Older Water", "slug": "older", "object_type": "kamihime", "element": "water", "release_date": "26/06/01"},
            {"name": "Latest B", "slug": "latest-b", "object_type": "kamihime", "element": "water", "release_date": "26/07/30"},
            {"name": "Latest A", "slug": "latest-a", "object_type": "kamihime", "element": "water", "release_date": "2026/07/30"},
            {"name": "Newer Fire", "slug": "fire", "object_type": "kamihime", "element": "fire", "release_date": "26/08/01"},
        ]
    }
    selection = select_latest_catalog_items(
        ["kamihime"],
        ["water"],
        lambda object_type: records.get(object_type, []),
    )

    assert [item["slug"] for item in selection.items] == ["latest-a", "latest-b"]
    assert selection.matching_count == 3
    assert selection.valid_date_count == 3
    assert selection.latest_dates == {"kamihime": date(2026, 7, 30)}


def test_select_latest_reports_matching_records_without_valid_dates():
    item = {"name": "Unknown", "slug": "unknown", "object_type": "eidolon", "element": "water", "release_date": "-"}
    selection = select_latest_catalog_items(
        ["eidolon"],
        ["water"],
        lambda _object_type: [item],
    )

    assert selection.items == []
    assert selection.matching_count == 1
    assert selection.valid_date_count == 0


def test_select_latest_computes_a_maximum_per_explicit_object_type():
    records = {
        "kamihime": [{"name": "K", "slug": "k", "object_type": "kamihime", "element": "water", "release_date": "26/07/30"}],
        "weapon": [{"name": "W", "slug": "w", "object_type": "weapon", "element": "water", "release_date": "26/06/01"}],
    }
    selection = select_latest_catalog_items(
        ["kamihime", "weapon"],
        ["water"],
        lambda object_type: records.get(object_type, []),
    )

    assert {item["slug"] for item in selection.items} == {"k", "w"}
```

- [ ] **Step 2: Run the selector tests and verify import failure**

Run:

```powershell
uv run pytest tests/test_catalog_query.py -v
```

Expected: collection fails because `kami.agent.catalog_query` does not exist.

- [ ] **Step 3: Implement the focused catalog selector**

Create `kami/agent/catalog_query.py` with this structure:

```python
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from typing import Any, Sequence

from .documents import CatalogLoader


@dataclass(frozen=True)
class CatalogSelection:
    items: list[dict[str, Any]]
    matching_count: int
    valid_date_count: int
    latest_dates: dict[str, date]


def parse_release_date(value: object) -> date | None:
    match = re.fullmatch(
        r"\s*(\d{2}|\d{4})[/-](\d{1,2})[/-](\d{1,2})\s*",
        str(value or ""),
    )
    if not match:
        return None
    year, month, day = (int(part) for part in match.groups())
    if year < 100:
        year += 2000
    try:
        return date(year, month, day)
    except ValueError:
        return None


def select_latest_catalog_items(
    object_types: Sequence[str],
    elements: Sequence[str],
    loader: CatalogLoader,
) -> CatalogSelection:
    selected_elements = {value.casefold() for value in elements if value}
    matching_count = 0
    valid_date_count = 0
    latest_dates: dict[str, date] = {}
    winners: dict[str, list[dict[str, Any]]] = {}

    for object_type in dict.fromkeys(object_types):
        for item in loader(object_type):
            element = str(item.get("element") or "").casefold()
            if selected_elements and element not in selected_elements:
                continue
            matching_count += 1
            released = parse_release_date(item.get("release_date"))
            if released is None:
                continue
            valid_date_count += 1
            current = latest_dates.get(object_type)
            if current is None or released > current:
                latest_dates[object_type] = released
                winners[object_type] = [item]
            elif released == current:
                winners.setdefault(object_type, []).append(item)

    items = [
        item
        for object_type in dict.fromkeys(object_types)
        for item in sorted(
            winners.get(object_type, []),
            key=lambda value: (
                str(value.get("name") or "").casefold(),
                str(value.get("slug") or ""),
            ),
        )
    ]
    return CatalogSelection(
        items=items,
        matching_count=matching_count,
        valid_date_count=valid_date_count,
        latest_dates=latest_dates,
    )
```

- [ ] **Step 4: Run selector tests**

Run:

```powershell
uv run pytest tests/test_catalog_query.py -v
```

Expected: all tests pass.

- [ ] **Step 5: Commit the selector deliverable**

```powershell
git add kami/agent/catalog_query.py tests/test_catalog_query.py
git commit -m "feat: select latest catalog records"
```

---

### Task 3: Route latest plans around vector RAG and hydrate exact evidence

**Files:**
- Modify: `kami/agent/schemas.py`
- Modify: `kami/agent/retrieval.py`
- Modify: `kami/agent/graph.py`
- Modify: `kami/agent/providers.py`
- Test: `tests/test_agentic_rag.py`

**Interfaces:**
- Consumes: `CatalogSelection` and `select_latest_catalog_items(...)` from Task 2; `QueryPlan` fields from Task 1.
- Produces: `hydrate_catalog_items(items: list[dict[str, Any]], query: str, retrieval_mode: str = "catalog_latest") -> list[Evidence]` and `AgentState.retrieval_issue`.

- [ ] **Step 1: Write failing integration tests for exact retrieval**

Add to `tests/test_agentic_rag.py`:

```python
def test_latest_agent_returns_only_max_date_ties_without_vector_search(monkeypatch):
    monkeypatch.setenv("KAMI_AGENT_DISABLE_LLM_PLANNER", "1")
    newest_a = _complete_item("Newest A", "newest-a", "water")
    newest_b = _complete_item("Newest B", "newest-b", "water")
    older = _complete_item("Older", "older", "water")
    wrong_element = _complete_item("New Fire", "new-fire", "fire")
    newest_a["release_date"] = "26/07/30"
    newest_b["release_date"] = "2026/07/30"
    older["release_date"] = "26/06/01"
    wrong_element["release_date"] = "26/08/01"
    items = [older, wrong_element, newest_b, newest_a]
    loader = lambda object_type: items if object_type == "kamihime" else []

    def fail_vector_search(*_args, **_kwargs):
        raise AssertionError("vector search must not run")

    monkeypatch.setattr("kami.agent.retrieval._qdrant_search", fail_vector_search)
    contexts = []
    result = run_agent(
        session_id="latest-1",
        client_id="client-1",
        provider="gpt",
        model="test-model",
        message="tìm thông tin về kamihime mới nhất hệ nước",
        history=[],
        memory_state={},
        answer_callback=lambda *_args: contexts.append(_args[-1]) or "Kết quả.",
        loader=loader,
    )

    assert [source["name"] for source in result["sources"]] == ["Newest A", "Newest B"]
    assert all(source["retrieval_mode"] == "catalog_latest" for source in result["sources"])
    assert "Selected by maximum release date: 2026-07-30" in contexts[0]
    assert "Name: Older" not in contexts[0]


@pytest.mark.parametrize(
    ("object_type", "message", "expected_sections"),
    [
        ("kamihime", "latest water kamihime", {"basic", "Burst", "Ability", "Assist"}),
        ("eidolon", "latest water eidolon", {"basic", "Stats", "Summon Effect", "Main Effect", "Sub Effect"}),
        ("weapon", "latest water weapon", {"basic", "Stats", "Burst Effects", "Weapon Skills"}),
    ],
)
def test_latest_catalog_hydrates_all_type_sections(monkeypatch, object_type, message, expected_sections):
    monkeypatch.setenv("KAMI_AGENT_DISABLE_LLM_PLANNER", "1")
    item = _complete_item("Newest", "newest", "water", object_type)
    loader = lambda selected: [item] if selected == object_type else []
    result = run_agent(
        session_id=f"latest-{object_type}",
        client_id="client-1",
        provider="gpt",
        model="test-model",
        message=message,
        history=[],
        memory_state={},
        answer_callback=lambda *_args: "Done.",
        loader=loader,
    )

    assert set(result["sources"][0]["sections"]) == expected_sections


def test_latest_catalog_with_only_invalid_dates_reports_date_problem(monkeypatch):
    monkeypatch.setenv("KAMI_AGENT_DISABLE_LLM_PLANNER", "1")
    item = _complete_item("Unknown", "unknown", "water")
    item["release_date"] = "-"
    result = run_agent(
        session_id="latest-invalid",
        client_id="client-1",
        provider="gpt",
        model="test-model",
        message="tìm kamihime mới nhất hệ nước",
        history=[],
        memory_state={},
        answer_callback=lambda *_args: pytest.fail("answer model must not run"),
        loader=lambda object_type: [item] if object_type == "kamihime" else [],
    )

    assert result["sources"] == []
    assert "ngày phát hành" in result["answer"].casefold()
```

- [ ] **Step 2: Run integration tests and verify current retrieval returns the wrong candidate shape**

Run:

```powershell
uv run pytest tests/test_agentic_rag.py -k "latest_agent_returns or latest_catalog_hydrates or latest_catalog_with_only" -v
```

Expected: failures because latest plans still enter entity/hybrid retrieval.

- [ ] **Step 3: Add a public exact-item hydration helper**

Add to `kami/agent/retrieval.py` after `_annotate_series_coverage`:

```python
def hydrate_catalog_items(
    items: list[dict[str, Any]],
    query: str,
    retrieval_mode: str = "catalog_latest",
) -> list[Evidence]:
    evidence: list[Evidence] = []
    for item in items:
        evidence.extend(
            _hydrate_item(
                item,
                query,
                200.0,
                retrieval_mode,
                selection_mode=retrieval_mode,
            )
        )
    return _annotate_series_coverage(evidence)
```

- [ ] **Step 4: Route release-date plans before semantic retrieval**

In `kami/agent/graph.py`, import `select_latest_catalog_items` and
`hydrate_catalog_items`. At the start of `_retrieve_node`, branch on
`plan.sort_by == "release_date"`:

```python
    selection_note = ""
    retrieval_issue = ""
    if plan.sort_by == "release_date":
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
        dates = ", ".join(
            sorted({value.isoformat() for value in selection.latest_dates.values()})
        )
        selection_note = f"Selected by maximum release date: {dates}\n"
    else:
        # Keep the existing entity/thread-pool retrieval block unchanged.
```

Move only the existing entity/thread-pool block under the `else`; keep evidence
deduplication, source construction, and context construction shared by both
paths. Prefix each latest context group with `selection_note`.

Do not render series-coverage instructions when
`selection_mode == "catalog_latest"`; latest selection is not a series overview
and intentionally excludes older series members.

- [ ] **Step 5: Add typed retrieval issue state and localized missing-date copy**

Add to `AgentState` in `kami/agent/schemas.py`:

```python
    retrieval_issue: str
```

Update `_missing_node` in `kami/agent/graph.py` so
`release_date_unavailable` returns:

```python
answer = (
    "Các bản ghi phù hợp chưa có ngày phát hành hợp lệ để xác định đối tượng mới nhất."
    if _state_language(state) == "vi"
    else "The matching records do not have valid release dates, so the newest object cannot be determined."
)
```

Keep the existing insufficient-data copy for other missing cases.

- [ ] **Step 6: Reinforce the answer policy for exact catalog selection**

Append to `ANSWER_SYSTEM_PROMPT` in `kami/agent/providers.py`:

```text
When evidence says it was selected by maximum release date, answer only from
those selected records. All same-date records are tied for newest; do not add
older semantic candidates.
```

- [ ] **Step 7: Run focused integration tests**

Run:

```powershell
uv run pytest tests/test_agentic_rag.py -k "latest or retrieval_keeps_seven or candidate_routing" -v
```

Expected: latest tests pass and existing candidate-limit tests still pass.

- [ ] **Step 8: Commit the retrieval integration**

```powershell
git add kami/agent/schemas.py kami/agent/retrieval.py kami/agent/graph.py kami/agent/providers.py tests/test_agentic_rag.py
git commit -m "feat: retrieve newest catalog objects exactly"
```

---

### Task 4: Full regression and current-data acceptance check

**Files:**
- Verify: `tests/test_agentic_rag.py`
- Verify: `tests/test_catalog_query.py`
- Verify: `kami/data/kamihime/water/translated/deepl.jsonl`

**Interfaces:**
- Consumes: completed latest planning, catalog selection, and evidence hydration.
- Produces: verified behavior against both fixtures and the current local catalog.

- [ ] **Step 1: Run the full automated test suite**

Run:

```powershell
uv run pytest -q
```

Expected: all tests pass with no regressions.

- [ ] **Step 2: Verify the current Water Kamihime maximum directly**

Run:

```powershell
@'
from kami.agent.catalog_query import select_latest_catalog_items
from kami.data_store import load_catalog_items

selection = select_latest_catalog_items(
    ["kamihime"],
    ["water"],
    load_catalog_items,
)
print(selection.latest_dates)
for item in selection.items:
    print(item["release_date"], item["name"], item["slug"])
'@ | uv run python -
```

Expected with the catalog reviewed on 2026-08-03:

```text
{'kamihime': datetime.date(2026, 7, 30)}
26/07/30 Procyon procyon
```

If newer data has been crawled, the expected result is instead every record on
the new maximum date.

- [ ] **Step 3: Verify no index rebuild is required**

Run:

```powershell
git diff fc870ec -- kami/agent/documents.py kami/agent/retrieval.py | Select-String "INDEX_SCHEMA_VERSION"
```

Expected: no `INDEX_SCHEMA_VERSION` change.

- [ ] **Step 4: Inspect final repository state**

Run:

```powershell
git status --short
git log -4 --oneline
```

Expected: only intentional implementation commits and no uncommitted source or
test changes.
