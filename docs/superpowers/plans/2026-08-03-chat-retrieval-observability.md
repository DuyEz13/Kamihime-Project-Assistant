# Chat Retrieval Scope and Observability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make generic Kamihime, Eidolon, and Weapon questions retrieve every logical feature while moving all retrieval/graph/model diagnostics into a rotating structured JSONL trace.

**Architecture:** Ground section intent deterministically from the original user message and carry it as structured `QueryPlan` data through retrieval. Build answer evidence and retrieval diagnostics as separate graph outputs, then use a request-scoped trace collector to record graph nodes, evidence decisions, provider model calls, timings, and reported token usage without adding diagnostics to the model's answer context.

**Tech Stack:** Python 3.11, Pydantic 2, LangChain 1, LangGraph 1, standard-library `logging.handlers.RotatingFileHandler`, pytest 8, uv

## Global Constraints

- Generic information requests must load every logical section for the selected Kamihime, Eidolon, or Weapon.
- Section scope must be derived from the original user message, never from an LLM rewrite.
- Filter and sorting terms such as element, rarity, weapon type, and latest release date must not narrow section scope.
- Internal coverage, omission, retrieval-limit, database-gap, score, and graph details must not appear in normal answer context.
- The trace must record selected evidence identities, sections, scores, coverage, omissions, missing-data diagnoses, graph flow, provider/model, timings, and provider-reported token usage.
- Raw prompts, full evidence content, and final answer content are disabled by default and enabled only by `KAMI_CHAT_TRACE_INCLUDE_CONTENT=1`.
- Missing token usage must be recorded as unavailable/null and must not be estimated.
- JSONL tracing must be local, request-scoped, concurrency-safe, rotating, and non-fatal when logging itself fails.
- Preserve the user's existing uncommitted `.gitignore` change for `docs/`; stage only the new chat-trace ignore rule when committing that file.
- Use existing project dependencies; do not add an external telemetry service or package.

---

## File Structure

- Create `kami/agent/section_scope.py`: deterministic multilingual section-intent detection and type-specific canonical section keys.
- Create `kami/agent/tracing.py`: trace schema assembly, node timing, model-call recording, redaction/content policy, JSONL rotation, and safe finalization.
- Modify `kami/agent/schemas.py`: add grounded per-type section scope and retrieval diagnostics to graph state.
- Modify `kami/agent/graph.py`: ground section scope from the original message, propagate it into retrieval, separate answer context from diagnostics, and instrument graph nodes.
- Modify `kami/agent/retrieval.py`: consume explicit structured section scope instead of parsing rewritten retrieval queries.
- Modify `kami/agent/providers.py`: extract provider-reported usage and emit one telemetry event per planner/answer invocation.
- Modify `kami/chatbot.py`: create/finalize one trace per chat turn and label answer-language retries.
- Modify `tests/test_agentic_rag.py`: regression coverage for generic/explicit sections, LLM rewrites, clean answer context, and retrieval diagnoses.
- Create `tests/test_chat_tracing.py`: trace writer, redaction, rotation-safe behavior, model usage, retry, and error-path tests.
- Modify `tests/test_chatbot.py`: chat-boundary trace integration and non-fatal logging tests.
- Modify `.gitignore`: ignore `kami/data/chat_traces.jsonl*` without staging the user's unrelated `docs/` line.
- Modify `README.md`: document only the primary chat-trace configuration switches.

---

### Task 1: Ground Explicit Section Intent from the Original Message

**Files:**
- Create: `kami/agent/section_scope.py`
- Modify: `kami/agent/schemas.py:30-49`
- Modify: `kami/agent/graph.py:119-141`
- Test: `tests/test_agentic_rag.py`

**Interfaces:**
- Produces: `detect_requested_sections(message: str, object_types: Iterable[str]) -> dict[str, list[str]]`
- Produces: `QueryPlan.requested_sections: dict[str, list[str]]`
- Canonical keys: Kamihime `basic|burst|ability|assist`; Eidolon `basic|stats|summon_effect|main_effect|sub_effect`; Weapon `basic|stats|burst_effects|weapon_skills`
- Empty/missing mapping for an object type means full-object retrieval.

- [ ] **Step 1: Write failing section-grounding tests**

Add tests that prove generic/filter wording is not explicit and concrete feature wording is explicit:

```python
from kami.agent.section_scope import detect_requested_sections


@pytest.mark.parametrize(
    ("message", "object_type"),
    [
        ("tìm thông tin về eidolon mới nhất hệ nước", "eidolon"),
        ("Find information about the newest water element eidolon", "eidolon"),
        ("latest dark kamihime", "kamihime"),
        ("newest water sword weapon", "weapon"),
    ],
)
def test_generic_and_filter_words_do_not_select_sections(message, object_type):
    assert detect_requested_sections(message, [object_type]) == {}


@pytest.mark.parametrize(
    ("message", "object_type", "expected"),
    [
        ("cho tôi stats và summon effect", "eidolon", ["stats", "summon_effect"]),
        ("main effect và sub effect", "eidolon", ["main_effect", "sub_effect"]),
        ("show burst and assist", "kamihime", ["burst", "assist"]),
        ("cho tôi ability", "kamihime", ["ability"]),
        ("weapon skills", "weapon", ["weapon_skills"]),
        ("burst effects", "weapon", ["burst_effects"]),
    ],
)
def test_explicit_section_names_are_grounded(message, object_type, expected):
    assert detect_requested_sections(message, [object_type]) == {
        object_type: expected
    }
```

Also add one test for explicit Basic fields using unambiguous phrases such as `ngày phát hành`, `acquisition method`, `thuộc hệ nào`, and `unlock weapon`; do not treat a bare `element` inside `water element eidolon` as Basic.

- [ ] **Step 2: Run the focused tests and verify the module/field is missing**

Run:

```powershell
uv run pytest tests/test_agentic_rag.py -k "sections_are_grounded or do_not_select_sections or basic_fields" -v
```

Expected: FAIL because `kami.agent.section_scope` and `QueryPlan.requested_sections` do not exist.

- [ ] **Step 3: Implement canonical aliases and deterministic detection**

Create `section_scope.py` with ordered canonical section maps and phrase matching based on normalized whole phrases:

```python
from collections.abc import Iterable

from .retrieval import contains_normalized_phrase, normalize_text

SECTION_ORDER = {
    "kamihime": ("basic", "burst", "ability", "assist"),
    "eidolon": ("basic", "stats", "summon_effect", "main_effect", "sub_effect"),
    "weapon": ("basic", "stats", "burst_effects", "weapon_skills"),
}


def detect_requested_sections(
    message: str,
    object_types: Iterable[str],
) -> dict[str, list[str]]:
    normalized = normalize_text(message)
    result: dict[str, list[str]] = {}
    for object_type in dict.fromkeys(object_types):
        matched = {
            section
            for section, phrases in SECTION_PHRASES.get(object_type, {}).items()
            if any(contains_normalized_phrase(normalized, phrase) for phrase in phrases)
        }
        if matched:
            result[object_type] = [
                section for section in SECTION_ORDER[object_type] if section in matched
            ]
    return result
```

Define this explicit alias set. Keep bare constraint words such as `element`, `water`, `rarity`, and `type` out of Basic detection; only the unambiguous phrases below select Basic:

```python
SECTION_PHRASES = {
    "kamihime": {
        "basic": (
            "basic data", "thong tin co ban", "release date", "ngay phat hanh",
            "acquisition method", "cach nhan", "unlock weapon", "mo khoa weapon",
            "preferred weapon", "vu khi ua thich", "what element", "which element",
            "thuoc he nao", "he gi", "max level", "cap toi da",
        ),
        "burst": ("burst", "ougi"),
        "ability": ("ability", "abilities", "active skill", "ky nang chu dong"),
        "assist": ("assist", "passive", "ky nang bi dong"),
    },
    "eidolon": {
        "basic": (
            "basic data", "thong tin co ban", "release date", "ngay phat hanh",
            "acquisition method", "cach nhan", "return items", "vat pham quy doi",
            "what element", "which element", "thuoc he nao", "he gi",
        ),
        "stats": ("stat", "stats", "chi so", "hp", "attack", "tan cong", "max level", "cap toi da"),
        "summon_effect": ("summon effect", "summoning effect", "hieu ung trieu hoi"),
        "main_effect": ("main effect", "hieu ung chinh"),
        "sub_effect": ("sub effect", "sub effects", "hieu ung phu"),
    },
    "weapon": {
        "basic": (
            "basic data", "thong tin co ban", "release date", "ngay phat hanh",
            "acquisition method", "cach nhan", "unlock kamihime", "mo khoa kamihime",
            "weapon type", "loai vu khi", "what element", "which element",
            "thuoc he nao", "he gi",
        ),
        "stats": ("stat", "stats", "chi so", "hp", "attack", "tan cong", "max level", "cap toi da"),
        "burst_effects": ("burst", "burst effect", "burst effects", "ougi"),
        "weapon_skills": ("weapon skill", "weapon skills", "ky nang vu khi"),
    },
}
```

Add to `QueryPlan`:

```python
requested_sections: dict[str, list[str]] = Field(default_factory=dict)
```

At the end of `_ground_query_constraints`, overwrite any LLM-provided value:

```python
plan.requested_sections = detect_requested_sections(
    message,
    plan.target_types or detected_types,
)
```

- [ ] **Step 4: Run the focused tests**

Run:

```powershell
uv run pytest tests/test_agentic_rag.py -k "sections_are_grounded or do_not_select_sections or basic_fields" -v
```

Expected: PASS.

- [ ] **Step 5: Commit the grounded section intent**

```powershell
git add kami/agent/section_scope.py kami/agent/schemas.py kami/agent/graph.py tests/test_agentic_rag.py
git commit -m "fix: ground chat section intent from user input"
```

---

### Task 2: Propagate Structured Section Scope Through Retrieval

**Files:**
- Modify: `kami/agent/retrieval.py:690-851, 987-1010, 1136-1148`
- Modify: `kami/agent/graph.py:513-561`
- Test: `tests/test_agentic_rag.py`

**Interfaces:**
- Consumes: `QueryPlan.requested_sections`
- Produces: `sections_for_type(requested_sections: dict[str, list[str]], object_type: str) -> set[str] | None`
- Changes: `hydrate_catalog_items(..., requested_sections: dict[str, list[str]] | None = None)`
- Changes: `retrieve_entity(..., requested_sections: dict[str, list[str]] | None = None)`
- `None` means all documents; an explicit set means only exact matching logical documents.

- [ ] **Step 1: Write failing end-to-end retrieval tests**

Add an LLM-planner regression that deliberately rewrites the query with `water element`:

```python
def test_llm_rewrite_cannot_narrow_generic_latest_eidolon(monkeypatch):
    item = _complete_item("The Little Mermaid", "the-little-mermaid", "water", "eidolon")
    model_plan = QueryPlan(
        in_domain=True,
        standalone_question="Find information about the newest water element eidolon",
        target_types=["eidolon"],
        elements=["water"],
        sort_by="release_date",
        include_ties=True,
    )
    monkeypatch.setattr("kami.agent.graph.model_info", lambda _provider: SimpleNamespace(configured=True))
    monkeypatch.setattr("kami.agent.graph.plan_with_model", lambda *_args, **_kwargs: model_plan)

    result = run_agent(
        session_id="latest-water-eidolon",
        client_id="client-1",
        provider="deepseek",
        model="deepseek-chat",
        message="tìm thông tin về eidolon mới nhất hệ nước",
        history=[],
        memory_state={},
        answer_callback=lambda *_args: "done",
        loader=lambda selected: [item] if selected == "eidolon" else [],
    )

    assert set(result["sources"][0]["sections"]) == {
        "basic", "Stats", "Summon Effect", "Main Effect", "Sub Effect"
    }
```

Add this parameterized explicit-section retrieval test and assert that no implicit Basic document is added:

```python
@pytest.mark.parametrize(
    ("object_type", "message", "expected_sections"),
    [
        ("kamihime", "show burst and assist", {"Burst", "Assist"}),
        ("eidolon", "show stats and summon effect", {"Stats", "Summon Effect"}),
        ("weapon", "show weapon skills", {"Weapon Skills"}),
    ],
)
def test_explicit_section_retrieval_does_not_add_basic(
    monkeypatch, object_type, message, expected_sections
):
    monkeypatch.setenv("KAMI_AGENT_DISABLE_LLM_PLANNER", "1")
    item = _complete_item("Target", "target", "water", object_type)
    result = run_agent(
        session_id=f"explicit-{object_type}",
        client_id="client-1",
        provider="gpt",
        model="test-model",
        message=f"{message} for {object_type} Target",
        history=[],
        memory_state={},
        answer_callback=lambda *_args: "done",
        loader=lambda selected: [item] if selected == object_type else [],
    )
    assert set(result["sources"][0]["sections"]) == expected_sections
```

- [ ] **Step 2: Run the new tests and verify the existing query parser causes failure**

```powershell
uv run pytest tests/test_agentic_rag.py -k "rewrite_cannot_narrow or explicit_section_retrieval" -v
```

Expected: the Eidolon test returns only `basic`, and explicit requests include unwanted Basic or parse the rewritten query.

- [ ] **Step 3: Replace query reparsing with structured scope**

Delete `_requested_sections(query, object_type)`. Add:

```python
def sections_for_type(
    requested_sections: dict[str, list[str]] | None,
    object_type: str,
) -> set[str] | None:
    values = (requested_sections or {}).get(object_type)
    return set(values) if values else None
```

Pass the mapping from `_retrieve_node` into both latest hydration and each entity retrieval. Update `_hydrate_item` to accept `requested: set[str] | None` and select exact requested logical documents:

```python
if requested is None and series_overview and allowed_series_sections is not None:
    selected_documents = [
        doc for doc in documents
        if _section_key(str(doc.metadata.get("section") or ""))
        in allowed_series_sections
    ]
elif requested is None:
    selected_documents = documents
else:
    selected_documents = [
        doc for doc in documents
        if _section_key(str(doc.metadata.get("section") or "")) in requested
    ]
```

Change `_series_section_policy` to consume the already-grounded requested set rather than a query string. Preserve series context budgeting only when no explicit scope exists.

- [ ] **Step 4: Run retrieval regressions**

```powershell
uv run pytest tests/test_agentic_rag.py -k "latest_catalog or explicit_section or generic_series or budgeted_overview or rewrite_cannot_narrow" -v
```

Expected: PASS, including all five Eidolon sections for the exact failing query.

- [ ] **Step 5: Commit structured retrieval scope**

```powershell
git add kami/agent/retrieval.py kami/agent/graph.py tests/test_agentic_rag.py
git commit -m "fix: propagate explicit section scope through retrieval"
```

---

### Task 3: Separate Answer Evidence from Retrieval Diagnostics

**Files:**
- Modify: `kami/agent/schemas.py:84-97`
- Modify: `kami/agent/graph.py:478-710, 750-769`
- Modify: `kami/agent/providers.py:172-204`
- Test: `tests/test_agentic_rag.py`

**Interfaces:**
- Produces: `AgentState.retrieval_diagnostics: dict[str, Any]`
- Produces: `_retrieval_diagnostics(plan, unique, sources, selection, retrieval_issue) -> dict[str, Any]`
- Answer `context` contains evidence content and source identity only; operational metadata exists solely in `retrieval_diagnostics` and source payloads needed by the UI.

- [ ] **Step 1: Write failing context/diagnostic separation tests**

Capture the answer callback context for the exact latest Eidolon case and assert:

```python
for forbidden in (
    "Available database sections:",
    "Included sections:",
    "Evidence coverage:",
    "Context selection mode:",
    "Intentionally omitted sections:",
    "Omission reason:",
    "Retrieval mode:",
    "Series coverage:",
):
    assert forbidden not in captured_context

diagnostics = result["retrieval_diagnostics"]
assert diagnostics["mode"] == "catalog_latest"
assert diagnostics["latest_dates"]["eidolon"] == "2026-06-22"
assert diagnostics["evidence"][0]["coverage_complete"] is True
assert diagnostics["evidence"][0]["omitted_sections"] == []
```

Add missing-data tests for `no_matching_catalog_records` and `release_date_unavailable`; require stable `reason_code` values in diagnostics while the public response does not mention graph, retrieval, coverage, or candidate limits.

- [ ] **Step 2: Run the tests and verify diagnostics are currently embedded in context**

```powershell
uv run pytest tests/test_agentic_rag.py -k "diagnostic or answer_context or no_matching_catalog" -v
```

Expected: FAIL because internal lines are in `context` and no `retrieval_diagnostics` state exists.

- [ ] **Step 3: Build a machine-readable retrieval diagnosis**

Add `retrieval_diagnostics: dict[str, Any]` to `AgentState`. Build a dictionary containing query constraints, latest-selection counts/dates/ties, reason code, and an evidence entry per selected object:

```python
{
    "mode": "catalog_latest",
    "reason_code": "selected_maximum_release_date",
    "matching_counts": dict(selection.matching_counts),
    "valid_date_counts": dict(selection.valid_date_counts),
    "latest_dates": {
        key: value.isoformat() for key, value in selection.latest_dates.items()
    },
    "selected_count": len(sources),
    "evidence": [
        {
            "name": source["name"],
            "object_type": source["object_type"],
            "slug": source["slug"],
            "sections": source["sections"],
            "available_sections": source["available_sections"],
            "coverage_complete": source["coverage_complete"],
            "omitted_sections": source["omitted_sections"],
            "omission_reason": source["omission_reason"],
            "selection_mode": source["selection_mode"],
            "score": source["score"],
            "local_url": source["local_url"],
        }
        for source in sources
    ],
}
```

For non-latest and series retrieval, copy these exact source fields into each diagnostic evidence entry:

```python
for key in (
    "series_key",
    "series_name",
    "series_lifecycle",
    "series_catalog_elements",
    "series_catalog_member_count",
    "series_retrieved_member_count",
    "series_missing_elements",
    "series_unreleased_elements",
    "series_coverage_complete",
):
    diagnostic_item[key] = source.get(key)
```

- [ ] **Step 4: Remove diagnostics from answer context and system instructions**

Replace the context header with source identity plus data content:

```python
context_blocks.append(
    f"[S{index}] {first['object_type'].title()} / {first['name']} / "
    f"{first['element']}\n\n{sections}"
)
```

Do not prepend `_catalog_selection_note`. Keep the shared-series summarization directive in the system prompt, but replace coverage-reporting instructions with:

```text
Never expose internal retrieval diagnostics, coverage flags, omitted-section
notes, ranking details, candidate limits, graph execution, or database pipeline
status unless the user explicitly asks to debug the system. Answer only with
the supplied game facts relevant to the question.
```

Make `_missing_node` return a concise localized “no matching information found” answer while retaining the exact failure reason in diagnostics.

- [ ] **Step 5: Run context and existing series/latest tests**

```powershell
uv run pytest tests/test_agentic_rag.py -k "diagnostic or context or latest or series or missing" -v
```

Expected: PASS; internal diagnostics exist in graph output but not answer context.

- [ ] **Step 6: Commit the public/internal context split**

```powershell
git add kami/agent/schemas.py kami/agent/graph.py kami/agent/providers.py tests/test_agentic_rag.py
git commit -m "fix: keep retrieval diagnostics out of chat answers"
```

---

### Task 4: Add a Request-Scoped Rotating JSONL Trace Collector

**Files:**
- Create: `kami/agent/tracing.py`
- Create: `tests/test_chat_tracing.py`

**Interfaces:**
- Produces: `ChatTrace(session_id: str, client_id: str, provider: str, model: str)`
- Produces: `ChatTrace.node(name: str)` context manager
- Produces: `record_plan(plan: QueryPlan)`, `record_retrieval(diagnostics: dict, evidence: list[Evidence])`, `record_model_call(event: dict)`, `record_response(...)`, `record_error(exc: Exception)`, and `finalize()`
- Produces: `create_chat_trace(...) -> ChatTrace | NullChatTrace`
- Uses: `KAMI_CHAT_TRACE_ENABLED`, `KAMI_CHAT_TRACE_PATH`, `KAMI_CHAT_TRACE_MAX_BYTES`, `KAMI_CHAT_TRACE_BACKUP_COUNT`, `KAMI_CHAT_TRACE_INCLUDE_CONTENT`

- [ ] **Step 1: Write failing collector tests**

Create tests using `tmp_path` and environment overrides:

```python
def test_trace_writes_one_redacted_json_record(tmp_path, monkeypatch):
    path = tmp_path / "chat_traces.jsonl"
    monkeypatch.setenv("KAMI_CHAT_TRACE_PATH", str(path))
    monkeypatch.setenv("KAMI_CHAT_TRACE_INCLUDE_CONTENT", "0")
    trace = create_chat_trace("session-1", "client-1", "deepseek", "deepseek-chat")

    with trace.node("retrieve"):
        trace.record_retrieval(
            {"mode": "catalog_latest", "reason_code": "selected_maximum_release_date"},
            [{"name": "The Little Mermaid", "section": "Stats", "content": "secret evidence"}],
        )
    trace.record_response("vi", "vi", "secret final answer", 0)
    trace.finalize()

    record = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    assert record["retrieval"]["evidence"][0]["name"] == "The Little Mermaid"
    assert "content" not in record["retrieval"]["evidence"][0]
    assert "answer" not in record["response"]
    assert record["graph"]["nodes"][0]["name"] == "retrieve"
```

Add these concrete cases:

```python
def test_trace_content_is_opt_in(tmp_path, monkeypatch):
    monkeypatch.setenv("KAMI_CHAT_TRACE_PATH", str(tmp_path / "trace.jsonl"))
    monkeypatch.setenv("KAMI_CHAT_TRACE_INCLUDE_CONTENT", "1")
    trace = create_chat_trace("s", "c", "gpt", "test-model")
    trace.record_retrieval({}, [{"name": "Nike", "section": "Ability", "content": "Restores HP"}])
    trace.record_response("en", "en", "Nike restores HP", 0)
    trace.finalize()
    record = json.loads((tmp_path / "trace.jsonl").read_text(encoding="utf-8"))
    assert record["retrieval"]["evidence"][0]["content"] == "Restores HP"
    assert record["response"]["answer"] == "Nike restores HP"


def test_disabled_trace_uses_noop_collector(tmp_path, monkeypatch):
    monkeypatch.setenv("KAMI_CHAT_TRACE_ENABLED", "0")
    monkeypatch.setenv("KAMI_CHAT_TRACE_PATH", str(tmp_path / "trace.jsonl"))
    trace = create_chat_trace("s", "c", "gpt", "test-model")
    with trace.node("plan"):
        pass
    trace.finalize()
    assert not (tmp_path / "trace.jsonl").exists()


def test_trace_aggregates_only_reported_tokens(tmp_path, monkeypatch):
    monkeypatch.setenv("KAMI_CHAT_TRACE_PATH", str(tmp_path / "trace.jsonl"))
    trace = create_chat_trace("s", "c", "deepseek", "deepseek-chat")
    trace.record_model_call({"step": "planner", "input_tokens": 10, "output_tokens": 2, "total_tokens": 12, "usage_available": True})
    trace.record_model_call({"step": "answer", "input_tokens": None, "output_tokens": None, "total_tokens": None, "usage_available": False})
    trace.finalize()
    record = json.loads((tmp_path / "trace.jsonl").read_text(encoding="utf-8"))
    assert record["totals"]["reported_total_tokens"] == 12
    assert record["totals"]["calls_with_usage"] == 1
```

For writer failure, monkeypatch the collector's `_write_record` to raise `OSError("disk full")`, call `finalize()`, assert no exception escapes, and assert `caplog` contains `disk full`. For exception recording, call `record_error(RuntimeError("provider failed"))` and assert `status == "error"`, `error.type == "RuntimeError"`, and raw traceback/prompt content is absent.

- [ ] **Step 2: Run the collector tests and verify the module is missing**

```powershell
uv run pytest tests/test_chat_tracing.py -k "trace" -v
```

Expected: FAIL because `kami.agent.tracing` does not exist.

- [ ] **Step 3: Implement trace records and safe JSONL rotation**

Use a request-local object with no shared mutable record. Implement the node timer as:

```python
@contextmanager
def node(self, name: str):
    started = time.perf_counter()
    status = "ok"
    error_type = None
    try:
        yield
    except Exception as exc:
        status = "error"
        error_type = type(exc).__name__
        raise
    finally:
        self._record["graph"]["nodes"].append({
            "name": name,
            "status": status,
            "duration_ms": round((time.perf_counter() - started) * 1000, 3),
            "error_type": error_type,
        })
```

Create one cached logger/`RotatingFileHandler` per resolved path, using a module lock only around handler creation and emission. Format each event as one compact UTF-8 JSON line. `finalize()` must be idempotent and catch writer errors.

When content logging is disabled, keep evidence identity/section/score/coverage fields but remove `content`; record prompt/answer lengths and hashes rather than text.

- [ ] **Step 4: Run all collector tests**

```powershell
uv run pytest tests/test_chat_tracing.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit the trace collector**

```powershell
git add kami/agent/tracing.py tests/test_chat_tracing.py
git commit -m "feat: add structured rotating chat traces"
```

---

### Task 5: Capture Planner and Answer Model Usage

**Files:**
- Modify: `kami/agent/providers.py:1-245`
- Modify: `kami/agent/graph.py:404-427, 713-743`
- Modify: `tests/test_chat_tracing.py`

**Interfaces:**
- Produces: `ModelTelemetry = Callable[[dict[str, Any]], None]`
- Produces: `extract_token_usage(message: Any) -> dict[str, int | bool | None]`
- Changes: `plan_with_model(..., telemetry: ModelTelemetry | None = None) -> QueryPlan`
- Changes: `answer_with_model(..., telemetry: ModelTelemetry | None = None, step: str = "answer") -> str`
- Existing parsed/string return types remain unchanged.

- [ ] **Step 1: Write failing usage extraction and emission tests**

Use fake LangChain messages to cover both metadata layouts:

```python
@pytest.mark.parametrize(
    ("message", "expected"),
    [
        (
            AIMessage(content="ok", usage_metadata={
                "input_tokens": 10, "output_tokens": 4, "total_tokens": 14
            }),
            {"input_tokens": 10, "output_tokens": 4, "total_tokens": 14},
        ),
        (
            AIMessage(content="ok", response_metadata={"token_usage": {
                "prompt_tokens": 8, "completion_tokens": 3, "total_tokens": 11
            }}),
            {"input_tokens": 8, "output_tokens": 3, "total_tokens": 11},
        ),
    ],
)
def test_extract_token_usage_supports_provider_metadata(message, expected):
    usage = extract_token_usage(message)
    assert {key: usage[key] for key in expected} == expected
    assert usage["usage_available"] is True
```

Add a missing-usage assertion:

```python
def test_extract_token_usage_does_not_estimate_missing_values():
    usage = extract_token_usage(AIMessage(content="ok"))
    assert usage == {
        "input_tokens": None,
        "output_tokens": None,
        "total_tokens": None,
        "usage_available": False,
    }
```

Mock `create_chat_model` with a fake whose answer `invoke` returns an `AIMessage` containing usage and whose planner `with_structured_output(..., include_raw=True)` returns this exact value:

```python
{
    "raw": AIMessage(
        content="",
        usage_metadata={"input_tokens": 20, "output_tokens": 5, "total_tokens": 25},
    ),
    "parsed": QueryPlan(
        in_domain=True,
        standalone_question="Tell me about Nike",
        target_types=["kamihime"],
    ),
    "parsing_error": None,
}
```

Assert emitted events contain `step`, `provider`, `model`, `status == "ok"`, non-negative `duration_ms`, and the exact usage values above.

- [ ] **Step 2: Run provider telemetry tests and verify failure**

```powershell
uv run pytest tests/test_chat_tracing.py -k "token_usage or model_telemetry" -v
```

Expected: FAIL because token extraction and telemetry parameters do not exist.

- [ ] **Step 3: Implement provider-neutral usage collection**

Add a timed invocation helper that always emits an event in `finally`. Use `include_raw=True` for planner structured output so the parsed plan is returned normally while usage comes from the raw AI message:

```python
structured = llm.with_structured_output(QueryPlan, include_raw=True)
result = structured.invoke(messages)
raw = result.get("raw") if isinstance(result, dict) else None
parsed = result.get("parsed") if isinstance(result, dict) else result
emit_model_telemetry(
    telemetry,
    step="planner",
    provider=provider,
    model=model,
    response=raw,
    started=started,
    error=error,
)
return parsed if isinstance(parsed, QueryPlan) else QueryPlan.model_validate(parsed)
```

For answers, extract usage from the returned AI message before converting its content to a string. Do not estimate absent values.

- [ ] **Step 4: Pass planner telemetry through the graph without changing test callbacks**

Extend `_plan_node` and `run_agent` with an optional telemetry callback. Keep the existing five-argument answer callback contract so existing test fakes remain valid. Answer call labeling will be provided by the per-request wrapper in Task 6.

- [ ] **Step 5: Run provider and agent tests**

```powershell
uv run pytest tests/test_chat_tracing.py -k "token_usage or model_telemetry" -v
uv run pytest tests/test_agentic_rag.py -v
```

Expected: PASS.

- [ ] **Step 6: Commit model-call telemetry**

```powershell
git add kami/agent/providers.py kami/agent/graph.py tests/test_chat_tracing.py
git commit -m "feat: record chat model token usage"
```

---

### Task 6: Integrate Trace Lifecycle with LangGraph and Chat Persistence

**Files:**
- Modify: `kami/agent/graph.py:772-818`
- Modify: `kami/chatbot.py:580-587, 616-731`
- Modify: `tests/test_chat_tracing.py`
- Modify: `tests/test_chatbot.py`

**Interfaces:**
- Consumes: `ChatTrace`, graph `retrieval_diagnostics`, and provider telemetry callbacks.
- Changes: `run_agent(..., trace: ChatTrace | None = None) -> dict[str, Any]`
- Produces: one finalized trace for every `answer_chat` attempt, including exceptions.

- [ ] **Step 1: Write failing chat-boundary integration tests**

Use a temporary trace path and deterministic planner. Assert one JSONL record contains `plan`, ordered graph nodes, retrieval evidence, and response language. Add a fake answer model that returns English once and Vietnamese second; assert two model-call entries named `answer` and `answer_language_retry` and `language_retry_count == 1`.

Add an exception-path test:

```python
def test_answer_chat_finalizes_trace_when_model_fails(tmp_path, monkeypatch):
    trace_path = tmp_path / "chat_traces.jsonl"
    monkeypatch.setenv("KAMI_CHAT_TRACE_PATH", str(trace_path))
    monkeypatch.setenv("KAMI_AGENT_DISABLE_LLM_PLANNER", "1")
    monkeypatch.setattr(chatbot, "CHAT_MEMORY_PATH", tmp_path / "sessions.json")
    monkeypatch.setattr(chatbot, "load_characters", _characters)
    monkeypatch.setattr(
        chatbot,
        "_call_model",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("provider failed")),
    )

    with pytest.raises(RuntimeError, match="provider failed"):
        chatbot.answer_chat("Tell me about Nike", provider="gpt")

    record = json.loads(trace_path.read_text(encoding="utf-8").splitlines()[0])
    assert record["status"] == "error"
    assert record["error"]["type"] == "RuntimeError"
```

- [ ] **Step 2: Run integration tests and verify no trace is written**

```powershell
uv run pytest tests/test_chatbot.py tests/test_chat_tracing.py -k "trace or language_retry" -v
```

Expected: FAIL because `answer_chat` does not create or finalize a trace.

- [ ] **Step 3: Instrument each graph node with the request trace**

Wrap registered node functions without placing the trace collector inside serialized graph state:

```python
def traced_node(name: str, callback: Callable[[AgentState], dict[str, Any]]):
    def invoke(state: AgentState) -> dict[str, Any]:
        if trace is None:
            return callback(state)
        with trace.node(name):
            return callback(state)
    return invoke
```

After graph invocation, call `trace.record_plan(result["plan"])` and `trace.record_retrieval(result.get("retrieval_diagnostics", {}), result.get("evidence", []))`.

- [ ] **Step 4: Create and finalize one trace at the chat boundary**

In `answer_chat`, create the trace after provider/model/session selection. Wrap `_call_model` with a per-request counter so the first call uses step `answer` and a second language-correction call uses `answer_language_retry`. Pass `trace.record_model_call` to `answer_with_model`.

Use `try/except/finally` so the result or exception is recorded and `trace.finalize()` always runs. Record response language after the graph returns and before session persistence. A trace-write failure must not change the user response or raised provider error.

- [ ] **Step 5: Run chat and trace integration tests**

```powershell
uv run pytest tests/test_chatbot.py tests/test_chat_tracing.py -v
```

Expected: PASS.

- [ ] **Step 6: Commit graph/chat integration**

```powershell
git add kami/agent/graph.py kami/chatbot.py tests/test_chatbot.py tests/test_chat_tracing.py
git commit -m "feat: trace each chatbot graph execution"
```

---

### Task 7: Add Configuration, Ignore Runtime Traces, and Run Full Verification

**Files:**
- Modify: `.gitignore`
- Modify: `README.md`
- Test: `tests/test_agentic_rag.py`
- Test: `tests/test_chat_tracing.py`
- Test: `tests/test_chatbot.py`

**Interfaces:**
- Documents: primary enable/path/content configuration only.
- Ignores: `kami/data/chat_traces.jsonl*` including rotated backups.

- [ ] **Step 1: Add the runtime ignore rule without staging the user's `docs/` change**

Append exactly:

```gitignore
kami/data/chat_traces.jsonl*
```

Inspect the diff before staging:

```powershell
git diff -- .gitignore
```

During commit, stage only the `kami/data/chat_traces.jsonl*` hunk. Leave the existing uncommitted `docs/` hunk unstaged.

- [ ] **Step 2: Add concise README configuration**

In the chatbot configuration section, document:

```dotenv
KAMI_CHAT_TRACE_ENABLED=1
KAMI_CHAT_TRACE_PATH=kami/data/chat_traces.jsonl
KAMI_CHAT_TRACE_INCLUDE_CONTENT=0
```

State in one sentence that the file contains local graph/retrieval/model/token diagnostics and that raw content is written only when `INCLUDE_CONTENT=1`. Do not add a pipeline internals tutorial.

- [ ] **Step 3: Run focused regression tests**

```powershell
uv run pytest tests/test_agentic_rag.py tests/test_chat_tracing.py tests/test_chatbot.py -v
```

Expected: PASS.

- [ ] **Step 4: Run the full test suite**

```powershell
uv run pytest -v
```

Expected: all tests PASS with no live provider calls.

- [ ] **Step 5: Verify the exact bug scenario structurally**

Run the regression test alone:

```powershell
uv run pytest tests/test_agentic_rag.py::test_llm_rewrite_cannot_narrow_generic_latest_eidolon -v
```

Expected: PASS and the source contains Basic, Stats, Summon Effect, Main Effect, and Sub Effect.

- [ ] **Step 6: Commit configuration and documentation**

Stage `README.md`, the three test files if final verification required adjustments, and only the trace ignore hunk from `.gitignore`:

```powershell
git add README.md tests/test_agentic_rag.py tests/test_chat_tracing.py tests/test_chatbot.py
git add -p .gitignore
git commit -m "docs: configure local chat diagnostics"
```

- [ ] **Step 7: Confirm final worktree state**

```powershell
git status --short
git log --oneline -8
```

Expected: only the user's pre-existing `.gitignore` `docs/` change remains unstaged; implementation commits are present and no runtime trace file is tracked.

---

## Self-Review Results

- Spec coverage: section grounding, all three object types, public/internal context separation, retrieval result logging, graph flow, model identity, token usage, language retries, errors, content opt-in, rotation, Git ignore, and verification are each assigned to a task.
- Placeholder scan: no TBD/TODO or unspecified implementation steps remain.
- Type consistency: `QueryPlan.requested_sections`, `AgentState.retrieval_diagnostics`, `ChatTrace`, and telemetry callback signatures are introduced before their consumers.
- Scope: retrieval correctness and observability share the same structured section/diagnostic data and are kept in one implementation plan intentionally.
