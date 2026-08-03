# Chat Retrieval Scope and Observability Design

**Date:** 2026-08-03  
**Status:** Approved  
**Scope:** Agent section selection, answer-context boundaries, per-turn graph tracing, model/token telemetry

## Problem

The query `tìm thông tin về eidolon mới nhất hệ nước` selected only the Eidolon `basic` document even though the user asked for general information. The LLM planner rewrote the request as `Find information about the newest water element eidolon`; the existing section selector interpreted the filter word `element` as an explicit request for the Basic section.

The answer also exposed internal retrieval diagnostics. The graph currently appends section coverage, omissions, retrieval mode, series coverage, and related operational notes directly to the answer context. The answer system prompt then instructs the model to discuss those details.

## Goals

1. General object-information questions load every logical feature document for the selected object type.
2. Retrieval is narrowed only when the original user message explicitly requests a section or field.
3. Filtering and sorting language must not accidentally narrow document sections.
4. Internal retrieval diagnostics must not appear in normal user answers.
5. Every chat turn must produce a structured local trace containing graph flow, retrieval decisions, model identity, timing, and provider-reported token usage.
6. Retrieved evidence identities, sections, scores, coverage, omissions, and missing-data diagnoses must be available in the trace.
7. Raw prompts, full evidence text, and answer text remain disabled by default and can be enabled for deep debugging.

## Non-goals

- Sending telemetry to an external observability service.
- Estimating token counts when a provider does not return usage metadata.
- Replacing the existing chat-session persistence format.
- Exposing trace diagnostics in the normal chatbot response.

## Section Selection

### Source of truth

Section intent is derived deterministically from the original user message. The LLM-generated `standalone_question` and per-entity retrieval queries remain useful for semantic retrieval, but they must never decide which logical sections are loaded.

The grounded section request is stored as structured state and passed explicitly to catalog hydration and entity retrieval. Retrieval code no longer re-parses rewritten English queries to infer section scope.

### General requests

Generic phrases such as `tìm thông tin`, `cho tôi biết`, `information`, `details`, `latest`, and `newest` do not select a section. When no explicit section is detected, the complete type-specific document set is loaded:

- Kamihime: Basic, Burst, Ability, Assist.
- Eidolon: Basic, Stats, Summon Effect, Main Effect, Sub Effect.
- Weapon: Basic, Stats, Burst Effects, Weapon Skills.

### Explicit requests

Concrete section names and unambiguous field requests select only the applicable logical section or sections. Examples include `stats`, `summon effect`, `main effect`, `sub effect`, `burst`, `ability`, `assist`, and `weapon skills` in supported user languages.

Filter terms are not section requests. For example, `water element`, a weapon type, a rarity constraint, or sorting by release date must not implicitly select Basic. An explicit field question such as `Eidolon này thuộc hệ nào?` may select Basic because the field itself is the requested answer.

Object identity metadata—name, type, slug, element, and local URL—remains available independently of section document content.

### Multiple object types

Section requests are represented per object type so that one message can ask different features of Kamihime, Eidolon, and Weapon without applying an invalid section across all types.

## Public Answer Context and Internal Diagnostics

The graph produces two logically separate outputs:

1. **Answer evidence:** database content needed to answer the user.
2. **Retrieval diagnostics:** operational metadata written to the trace.

Normal answer context excludes:

- available/included/omitted section reports;
- coverage-complete or partial labels;
- omission reasons and context-budget notes;
- retrieval mode, scores, candidate counts, and ranking details;
- series catalog gaps and retrieval completeness diagnostics;
- graph node names and execution details.

Latest selection continues to be deterministic. Only maximum-date records, including all ties, are supplied as answer evidence. The model therefore does not need an operational explanation of how candidates were rejected.

Series-answer behavior remains concise: shared mechanics are summarized once and per-member progression is expanded only when requested. Operational coverage facts are retained in diagnostics rather than displayed as unsolicited notes.

When no evidence can answer the question, the public response remains concise and does not speculate about pipeline or database internals. The exact reason is recorded in the trace.

## Trace Storage

### File and rotation

The default destination is:

`kami/data/chat_traces.jsonl`

The file and rotated backups are ignored by Git. Rotation is size-based with configurable maximum size and backup count.

Suggested configuration:

- `KAMI_CHAT_TRACE_ENABLED=1`
- `KAMI_CHAT_TRACE_PATH=kami/data/chat_traces.jsonl`
- `KAMI_CHAT_TRACE_MAX_BYTES=10485760`
- `KAMI_CHAT_TRACE_BACKUP_COUNT=5`
- `KAMI_CHAT_TRACE_INCLUDE_CONTENT=0`

### Record boundary

One JSON object is appended per chat turn. The trace is finalized in success, refusal, clarification, missing-evidence, and exception paths. A unique `trace_id` correlates the record with runtime results without rendering diagnostics to the user.

### Schema

Each record contains:

- schema version, trace/session/client IDs, timestamps, and total duration;
- detected request and response language metadata;
- grounded query plan: intent, object types, entities, elements, sorting, ties, and requested sections;
- ordered graph-node executions with status and duration;
- retrieval mode, filters, candidate counts, date validity, maximum dates, and tie counts;
- every selected evidence identity with object name/type/slug, section, score, URL, available/included/omitted sections, coverage, omission reason, series diagnostics, and selection reason;
- model calls separated by step (`planner`, `answer`, and language-correction retries), including provider, model, duration, status, and provider-reported input/output/total tokens;
- final status, expected/actual response language, retry count, and error details when applicable;
- aggregate model-call count, duration, and token totals where usage is available.

By default the trace records evidence metadata and diagnostics but not original prompts, full retrieved document text, or final answer text. When `KAMI_CHAT_TRACE_INCLUDE_CONTENT=1`, the trace additionally includes those raw debugging fields.

## Model Usage Collection

Provider calls retain their current parsed return values while also emitting structured call telemetry. For normal chat responses, usage is read from the returned AI message metadata. For structured planner responses, the raw provider message must be retained alongside the parsed `QueryPlan` so its usage metadata can be captured.

Usage extraction supports the common LangChain fields `usage_metadata` and provider-specific `response_metadata` token fields. Missing usage is represented as `null` with an availability flag; it is never estimated silently.

Each retry is a separate model-call entry. Aggregate tokens are computed only from reported values, preventing a language-correction retry from being hidden in the totals.

## Graph Instrumentation

A per-turn trace collector is created at the chat boundary and passed explicitly through the agent execution path. Graph nodes and provider calls report events to this collector. Explicit propagation is preferred over global mutable state so concurrent chat requests cannot mix telemetry.

Node instrumentation records at least:

- plan;
- refuse or clarify when selected;
- retrieve;
- answer;
- missing-evidence handling.

The collector finalizes and appends a JSONL record in a `finally` path. Logging failures must not break the chatbot response; they are reported through the normal Python logger.

## Retrieval Diagnosis Semantics

Trace diagnostics distinguish these conditions:

- all available logical sections were selected;
- sections were intentionally narrowed by an explicit user request;
- sections were omitted by a series context budget;
- a logical section is absent from the database object;
- catalog records matched but have no valid release date;
- no catalog records matched the filters;
- latest records were selected, including the number of tied winners;
- series members exist in the catalog but were not retrieved;
- expected series members have not yet been released;
- provider, parsing, or graph execution failed.

These conditions use stable machine-readable codes plus human-readable detail in the trace. They are not copied into normal user answers.

## Verification

Automated tests will cover:

1. The exact failing Vietnamese query with an LLM rewrite containing `water element` loads all five Eidolon sections.
2. Generic latest queries load all logical sections for Kamihime, Eidolon, and Weapon.
3. Explicit section requests narrow correctly for all three object types.
4. Filter words such as element, rarity, weapon type, and release sorting do not narrow sections.
5. Internal selection and coverage diagnostics are absent from the answer context but present in the JSONL trace.
6. Latest ties and missing/invalid-date outcomes are recorded accurately.
7. Planner, answer, and language-retry calls record separate timing and usage entries.
8. Missing provider usage remains null rather than being estimated.
9. Trace records are finalized for successful and exceptional graph runs.
10. Content logging remains off by default and can be enabled explicitly.

