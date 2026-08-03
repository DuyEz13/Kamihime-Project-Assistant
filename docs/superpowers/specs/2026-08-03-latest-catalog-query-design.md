# Latest Catalog Query Design

## Context

The query `tìm thông tin về kamihime mới nhất hệ nước` currently returns seven
semantically related Water Kamihime instead of the newest Water Kamihime. The
recorded chat session returned seven sources and omitted newer 2026 records,
including Procyon (`26/07/30`).

The current planner can represent object types, entities, and elements, but it
cannot represent ordering or result limits. When the query contains no named
entity, retrieval falls through to hybrid vector search. That path ranks by
semantic relevance and applies the normal Kamihime candidate quota of seven.
`QueryPlan.elements` is also not applied to that fallback search. The answer
model therefore receives seven relevance-ranked objects and has no reliable
way to discover the catalog maximum release date.

## Goals

- Answer singular latest/newest queries using exact catalog data rather than
  semantic similarity.
- Filter by the requested object type and optional element before selecting a
  result.
- Return every object tied on the newest valid release date.
- Apply the same behavior independently to Kamihime, Eidolon, and Weapon.
- Preserve the existing 7/7/24 candidate limits for named-object and series
  retrieval.
- Keep selected objects in the existing Evidence/source/context pipeline so
  answer generation and local links continue to work normally.
- Support Vietnamese and English latest wording without relying solely on the
  configured LLM planner.

## Non-goals

- Ranking subjective notions such as strongest, best, or most useful.
- Replacing vector RAG for semantic questions.
- Adding arbitrary date ranges, oldest-first queries, or a general query
  language in this change.
- Rebuilding or changing the vector index schema.

## Considered Approaches

### 1. Structured catalog retrieval (selected)

Represent the latest constraint in the plan, filter the in-memory catalog, and
select the maximum parsed date deterministically. This is exact, testable, and
does not send irrelevant records to the answer model.

### 2. Planner prompt only

Ask the LLM planner or answer model to infer which retrieved result is newest.
This leaves selection dependent on model behavior and cannot recover records
that vector search did not retrieve.

### 3. Recency-boosted vector search

Mix semantic similarity with a release-date score. This makes an exact
superlative query approximate and introduces a weighting problem with no user
benefit.

## Query Plan

`QueryPlan` will gain explicit structured selection fields:

- `sort_by`: `relevance` by default; `release_date` for latest queries.
- `sort_order`: descending for latest queries.
- `result_limit`: one logical date rank for a singular latest request.
- `include_ties`: true for latest queries so all objects on the maximum date
  are retained.

The LLM planner prompt and schema will expose these fields, but correctness will
not depend on the LLM. A deterministic grounding step will inspect the original
user message after either planner runs. Normalized Vietnamese phrases such as
`mới nhất` and English phrases such as `latest`, `newest`, and `most recent`
will force the release-date selection constraints.

The same grounding step will extract explicit element names from the original
message. This prevents an LLM planner from dropping or changing a concrete
element constraint. Existing deterministic entity and series grounding remains
unchanged.

If a latest query does not identify any object type, the system will request
clarification rather than compare unlike Kamihime, Eidolon, and Weapon records.
If multiple object types are explicitly requested, latest selection is
performed independently for each type.

## Catalog Selection

Latest selection will be a dedicated retrieval operation, separate from hybrid
RAG:

1. Load catalog records for each requested object type using the existing
   cached catalog loader.
2. Filter by the requested element when present.
3. Parse each top-level `release_date` into a date value.
4. Scan candidates once, tracking the maximum date and all records tied on that
   date.
5. Hydrate only the winning records into their complete type-specific logical
   documents.

The scan is O(n) and does not require sorting the full catalog. The database is
not sent to the answer LLM. Catalog loaders are already cached by source-file
modification time, so repeated queries do not repeatedly parse unchanged JSONL
files.

Date parsing will accept the formats currently present in the project,
including `YY/MM/DD`, `YYYY/MM/DD`, and their hyphen-separated equivalents.
Two-digit years are interpreted as 2000-based because all Kamihime Project
release records are from 2016 onward. Empty, placeholder, and malformed dates
are excluded from selection.

For each object type, all records whose parsed date equals the maximum valid
date are returned. The order within a tie is deterministic by normalized name
and slug so tests and UI source ordering remain stable.

## Evidence and Answer Flow

Selected catalog records will be hydrated through the same type-aware document
builder used by resolved entities:

- Kamihime: Basic, Burst, Ability, Assist/other available skill sections.
- Eidolon: Basic, Stats, Summon Effect, Main Effect, Sub Effect.
- Weapon: Basic, Stats, Burst Effects, Weapon Skills.

Evidence will identify the retrieval mode as `catalog_latest`. Sources will
contain only the winning records. The generated context will explicitly state
that records were selected by maximum release date and that same-date records
are ties. The answer prompt will instruct the model not to expand beyond those
selected sources.

Named-object, comparison, series, and general semantic questions continue down
their existing entity or hybrid retrieval paths. Their configured candidate
limits are not changed.

## Error Handling

- No matching records after type/element filtering: return the existing
  localized insufficient-data response.
- Matching records but no valid release dates: return a localized message that
  release-date data is unavailable, without falling back to semantic search or
  guessing.
- Mixed valid and invalid dates: ignore invalid dates and select from valid
  records.
- A malformed planner result: deterministic latest and element grounding takes
  precedence over model output.

## Testing

Implementation will follow test-driven development. Tests will cover:

- Vietnamese `mới nhất` planning for Water Kamihime.
- English `latest`, `newest`, and `most recent` planning.
- Deterministic grounding correcting a model plan that omitted latest or the
  requested element.
- Correct selection across older and newer candidates.
- Returning every object tied on the maximum date.
- Parsing two-digit, four-digit, slash, and hyphen date formats.
- Ignoring malformed dates without selecting an older semantic match.
- Independent support for Kamihime, Eidolon, and Weapon.
- Full type-specific evidence hydration for selected records.
- Unchanged seven-variant named Kamihime behavior and unchanged series/weapon
  candidate limits.
- No vector-index call on the catalog-latest path.

## Acceptance Criteria

- The tested query returns Procyon from the current Water Kamihime catalog,
  unless another record with a later date is added before execution.
- If another Water Kamihime shares the maximum date, both are returned.
- No older Water Kamihime appears in sources for the singular latest query.
- The answer is produced in the user's language through the existing language
  guard.
- Existing agentic RAG tests continue to pass.
- No RAG index rebuild is required after deployment.
