# KamiWiki

FastAPI wiki for Kamihime Project character data.

## Data pipeline

1. The web UI starts a background refresh.
2. The crawler downloads and parses the configured Japanese wiki pages.
3. Each element is written atomically to its own file:
   `kami/data/kamihime_<element>_raw.jsonl`.
4. The wiki reloads and renders all element files together.

Automatic Gemini translation is temporarily disabled to avoid API quota usage.
`kami/translator.py` remains available for re-enabling translation later.

## Setup with uv

```powershell
uv sync
```

Optional features:

```powershell
# Gemini translation support
uv sync --extra translation

# FAISS/RAG indexing support
uv sync --extra rag

# Install every optional feature
uv sync --all-extras
```

Copy `.env.example` to `.env` and configure values as needed:

```dotenv
KAMI_ELEMENTS=fire,water,wind,thunder,light,dark
```

Individual source URLs can be overridden with environment variables such as
`KAMI_SOURCE_URL_FIRE` or `KAMI_SOURCE_URL_WATER`.

Full database crawling uses four concurrent detail requests by default. Adjust
`KAMI_CRAWL_WORKERS`, `KAMI_CRAWL_DELAY_MIN`, and `KAMI_CRAWL_DELAY_MAX` if the
source site requires a slower request rate. `KAMI_REQUEST_INTERVAL` applies a
global delay between requests, and HTTP 429/5xx responses are retried with
backoff according to `KAMI_HTTP_RETRIES`.

## Run

```powershell
uv run uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000/`.

For a production-style local process:

```powershell
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000
```

The element pages provide two update modes:

- **Update latest characters** checks all six list pages, reuses existing detail
  records, and crawls detail pages only for newly discovered entries.
- **Update Database** crawls every character detail page again so edits to
  existing skills, stats, and flavor text are captured.

Existing element files are replaced atomically only after an update succeeds.
