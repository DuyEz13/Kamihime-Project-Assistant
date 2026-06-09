# KamiWiki

A simplified wiki website about Kamihime Project that integrates a chatbot to assist with questions about in-game character information or how to build a team and weapon grid.

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

## Project Structure

```text
KamiWiki/
|-- app/
|   |-- main.py                 # FastAPI application, routes, and static mounts
|   |-- static/
|   |   |-- wiki.css            # Website layout and component styles
|   |   `-- wiki.js             # Client-side update status and UI behavior
|   `-- templates/
|       |-- base.html            # Shared page layout and element sidebar
|       |-- index.html           # Home and chat-style landing page
|       |-- element.html         # Character list page for one element
|       `-- character.html       # Character information and skill page
|-- kami/
|   |-- data/
|   |   `-- kamihime_*_raw.jsonl # Active character data, split by element
|   |-- crawler.py              # Crawls character lists and detail pages
|   |-- pipeline.py             # Runs latest/full updates in the background
|   |-- data_store.py           # Loads, normalizes, filters, and finds characters
|   |-- data_loader.py          # Generic JSONL record iterator
|   |-- translator.py           # Optional Japanese-to-English Gemini pipeline
|   |-- build_index.py          # Optional FAISS/RAG index builder
|   |-- kamihime_raw.jsonl      # Legacy combined raw-data fallback
|   |-- all_kami_data.jsonl     # Legacy JSONL data fallback
|   `-- all_kami_data.json      # Legacy JSON data snapshot
|-- img/                        # Element icons used by the sidebar
|-- test.ipynb                  # Experimental crawler and data inspection notebook
|-- .env.example                # Example environment variables
|-- .python-version             # Python version selected by uv
|-- pyproject.toml              # Project metadata and dependency definitions
|-- requirements.txt            # Core pip-compatible dependency list
|-- uv.lock                     # Reproducible dependency lockfile
`-- README.md                   # Project documentation
```

The application normally reads the six element-specific files under
`kami/data/`. The combined files directly under `kami/` are retained as
backward-compatible fallbacks and are not the primary crawl output.
