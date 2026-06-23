# KamiWiki

A simplified wiki website about Kamihime Project that integrates a chatbot to assist with questions about in-game character information or how to build a team and weapon grid.

## Setup with uv

```powershell
uv sync
```

Optional features:

```powershell
# Local Japanese-to-English translation
uv sync --extra translation

# FAISS/RAG indexing support
uv sync --extra rag

# Install every optional feature
uv sync --all-extras
```

Configure local translation with:

```dotenv
KAMI_TRANSLATION_MODEL=Qwen/Qwen2.5-14B-Instruct-AWQ
KAMI_TRANSLATION_DEVICE=cuda
KAMI_TRANSLATION_BATCH_SIZE=8
KAMI_TRANSLATION_MAX_CHARS=500
KAMI_TRANSLATION_MAX_NEW_TOKENS=2048
KAMI_TRANSLATION_MEMORY_EXAMPLES=6
KAMI_TRANSLATION_MEMORY_SCAN=500
```

The translator uses `Qwen/Qwen2.5-14B-Instruct-AWQ` and requires an NVIDIA CUDA
GPU on Linux.

### DeepL API alternative

DeepL API Free currently includes up to 1,000,000 translated source characters
per account. To test DeepL without removing the Qwen pipeline:

```powershell
uv sync --extra deepl
```

Set the provider and API key in `.env`:

```dotenv
KAMI_TRANSLATION_PROVIDER=deepl
DEEPL_AUTH_KEY=your_deepl_api_key
DEEPL_MODEL_TYPE=prefer_quality_optimized
DEEPL_TRANSLATION_BATCH_SIZE=50
DEEPL_REQUIRE_GLOSSARY=1
```

Test a few values before running a full update:

```powershell
uv run python scripts/test_translation.py --provider deepl --element fire --count 5
```

### Google Translate API alternative

Google Translate uses the existing `httpx` dependency and writes output to its
own provider folder. Set an API key in `.env`:

```dotenv
KAMI_TRANSLATION_PROVIDER=google
GOOGLE_TRANSLATE_API_KEY=your_google_translate_api_key
GOOGLE_TRANSLATE_BATCH_SIZE=50
```

Test a few values before running a full update:

```powershell
uv run python scripts/test_translation.py --provider google --element fire --count 5
```

To translate the existing Japanese database without crawling the source wiki
again, open an element page, choose **DeepL**, **Google Translate**, or
**Qwen** from the provider dropdown, then click **Translate Database**. This
reads the existing `kami/data/raw/kamihime_*_raw.jsonl` files and rewrites the
corresponding provider output under
`kami/data/translated/<provider>/kamihime_*_en.jsonl` only after translation
succeeds.

Add or correct game terminology in `kami/translation_glossary.json`. For
example, the default glossary fixes `レイジング状態` as
`Raging state`. If automatic glossary creation is unavailable for the account,
set `DEEPL_GLOSSARY_ID` to an existing multilingual glossary ID. Set
`DEEPL_REQUIRE_GLOSSARY=0` only when testing without terminology enforcement.

Test a small translation sample without rebuilding or overwriting the English
element files:

```powershell
# Translate five random values from the Fire raw data
uv run python scripts/test_translation.py --element fire --count 5

# Translate specific values
uv run python scripts/test_translation.py `
  --text "敵全体に火属性ダメージ" `
  --text "味方全体のバーストゲージUP"
```

Individual source URLs can be overridden with environment variables such as
`KAMI_SOURCE_URL_FIRE` or `KAMI_SOURCE_URL_WATER`.

Full database crawling is intentionally conservative because the source wiki
can return HTTP 429 when requests arrive too quickly. The default setup uses
one detail worker, a global request interval, randomized per-character delay,
and exponential backoff with jitter:

```dotenv
KAMI_CRAWL_WORKERS=1
KAMI_CRAWL_DELAY_MIN=0.8
KAMI_CRAWL_DELAY_MAX=1.6
KAMI_REQUEST_INTERVAL=1.2
KAMI_HTTP_RETRIES=8
KAMI_HTTP_BACKOFF_BASE=4
KAMI_HTTP_BACKOFF_MAX=180
KAMI_HTTP_BACKOFF_JITTER=0.35
KAMI_HTTP_429_COOLDOWN=45
```

If the wiki still returns 429, increase `KAMI_REQUEST_INTERVAL` and
`KAMI_HTTP_429_COOLDOWN` before increasing workers. `Retry-After` headers are
honored when the site provides them.

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
  records, crawls detail pages only for newly discovered entries, and locally
  translates new or changed text.
- **Update Database** crawls every character detail page again so edits to
  existing skills, stats, and flavor text are captured, then rebuilds the
  translated element files.

Existing element files are replaced atomically only after an update succeeds.
Raw crawl output is stored under `kami/data/raw/` as
`kamihime_<element>_raw.jsonl`; translated output is stored under
`kami/data/translated/<provider>/` as `kamihime_<element>_en.jsonl`. The web
application prefers the provider selected by `KAMI_RENDER_TRANSLATION_PROVIDER`
or `KAMI_TRANSLATION_PROVIDER`, then falls back to other provider folders and
finally to raw Japanese data until translation is available.

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
|   |   |-- raw/
|   |   |   `-- kamihime_*_raw.jsonl # Japanese crawl data, split by element
|   |   |-- translated/
|   |   |   |-- deepl/                # DeepL translated element JSONL files
|   |   |   |-- google/               # Google Translate element JSONL files
|   |   |   `-- qwen/                 # Qwen translated element JSONL files
|   |   |-- chat_sessions.json      # Per-conversation chatbot memory
|   |   `-- .translation_cache.json # Shared translation-memory cache
|   |-- chatbot.py              # RAG retrieval, chat memory and calls
|   |-- crawler.py              # Crawls character lists and detail pages
|   |-- pipeline.py             # Runs latest/full updates in the background
|   |-- data_store.py           # Loads, normalizes, filters, and finds characters
|   |-- data_loader.py          # Generic JSONL record iterator
|   |-- paths.py                # Shared data directory and element file paths
|   |-- translator.py           # Qwen, DeepL, Google translation pipelines
|   |-- translation_glossary.json # Canonical English game terminology
|   |-- build_index.py          # Optional FAISS/RAG index builder
|   |-- kamihime_raw.jsonl      # Legacy combined raw-data fallback
|   |-- all_kami_data.jsonl     # Legacy JSONL data fallback
|   `-- all_kami_data.json      # Legacy JSON data snapshot
|-- img/                        # Element icons used by the sidebar
|-- scripts/
|   `-- test_translation.py     # Test a few translations without rebuilding data
|-- test.ipynb                  # Experimental crawler and data inspection notebook
|-- .env.example                # Example environment variables
|-- .python-version             # Python version selected by uv
|-- pyproject.toml              # Project metadata and dependency definitions
|-- requirements.txt            # Core pip-compatible dependency list
|-- uv.lock                     # Reproducible dependency lockfile
`-- README.md                   # Project documentation
```
