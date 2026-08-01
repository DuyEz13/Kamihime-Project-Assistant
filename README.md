# KamiWiki

A simplified wiki website about Kamihime Project that integrates a chatbot to
assist with character information. The data pipeline supports the three SSR
object catalogs: Kamihime, Eidolons, and Weapons.

## Setup with uv

```powershell
uv sync
```

Optional features:

```powershell
# Local Japanese-to-English translation
uv sync --extra translation

# Local dense/sparse embeddings and reranking
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
uv sync
```

Set the provider and API key in `.env`:

```dotenv
KAMI_TRANSLATION_PROVIDER=deepl
DEEPL_AUTH_KEY=your_deepl_api_key
DEEPL_MODEL_TYPE=prefer_quality_optimized
DEEPL_TRANSLATION_BATCH_SIZE=50
DEEPL_REQUIRE_GLOSSARY=1
DEEPL_GLOSSARY_NAME=KamiWiki JA-EN
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

Add or correct game terminology in `kami/translation_glossary.json`. Before
translation, KamiWiki creates or reuses the stable glossary named by
`DEEPL_GLOSSARY_NAME` and synchronizes its complete JA-EN dictionary with this
file. Legacy hash-named KamiWiki glossaries are renamed automatically during
the first synchronization. To manage a specific existing multilingual
glossary instead, set `DEEPL_GLOSSARY_ID`; its JA-EN dictionary will be
synchronized too. Set `DEEPL_REQUIRE_GLOSSARY=0` only when testing without
terminology enforcement.

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

## Agentic RAG chatbot

Configure at least one chat provider in `.env`:

```dotenv
KAMI_CHAT_PROVIDER=gpt

OPENAI_API_KEY=your_openai_api_key
OPENAI_CHAT_MODEL=gpt-4.1-mini

GEMINI_API_KEY=your_gemini_api_key
GEMINI_CHAT_MODEL=gemini-2.5-flash

DEEPSEEK_API_KEY=your_deepseek_api_key
DEEPSEEK_CHAT_MODEL=deepseek-chat
```

Install the RAG dependencies and build the local index:

```powershell
uv sync --extra rag
uv run python scripts/build_rag_index.py
```

Main RAG options:

```dotenv
KAMI_RAG_DEVICE=auto
KAMI_RAG_EMBED_MODEL=intfloat/multilingual-e5-base
KAMI_RAG_SPARSE_MODEL=Qdrant/bm25
KAMI_RAG_RERANK_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
KAMI_RAG_OBJECT_CANDIDATES_KAMIHIME=7
KAMI_RAG_OBJECT_CANDIDATES_EIDOLON=7
KAMI_RAG_OBJECT_CANDIDATES_WEAPON=24
KAMI_RAG_RETRIEVAL_K=20
KAMI_RAG_SERIES_CONTEXT_CHARS=48000
KAMI_RAG_RERANK=1
KAMI_RAG_INDEX_BATCH_SIZE=64
```

To build with an existing CUDA-enabled Python environment:

```powershell
python scripts/build_rag_index.py --device cuda
```

## Project Structure

```text
KamiWiki/
|-- app/
|   |-- main.py                 # FastAPI application, routes, and static mounts
|   |-- static/
|   |   |-- wiki.css            # Website layout and component styles
|   |   `-- wiki.js             # Client-side update status and UI behavior
|   `-- templates/
|       |-- base.html            # Shared layout and catalog dropdown sidebar
|       |-- index.html           # Home and chat-style landing page
|       |-- catalog.html         # Shared object list page for one element
|       |-- character.html       # Kamihime information and skill page
|       `-- object_detail.html   # Eidolon and Weapon detail page
|-- kami/
|   |-- data/
|   |   |-- kamihime/                 # Six element folders
|   |   |   `-- <element>/
|   |   |       |-- raw.jsonl
|   |   |       `-- translated/<provider>.jsonl
|   |   |-- eidolon/                  # Six elements plus Phantom
|   |   |   `-- <element>/...
|   |   |-- weapon/                   # Six elements plus Phantom
|   |   |   `-- <element>/...
|   |   |-- chat_memory.sqlite3     # Durable conversation and agent state
|   |   |-- chat_sessions.json      # Migrated compatibility mirror
|   |   |-- rag_index/              # Local Qdrant hybrid index
|   |   `-- .translation_cache.json # Shared translation-memory cache
|   |-- chatbot.py              # Backward-compatible chatbot/API facade
|   |-- agent/                  # LangGraph, providers, memory and hybrid RAG
|   |-- crawler.py              # Crawls all three object catalogs and details
|   |-- pipeline.py             # Runs latest/full updates in the background
|   |-- data_store.py           # Loads and normalizes catalog view models
|   |-- data_loader.py          # Generic JSONL record iterator
|   |-- paths.py                # Shared data directory and element file paths
|   |-- translator.py           # Qwen, DeepL, Google translation pipelines
|   |-- translation_glossary.json # Canonical English game terminology
|   |-- build_index.py          # Compatibility hybrid-index entry point
|   |-- kamihime_raw.jsonl      # Legacy combined raw-data fallback
|   |-- all_kami_data.jsonl     # Legacy JSONL data fallback
|   `-- all_kami_data.json      # Legacy JSON data snapshot
|-- img/                        # Element icons used by the sidebar
|-- scripts/
|   |-- build_rag_index.py      # Build the unified Qdrant hybrid index
|   |-- crawl_data.py           # Crawl raw data by object type and element
|   `-- test_translation.py     # Test a few translations without rebuilding data
|-- test.ipynb                  # Experimental crawler and data inspection notebook
|-- .env.example                # Example environment variables
|-- .python-version             # Python version selected by uv
|-- pyproject.toml              # Project metadata and dependency definitions
|-- requirements.txt            # Core pip-compatible dependency list
|-- uv.lock                     # Reproducible dependency lockfile
`-- README.md                   # Project documentation
```
