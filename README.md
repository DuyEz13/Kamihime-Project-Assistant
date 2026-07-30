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

For a production-style local process:

```powershell
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000
```

The sidebar groups data into Kamihime, Eidolons, and Weapons. Each catalog
expands into its supported elements and opens routes such as
`/catalog/eidolon/phantom` or `/catalog/weapon/fire`. Catalog pages provide
two update modes:

- **Update latest records** checks every configured element list for the
  selected object type, reuses existing detail records, crawls detail pages
  only for newly discovered entries, and translates new or changed text.
- **Update Database** crawls every object detail page again so edits to
  existing skills, stats, effects, and flavor text are captured, then rebuilds
  the translated element files.

Each element file is replaced atomically only after that element crawl
succeeds, so an empty or failed crawl does not overwrite its previous data.
Raw and translated data use an object/element layout:

```text
kami/data/<object_type>/<element>/raw.jsonl
kami/data/<object_type>/<element>/translated/<provider>.jsonl
```

`object_type` is `kamihime`, `eidolon`, or `weapon`. Kamihime has six
elements; Eidolons and Weapons also have `phantom`. The web application
prefers the provider selected by `KAMI_RENDER_TRANSLATION_PROVIDER` or
`KAMI_TRANSLATION_PROVIDER`, then falls back to other providers and finally
to raw Japanese data until translation is available.

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
|   |   |-- chat_sessions.json      # Per-conversation chatbot memory
|   |   `-- .translation_cache.json # Shared translation-memory cache
|   |-- chatbot.py              # RAG retrieval, chat memory and calls
|   |-- crawler.py              # Crawls all three object catalogs and details
|   |-- pipeline.py             # Runs latest/full updates in the background
|   |-- data_store.py           # Loads and normalizes catalog view models
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
