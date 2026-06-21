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

Copy `.env.example` to `.env` and configure values as needed:

```dotenv
KAMI_ELEMENTS=fire,water,wind,thunder,light,dark
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
GPU on Linux. AutoAWQ depends on Triton, which does not provide Windows wheels.
A Colab T4 runtime can run the quantized model. Translation is deterministic
and processes multiple data points in one JSON batch.

AutoAWQ is deprecated and only supports a narrow dependency range. The
translation extra intentionally pins the last tested stack:

- `autoawq==0.2.9`
- `torch==2.6.0`
- `transformers==4.51.3`

Do not upgrade Transformers independently. Newer releases remove activation
classes that AutoAWQ still imports and cause errors such as
`cannot import name 'PytorchGELUTanh'`.

Consistency is enforced by:

- `kami/translation_glossary.json`, which defines canonical Kamihime terms.
- A versioned cache that reuses exact translations.
- Relevant translation-memory examples injected into later prompts.
- A shared system prompt that prohibits stylistic variation for recurring
  effects.

Changing the glossary automatically invalidates affected cache namespace
instead of silently reusing output generated under the old terminology.
During translation, the element page displays GPU, model name, translated
chunk count, percentage, and a progress bar.

For Google Colab, select a GPU runtime and install the translation extra:

```bash
!pip install uv
!uv sync --extra translation
```

After changing an existing Colab environment, recreate `.venv` or force an
exact sync so the incompatible Transformers version is removed:

```bash
!rm -rf .venv
!uv sync --extra translation
!uv run python -c "import torch, transformers; print(torch.__version__, transformers.__version__)"
```

The expected versions are `2.6.0` and `4.51.3`.

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

To translate the existing Japanese database without crawling the source wiki
again, open an element page, choose **DeepL** or **Qwen** from the provider
dropdown, then click **Translate Database**. This reads the existing
`kami/data/raw/kamihime_*_raw.jsonl` files and rewrites the corresponding
`kami/data/translated/kamihime_*_en.jsonl` files only after translation
succeeds.

The DeepL backend:

- Uses a separate cache namespace, so cached Qwen results are never mixed in.
- Sends text in batches and only bills uncached source text.
- Automatically creates and reuses a versioned JA-EN glossary from
  `kami/translation_glossary.json`.
- Sends matching canonical terms as unbilled translation context.
- Prints current character usage after the test.

Add or correct game terminology in `kami/translation_glossary.json`. For
example, the default glossary fixes `ニケ` as `Nike` and `レイジング状態` as
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
`kami/data/translated/` as `kamihime_<element>_en.jsonl`. The web application
prefers each English file and falls back to its raw file until translation is
available.

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
|   |   |   `-- kamihime_*_en.jsonl  # Translated data rendered by the web
|   |   `-- .translation_cache.json # Shared translation-memory cache
|   |-- crawler.py              # Crawls character lists and detail pages
|   |-- pipeline.py             # Runs latest/full updates in the background
|   |-- data_store.py           # Loads, normalizes, filters, and finds characters
|   |-- data_loader.py          # Generic JSONL record iterator
|   |-- paths.py                # Shared data directory and element file paths
|   |-- translator.py           # Qwen AWQ translation and translation memory
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
