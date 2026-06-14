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

No translation API key is required. The refresh pipeline uses
[`google/madlad400-3b-mt`](https://huggingface.co/google/madlad400-3b-mt)
through Hugging Face Transformers. MADLAD-400 is a 3B-parameter multilingual
translation model and produces substantially better general-purpose output
than the previous small MarianMT model. It is still a research model, so its
quality is not guaranteed to match the production Google Translate service,
especially for Kamihime-specific names and terminology.

The original model weights are about 11.8 GB. The model is downloaded to the
local Hugging Face cache on the first translation and reused afterward. CPU
translation requires substantial RAM and is slow; a CUDA GPU with enough VRAM
is strongly preferred.

Configure local translation with:

```dotenv
KAMI_TRANSLATION_MODEL=google/madlad400-3b-mt
KAMI_TRANSLATION_DEVICE=auto
KAMI_TRANSLATION_BATCH_SIZE=1
KAMI_TRANSLATION_MAX_CHARS=350
KAMI_TRANSLATION_BEAMS=4
```

`KAMI_TRANSLATION_DEVICE=auto` uses CUDA when available and otherwise runs on
the CPU. Translation results are cached by source text, so later latest updates
only invoke the model for new or changed Japanese text. During translation,
the element page displays the selected CPU/GPU, model name, translated chunk
count, percentage, and a progress bar.

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

Use `--device cpu` or `--device cuda` to override automatic device selection.
The script prints the model, device, progress, Japanese source, and English
result. It may update `.translation_cache.json`, but it never writes an element
`_en.jsonl` file.

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
  records, crawls detail pages only for newly discovered entries, and locally
  translates new or changed text.
- **Update Database** crawls every character detail page again so edits to
  existing skills, stats, and flavor text are captured, then rebuilds the
  translated element files.

Existing element files are replaced atomically only after an update succeeds.
Raw crawl output is stored as `kamihime_<element>_raw.jsonl`; translated output
is stored as `kamihime_<element>_en.jsonl`. The web application prefers each
English file and falls back to its raw file until translation is available.

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
|   |   |-- kamihime_*_raw.jsonl # Japanese crawl data, split by element
|   |   `-- kamihime_*_en.jsonl  # Locally translated data rendered by the web
|   |-- crawler.py              # Crawls character lists and detail pages
|   |-- pipeline.py             # Runs latest/full updates in the background
|   |-- data_store.py           # Loads, normalizes, filters, and finds characters
|   |-- data_loader.py          # Generic JSONL record iterator
|   |-- translator.py           # Local MADLAD-400 translation and text cache
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

The application normally reads the six translated element files under
`kami/data/`, falling back to the corresponding Japanese raw file when needed.
The combined files directly under `kami/` are retained as backward-compatible
fallbacks and are not the primary crawl output.
