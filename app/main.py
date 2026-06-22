from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from kami.data_store import get_character, load_characters
from kami.pipeline import get_refresh_status, start_translation, start_update
from kami.paths import TRANSLATION_PROVIDERS


load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
ASSET_VERSION = str(
    max(
        (BASE_DIR / "static" / "wiki.css").stat().st_mtime_ns,
        (BASE_DIR / "static" / "wiki.js").stat().st_mtime_ns,
    )
)
ELEMENTS = (
    {"key": "fire", "label": "Fire", "image": "火.jpg"},
    {"key": "water", "label": "Water", "image": "水.jpg"},
    {"key": "wind", "label": "Wind", "image": "風.jpg"},
    {"key": "thunder", "label": "Thunder", "image": "雷.jpg"},
    {"key": "light", "label": "Light", "image": "光.jpg"},
    {"key": "dark", "label": "Dark", "image": "闇.jpg"},
)

app = FastAPI(title="KamiWiki")
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
app.mount(
    "/img",
    StaticFiles(directory=BASE_DIR.parent / "img"),
    name="element_images",
)
templates = Jinja2Templates(directory=BASE_DIR / "templates")
templates.env.globals["asset_version"] = ASSET_VERSION


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "elements": ELEMENTS,
            "active_element": None,
            "refresh": get_refresh_status(),
        },
    )


@app.get("/elements/{element}", response_class=HTMLResponse)
def element_characters(request: Request, element: str, q: str = ""):
    element_meta = next(
        (item for item in ELEMENTS if item["key"] == element),
        None,
    )
    if element_meta is None:
        raise HTTPException(status_code=404, detail="Element not found")

    characters = [
        character
        for character in load_characters()
        if character["element"] == element
    ]
    query = q.strip().casefold()
    if query:
        characters = [
            character
            for character in characters
            if query in character["name"].casefold()
        ]

    return templates.TemplateResponse(
        request=request,
        name="element.html",
        context={
            "elements": ELEMENTS,
            "active_element": element,
            "element": element_meta,
            "characters": characters,
            "query": q,
            "refresh": get_refresh_status(),
        },
    )


@app.get("/characters/{slug}", response_class=HTMLResponse)
def character_detail(request: Request, slug: str):
    character = get_character(slug)
    if character is None:
        raise HTTPException(status_code=404, detail="Character not found")

    return templates.TemplateResponse(
        request=request,
        name="character.html",
        context={
            "elements": ELEMENTS,
            "active_element": character["element"],
            "character": character,
        },
    )


@app.post("/api/update/{mode}", status_code=202)
def update_data(mode: str):
    if mode not in {"latest", "database"}:
        raise HTTPException(status_code=404, detail="Update mode not found")
    started = start_update(mode)
    if not started:
        raise HTTPException(status_code=409, detail="An update is already running")
    return get_refresh_status()


@app.get("/api/update/status")
def update_status():
    return get_refresh_status()


@app.post("/api/translate/{provider}", status_code=202)
def translate_database(provider: str):
    if provider not in TRANSLATION_PROVIDERS:
        raise HTTPException(status_code=404, detail="Translation provider not found")
    started = start_translation(provider)
    if not started:
        raise HTTPException(status_code=409, detail="An update is already running")
    return get_refresh_status()


@app.post("/api/refresh", status_code=202)
def refresh_data():
    """Backward-compatible endpoint for a full database update."""
    return update_data("database")


@app.get("/api/refresh/status")
def refresh_status():
    return update_status()
