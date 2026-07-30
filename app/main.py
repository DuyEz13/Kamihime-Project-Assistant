from pathlib import Path
from urllib.parse import unquote, urlsplit

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from kami.chatbot import (
    answer_chat,
    available_chat_models,
    delete_session,
    get_session,
    list_sessions,
)
from kami.data_store import (
    get_catalog_item,
    load_catalog_items,
)
from kami.pipeline import get_refresh_status, start_translation, start_update
from kami.paths import (
    OBJECT_ELEMENTS,
    TRANSLATION_PROVIDERS,
    normalize_object_type,
)


load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
ASSET_VERSION = str(
    max(
        (BASE_DIR / "static" / "wiki.css").stat().st_mtime_ns,
        (BASE_DIR / "static" / "wiki.js").stat().st_mtime_ns,
    )
)
ELEMENTS = {
    "fire": {"key": "fire", "label": "Fire", "image": "火.jpg"},
    "water": {"key": "water", "label": "Water", "image": "水.jpg"},
    "wind": {"key": "wind", "label": "Wind", "image": "風.jpg"},
    "thunder": {"key": "thunder", "label": "Thunder", "image": "雷.jpg"},
    "light": {"key": "light", "label": "Light", "image": "光.jpg"},
    "dark": {"key": "dark", "label": "Dark", "image": "闇.jpg"},
    "phantom": {
        "key": "phantom",
        "label": "Phantom",
        "image": "幻.webp",
    },
}
CATALOGS = (
    {
        "key": "kamihime",
        "label": "Kamihime",
        "singular": "Kamihime",
        "elements": tuple(ELEMENTS[key] for key in OBJECT_ELEMENTS["kamihime"]),
    },
    {
        "key": "eidolon",
        "label": "Eidolons",
        "singular": "Eidolon",
        "elements": tuple(ELEMENTS[key] for key in OBJECT_ELEMENTS["eidolon"]),
    },
    {
        "key": "weapon",
        "label": "Weapons",
        "singular": "Weapon",
        "elements": tuple(ELEMENTS[key] for key in OBJECT_ELEMENTS["weapon"]),
    },
)
CATALOG_BY_KEY = {catalog["key"]: catalog for catalog in CATALOGS}

app = FastAPI(title="KamiWiki")
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
app.mount(
    "/img",
    StaticFiles(directory=BASE_DIR.parent / "img"),
    name="element_images",
)
templates = Jinja2Templates(directory=BASE_DIR / "templates")
templates.env.globals["asset_version"] = ASSET_VERSION


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    session_id: str | None = None
    provider: str = "gpt"


def _navigation_context(
    active_object_type: str | None = None,
    active_element: str | None = None,
) -> dict:
    return {
        "catalogs": CATALOGS,
        "active_object_type": active_object_type,
        "active_element": active_element,
    }


def _source_url_key(value: str | None) -> tuple[str, str] | None:
    if not value:
        return None
    parsed = urlsplit(str(value).strip())
    if not parsed.netloc or not parsed.path:
        return None
    return (
        parsed.netloc.casefold(),
        unquote(parsed.path).rstrip("/").casefold(),
    )


def _display_info_rows(item: dict) -> list[dict[str, str | None]]:
    link_fields = {
        "Unlock Weapon": ("unlock_weapon_url", "weapon"),
        "解放武器": ("unlock_weapon_url", "weapon"),
        "Unlock Kamihime": ("unlock_kamihime_url", "kamihime"),
        "解放神姫": ("unlock_kamihime_url", "kamihime"),
    }
    info = item.get("info") if isinstance(item.get("info"), dict) else {}
    display_info = (
        item.get("display_info")
        if isinstance(item.get("display_info"), dict)
        else {}
    )
    rows: list[dict[str, str | None]] = []
    target_indexes: dict[str, dict[tuple[str, str], dict]] = {}

    for label, value in display_info.items():
        href = None
        link_config = link_fields.get(str(label))
        if link_config is not None:
            url_field, target_type = link_config
            target_key = _source_url_key(info.get(url_field))
            if target_key is not None:
                if target_type not in target_indexes:
                    target_indexes[target_type] = {
                        source_key: target
                        for target in load_catalog_items(target_type)
                        if (
                            source_key := _source_url_key(
                                target.get("info", {}).get("source_url")
                            )
                        )
                        is not None
                    }
                target = target_indexes[target_type].get(target_key)
                if target is not None:
                    href = (
                        f"/objects/{target_type}/"
                        f"{target['slug']}"
                    )
        rows.append(
            {
                "label": str(label),
                "value": str(value),
                "href": href,
            }
        )
    return rows


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            **_navigation_context(),
            "refresh": get_refresh_status(),
        },
    )


@app.get("/api/chat/models")
def chat_models():
    return {
        "models": [
            {
                "provider": model.provider,
                "label": model.label,
                "model": model.model,
                "configured": model.configured,
            }
            for model in available_chat_models()
        ]
    }


@app.get("/api/chat/sessions")
def chat_sessions():
    return {"sessions": list_sessions()}


@app.get("/api/chat/{session_id}")
def chat_session(session_id: str):
    return get_session(session_id)


@app.delete("/api/chat/{session_id}")
def remove_chat_session(session_id: str):
    if not delete_session(session_id):
        raise HTTPException(status_code=404, detail="Chat session not found")
    return {"deleted": True, "session_id": session_id}


@app.post("/api/chat")
def chat(request: ChatRequest):
    try:
        return answer_chat(
            message=request.message,
            session_id=request.session_id,
            provider=request.provider,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.get("/elements/{element}", response_class=HTMLResponse)
def element_characters(request: Request, element: str, q: str = ""):
    """Backward-compatible Kamihime catalog route."""
    return catalog_items(request, "kamihime", element, q)


@app.get(
    "/catalog/{object_type}/{element}",
    response_class=HTMLResponse,
)
def catalog_items(
    request: Request,
    object_type: str,
    element: str,
    q: str = "",
):
    try:
        selected_object_type = normalize_object_type(object_type)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    if element not in OBJECT_ELEMENTS[selected_object_type]:
        raise HTTPException(status_code=404, detail="Element not found")

    catalog = CATALOG_BY_KEY[selected_object_type]
    element_meta = ELEMENTS[element]
    objects = load_catalog_items(selected_object_type, element)
    query = q.strip().casefold()
    if query:
        objects = [
            item
            for item in objects
            if query in item["name"].casefold()
        ]

    return templates.TemplateResponse(
        request=request,
        name="catalog.html",
        context={
            **_navigation_context(selected_object_type, element),
            "catalog": catalog,
            "element": element_meta,
            "objects": objects,
            "query": q,
            "refresh": get_refresh_status(),
        },
    )


@app.get("/characters/{slug}", response_class=HTMLResponse)
def character_detail(request: Request, slug: str):
    """Backward-compatible Kamihime detail route."""
    return object_detail(request, "kamihime", slug)


@app.get(
    "/objects/{object_type}/{slug}",
    response_class=HTMLResponse,
)
def object_detail(request: Request, object_type: str, slug: str):
    try:
        selected_object_type = normalize_object_type(object_type)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    item = get_catalog_item(selected_object_type, slug)
    if item is None:
        raise HTTPException(status_code=404, detail="Object not found")
    catalog = CATALOG_BY_KEY[selected_object_type]

    return templates.TemplateResponse(
        request=request,
        name=(
            "character.html"
            if selected_object_type == "kamihime"
            else "object_detail.html"
        ),
        context={
            **_navigation_context(
                selected_object_type,
                item["element"],
            ),
            "catalog": catalog,
            "item": item,
            "character": item,
            "display_info_rows": _display_info_rows(item),
        },
    )


@app.post("/api/update/{mode}", status_code=202)
def update_data(mode: str, provider: str = "deepl"):
    if mode not in {"latest", "database"}:
        raise HTTPException(status_code=404, detail="Update mode not found")
    selected_provider = provider.strip().lower()
    if selected_provider not in TRANSLATION_PROVIDERS:
        raise HTTPException(status_code=404, detail="Translation provider not found")
    started = start_update(mode, provider=selected_provider)
    if not started:
        raise HTTPException(status_code=409, detail="An update is already running")
    return get_refresh_status()


@app.post("/api/update/{object_type}/{mode}", status_code=202)
def update_object_data(
    object_type: str,
    mode: str,
    provider: str = "deepl",
):
    if mode not in {"latest", "database"}:
        raise HTTPException(status_code=404, detail="Update mode not found")
    selected_provider = provider.strip().lower()
    if selected_provider not in TRANSLATION_PROVIDERS:
        raise HTTPException(status_code=404, detail="Translation provider not found")
    try:
        selected_object_type = normalize_object_type(object_type)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    started = start_update(mode, selected_object_type, selected_provider)
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


@app.post("/api/translate/{object_type}/{provider}", status_code=202)
def translate_object_database(object_type: str, provider: str):
    if provider not in TRANSLATION_PROVIDERS:
        raise HTTPException(status_code=404, detail="Translation provider not found")
    try:
        selected_object_type = normalize_object_type(object_type)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    started = start_translation(provider, selected_object_type)
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
