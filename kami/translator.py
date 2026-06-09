import json
import os
import re
from pathlib import Path


TRANSLATION_PROMPT = """Translate this Kamihime Project character record from Japanese to natural English.
Return JSON only. Preserve the exact JSON structure and all numeric values, URLs, game abbreviations,
percentages, turn notation, and proper nouns when no established English form is known.
Translate nested field names as well, but keep these top-level keys unchanged: info, skill, flavor.

JSON:
{payload}
"""


def _extract_json(text: str) -> dict:
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    value = json.loads(cleaned)
    if not isinstance(value, dict):
        raise ValueError("Translation response was not a JSON object")
    return value


def _write_jsonl(records: list[dict], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary.replace(destination)


def translate_jsonl(source: Path, destination: Path) -> int:
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Set GEMINI_API_KEY or GOOGLE_API_KEY before refreshing data"
        )

    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
    except ImportError as exc:
        raise RuntimeError(
            "langchain-google-genai is required for translation"
        ) from exc

    model = ChatGoogleGenerativeAI(
        model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
        google_api_key=api_key,
        temperature=0,
    )

    translated: list[dict] = []
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            prompt = TRANSLATION_PROMPT.format(
                payload=json.dumps(record, ensure_ascii=False)
            )
            response = model.invoke(prompt)
            content = response.content
            if isinstance(content, list):
                content = "".join(
                    part.get("text", "") if isinstance(part, dict) else str(part)
                    for part in content
                )
            try:
                translated.append(_extract_json(str(content)))
            except (json.JSONDecodeError, ValueError) as exc:
                raise RuntimeError(
                    f"Invalid translation JSON at record {line_number}"
                ) from exc

    _write_jsonl(translated, destination)
    return len(translated)
