from __future__ import annotations

import re
import unicodedata


VIETNAMESE_CHARACTERS = set(
    "ăâđêôơư"
    "áàảãạấầẩẫậắằẳẵặ"
    "éèẻẽẹếềểễệ"
    "íìỉĩị"
    "óòỏõọốồổỗộớờởỡợ"
    "úùủũụứừửữự"
    "ýỳỷỹỵ"
)
VIETNAMESE_TOKENS = {
    "ban",
    "biet",
    "cai",
    "chi",
    "cho",
    "co",
    "cua",
    "gioi",
    "hay",
    "he",
    "hieu",
    "khong",
    "ky",
    "la",
    "nao",
    "nhan",
    "nhung",
    "noi",
    "so",
    "thong",
    "thu",
    "tim",
    "toi",
    "trong",
    "ve",
    "va",
    "voi",
    "vu",
}
ENGLISH_TOKENS = {
    "about",
    "and",
    "are",
    "can",
    "compare",
    "does",
    "data",
    "effect",
    "find",
    "give",
    "how",
    "has",
    "includes",
    "information",
    "is",
    "list",
    "members",
    "no",
    "only",
    "of",
    "show",
    "series",
    "skill",
    "tell",
    "the",
    "there",
    "this",
    "what",
    "which",
    "who",
}


def _ascii_tokens(value: str) -> list[str]:
    normalized = unicodedata.normalize("NFKD", value.casefold())
    ascii_value = "".join(
        character for character in normalized if not unicodedata.combining(character)
    )
    return re.findall(r"[a-z0-9]+", ascii_value)


def detect_response_language(value: str, fallback: str = "en") -> str:
    lowered = value.casefold()
    if any(character in VIETNAMESE_CHARACTERS for character in lowered):
        return "vi"
    tokens = _ascii_tokens(value)
    vietnamese_score = sum(token in VIETNAMESE_TOKENS for token in tokens)
    english_score = sum(token in ENGLISH_TOKENS for token in tokens)
    if vietnamese_score >= 2 and vietnamese_score > english_score:
        return "vi"
    if english_score >= 2 and english_score >= vietnamese_score:
        return "en"
    return fallback if fallback in {"vi", "en"} else "en"


def language_name(code: str) -> str:
    return "Vietnamese" if code == "vi" else "English"


def guarded_question(original: str, standalone: str, language: str) -> str:
    selected = language_name(language)
    return (
        f"Required response language: {selected} ({language}). "
        f"Write the entire answer in {selected}; keep only official game names "
        "and unavoidable technical terms in their original form.\n"
        f"Original user message: {original}\n"
        "Grounded standalone query for retrieval only (do not infer the response "
        f"language from this line): {standalone}"
    )
