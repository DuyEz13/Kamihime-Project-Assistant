from __future__ import annotations

from collections.abc import Iterable

from .retrieval import contains_normalized_phrase, normalize_text


SECTION_ORDER = {
    "kamihime": ("basic", "burst", "ability", "assist"),
    "eidolon": ("basic", "stats", "summon_effect", "main_effect", "sub_effect"),
    "weapon": ("basic", "stats", "burst_effects", "weapon_skills"),
}

SECTION_PHRASES = {
    "kamihime": {
        "basic": (
            "basic data",
            "thong tin co ban",
            "release date",
            "ngay phat hanh",
            "acquisition method",
            "cach nhan",
            "unlock weapon",
            "mo khoa weapon",
            "preferred weapon",
            "vu khi ua thich",
            "what element",
            "which element",
            "thuoc he nao",
            "he gi",
            "max level",
            "cap toi da",
        ),
        "burst": ("burst", "ougi"),
        "ability": ("ability", "abilities", "active skill", "ky nang chu dong"),
        "assist": ("assist", "passive", "ky nang bi dong"),
    },
    "eidolon": {
        "basic": (
            "basic data",
            "thong tin co ban",
            "release date",
            "ngay phat hanh",
            "acquisition method",
            "cach nhan",
            "return items",
            "vat pham quy doi",
            "what element",
            "which element",
            "thuoc he nao",
            "he gi",
        ),
        "stats": (
            "stat",
            "stats",
            "chi so",
            "hp",
            "attack",
            "tan cong",
            "max level",
            "cap toi da",
        ),
        "summon_effect": (
            "summon effect",
            "summoning effect",
            "hieu ung trieu hoi",
        ),
        "main_effect": ("main effect", "hieu ung chinh"),
        "sub_effect": ("sub effect", "sub effects", "hieu ung phu"),
    },
    "weapon": {
        "basic": (
            "basic data",
            "thong tin co ban",
            "release date",
            "ngay phat hanh",
            "acquisition method",
            "cach nhan",
            "unlock kamihime",
            "mo khoa kamihime",
            "weapon type",
            "loai vu khi",
            "what element",
            "which element",
            "thuoc he nao",
            "he gi",
        ),
        "stats": (
            "stat",
            "stats",
            "chi so",
            "hp",
            "attack",
            "tan cong",
            "max level",
            "cap toi da",
        ),
        "burst_effects": ("burst", "burst effect", "burst effects", "ougi"),
        "weapon_skills": ("weapon skill", "weapon skills", "ky nang vu khi"),
    },
}


def detect_requested_sections(
    message: str,
    object_types: Iterable[str],
) -> dict[str, list[str]]:
    normalized = normalize_text(message)
    result: dict[str, list[str]] = {}
    for object_type in dict.fromkeys(object_types):
        matched = {
            section
            for section, phrases in SECTION_PHRASES.get(object_type, {}).items()
            if any(
                contains_normalized_phrase(normalized, phrase)
                for phrase in phrases
            )
        }
        if matched:
            result[object_type] = [
                section
                for section in SECTION_ORDER[object_type]
                if section in matched
            ]
    return result
