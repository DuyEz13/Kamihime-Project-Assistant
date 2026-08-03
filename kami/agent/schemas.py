from __future__ import annotations

from typing import Literal, TypedDict

from pydantic import BaseModel, Field


ObjectType = Literal["kamihime", "eidolon", "weapon"]


class EntityQuery(BaseModel):
    mention: str = Field(description="Entity mention in the user message")
    name: str | None = Field(
        default=None,
        description="Resolved or normalized object name when known",
    )
    object_type: ObjectType | None = None
    series_key: str | None = Field(
        default=None,
        description="Deterministically grounded series identity when known",
    )
    element: str | None = None
    skills: list[str] = Field(default_factory=list)
    retrieval_query: str = Field(
        default="",
        description="Standalone English retrieval query for this entity",
    )


class QueryPlan(BaseModel):
    in_domain: bool
    standalone_question: str
    intent: Literal[
        "lookup",
        "compare",
        "recommend",
        "filter",
        "relationship",
        "explain",
    ] = "lookup"
    target_types: list[ObjectType] = Field(default_factory=list)
    entities: list[EntityQuery] = Field(default_factory=list)
    elements: list[str] = Field(default_factory=list)
    sort_by: Literal["relevance", "release_date"] = "relevance"
    sort_order: Literal["asc", "desc"] = "desc"
    result_limit: int | None = Field(default=None, ge=1, le=100)
    include_ties: bool = False
    needs_clarification: bool = False
    clarification_question: str | None = None


class Evidence(TypedDict):
    content: str
    score: float
    object_type: str
    slug: str
    name: str
    element: str
    section: str
    local_url: str
    available_sections: list[str]
    included_sections: list[str]
    coverage_complete: bool
    retrieval_mode: str
    selection_mode: str
    omitted_sections: list[str]
    omission_reason: str
    effect_group_id: str
    effect_is_shared: bool
    effect_variant_fields: list[str]
    series_key: str
    series_name: str
    series_elements: list[str]
    series_expected_elements: list[str]
    series_lifecycle: str
    series_catalog_elements: list[str]
    series_catalog_member_count: int
    series_retrieved_member_count: int
    series_unreleased_elements: list[str]
    series_missing_elements: list[str]
    series_coverage_complete: bool


class AgentState(TypedDict, total=False):
    session_id: str
    client_id: str
    provider: str
    model: str
    message: str
    history: list[dict]
    memory_state: dict
    plan: QueryPlan
    evidence: list[Evidence]
    sources: list[dict]
    context: str
    answer: str
