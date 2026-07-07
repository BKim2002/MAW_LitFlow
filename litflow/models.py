from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class RunConfig:
    topic: str
    out: str
    year_from: int | None = None
    year_to: int | None = None
    language: str = "ko"
    max_raw: int = 1500
    max_deep: int = 50
    user_files: str | None = None
    per_source_cap: int = 300
    agent_mode: str = "auto"
    agent_model: str | None = None
    recall_mode: str = "high"
    web_search_provider: str = "serpapi"
    web_search_token_env: str = "SERPAPI_API_KEY"
    output_format: str = "files"
    notion_parent: str | None = None
    notion_run_page: str | None = None
    notion_token_env: str = "NOTION_TOKEN"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SearchQuery:
    source: str
    query: str
    intent: str
    search_round: int = 1
    facets: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RawRecord:
    source: str
    source_id: str
    title: str
    authors: list[str] = field(default_factory=list)
    year: int | None = None
    venue: str = ""
    doi: str = ""
    abstract: str = ""
    url: str = ""
    citation_count: int | None = None
    source_database: str = ""
    open_access_pdf: str = ""
    found_by: str = ""
    search_round: int = 1
    facet_matches: list[str] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Record:
    record_id: str
    title: str
    authors: list[str] = field(default_factory=list)
    year: int | None = None
    venue: str = ""
    doi: str = ""
    source_ids: dict[str, str] = field(default_factory=dict)
    abstract: str = ""
    url: str = ""
    citation_count: int | None = None
    source_database: list[str] = field(default_factory=list)
    evidence_level: str = "metadata_only"
    relevance_score: int = 0
    inclusion_status: str = "unscreened"
    exclusion_reason: str = ""
    summary_status: str = "pending"
    open_access_pdf: str = ""
    found_by: list[str] = field(default_factory=list)
    search_round: int = 1
    facet_matches: list[str] = field(default_factory=list)
    coverage_warning: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Record":
        return cls(**data)


@dataclass
class SearchLogEntry:
    source: str
    query: str
    timestamp: str
    endpoint: str
    result_count: int = 0
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
