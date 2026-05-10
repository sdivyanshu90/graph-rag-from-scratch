from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from .models import EnrichmentReport, IngestionReport, Neo4jSyncReport, QueryResult, RetrievedChunk


class IngestRequest(BaseModel):
    source_id: str
    text: str

    @field_validator("source_id", "text")
    @classmethod
    def strip_required_fields(cls, value: str) -> str:
        cleaned_value = value.strip()
        if not cleaned_value:
            raise ValueError("request fields must not be empty")
        return cleaned_value


class GraphStatsResponse(BaseModel):
    node_count: int
    edge_count: int
    chunk_count: int
    community_count: int


class IngestResponse(BaseModel):
    ingestion: IngestionReport
    enrichment: EnrichmentReport
    graph_stats: GraphStatsResponse
    neo4j_sync: Neo4jSyncReport | None = None
    neo4j_sync_error: str | None = None


class QueryRequest(BaseModel):
    question: str
    mode: Literal["local", "global"] = "local"
    top_k: int | None = None
    max_hops: int | None = None
    max_chunks: int | None = None

    @field_validator("question")
    @classmethod
    def strip_question(cls, value: str) -> str:
        cleaned_value = value.strip()
        if not cleaned_value:
            raise ValueError("question must not be empty")
        return cleaned_value

    @model_validator(mode="after")
    def validate_optional_limits(self) -> "QueryRequest":
        if self.top_k is not None and self.top_k <= 0:
            raise ValueError("top_k must be positive when provided")
        if self.max_hops is not None and self.max_hops < 0:
            raise ValueError("max_hops must not be negative when provided")
        if self.max_chunks is not None and self.max_chunks <= 0:
            raise ValueError("max_chunks must be positive when provided")
        return self


class EntityNeighbor(BaseModel):
    neighbor_name: str
    relations: list[str] = Field(default_factory=list)
    direction: Literal["incoming", "outgoing", "both"]

    @field_validator("neighbor_name")
    @classmethod
    def validate_neighbor_name(cls, value: str) -> str:
        cleaned_value = value.strip()
        if not cleaned_value:
            raise ValueError("neighbor_name must not be empty")
        return cleaned_value

    @field_validator("relations")
    @classmethod
    def normalize_relations(cls, values: list[str]) -> list[str]:
        return sorted({value.strip() for value in values if value.strip()}, key=str.casefold)


class EntityDetailResponse(BaseModel):
    canonical_name: str
    aliases: list[str] = Field(default_factory=list)
    community_id: int | None = None
    source_ids: list[str] = Field(default_factory=list)
    neighbors: list[EntityNeighbor] = Field(default_factory=list)
    chunk_excerpts: list[RetrievedChunk] = Field(default_factory=list)

    @field_validator("canonical_name")
    @classmethod
    def validate_canonical_name(cls, value: str) -> str:
        cleaned_value = value.strip()
        if not cleaned_value:
            raise ValueError("canonical_name must not be empty")
        return cleaned_value

    @field_validator("aliases", "source_ids")
    @classmethod
    def normalize_string_lists(cls, values: list[str]) -> list[str]:
        cleaned_values: list[str] = []
        seen_values: set[str] = set()
        for value in values:
            cleaned_value = value.strip()
            if cleaned_value and cleaned_value not in seen_values:
                cleaned_values.append(cleaned_value)
                seen_values.add(cleaned_value)
        return cleaned_values