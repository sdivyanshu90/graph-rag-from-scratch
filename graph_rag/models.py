from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator


def _dedupe_preserving_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        if value not in seen:
            deduped.append(value)
            seen.add(value)
    return deduped


class RelationshipTriple(BaseModel):
    source: str
    relation: str
    target: str

    @field_validator("source", "relation", "target")
    @classmethod
    def strip_and_validate(cls, value: str) -> str:
        cleaned_value = value.strip()
        if not cleaned_value:
            raise ValueError("triple fields must not be empty")
        return cleaned_value


class ExtractionResult(BaseModel):
    entities: list[str] = Field(default_factory=list)
    relationships: list[RelationshipTriple] = Field(default_factory=list)

    @field_validator("entities")
    @classmethod
    def normalize_entities(cls, values: list[str]) -> list[str]:
        cleaned_values = [value.strip() for value in values if value.strip()]
        return _dedupe_preserving_order(cleaned_values)

    @model_validator(mode="after")
    def include_relationship_entities(self) -> "ExtractionResult":
        relationship_entities: list[str] = []
        for relationship in self.relationships:
            relationship_entities.extend([relationship.source, relationship.target])

        self.entities = _dedupe_preserving_order(self.entities + relationship_entities)
        return self


class TextChunk(BaseModel):
    chunk_id: str
    source_id: str
    text: str
    chunk_index: int
    token_start: int
    token_end: int


class ChunkMention(BaseModel):
    chunk_id: str
    source_id: str
    text: str


class ChunkingConfig(BaseModel):
    chunk_size: int = 512
    chunk_overlap: int = 64
    encoding_name: str = "cl100k_base"

    @model_validator(mode="after")
    def validate_overlap(self) -> "ChunkingConfig":
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if self.chunk_overlap < 0:
            raise ValueError("chunk_overlap must not be negative")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        return self


class IngestionReport(BaseModel):
    source_id: str
    chunk_count: int
    entity_count: int
    relationship_count: int


class EntityMergeGroup(BaseModel):
    canonical_name: str
    merged_names: list[str] = Field(default_factory=list)

    @field_validator("merged_names")
    @classmethod
    def normalize_merged_names(cls, values: list[str]) -> list[str]:
        cleaned_values = [value.strip() for value in values if value.strip()]
        return _dedupe_preserving_order(cleaned_values)


class CommunityRecord(BaseModel):
    community_id: int
    node_names: list[str] = Field(default_factory=list)
    summary: str | None = None

    @field_validator("node_names")
    @classmethod
    def normalize_node_names(cls, values: list[str]) -> list[str]:
        cleaned_values = [value.strip() for value in values if value.strip()]
        return _dedupe_preserving_order(cleaned_values)


class EnrichmentReport(BaseModel):
    merge_groups: list[EntityMergeGroup] = Field(default_factory=list)
    communities: list[CommunityRecord] = Field(default_factory=list)
    embedded_node_count: int = 0
    summarized_community_count: int = 0


class RetrievedChunk(BaseModel):
    chunk_id: str
    source_id: str
    text: str


class NodeMatch(BaseModel):
    node_name: str
    score: float


class CommunityMatch(BaseModel):
    community_id: int
    score: float
    summary: str
    node_names: list[str] = Field(default_factory=list)

    @field_validator("node_names")
    @classmethod
    def normalize_match_node_names(cls, values: list[str]) -> list[str]:
        cleaned_values = [value.strip() for value in values if value.strip()]
        return _dedupe_preserving_order(cleaned_values)


class QueryProvenance(BaseModel):
    node_names: list[str] = Field(default_factory=list)
    community_ids: list[int] = Field(default_factory=list)
    chunk_ids: list[str] = Field(default_factory=list)
    source_ids: list[str] = Field(default_factory=list)

    @field_validator("node_names", "chunk_ids", "source_ids")
    @classmethod
    def normalize_string_lists(cls, values: list[str]) -> list[str]:
        cleaned_values = [value.strip() for value in values if value.strip()]
        return _dedupe_preserving_order(cleaned_values)

    @field_validator("community_ids")
    @classmethod
    def normalize_int_lists(cls, values: list[int]) -> list[int]:
        seen_ids: set[int] = set()
        deduped_ids: list[int] = []
        for value in values:
            if value not in seen_ids:
                deduped_ids.append(value)
                seen_ids.add(value)
        return deduped_ids


class QueryResult(BaseModel):
    mode: Literal["local", "global"]
    question: str
    answer: str
    context_text: str
    retrieved_chunks: list[RetrievedChunk] = Field(default_factory=list)
    node_matches: list[NodeMatch] = Field(default_factory=list)
    community_matches: list[CommunityMatch] = Field(default_factory=list)
    provenance: QueryProvenance = Field(default_factory=QueryProvenance)


class EntitySearchRecord(BaseModel):
    node_name: str
    aliases: list[str] = Field(default_factory=list)
    embedding: list[float] = Field(default_factory=list)

    @field_validator("node_name")
    @classmethod
    def validate_node_name(cls, value: str) -> str:
        cleaned_value = value.strip()
        if not cleaned_value:
            raise ValueError("node_name must not be empty")
        return cleaned_value

    @field_validator("aliases")
    @classmethod
    def normalize_aliases(cls, values: list[str]) -> list[str]:
        cleaned_values = [value.strip() for value in values if value.strip()]
        return _dedupe_preserving_order(cleaned_values)


class NodeHop(BaseModel):
    node_name: str
    hop_distance: int

    @field_validator("node_name")
    @classmethod
    def validate_hop_node_name(cls, value: str) -> str:
        cleaned_value = value.strip()
        if not cleaned_value:
            raise ValueError("node_name must not be empty")
        return cleaned_value


class NeighborhoodRelationship(BaseModel):
    source: str
    relation: str
    target: str
    chunk_id: str
    source_id: str
    chunk_text: str

    @field_validator("source", "relation", "target", "chunk_id", "source_id", "chunk_text")
    @classmethod
    def validate_relationship_fields(cls, value: str) -> str:
        cleaned_value = value.strip()
        if not cleaned_value:
            raise ValueError("neighborhood relationship fields must not be empty")
        return cleaned_value


class LocalSearchBundle(BaseModel):
    node_hops: list[NodeHop] = Field(default_factory=list)
    relationships: list[NeighborhoodRelationship] = Field(default_factory=list)
    retrieved_chunks: list[RetrievedChunk] = Field(default_factory=list)


class Neo4jSyncReport(BaseModel):
    entity_count: int
    relationship_count: int
    chunk_count: int
    community_count: int