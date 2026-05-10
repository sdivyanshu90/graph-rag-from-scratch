from __future__ import annotations

import networkx as nx

from .models import ChunkMention, ExtractionResult, TextChunk


class NetworkXKnowledgeGraph:
    def __init__(self) -> None:
        self.graph = nx.MultiDiGraph()
        self.graph.graph["chunks"] = {}

    def add_extraction(self, *, chunk: TextChunk, extraction: ExtractionResult) -> None:
        self.graph.graph["chunks"][chunk.chunk_id] = chunk.model_dump()

        for entity_name in extraction.entities:
            self._upsert_entity(entity_name=entity_name, chunk=chunk)

        for relationship_index, relationship in enumerate(extraction.relationships):
            self._upsert_entity(entity_name=relationship.source, chunk=chunk)
            self._upsert_entity(entity_name=relationship.target, chunk=chunk)

            self.graph.add_edge(
                relationship.source,
                relationship.target,
                key=f"{chunk.chunk_id}:{relationship_index}",
                relation=relationship.relation,
                chunk_id=chunk.chunk_id,
                source_id=chunk.source_id,
                chunk_text=chunk.text,
            )

    def stats(self) -> dict[str, int]:
        return {
            "node_count": self.graph.number_of_nodes(),
            "edge_count": self.graph.number_of_edges(),
            "chunk_count": len(self.graph.graph["chunks"]),
            "community_count": len(self.graph.graph.get("communities", {})),
        }

    def _upsert_entity(self, *, entity_name: str, chunk: TextChunk) -> None:
        if not self.graph.has_node(entity_name):
            self.graph.add_node(
                entity_name,
                name=entity_name,
                canonical_name=entity_name,
                aliases=[],
                mentions=[],
                source_ids=[],
            )

        node_data = self.graph.nodes[entity_name]
        existing_mentions: list[dict[str, str]] = node_data["mentions"]
        new_mention = ChunkMention(
            chunk_id=chunk.chunk_id,
            source_id=chunk.source_id,
            text=chunk.text,
        ).model_dump()

        if all(mention["chunk_id"] != chunk.chunk_id for mention in existing_mentions):
            existing_mentions.append(new_mention)

        source_ids: list[str] = node_data["source_ids"]
        if chunk.source_id not in source_ids:
            source_ids.append(chunk.source_id)