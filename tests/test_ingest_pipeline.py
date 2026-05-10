from __future__ import annotations

import json

from graph_rag.chunking import TokenChunker
from graph_rag.extractor import EntityRelationshipExtractor
from graph_rag.graph_store import NetworkXKnowledgeGraph
from graph_rag.ingest import IngestionPipeline
from graph_rag.models import ChunkingConfig


class FakeLLMClient:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload

    def complete_json(self, *, system_prompt: str, user_prompt: str) -> str:
        del system_prompt
        del user_prompt
        return json.dumps(self.payload)


def test_ingestion_pipeline_builds_graph_with_chunk_provenance() -> None:
    pipeline = IngestionPipeline(
        chunker=TokenChunker(ChunkingConfig(chunk_size=64, chunk_overlap=8)),
        extractor=EntityRelationshipExtractor(
            FakeLLMClient(
                {
                    "entities": ["Alice", "Project Atlas"],
                    "relationships": [
                        {
                            "source": "Alice",
                            "relation": "worked_on",
                            "target": "Project Atlas",
                        }
                    ],
                }
            )
        ),
        graph_store=NetworkXKnowledgeGraph(),
    )

    report = pipeline.ingest_text(
        source_id="doc-1",
        text="Alice worked on Project Atlas and delivered the prototype.",
    )
    graph = pipeline.graph_store.graph

    assert report.chunk_count == 1
    assert report.entity_count == 2
    assert report.relationship_count == 1
    assert graph.has_node("Alice")
    assert graph.has_node("Project Atlas")
    assert graph.nodes["Alice"]["mentions"][0]["chunk_id"] == "doc-1:0"
    assert graph.nodes["Alice"]["mentions"][0]["text"] == (
        "Alice worked on Project Atlas and delivered the prototype."
    )

    edges = list(graph.edges(data=True, keys=True))
    assert len(edges) == 1
    _, _, _, edge_data = edges[0]
    assert edge_data["relation"] == "worked_on"
    assert edge_data["chunk_text"] == (
        "Alice worked on Project Atlas and delivered the prototype."
    )


def test_ingestion_pipeline_keeps_empty_extractions_from_crashing() -> None:
    pipeline = IngestionPipeline(
        chunker=TokenChunker(ChunkingConfig(chunk_size=64, chunk_overlap=8)),
        extractor=EntityRelationshipExtractor(
            FakeLLMClient({"entities": [], "relationships": []})
        ),
        graph_store=NetworkXKnowledgeGraph(),
    )

    report = pipeline.ingest_text(source_id="doc-2", text="It rained all morning.")

    assert report.chunk_count == 1
    assert report.entity_count == 0
    assert report.relationship_count == 0
    assert pipeline.graph_store.stats()["chunk_count"] == 1
    assert pipeline.graph_store.stats()["node_count"] == 0