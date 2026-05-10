from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from graph_rag.enrichment import EntityMerger, GraphEnrichmentPipeline
from graph_rag.graph_store import NetworkXKnowledgeGraph
from graph_rag.models import ExtractionResult, RelationshipTriple, TextChunk


class FakeCommunityDetector:
    def detect(self, graph) -> list[list[str]]:
        del graph
        return [
            ["Acme Corp", "Alice", "River County"],
            ["Solar Lab", "Bob", "Metro Grid"],
        ]


class FakeNodeEmbedder:
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [[float(index), float(len(text))] for index, text in enumerate(texts)]


class FakeCommunitySummarizer:
    def summarize(self, community) -> str:
        members = ", ".join(community.node_names)
        return f"Community {community.community_id} centers on {members}."


def demo_similarity(left_name: str, right_name: str) -> float:
    normalized_left = left_name.casefold().replace(".", "").strip()
    normalized_right = right_name.casefold().replace(".", "").strip()
    return 100.0 if normalized_left == normalized_right else 0.0


def build_chunk(chunk_id: str, source_id: str, text: str) -> TextChunk:
    return TextChunk(
        chunk_id=chunk_id,
        source_id=source_id,
        text=text,
        chunk_index=int(chunk_id.split(":")[-1]),
        token_start=0,
        token_end=len(text.split()),
    )


def main() -> None:
    graph_store = NetworkXKnowledgeGraph()

    graph_store.add_extraction(
        chunk=build_chunk(
            chunk_id="doc-1:0",
            source_id="doc-1",
            text="Alice from Acme Corp worked with River County on flood sensors.",
        ),
        extraction=ExtractionResult(
            entities=["Alice", "Acme Corp", "River County"],
            relationships=[
                RelationshipTriple(
                    source="Alice",
                    relation="worked_for",
                    target="Acme Corp",
                ),
                RelationshipTriple(
                    source="Acme Corp",
                    relation="partnered_with",
                    target="River County",
                ),
            ],
        ),
    )
    graph_store.add_extraction(
        chunk=build_chunk(
            chunk_id="doc-2:0",
            source_id="doc-2",
            text="Acme Corp. deployed flood sensors across River County.",
        ),
        extraction=ExtractionResult(
            entities=["Acme Corp.", "River County"],
            relationships=[
                RelationshipTriple(
                    source="Acme Corp.",
                    relation="deployed_in",
                    target="River County",
                )
            ],
        ),
    )
    graph_store.add_extraction(
        chunk=build_chunk(
            chunk_id="doc-3:0",
            source_id="doc-3",
            text="Bob manages Solar Lab, which supplies Metro Grid.",
        ),
        extraction=ExtractionResult(
            entities=["Bob", "Solar Lab", "Metro Grid"],
            relationships=[
                RelationshipTriple(
                    source="Bob",
                    relation="manages",
                    target="Solar Lab",
                ),
                RelationshipTriple(
                    source="Solar Lab",
                    relation="supplies",
                    target="Metro Grid",
                ),
            ],
        ),
    )

    enrichment_pipeline = GraphEnrichmentPipeline(
        merger=EntityMerger(threshold=95.0, similarity_calculator=demo_similarity),
        community_detector=FakeCommunityDetector(),
        node_embedder=FakeNodeEmbedder(),
        community_summarizer=FakeCommunitySummarizer(),
    )
    report = enrichment_pipeline.enrich(graph_store)

    print("Enrichment report:")
    print(report.model_dump_json(indent=2))

    print("\nGraph stats:")
    print(graph_store.stats())

    print("\nNodes after merge and enrichment:")
    for node_name, attrs in graph_store.graph.nodes(data=True):
        print(f"  - {node_name}: {attrs}")

    print("\nCommunity summaries:")
    for community_id, summary in graph_store.graph.graph["community_summaries"].items():
        print(f"  - {community_id}: {summary}")


if __name__ == "__main__":
    main()