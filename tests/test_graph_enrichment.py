from __future__ import annotations

from graph_rag.enrichment import EntityMerger, GraphEnrichmentPipeline
from graph_rag.graph_store import NetworkXKnowledgeGraph
from graph_rag.models import ExtractionResult, RelationshipTriple, TextChunk


class FakeCommunityDetector:
    def detect(self, graph) -> list[list[str]]:
        del graph
        return [["Acme Corp", "Alice", "River County"], ["Solar Lab", "Bob", "Metro Grid"]]


class FakeNodeEmbedder:
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [[float(index), float(len(text))] for index, text in enumerate(texts)]


class FakeCommunitySummarizer:
    def summarize(self, community) -> str:
        return f"Summary for {', '.join(community.node_names)}"


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


def test_entity_merger_merges_fuzzy_duplicates_and_preserves_provenance() -> None:
    graph_store = NetworkXKnowledgeGraph()
    graph_store.add_extraction(
        chunk=build_chunk(
            chunk_id="doc-1:0",
            source_id="doc-1",
            text="Alice works for Acme Corp.",
        ),
        extraction=ExtractionResult(
            entities=["Alice", "Acme Corp"],
            relationships=[
                RelationshipTriple(
                    source="Alice",
                    relation="worked_for",
                    target="Acme Corp",
                )
            ],
        ),
    )
    graph_store.add_extraction(
        chunk=build_chunk(
            chunk_id="doc-2:0",
            source_id="doc-2",
            text="Acme Corp. partnered with River County.",
        ),
        extraction=ExtractionResult(
            entities=["Acme Corp.", "River County"],
            relationships=[
                RelationshipTriple(
                    source="Acme Corp.",
                    relation="partnered_with",
                    target="River County",
                )
            ],
        ),
    )

    merge_groups = EntityMerger(
        threshold=95.0,
        similarity_calculator=demo_similarity,
    ).merge_graph(graph_store.graph)
    graph = graph_store.graph

    assert len(merge_groups) == 1
    assert merge_groups[0].canonical_name == "Acme Corp"
    assert merge_groups[0].merged_names == ["Acme Corp."]
    assert graph.has_node("Acme Corp")
    assert not graph.has_node("Acme Corp.")
    assert graph.nodes["Acme Corp"]["aliases"] == ["Acme Corp."]
    assert len(graph.nodes["Acme Corp"]["mentions"]) == 2

    edges = list(graph.edges(data=True, keys=True))
    edge_relations = {(source, edge_data["relation"], target) for source, target, _, edge_data in edges}
    assert ("Alice", "worked_for", "Acme Corp") in edge_relations
    assert ("Acme Corp", "partnered_with", "River County") in edge_relations


def test_graph_enrichment_pipeline_assigns_communities_embeddings_and_summaries() -> None:
    graph_store = NetworkXKnowledgeGraph()
    graph_store.add_extraction(
        chunk=build_chunk(
            chunk_id="doc-1:0",
            source_id="doc-1",
            text="Alice works for Acme Corp in River County.",
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
                    relation="located_in",
                    target="River County",
                ),
            ],
        ),
    )
    graph_store.add_extraction(
        chunk=build_chunk(
            chunk_id="doc-2:0",
            source_id="doc-2",
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

    pipeline = GraphEnrichmentPipeline(
        merger=EntityMerger(threshold=95.0, similarity_calculator=demo_similarity),
        community_detector=FakeCommunityDetector(),
        node_embedder=FakeNodeEmbedder(),
        community_summarizer=FakeCommunitySummarizer(),
    )

    report = pipeline.enrich(graph_store)
    graph = graph_store.graph

    assert len(report.merge_groups) == 0
    assert len(report.communities) == 2
    assert report.embedded_node_count == 6
    assert report.summarized_community_count == 2
    assert graph.nodes["Alice"]["community_id"] == 0
    assert graph.nodes["Solar Lab"]["community_id"] == 1
    assert graph.nodes["Acme Corp"]["embedding_text"].startswith("Entity: Acme Corp")
    assert len(graph.nodes["Acme Corp"]["embedding"]) == 2
    assert graph.graph["community_summaries"][0] == "Summary for Acme Corp, Alice, River County"
    assert graph_store.stats()["community_count"] == 2