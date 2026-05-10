from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from graph_rag.config import get_settings
from graph_rag.enrichment import EntityMerger, GraphEnrichmentPipeline
from graph_rag.graph_store import NetworkXKnowledgeGraph
from graph_rag.models import ExtractionResult, RelationshipTriple, TextChunk
from graph_rag.neo4j_query import Neo4jQueryEngine
from graph_rag.neo4j_store import Neo4jKnowledgeGraph


class KeywordEmbedder:
    KEYWORDS = [
        "alice",
        "acme",
        "river",
        "flood",
        "solar",
        "metro",
        "energy",
        "county",
    ]

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            normalized_text = text.casefold()
            vectors.append([float(keyword in normalized_text) for keyword in self.KEYWORDS])
        return vectors


class FakeCommunityDetector:
    def detect(self, graph) -> list[list[str]]:
        del graph
        return [
            ["Acme Corp", "Alice", "River County"],
            ["Bob", "Metro Grid", "Solar Lab"],
        ]


class FakeCommunitySummarizer:
    def summarize(self, community) -> str:
        if community.community_id == 0:
            return "Acme Corp, Alice, and River County revolve around flood-sensor work and county resilience."
        return "Bob, Solar Lab, and Metro Grid revolve around energy supply and grid operations."


class FakeAnswerLLM:
    def complete_text(self, *, system_prompt: str, user_prompt: str) -> str:
        del user_prompt
        if "entity-focused" in system_prompt:
            return "Neo4j local answer: Alice worked with Acme Corp on flood-sensor work in River County."
        return "Neo4j global answer: The graph centers on flood resilience work, with a separate energy-supply theme."


def build_chunk(chunk_id: str, source_id: str, text: str) -> TextChunk:
    return TextChunk(
        chunk_id=chunk_id,
        source_id=source_id,
        text=text,
        chunk_index=int(chunk_id.split(":")[-1]),
        token_start=0,
        token_end=len(text.split()),
    )


def build_enriched_graph() -> NetworkXKnowledgeGraph:
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
                RelationshipTriple(source="Alice", relation="worked_for", target="Acme Corp"),
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
            text="Acme Corp deployed flood sensors across River County.",
        ),
        extraction=ExtractionResult(
            entities=["Acme Corp", "River County"],
            relationships=[
                RelationshipTriple(
                    source="Acme Corp",
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
                RelationshipTriple(source="Bob", relation="manages", target="Solar Lab"),
                RelationshipTriple(
                    source="Solar Lab",
                    relation="supplies",
                    target="Metro Grid",
                ),
            ],
        ),
    )

    GraphEnrichmentPipeline(
        merger=EntityMerger(),
        community_detector=FakeCommunityDetector(),
        node_embedder=KeywordEmbedder(),
        community_summarizer=FakeCommunitySummarizer(),
    ).enrich(graph_store)
    return graph_store


def print_query_comparisons() -> None:
    print("NetworkX vs Cypher comparison:\n")
    print("1. Graph stats")
    print("NetworkX:")
    print("  graph.number_of_nodes(), graph.number_of_edges(), len(graph.graph['chunks'])")
    print("Cypher:")
    print("  MATCH (entity:Entity) WITH count(entity) AS node_count")
    print("  MATCH ()-[rel:RELATES_TO]->() WITH node_count, count(rel) AS edge_count")
    print("  MATCH (chunk:Chunk) RETURN node_count, edge_count, count(chunk) AS chunk_count")

    print("\n2. One-hop neighborhood around Alice")
    print("NetworkX:")
    print("  projection = graph.to_undirected()")
    print("  nx.single_source_shortest_path_length(projection, 'Alice', cutoff=1)")
    print("Cypher:")
    print("  MATCH path = (:Entity {canonical_name: $name})-[:RELATES_TO*0..1]-(:Entity)")
    print("  RETURN path")

    print("\n3. Community summaries for global search")
    print("NetworkX:")
    print("  graph.graph['community_summaries']")
    print("Cypher:")
    print("  MATCH (community:Community) RETURN community.community_id, community.summary")


def main() -> None:
    print_query_comparisons()

    settings = get_settings()
    if not settings.neo4j_uri or not settings.neo4j_username or settings.neo4j_password is None:
        print("\nSkipping live Neo4j sync because NEO4J_URI, NEO4J_USERNAME, or NEO4J_PASSWORD is not set.")
        return

    graph_store = build_enriched_graph()
    neo4j_store = Neo4jKnowledgeGraph.from_settings(settings)
    try:
        sync_report = neo4j_store.sync_from_networkx(graph_store)
        print("\nNeo4j sync report:")
        print(sync_report.model_dump_json(indent=2))
        print("Neo4j stats:", neo4j_store.stats())

        query_engine = Neo4jQueryEngine(
            query_embedder=KeywordEmbedder(),
            answer_llm=FakeAnswerLLM(),
        )
        question = "What did Alice do in River County?"

        local_result = query_engine.local_search(
            graph_store=neo4j_store,
            question=question,
            top_k=2,
            max_hops=1,
            max_chunks=3,
        )
        print("\nNeo4j local search context:\n" + local_result.context_text)
        print("\nNeo4j local answer:", local_result.answer)

        global_result = query_engine.global_search(
            graph_store=neo4j_store,
            question=question,
            top_k_communities=2,
        )
        print("\nNeo4j global search context:\n" + global_result.context_text)
        print("\nNeo4j global answer:", global_result.answer)
    finally:
        neo4j_store.close()


if __name__ == "__main__":
    main()