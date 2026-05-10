from __future__ import annotations

from graph_rag.enrichment import EntityMerger, GraphEnrichmentPipeline
from graph_rag.graph_store import NetworkXKnowledgeGraph
from graph_rag.models import ExtractionResult, RelationshipTriple, TextChunk
from graph_rag.neo4j_store import Neo4jKnowledgeGraph


class FakeResult:
    def __init__(self, rows: list[dict[str, object]]) -> None:
        self.rows = rows

    def data(self) -> list[dict[str, object]]:
        return list(self.rows)


class FakeSession:
    def __init__(self, *, scripted_rows: list[list[dict[str, object]]], calls: list[dict[str, object]]) -> None:
        self.scripted_rows = scripted_rows
        self.calls = calls

    def run(self, query: str, parameters: dict[str, object] | None = None) -> FakeResult:
        self.calls.append({"query": query, "parameters": parameters or {}})
        rows = self.scripted_rows.pop(0) if self.scripted_rows else []
        return FakeResult(rows)

    def __enter__(self) -> "FakeSession":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        del exc_type
        del exc
        del tb


class FakeDriver:
    def __init__(self, scripted_rows: list[list[dict[str, object]]] | None = None) -> None:
        self.scripted_rows = list(scripted_rows or [])
        self.calls: list[dict[str, object]] = []
        self.closed = False

    def session(self, *, database: str) -> FakeSession:
        self.calls.append({"database": database})
        return FakeSession(scripted_rows=self.scripted_rows, calls=self.calls)

    def close(self) -> None:
        self.closed = True


class FakeCommunityDetector:
    def detect(self, graph) -> list[list[str]]:
        del graph
        return [["Acme Corp", "Alice", "River County"]]


class FakeNodeEmbedder:
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [[float(index), float(len(text))] for index, text in enumerate(texts)]


class FakeCommunitySummarizer:
    def summarize(self, community) -> str:
        return f"Summary for {', '.join(community.node_names)}"


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

    GraphEnrichmentPipeline(
        merger=EntityMerger(),
        community_detector=FakeCommunityDetector(),
        node_embedder=FakeNodeEmbedder(),
        community_summarizer=FakeCommunitySummarizer(),
    ).enrich(graph_store)
    return graph_store


def test_neo4j_sync_from_networkx_writes_entities_relationships_and_communities() -> None:
    fake_driver = FakeDriver()
    store = Neo4jKnowledgeGraph(driver=fake_driver, database="neo4j")

    report = store.sync_from_networkx(build_enriched_graph())

    assert report.entity_count == 3
    assert report.relationship_count == 2
    assert report.chunk_count == 1
    assert report.community_count == 1

    executed_queries = [call["query"] for call in fake_driver.calls if "query" in call]
    assert any("DETACH DELETE node" in query for query in executed_queries)
    assert any("MERGE (entity_node:Entity" in query for query in executed_queries)
    assert any("MERGE (community_node:Community" in query for query in executed_queries)


def test_neo4j_store_reads_stats_entities_neighborhood_and_communities() -> None:
    fake_driver = FakeDriver(
        scripted_rows=[
            [{"node_count": 3, "edge_count": 2, "chunk_count": 1, "community_count": 1}],
            [{"node_name": "Alice", "aliases": [], "embedding": [1.0, 0.0]}],
            [
                {"node_name": "Alice", "hop_distance": 0},
                {"node_name": "Acme Corp", "hop_distance": 1},
            ],
            [
                {
                    "source": "Alice",
                    "relation": "worked_for",
                    "target": "Acme Corp",
                    "chunk_id": "doc-1:0",
                    "source_id": "doc-1",
                    "chunk_text": "Alice from Acme Corp worked with River County on flood sensors.",
                }
            ],
            [
                {
                    "chunk_id": "doc-1:0",
                    "source_id": "doc-1",
                    "text": "Alice from Acme Corp worked with River County on flood sensors.",
                }
            ],
            [
                {
                    "community_id": 0,
                    "summary": "Summary for Acme Corp, Alice, River County",
                    "node_names": ["Acme Corp", "Alice", "River County"],
                }
            ],
        ]
    )
    store = Neo4jKnowledgeGraph(driver=fake_driver, database="neo4j")

    assert store.stats() == {
        "node_count": 3,
        "edge_count": 2,
        "chunk_count": 1,
        "community_count": 1,
    }

    entity_records = store.list_entity_records()
    assert entity_records[0].node_name == "Alice"

    bundle = store.get_local_search_bundle(
        seed_node_names=["Alice"],
        max_hops=1,
        max_chunks=2,
    )
    assert bundle.node_hops[0].node_name == "Alice"
    assert bundle.relationships[0].relation == "worked_for"
    assert bundle.retrieved_chunks[0].chunk_id == "doc-1:0"

    communities = store.list_community_records()
    assert communities[0].community_id == 0
    assert communities[0].node_names == ["Acme Corp", "Alice", "River County"]