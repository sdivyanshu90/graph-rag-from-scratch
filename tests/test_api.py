from __future__ import annotations

from fastapi.testclient import TestClient

from graph_rag.api import create_app
from graph_rag.api_models import EntityDetailResponse, EntityNeighbor, GraphStatsResponse, IngestResponse
from graph_rag.api_service import DuplicateSourceError, EntityNotFoundError, GraphNotReadyError
from graph_rag.models import EnrichmentReport, IngestionReport, QueryProvenance, QueryResult, RetrievedChunk


class FakeGraphRAGAPIService:
    def close(self) -> None:
        return None

    def ingest_text(self, *, source_id: str, text: str) -> IngestResponse:
        if source_id == "duplicate-doc":
            raise DuplicateSourceError("source_id 'duplicate-doc' already exists")
        return IngestResponse(
            ingestion=IngestionReport(
                source_id=source_id,
                chunk_count=1,
                entity_count=2,
                relationship_count=1,
            ),
            enrichment=EnrichmentReport(
                merge_groups=[],
                communities=[],
                embedded_node_count=2,
                summarized_community_count=1,
            ),
            graph_stats=GraphStatsResponse(
                node_count=2,
                edge_count=1,
                chunk_count=1,
                community_count=1,
            ),
            neo4j_sync=None,
            neo4j_sync_error=None,
        )

    def query(
        self,
        *,
        question: str,
        mode: str,
        top_k: int | None,
        max_hops: int | None,
        max_chunks: int | None,
    ) -> QueryResult:
        del top_k
        del max_hops
        del max_chunks
        if question == "empty":
            raise GraphNotReadyError("No node embeddings found. Run graph enrichment before local search.")
        return QueryResult(
            mode=mode,
            question=question,
            answer="Alice worked with Acme Corp in River County.",
            context_text="Supporting chunk excerpts:\n[doc-1:0 | doc-1] Alice worked with Acme Corp in River County.",
            retrieved_chunks=[
                RetrievedChunk(
                    chunk_id="doc-1:0",
                    source_id="doc-1",
                    text="Alice worked with Acme Corp in River County.",
                )
            ],
            provenance=QueryProvenance(
                node_names=["Alice", "Acme Corp"],
                community_ids=[0] if mode == "global" else [],
                chunk_ids=["doc-1:0"],
                source_ids=["doc-1"],
            ),
        )

    def graph_stats(self) -> GraphStatsResponse:
        return GraphStatsResponse(
            node_count=2,
            edge_count=1,
            chunk_count=1,
            community_count=1,
        )

    def entity_detail(self, *, name: str) -> EntityDetailResponse:
        if name == "Missing":
            raise EntityNotFoundError("entity 'Missing' was not found in the graph")
        return EntityDetailResponse(
            canonical_name="Alice",
            aliases=["Alice A."],
            community_id=0,
            source_ids=["doc-1"],
            neighbors=[
                EntityNeighbor(
                    neighbor_name="Acme Corp",
                    relations=["worked_for"],
                    direction="outgoing",
                )
            ],
            chunk_excerpts=[
                RetrievedChunk(
                    chunk_id="doc-1:0",
                    source_id="doc-1",
                    text="Alice worked with Acme Corp in River County.",
                )
            ],
        )


def test_post_ingest_returns_201_with_ingestion_and_enrichment_reports() -> None:
    client = TestClient(create_app(service=FakeGraphRAGAPIService()))
    response = client.post(
        "/ingest",
        json={
            "source_id": "doc-1",
            "text": "Alice worked with Acme Corp in River County.",
        },
    )

    assert response.status_code == 201
    payload = response.json()
    assert payload["ingestion"]["source_id"] == "doc-1"
    assert payload["graph_stats"]["node_count"] == 2


def test_post_ingest_returns_409_for_duplicate_source_id() -> None:
    client = TestClient(create_app(service=FakeGraphRAGAPIService()))
    response = client.post(
        "/ingest",
        json={
            "source_id": "duplicate-doc",
            "text": "Repeated text.",
        },
    )

    assert response.status_code == 409
    assert "already exists" in response.json()["detail"]


def test_post_query_returns_answer_and_provenance() -> None:
    client = TestClient(create_app(service=FakeGraphRAGAPIService()))
    response = client.post(
        "/query",
        json={
            "question": "What did Alice do?",
            "mode": "local",
            "top_k": 2,
            "max_hops": 1,
            "max_chunks": 3,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["mode"] == "local"
    assert payload["provenance"]["node_names"] == ["Alice", "Acme Corp"]


def test_post_query_returns_409_when_graph_is_not_ready() -> None:
    client = TestClient(create_app(service=FakeGraphRAGAPIService()))
    response = client.post(
        "/query",
        json={
            "question": "empty",
            "mode": "local",
        },
    )

    assert response.status_code == 409
    assert "Run graph enrichment" in response.json()["detail"]


def test_get_graph_stats_returns_counts() -> None:
    client = TestClient(create_app(service=FakeGraphRAGAPIService()))
    response = client.get("/graph/stats")

    assert response.status_code == 200
    assert response.json() == {
        "node_count": 2,
        "edge_count": 1,
        "chunk_count": 1,
        "community_count": 1,
    }


def test_get_graph_entity_returns_neighbors_and_chunks() -> None:
    client = TestClient(create_app(service=FakeGraphRAGAPIService()))
    response = client.get("/graph/entity/Alice")

    assert response.status_code == 200
    payload = response.json()
    assert payload["canonical_name"] == "Alice"
    assert payload["neighbors"][0]["neighbor_name"] == "Acme Corp"
    assert payload["chunk_excerpts"][0]["chunk_id"] == "doc-1:0"


def test_get_graph_entity_returns_404_when_missing() -> None:
    client = TestClient(create_app(service=FakeGraphRAGAPIService()))
    response = client.get("/graph/entity/Missing")

    assert response.status_code == 404
    assert "was not found" in response.json()["detail"]