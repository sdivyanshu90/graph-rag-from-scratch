from __future__ import annotations

from graph_rag.models import CommunityRecord, EntitySearchRecord, LocalSearchBundle, NeighborhoodRelationship, NodeHop, RetrievedChunk
from graph_rag.neo4j_query import Neo4jQueryEngine


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


class FakeAnswerLLM:
    def complete_text(self, *, system_prompt: str, user_prompt: str) -> str:
        del user_prompt
        if "entity-focused" in system_prompt:
            return "Neo4j local answer: Alice worked with Acme Corp on flood-sensor work in River County."
        return "Neo4j global answer: The graph centers on flood resilience work, with a separate energy-supply theme."

    def stream_text(self, *, system_prompt: str, user_prompt: str):
        del user_prompt
        if "entity-focused" in system_prompt:
            yield "Neo4j local answer: Alice worked with Acme Corp "
            yield "on flood-sensor work in River County."
            return

        yield "Neo4j global answer: The graph centers on flood resilience work, "
        yield "with a separate energy-supply theme."


class FakeNeo4jStore:
    def list_entity_records(self) -> list[EntitySearchRecord]:
        return [
            EntitySearchRecord(
                node_name="Alice",
                aliases=[],
                embedding=[1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            ),
            EntitySearchRecord(
                node_name="Acme Corp",
                aliases=[],
                embedding=[0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            ),
            EntitySearchRecord(
                node_name="River County",
                aliases=[],
                embedding=[0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            ),
            EntitySearchRecord(
                node_name="Solar Lab",
                aliases=[],
                embedding=[0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            ),
        ]

    def get_local_search_bundle(self, *, seed_node_names: list[str], max_hops: int, max_chunks: int) -> LocalSearchBundle:
        del seed_node_names
        del max_hops
        del max_chunks
        return LocalSearchBundle(
            node_hops=[
                NodeHop(node_name="Alice", hop_distance=0),
                NodeHop(node_name="Acme Corp", hop_distance=1),
                NodeHop(node_name="River County", hop_distance=1),
            ],
            relationships=[
                NeighborhoodRelationship(
                    source="Alice",
                    relation="worked_for",
                    target="Acme Corp",
                    chunk_id="doc-1:0",
                    source_id="doc-1",
                    chunk_text="Alice from Acme Corp worked with River County on flood sensors.",
                )
            ],
            retrieved_chunks=[
                RetrievedChunk(
                    chunk_id="doc-1:0",
                    source_id="doc-1",
                    text="Alice from Acme Corp worked with River County on flood sensors.",
                ),
                RetrievedChunk(
                    chunk_id="doc-2:0",
                    source_id="doc-2",
                    text="Acme Corp deployed flood sensors across River County.",
                ),
            ],
        )

    def list_community_records(self) -> list[CommunityRecord]:
        return [
            CommunityRecord(
                community_id=0,
                node_names=["Acme Corp", "Alice", "River County"],
                summary="Acme Corp, Alice, and River County revolve around flood-sensor work and county resilience.",
            ),
            CommunityRecord(
                community_id=1,
                node_names=["Bob", "Metro Grid", "Solar Lab"],
                summary="Bob, Solar Lab, and Metro Grid revolve around energy supply and grid operations.",
            ),
        ]


def test_neo4j_local_search_reconstructs_context_and_streams_answer() -> None:
    query_engine = Neo4jQueryEngine(
        query_embedder=KeywordEmbedder(),
        answer_llm=FakeAnswerLLM(),
    )
    streamed_tokens: list[str] = []

    result = query_engine.local_search(
        graph_store=FakeNeo4jStore(),
        question="What did Alice do in River County?",
        top_k=2,
        max_hops=1,
        max_chunks=3,
        on_token=streamed_tokens.append,
    )

    assert result.mode == "local"
    assert result.node_matches[0].node_name == "Alice"
    assert "Alice from Acme Corp worked with River County on flood sensors." in result.context_text
    assert result.provenance.chunk_ids == ["doc-1:0", "doc-2:0"]
    assert set(result.provenance.node_names) == {"Alice", "Acme Corp", "River County"}
    assert "Neo4j local answer" in result.answer
    assert "".join(streamed_tokens).strip() == result.answer


def test_neo4j_global_search_uses_community_reports() -> None:
    query_engine = Neo4jQueryEngine(
        query_embedder=KeywordEmbedder(),
        answer_llm=FakeAnswerLLM(),
    )
    streamed_tokens: list[str] = []

    result = query_engine.global_search(
        graph_store=FakeNeo4jStore(),
        question="What are the main themes?",
        top_k_communities=1,
        on_token=streamed_tokens.append,
    )

    assert result.mode == "global"
    assert result.community_matches[0].community_id == 0
    assert "Selected community reports:" in result.context_text
    assert "Summary: Acme Corp, Alice, and River County revolve around flood-sensor work and county resilience." in result.context_text
    assert result.retrieved_chunks == []
    assert result.provenance.community_ids == [0]
    assert "Neo4j global answer" in result.answer
    assert "".join(streamed_tokens).strip() == result.answer