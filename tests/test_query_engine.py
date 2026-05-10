from __future__ import annotations

from graph_rag.enrichment import EntityMerger, GraphEnrichmentPipeline
from graph_rag.graph_store import NetworkXKnowledgeGraph
from graph_rag.models import ExtractionResult, RelationshipTriple, TextChunk
from graph_rag.query import GraphQueryEngine


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
            vectors.append([
                float(keyword in normalized_text)
                for keyword in self.KEYWORDS
            ])
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
            return "Local answer: Alice worked with Acme Corp on flood-sensor work in River County."
        return "Global answer: The strongest theme is flood resilience work, with a separate energy-supply theme elsewhere in the graph."

    def stream_text(self, *, system_prompt: str, user_prompt: str):
        del user_prompt
        if "entity-focused" in system_prompt:
            yield "Local answer: Alice worked with Acme Corp "
            yield "on flood-sensor work in River County."
            return

        yield "Global answer: The strongest theme is flood resilience work, "
        yield "with a separate energy-supply theme elsewhere in the graph."


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


def test_local_search_uses_node_similarity_hops_and_chunk_context() -> None:
    query_engine = GraphQueryEngine(
        query_embedder=KeywordEmbedder(),
        answer_llm=FakeAnswerLLM(),
    )
    streamed_tokens: list[str] = []

    result = query_engine.local_search(
        graph_store=build_enriched_graph(),
        question="What did Alice do in River County?",
        top_k=2,
        max_hops=1,
        max_chunks=3,
        on_token=streamed_tokens.append,
    )

    assert result.mode == "local"
    assert result.node_matches[0].node_name == "Alice"
    assert "Alice from Acme Corp worked with River County on flood sensors." in result.context_text
    assert "Selected community reports:" not in result.context_text
    assert "Solar Lab" not in result.context_text
    assert result.provenance.chunk_ids == ["doc-1:0", "doc-2:0"]
    assert result.provenance.node_names[0] == "Alice"
    assert set(result.provenance.node_names) == {"Alice", "Acme Corp", "River County"}
    assert "Alice worked with Acme Corp" in result.answer
    assert "".join(streamed_tokens).strip() == result.answer


def test_global_search_uses_community_summaries_instead_of_raw_chunks() -> None:
    query_engine = GraphQueryEngine(
        query_embedder=KeywordEmbedder(),
        answer_llm=FakeAnswerLLM(),
    )
    streamed_tokens: list[str] = []

    result = query_engine.global_search(
        graph_store=build_enriched_graph(),
        question="What did Alice do in River County?",
        top_k_communities=1,
        on_token=streamed_tokens.append,
    )

    assert result.mode == "global"
    assert result.community_matches[0].community_id == 0
    assert result.community_matches[0].node_names == ["Acme Corp", "Alice", "River County"]
    assert "Selected community reports:" in result.context_text
    assert "Summary: Acme Corp, Alice, and River County revolve around flood-sensor work and county resilience." in result.context_text
    assert "Alice from Acme Corp worked with River County on flood sensors." not in result.context_text
    assert result.provenance.community_ids == [0]
    assert result.retrieved_chunks == []
    assert "flood resilience work" in result.answer
    assert "".join(streamed_tokens).strip() == result.answer