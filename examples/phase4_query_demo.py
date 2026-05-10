from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

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


def build_demo_graph() -> NetworkXKnowledgeGraph:
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


def main() -> None:
    graph_store = build_demo_graph()
    query_engine = GraphQueryEngine(
        query_embedder=KeywordEmbedder(),
        answer_llm=FakeAnswerLLM(),
    )
    question = "What did Alice do in River County?"

    print("Question:", question)

    print("\nLocal search result:")
    local_tokens: list[str] = []
    local_result = query_engine.local_search(
        graph_store=graph_store,
        question=question,
        top_k=2,
        max_hops=1,
        max_chunks=3,
        on_token=local_tokens.append,
    )
    print("Context:\n" + local_result.context_text)
    print("\nAnswer:", local_result.answer)
    print("Streamed tokens:", local_tokens)

    print("\nGlobal search result:")
    global_tokens: list[str] = []
    global_result = query_engine.global_search(
        graph_store=graph_store,
        question=question,
        top_k_communities=2,
        on_token=global_tokens.append,
    )
    print("Context:\n" + global_result.context_text)
    print("\nAnswer:", global_result.answer)
    print("Streamed tokens:", global_tokens)


if __name__ == "__main__":
    main()