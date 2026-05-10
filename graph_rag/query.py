from __future__ import annotations

import math
from collections.abc import Callable, Iterator
from typing import Protocol

import networkx as nx

from .config import Settings, get_settings
from .graph_store import NetworkXKnowledgeGraph
from .models import CommunityMatch, NodeMatch, QueryProvenance, QueryResult, RetrievedChunk

LOCAL_SEARCH_SYSTEM_PROMPT = """
You answer specific, entity-focused questions using local graph context.

Use only the provided context.
If the context does not fully support the answer, say what is missing.
Prefer concrete details over broad themes.
""".strip()

GLOBAL_SEARCH_SYSTEM_PROMPT = """
You answer high-level, thematic questions using community reports from a knowledge graph.

Synthesize across the selected communities.
Do not invent entity-level facts that are not supported by the summaries.
If the summaries are too sparse, say so plainly.
""".strip()


class TextEmbedder(Protocol):
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        ...


class AnswerGenerationModel(Protocol):
    def complete_text(self, *, system_prompt: str, user_prompt: str) -> str:
        ...


class StreamingAnswerGenerationModel(AnswerGenerationModel, Protocol):
    def stream_text(self, *, system_prompt: str, user_prompt: str) -> Iterator[str]:
        ...


class GraphQueryEngine:
    def __init__(
        self,
        *,
        query_embedder: TextEmbedder,
        answer_llm: AnswerGenerationModel,
        settings: Settings | None = None,
    ) -> None:
        self.query_embedder = query_embedder
        self.answer_llm = answer_llm
        self.settings = settings or get_settings()

    def local_search(
        self,
        *,
        graph_store: NetworkXKnowledgeGraph,
        question: str,
        top_k: int | None = None,
        max_hops: int | None = None,
        max_chunks: int | None = None,
        on_token: Callable[[str], None] | None = None,
    ) -> QueryResult:
        resolved_top_k = top_k or self.settings.local_search_top_k
        resolved_max_hops = (
            max_hops if max_hops is not None else self.settings.local_search_max_hops
        )
        resolved_max_chunks = (
            max_chunks if max_chunks is not None else self.settings.local_search_max_chunks
        )

        if resolved_top_k <= 0:
            raise ValueError("top_k must be positive")
        if resolved_max_hops < 0:
            raise ValueError("max_hops must not be negative")
        if resolved_max_chunks <= 0:
            raise ValueError("max_chunks must be positive")

        graph = graph_store.graph
        query_embedding = self._embed_query(question)
        node_matches = self._rank_nodes(graph, question, query_embedding, resolved_top_k)
        if not node_matches:
            raise ValueError("No node embeddings found. Run graph enrichment before local search.")

        neighborhood_distances = self._expand_neighborhood(
            graph,
            [match.node_name for match in node_matches],
            resolved_max_hops,
        )
        retrieved_chunks = self._collect_local_chunks(
            graph,
            neighborhood_distances=neighborhood_distances,
            max_chunks=resolved_max_chunks,
        )
        context_text = self._build_local_context(
            question=question,
            graph=graph,
            node_matches=node_matches,
            neighborhood_distances=neighborhood_distances,
            retrieved_chunks=retrieved_chunks,
        )
        answer = self._answer_question(
            system_prompt=LOCAL_SEARCH_SYSTEM_PROMPT,
            question=question,
            context_text=context_text,
            on_token=on_token,
        )

        provenance = QueryProvenance(
            node_names=list(neighborhood_distances.keys()),
            chunk_ids=[chunk.chunk_id for chunk in retrieved_chunks],
            source_ids=[chunk.source_id for chunk in retrieved_chunks],
        )
        return QueryResult(
            mode="local",
            question=question,
            answer=answer,
            context_text=context_text,
            retrieved_chunks=retrieved_chunks,
            node_matches=node_matches,
            provenance=provenance,
        )

    def global_search(
        self,
        *,
        graph_store: NetworkXKnowledgeGraph,
        question: str,
        top_k_communities: int | None = None,
        on_token: Callable[[str], None] | None = None,
    ) -> QueryResult:
        resolved_top_k = top_k_communities or self.settings.global_search_top_k
        if resolved_top_k <= 0:
            raise ValueError("top_k_communities must be positive")

        graph = graph_store.graph
        community_matches = self._rank_communities(graph, question, resolved_top_k)
        if not community_matches:
            raise ValueError(
                "No community summaries found. Run graph enrichment with summaries before global search."
            )

        context_text = self._build_global_context(question=question, community_matches=community_matches)
        answer = self._answer_question(
            system_prompt=GLOBAL_SEARCH_SYSTEM_PROMPT,
            question=question,
            context_text=context_text,
            on_token=on_token,
        )

        source_ids: list[str] = []
        for community_match in community_matches:
            for node_name in community_match.node_names:
                node_source_ids = graph.nodes[node_name].get("source_ids", [])
                if isinstance(node_source_ids, list):
                    source_ids.extend(str(source_id) for source_id in node_source_ids)

        provenance = QueryProvenance(
            node_names=[
                node_name
                for community_match in community_matches
                for node_name in community_match.node_names
            ],
            community_ids=[match.community_id for match in community_matches],
            source_ids=source_ids,
        )
        return QueryResult(
            mode="global",
            question=question,
            answer=answer,
            context_text=context_text,
            community_matches=community_matches,
            provenance=provenance,
        )

    def _embed_query(self, question: str) -> list[float]:
        embeddings = self.query_embedder.embed_texts([question])
        if len(embeddings) != 1:
            raise ValueError("query embedder returned the wrong number of vectors")
        return embeddings[0]

    def _rank_nodes(
        self,
        graph: nx.MultiDiGraph,
        question: str,
        query_embedding: list[float],
        top_k: int,
    ) -> list[NodeMatch]:
        ranked_nodes: list[NodeMatch] = []
        for node_name, node_data in graph.nodes(data=True):
            node_embedding = node_data.get("embedding")
            if not isinstance(node_embedding, list) or not node_embedding:
                continue

            score = self._cosine_similarity(query_embedding, node_embedding) + self._surface_form_bonus(
                question,
                node_name,
                node_data,
            )
            ranked_nodes.append(NodeMatch(node_name=node_name, score=score))

        ranked_nodes.sort(key=lambda match: (-match.score, match.node_name.casefold()))
        return ranked_nodes[:top_k]

    @staticmethod
    def _surface_form_bonus(
        question: str,
        node_name: str,
        node_data: dict[str, object],
    ) -> float:
        normalized_question = " ".join(question.casefold().split())
        candidate_names = [node_name]
        aliases = node_data.get("aliases", [])
        if isinstance(aliases, list):
            candidate_names.extend(str(alias) for alias in aliases)

        for candidate_name in candidate_names:
            normalized_candidate = " ".join(candidate_name.casefold().split())
            if normalized_candidate and normalized_candidate in normalized_question:
                return 0.15

        return 0.0

    def _expand_neighborhood(
        self,
        graph: nx.MultiDiGraph,
        seed_nodes: list[str],
        max_hops: int,
    ) -> dict[str, int]:
        if not seed_nodes:
            return {}

        projection = graph.to_undirected()
        neighborhood_distances: dict[str, int] = {}
        for seed_node in seed_nodes:
            if not projection.has_node(seed_node):
                continue

            path_lengths = nx.single_source_shortest_path_length(
                projection,
                seed_node,
                cutoff=max_hops,
            )
            for node_name, distance in path_lengths.items():
                current_distance = neighborhood_distances.get(node_name)
                if current_distance is None or distance < current_distance:
                    neighborhood_distances[node_name] = distance

        return dict(
            sorted(
                neighborhood_distances.items(),
                key=lambda item: (item[1], item[0].casefold()),
            )
        )

    def _collect_local_chunks(
        self,
        graph: nx.MultiDiGraph,
        *,
        neighborhood_distances: dict[str, int],
        max_chunks: int,
    ) -> list[RetrievedChunk]:
        seen_chunk_ids: set[str] = set()
        retrieved_chunks: list[RetrievedChunk] = []

        for node_name, _distance in neighborhood_distances.items():
            mentions = graph.nodes[node_name].get("mentions", [])
            if not isinstance(mentions, list):
                continue

            for mention in mentions:
                if not isinstance(mention, dict):
                    continue

                chunk_id = str(mention.get("chunk_id", "")).strip()
                source_id = str(mention.get("source_id", "")).strip()
                text = str(mention.get("text", "")).strip()
                if not chunk_id or not source_id or not text or chunk_id in seen_chunk_ids:
                    continue

                retrieved_chunks.append(
                    RetrievedChunk(chunk_id=chunk_id, source_id=source_id, text=text)
                )
                seen_chunk_ids.add(chunk_id)
                if len(retrieved_chunks) >= max_chunks:
                    return retrieved_chunks

        for source, target, edge_data in graph.edges(data=True):
            if source not in neighborhood_distances or target not in neighborhood_distances:
                continue

            chunk_id = str(edge_data.get("chunk_id", "")).strip()
            source_id = str(edge_data.get("source_id", "")).strip()
            text = str(edge_data.get("chunk_text", "")).strip()
            if not chunk_id or not source_id or not text or chunk_id in seen_chunk_ids:
                continue

            retrieved_chunks.append(RetrievedChunk(chunk_id=chunk_id, source_id=source_id, text=text))
            seen_chunk_ids.add(chunk_id)
            if len(retrieved_chunks) >= max_chunks:
                return retrieved_chunks

        return retrieved_chunks

    def _build_local_context(
        self,
        *,
        question: str,
        graph: nx.MultiDiGraph,
        node_matches: list[NodeMatch],
        neighborhood_distances: dict[str, int],
        retrieved_chunks: list[RetrievedChunk],
    ) -> str:
        lines = [
            f"Question: {question}",
            "",
            "Seed node matches:",
        ]
        for match in node_matches:
            lines.append(f"- {match.node_name} (cosine={match.score:0.3f})")

        lines.append("")
        lines.append("Expanded neighborhood:")
        for node_name, distance in neighborhood_distances.items():
            lines.append(f"- {node_name} (hop_distance={distance})")

        edge_lines = self._build_neighborhood_edges(graph, neighborhood_distances)
        if edge_lines:
            lines.append("")
            lines.append("Observed relationships:")
            lines.extend(edge_lines)

        lines.append("")
        lines.append("Supporting chunk excerpts:")
        for chunk in retrieved_chunks:
            lines.append(f"[{chunk.chunk_id} | {chunk.source_id}] {chunk.text}")

        return "\n".join(lines)

    def _rank_communities(
        self,
        graph: nx.MultiDiGraph,
        question: str,
        top_k: int,
    ) -> list[CommunityMatch]:
        community_summaries = graph.graph.get("community_summaries", {})
        if not isinstance(community_summaries, dict) or not community_summaries:
            return []

        sorted_community_ids = sorted(community_summaries)
        community_texts = [str(community_summaries[community_id]) for community_id in sorted_community_ids]
        embeddings = self.query_embedder.embed_texts([question, *community_texts])
        if len(embeddings) != len(community_texts) + 1:
            raise ValueError("query embedder returned the wrong number of community vectors")

        query_embedding = embeddings[0]
        community_embeddings = embeddings[1:]
        community_records = graph.graph.get("communities", {})
        matches: list[CommunityMatch] = []

        for community_id, summary, summary_embedding in zip(
            sorted_community_ids,
            community_texts,
            community_embeddings,
        ):
            community_record = community_records.get(community_id, {})
            node_names = community_record.get("node_names", []) if isinstance(community_record, dict) else []
            matches.append(
                CommunityMatch(
                    community_id=int(community_id),
                    score=self._cosine_similarity(query_embedding, summary_embedding),
                    summary=summary,
                    node_names=[str(node_name) for node_name in node_names],
                )
            )

        matches.sort(key=lambda match: (-match.score, match.community_id))
        return matches[:top_k]

    @staticmethod
    def _build_global_context(*, question: str, community_matches: list[CommunityMatch]) -> str:
        lines = [
            f"Question: {question}",
            "",
            "Selected community reports:",
        ]
        for match in community_matches:
            lines.append(f"Community {match.community_id} (cosine={match.score:0.3f})")
            lines.append("Entities: " + ", ".join(match.node_names))
            lines.append("Summary: " + match.summary)
            lines.append("")

        return "\n".join(lines).strip()

    def _answer_question(
        self,
        *,
        system_prompt: str,
        question: str,
        context_text: str,
        on_token: Callable[[str], None] | None,
    ) -> str:
        user_prompt = f"Question:\n{question}\n\nRetrieved context:\n{context_text}"
        stream_text = getattr(self.answer_llm, "stream_text", None)
        if on_token is not None and callable(stream_text):
            answer_parts: list[str] = []
            for token in stream_text(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
            ):
                answer_parts.append(token)
                on_token(token)

            return "".join(answer_parts).strip()

        answer = self.answer_llm.complete_text(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        ).strip()
        if on_token is not None and answer:
            on_token(answer)
        return answer

    @staticmethod
    def _build_neighborhood_edges(
        graph: nx.MultiDiGraph,
        neighborhood_distances: dict[str, int],
    ) -> list[str]:
        edge_lines: list[str] = []
        seen_edges: set[tuple[str, str, str, str]] = set()
        for source, target, edge_data in graph.edges(data=True):
            if source not in neighborhood_distances or target not in neighborhood_distances:
                continue

            relation = str(edge_data.get("relation", "related_to")).strip() or "related_to"
            edge_key = (source, relation, target, str(edge_data.get("chunk_id", "")))
            if edge_key in seen_edges:
                continue

            seen_edges.add(edge_key)
            edge_lines.append(f"- {source} -[{relation}]-> {target}")

        edge_lines.sort(key=str.casefold)
        return edge_lines

    @staticmethod
    def _cosine_similarity(left_vector: list[float], right_vector: list[float]) -> float:
        if len(left_vector) != len(right_vector):
            raise ValueError("embedding vectors must have the same dimensionality")

        dot_product = sum(left_value * right_value for left_value, right_value in zip(left_vector, right_vector))
        left_norm = math.sqrt(sum(value * value for value in left_vector))
        right_norm = math.sqrt(sum(value * value for value in right_vector))
        if left_norm == 0.0 or right_norm == 0.0:
            return 0.0
        return dot_product / (left_norm * right_norm)