from __future__ import annotations

import math
from collections.abc import Callable

from .config import Settings, get_settings
from .models import CommunityMatch, CommunityRecord, EntitySearchRecord, NodeMatch, QueryProvenance, QueryResult
from .neo4j_store import Neo4jKnowledgeGraph
from .query import GLOBAL_SEARCH_SYSTEM_PROMPT, LOCAL_SEARCH_SYSTEM_PROMPT


class Neo4jQueryEngine:
    def __init__(
        self,
        *,
        query_embedder,
        answer_llm,
        settings: Settings | None = None,
    ) -> None:
        self.query_embedder = query_embedder
        self.answer_llm = answer_llm
        self.settings = settings or get_settings()

    def local_search(
        self,
        *,
        graph_store: Neo4jKnowledgeGraph,
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

        entity_records = graph_store.list_entity_records()
        if not entity_records:
            raise ValueError("No node embeddings found in Neo4j. Sync an enriched graph first.")

        query_embedding = self._embed_query(question)
        node_matches = self._rank_nodes(question, query_embedding, entity_records, resolved_top_k)
        bundle = graph_store.get_local_search_bundle(
            seed_node_names=[match.node_name for match in node_matches],
            max_hops=resolved_max_hops,
            max_chunks=resolved_max_chunks,
        )
        context_text = self._build_local_context(question=question, node_matches=node_matches, bundle=bundle)
        answer = self._answer_question(
            system_prompt=LOCAL_SEARCH_SYSTEM_PROMPT,
            question=question,
            context_text=context_text,
            on_token=on_token,
        )

        provenance = QueryProvenance(
            node_names=[node_hop.node_name for node_hop in bundle.node_hops],
            chunk_ids=[chunk.chunk_id for chunk in bundle.retrieved_chunks],
            source_ids=[chunk.source_id for chunk in bundle.retrieved_chunks],
        )
        return QueryResult(
            mode="local",
            question=question,
            answer=answer,
            context_text=context_text,
            retrieved_chunks=bundle.retrieved_chunks,
            node_matches=node_matches,
            provenance=provenance,
        )

    def global_search(
        self,
        *,
        graph_store: Neo4jKnowledgeGraph,
        question: str,
        top_k_communities: int | None = None,
        on_token: Callable[[str], None] | None = None,
    ) -> QueryResult:
        resolved_top_k = top_k_communities or self.settings.global_search_top_k
        if resolved_top_k <= 0:
            raise ValueError("top_k_communities must be positive")

        communities = [
            community
            for community in graph_store.list_community_records()
            if community.summary is not None
        ]
        if not communities:
            raise ValueError("No community summaries found in Neo4j. Sync an enriched graph first.")

        community_matches = self._rank_communities(question, communities, resolved_top_k)
        context_text = self._build_global_context(question=question, community_matches=community_matches)
        answer = self._answer_question(
            system_prompt=GLOBAL_SEARCH_SYSTEM_PROMPT,
            question=question,
            context_text=context_text,
            on_token=on_token,
        )

        provenance = QueryProvenance(
            node_names=[
                node_name
                for community_match in community_matches
                for node_name in community_match.node_names
            ],
            community_ids=[community_match.community_id for community_match in community_matches],
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
        question: str,
        query_embedding: list[float],
        entity_records: list[EntitySearchRecord],
        top_k: int,
    ) -> list[NodeMatch]:
        ranked_nodes: list[NodeMatch] = []
        for entity_record in entity_records:
            score = self._cosine_similarity(query_embedding, entity_record.embedding) + self._surface_form_bonus(
                question,
                entity_record,
            )
            ranked_nodes.append(NodeMatch(node_name=entity_record.node_name, score=score))

        ranked_nodes.sort(key=lambda match: (-match.score, match.node_name.casefold()))
        return ranked_nodes[:top_k]

    def _rank_communities(
        self,
        question: str,
        communities: list[CommunityRecord],
        top_k: int,
    ) -> list[CommunityMatch]:
        summaries = [community.summary or "" for community in communities]
        embeddings = self.query_embedder.embed_texts([question, *summaries])
        if len(embeddings) != len(summaries) + 1:
            raise ValueError("query embedder returned the wrong number of community vectors")

        query_embedding = embeddings[0]
        summary_embeddings = embeddings[1:]
        matches: list[CommunityMatch] = []
        for community, summary_embedding in zip(communities, summary_embeddings):
            matches.append(
                CommunityMatch(
                    community_id=community.community_id,
                    score=self._cosine_similarity(query_embedding, summary_embedding),
                    summary=community.summary or "",
                    node_names=community.node_names,
                )
            )

        matches.sort(key=lambda match: (-match.score, match.community_id))
        return matches[:top_k]

    @staticmethod
    def _surface_form_bonus(question: str, entity_record: EntitySearchRecord) -> float:
        normalized_question = " ".join(question.casefold().split())
        candidate_names = [entity_record.node_name, *entity_record.aliases]
        for candidate_name in candidate_names:
            normalized_candidate = " ".join(candidate_name.casefold().split())
            if normalized_candidate and normalized_candidate in normalized_question:
                return 0.15
        return 0.0

    @staticmethod
    def _build_local_context(*, question: str, node_matches: list[NodeMatch], bundle) -> str:
        lines = [f"Question: {question}", "", "Seed node matches:"]
        for match in node_matches:
            lines.append(f"- {match.node_name} (cosine={match.score:0.3f})")

        lines.append("")
        lines.append("Expanded neighborhood:")
        for node_hop in bundle.node_hops:
            lines.append(f"- {node_hop.node_name} (hop_distance={node_hop.hop_distance})")

        if bundle.relationships:
            lines.append("")
            lines.append("Observed relationships:")
            for relationship in bundle.relationships:
                lines.append(
                    f"- {relationship.source} -[{relationship.relation}]-> {relationship.target}"
                )

        lines.append("")
        lines.append("Supporting chunk excerpts:")
        for chunk in bundle.retrieved_chunks:
            lines.append(f"[{chunk.chunk_id} | {chunk.source_id}] {chunk.text}")
        return "\n".join(lines)

    @staticmethod
    def _build_global_context(*, question: str, community_matches: list[CommunityMatch]) -> str:
        lines = [f"Question: {question}", "", "Selected community reports:"]
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
            for token in stream_text(system_prompt=system_prompt, user_prompt=user_prompt):
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
    def _cosine_similarity(left_vector: list[float], right_vector: list[float]) -> float:
        if len(left_vector) != len(right_vector):
            raise ValueError("embedding vectors must have the same dimensionality")

        dot_product = sum(left_value * right_value for left_value, right_value in zip(left_vector, right_vector))
        left_norm = math.sqrt(sum(value * value for value in left_vector))
        right_norm = math.sqrt(sum(value * value for value in right_vector))
        if left_norm == 0.0 or right_norm == 0.0:
            return 0.0
        return dot_product / (left_norm * right_norm)