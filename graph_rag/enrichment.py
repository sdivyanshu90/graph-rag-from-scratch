from __future__ import annotations

import re
from collections.abc import Callable
from typing import TYPE_CHECKING, Protocol

import networkx as nx

from .models import CommunityRecord, EntityMergeGroup, EnrichmentReport

if TYPE_CHECKING:
    from .graph_store import NetworkXKnowledgeGraph

SimilarityCalculator = Callable[[str, str], float]

COMMUNITY_SUMMARY_SYSTEM_PROMPT = """
You summarize the shared theme of one graph community.

Write exactly one short paragraph.
Use only the provided entity names.
If the entity list is sparse or ambiguous, say that plainly instead of inventing facts.
""".strip()


class CommunityDetector(Protocol):
    def detect(self, graph: nx.MultiDiGraph) -> list[list[str]]:
        ...


class NodeEmbedder(Protocol):
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        ...


class CommunitySummarizer(Protocol):
    def summarize(self, community: CommunityRecord) -> str:
        ...


class TextGenerationModel(Protocol):
    def complete_text(self, *, system_prompt: str, user_prompt: str) -> str:
        ...


class EntityMerger:
    def __init__(
        self,
        *,
        threshold: float = 90.0,
        similarity_calculator: SimilarityCalculator | None = None,
    ) -> None:
        self.threshold = threshold
        self.similarity_calculator = similarity_calculator

    def merge_graph(self, graph: nx.MultiDiGraph) -> list[EntityMergeGroup]:
        original_node_names = sorted(graph.nodes(), key=str.casefold)
        claimed_names: set[str] = set()
        merge_groups: list[EntityMergeGroup] = []

        for node_name in original_node_names:
            if node_name in claimed_names or not graph.has_node(node_name):
                continue

            cluster = [node_name]
            for other_name in original_node_names:
                if (
                    other_name == node_name
                    or other_name in claimed_names
                    or not graph.has_node(other_name)
                ):
                    continue

                similarity = self._similarity_score(node_name, other_name)
                if similarity >= self.threshold:
                    cluster.append(other_name)

            if len(cluster) == 1:
                continue

            canonical_name = self._choose_canonical_name(graph, cluster)
            merged_names = [name for name in cluster if name != canonical_name]
            self._merge_into_canonical(graph, canonical_name=canonical_name, merged_names=merged_names)

            merge_groups.append(
                EntityMergeGroup(
                    canonical_name=canonical_name,
                    merged_names=sorted(merged_names, key=str.casefold),
                )
            )
            claimed_names.update(cluster)

        graph.graph["merge_groups"] = [merge_group.model_dump() for merge_group in merge_groups]
        return merge_groups

    def _similarity_score(self, left_name: str, right_name: str) -> float:
        if self.similarity_calculator is not None:
            return self.similarity_calculator(left_name, right_name)

        from rapidfuzz.distance import Levenshtein

        normalized_left = self._normalize_name(left_name)
        normalized_right = self._normalize_name(right_name)
        return float(
            Levenshtein.normalized_similarity(normalized_left, normalized_right) * 100.0
        )

    @staticmethod
    def _normalize_name(name: str) -> str:
        return " ".join(name.casefold().split())

    @staticmethod
    def _clean_name(name: str) -> str:
        return re.sub(r"[^\w\s]", "", name).strip()

    def _choose_canonical_name(self, graph: nx.MultiDiGraph, cluster: list[str]) -> str:
        def canonical_sort_key(name: str) -> tuple[int, int, int, str]:
            mentions = graph.nodes[name].get("mentions", [])
            cleaned_name = self._clean_name(name)
            return (-len(mentions), -len(cleaned_name), len(name), name.casefold())

        return sorted(cluster, key=canonical_sort_key)[0]

    def _merge_into_canonical(
        self,
        graph: nx.MultiDiGraph,
        *,
        canonical_name: str,
        merged_names: list[str],
    ) -> None:
        canonical_data = graph.nodes[canonical_name]
        canonical_aliases = list(canonical_data.get("aliases", []))

        for merged_name in merged_names:
            if not graph.has_node(merged_name):
                continue

            merged_data = graph.nodes[merged_name]
            canonical_aliases = self._merge_unique_strings(
                canonical_aliases,
                [merged_name, *merged_data.get("aliases", [])],
            )
            canonical_data["mentions"] = self._merge_mentions(
                canonical_data.get("mentions", []),
                merged_data.get("mentions", []),
            )
            canonical_data["source_ids"] = self._merge_unique_strings(
                canonical_data.get("source_ids", []),
                merged_data.get("source_ids", []),
            )

            self._transfer_edges(graph, old_name=merged_name, canonical_name=canonical_name)
            graph.remove_node(merged_name)

        canonical_data["name"] = canonical_name
        canonical_data["canonical_name"] = canonical_name
        canonical_data["aliases"] = [
            alias
            for alias in self._merge_unique_strings(canonical_aliases, [])
            if alias != canonical_name
        ]

    def _transfer_edges(
        self,
        graph: nx.MultiDiGraph,
        *,
        old_name: str,
        canonical_name: str,
    ) -> None:
        incident_edges = self._collect_incident_edges(graph, old_name)

        for source, target, key, edge_data in incident_edges:
            new_source = canonical_name if source == old_name else source
            new_target = canonical_name if target == old_name else target

            if new_source == new_target:
                continue
            if self._has_equivalent_edge(graph, new_source, new_target, edge_data):
                continue

            new_key = self._unique_edge_key(graph, new_source, new_target, key)
            graph.add_edge(new_source, new_target, key=new_key, **edge_data)

    @staticmethod
    def _collect_incident_edges(
        graph: nx.MultiDiGraph,
        node_name: str,
    ) -> list[tuple[str, str, str, dict[str, str]]]:
        seen_edges: set[tuple[str, str, str]] = set()
        incident_edges: list[tuple[str, str, str, dict[str, str]]] = []

        for source, target, key, edge_data in list(graph.in_edges(node_name, keys=True, data=True)):
            edge_id = (source, target, key)
            if edge_id not in seen_edges:
                incident_edges.append((source, target, key, dict(edge_data)))
                seen_edges.add(edge_id)

        for source, target, key, edge_data in list(graph.out_edges(node_name, keys=True, data=True)):
            edge_id = (source, target, key)
            if edge_id not in seen_edges:
                incident_edges.append((source, target, key, dict(edge_data)))
                seen_edges.add(edge_id)

        return incident_edges

    @staticmethod
    def _has_equivalent_edge(
        graph: nx.MultiDiGraph,
        source: str,
        target: str,
        candidate_data: dict[str, str],
    ) -> bool:
        existing_edges = graph.get_edge_data(source, target, default={})
        for edge_data in existing_edges.values():
            if (
                edge_data.get("relation") == candidate_data.get("relation")
                and edge_data.get("chunk_id") == candidate_data.get("chunk_id")
                and edge_data.get("source_id") == candidate_data.get("source_id")
            ):
                return True
        return False

    @staticmethod
    def _unique_edge_key(
        graph: nx.MultiDiGraph,
        source: str,
        target: str,
        base_key: str,
    ) -> str:
        if not graph.has_edge(source, target, base_key):
            return base_key

        suffix = 1
        while graph.has_edge(source, target, f"{base_key}:merged:{suffix}"):
            suffix += 1
        return f"{base_key}:merged:{suffix}"

    @staticmethod
    def _merge_mentions(
        existing_mentions: list[dict[str, str]],
        new_mentions: list[dict[str, str]],
    ) -> list[dict[str, str]]:
        merged_mentions = list(existing_mentions)
        seen_chunk_ids = {mention["chunk_id"] for mention in existing_mentions}

        for mention in new_mentions:
            if mention["chunk_id"] not in seen_chunk_ids:
                merged_mentions.append(mention)
                seen_chunk_ids.add(mention["chunk_id"])

        return merged_mentions

    @staticmethod
    def _merge_unique_strings(existing_values: list[str], new_values: list[str]) -> list[str]:
        merged_values = list(existing_values)
        seen_values = set(existing_values)

        for value in new_values:
            if value not in seen_values:
                merged_values.append(value)
                seen_values.add(value)

        return merged_values


class LeidenCommunityDetector:
    def detect(self, graph: nx.MultiDiGraph) -> list[list[str]]:
        projection = self._build_projection(graph)
        if projection.number_of_nodes() == 0:
            return []
        if projection.number_of_edges() == 0:
            return [[node_name] for node_name in sorted(projection.nodes(), key=str.casefold)]

        from cdlib import algorithms

        community_partition = algorithms.leiden(projection)
        communities = [
            sorted(community, key=str.casefold)
            for community in community_partition.communities
            if community
        ]
        return sorted(communities, key=lambda community: [name.casefold() for name in community])

    @staticmethod
    def _build_projection(graph: nx.MultiDiGraph) -> nx.Graph:
        projection = nx.Graph()
        projection.add_nodes_from(graph.nodes())

        for source, target in graph.edges():
            if source == target:
                continue

            current_weight = projection.get_edge_data(source, target, {}).get("weight", 0)
            projection.add_edge(source, target, weight=current_weight + 1)

        return projection


class SentenceTransformerNodeEmbedder:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        embeddings = self.model.encode(texts, normalize_embeddings=True)
        return embeddings.tolist()


class LLMCommunitySummarizer:
    def __init__(self, llm_client: TextGenerationModel) -> None:
        self.llm_client = llm_client

    def summarize(self, community: CommunityRecord) -> str:
        entity_lines = "\n".join(f"- {node_name}" for node_name in community.node_names)
        return self.llm_client.complete_text(
            system_prompt=COMMUNITY_SUMMARY_SYSTEM_PROMPT,
            user_prompt=(
                "Summarize the likely theme connecting these entities.\n\n"
                f"Community ID: {community.community_id}\n"
                f"Entities:\n{entity_lines}"
            ),
        ).strip()


class GraphEnrichmentPipeline:
    def __init__(
        self,
        *,
        merger: EntityMerger,
        community_detector: CommunityDetector,
        node_embedder: NodeEmbedder,
        community_summarizer: CommunitySummarizer | None = None,
    ) -> None:
        self.merger = merger
        self.community_detector = community_detector
        self.node_embedder = node_embedder
        self.community_summarizer = community_summarizer

    def enrich(self, graph_store: NetworkXKnowledgeGraph) -> EnrichmentReport:
        graph = graph_store.graph
        merge_groups = self.merger.merge_graph(graph)
        communities = self._assign_communities(graph)
        embedded_node_count = self._embed_nodes(graph)
        communities, summarized_community_count = self._summarize_communities(graph, communities)

        return EnrichmentReport(
            merge_groups=merge_groups,
            communities=communities,
            embedded_node_count=embedded_node_count,
            summarized_community_count=summarized_community_count,
        )

    def _assign_communities(self, graph: nx.MultiDiGraph) -> list[CommunityRecord]:
        detected_communities = self.community_detector.detect(graph)
        assigned_nodes: set[str] = set()
        community_records: list[CommunityRecord] = []

        for community_index, node_names in enumerate(detected_communities):
            member_names = [node_name for node_name in node_names if graph.has_node(node_name)]
            if not member_names:
                continue

            community_record = CommunityRecord(
                community_id=community_index,
                node_names=sorted(member_names, key=str.casefold),
            )
            community_records.append(community_record)

            for node_name in community_record.node_names:
                graph.nodes[node_name]["community_id"] = community_index
                assigned_nodes.add(node_name)

        for node_name in sorted(graph.nodes(), key=str.casefold):
            if node_name in assigned_nodes:
                continue

            community_index = len(community_records)
            community_record = CommunityRecord(community_id=community_index, node_names=[node_name])
            community_records.append(community_record)
            graph.nodes[node_name]["community_id"] = community_index

        graph.graph["communities"] = {
            community.community_id: community.model_dump()
            for community in community_records
        }
        return community_records

    def _embed_nodes(self, graph: nx.MultiDiGraph) -> int:
        node_names = sorted(graph.nodes(), key=str.casefold)
        if not node_names:
            return 0

        embedding_texts = [
            self._build_node_embedding_text(node_name, graph.nodes[node_name])
            for node_name in node_names
        ]
        embeddings = self.node_embedder.embed_texts(embedding_texts)
        if len(embeddings) != len(node_names):
            raise ValueError("node embedder returned the wrong number of embeddings")

        for node_name, embedding_text, embedding in zip(node_names, embedding_texts, embeddings):
            graph.nodes[node_name]["embedding_text"] = embedding_text
            graph.nodes[node_name]["embedding"] = embedding

        return len(node_names)

    @staticmethod
    def _build_node_embedding_text(node_name: str, node_data: dict[str, object]) -> str:
        lines = [f"Entity: {node_name}"]
        aliases = node_data.get("aliases", [])
        if isinstance(aliases, list) and aliases:
            lines.append("Aliases: " + ", ".join(str(alias) for alias in aliases))

        mentions = node_data.get("mentions", [])
        if isinstance(mentions, list) and mentions:
            lines.append("Mentions:")
            for mention in mentions[:2]:
                if isinstance(mention, dict):
                    lines.append(str(mention.get("text", "")))

        return "\n".join(lines)

    def _summarize_communities(
        self,
        graph: nx.MultiDiGraph,
        communities: list[CommunityRecord],
    ) -> tuple[list[CommunityRecord], int]:
        if self.community_summarizer is None:
            graph.graph["community_summaries"] = {}
            return communities, 0

        summarized_communities: list[CommunityRecord] = []
        community_summaries: dict[int, str] = {}

        for community in communities:
            summary = self.community_summarizer.summarize(community).strip()
            summarized_community = CommunityRecord(
                community_id=community.community_id,
                node_names=community.node_names,
                summary=summary or None,
            )
            summarized_communities.append(summarized_community)

            if summarized_community.summary is not None:
                community_summaries[community.community_id] = summarized_community.summary

        graph.graph["community_summaries"] = community_summaries
        graph.graph["communities"] = {
            community.community_id: community.model_dump()
            for community in summarized_communities
        }
        return summarized_communities, len(community_summaries)