from __future__ import annotations

from .api_models import EntityDetailResponse, EntityNeighbor, GraphStatsResponse, IngestResponse
from .chunking import TokenChunker
from .config import Settings, get_settings
from .enrichment import (
    EntityMerger,
    GraphEnrichmentPipeline,
    LLMCommunitySummarizer,
    LeidenCommunityDetector,
    SentenceTransformerNodeEmbedder,
)
from .extractor import EntityRelationshipExtractor
from .graph_store import NetworkXKnowledgeGraph
from .ingest import IngestionPipeline
from .llm import OpenAIChatClient
from .models import ChunkingConfig, QueryResult, RetrievedChunk
from .neo4j_store import Neo4jKnowledgeGraph
from .query import GraphQueryEngine


class GraphRAGServiceError(Exception):
    pass


class DuplicateSourceError(GraphRAGServiceError):
    pass


class GraphNotReadyError(GraphRAGServiceError):
    pass


class EntityNotFoundError(GraphRAGServiceError):
    pass


class BackendUnavailableError(GraphRAGServiceError):
    pass


class GraphRAGAPIService:
    def __init__(
        self,
        *,
        ingestion_pipeline: IngestionPipeline,
        enrichment_pipeline: GraphEnrichmentPipeline,
        query_engine: GraphQueryEngine,
        graph_store: NetworkXKnowledgeGraph,
        neo4j_store: Neo4jKnowledgeGraph | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.ingestion_pipeline = ingestion_pipeline
        self.enrichment_pipeline = enrichment_pipeline
        self.query_engine = query_engine
        self.graph_store = graph_store
        self.neo4j_store = neo4j_store
        self.settings = settings or get_settings()

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> "GraphRAGAPIService":
        resolved_settings = settings or get_settings()
        llm_client = OpenAIChatClient.from_settings(resolved_settings)
        embedder = SentenceTransformerNodeEmbedder(resolved_settings.embedding_model)
        graph_store = NetworkXKnowledgeGraph()
        chunker = TokenChunker(
            ChunkingConfig(
                chunk_size=resolved_settings.chunk_size,
                chunk_overlap=resolved_settings.chunk_overlap,
                encoding_name=resolved_settings.token_encoding,
            )
        )
        ingestion_pipeline = IngestionPipeline(
            chunker=chunker,
            extractor=EntityRelationshipExtractor(llm_client),
            graph_store=graph_store,
        )
        enrichment_pipeline = GraphEnrichmentPipeline(
            merger=EntityMerger(threshold=resolved_settings.fuzzy_match_threshold),
            community_detector=LeidenCommunityDetector(),
            node_embedder=embedder,
            community_summarizer=LLMCommunitySummarizer(llm_client),
        )
        query_engine = GraphQueryEngine(
            query_embedder=embedder,
            answer_llm=llm_client,
            settings=resolved_settings,
        )

        neo4j_store: Neo4jKnowledgeGraph | None = None
        if (
            resolved_settings.enable_neo4j_sync
            and resolved_settings.neo4j_uri
            and resolved_settings.neo4j_uri.strip()
            and resolved_settings.neo4j_username
            and resolved_settings.neo4j_username.strip()
            and resolved_settings.neo4j_password is not None
            and resolved_settings.neo4j_password.get_secret_value().strip()
        ):
            neo4j_store = Neo4jKnowledgeGraph.from_settings(resolved_settings)

        return cls(
            ingestion_pipeline=ingestion_pipeline,
            enrichment_pipeline=enrichment_pipeline,
            query_engine=query_engine,
            graph_store=graph_store,
            neo4j_store=neo4j_store,
            settings=resolved_settings,
        )

    def close(self) -> None:
        if self.neo4j_store is not None:
            self.neo4j_store.close()

    def ingest_text(self, *, source_id: str, text: str) -> IngestResponse:
        cleaned_source_id = source_id.strip()
        cleaned_text = text.strip()
        if not cleaned_source_id:
            raise ValueError("source_id must not be empty")
        if not cleaned_text:
            raise ValueError("text must not be empty")
        if self._source_exists(cleaned_source_id):
            raise DuplicateSourceError(
                f"source_id '{cleaned_source_id}' already exists; use a new source_id for this learning build"
            )

        ingestion_report = self.ingestion_pipeline.ingest_text(
            source_id=cleaned_source_id,
            text=cleaned_text,
        )
        enrichment_report = self.enrichment_pipeline.enrich(self.graph_store)
        neo4j_sync = None
        neo4j_sync_error = None

        if self.neo4j_store is not None:
            try:
                neo4j_sync = self.neo4j_store.sync_from_networkx(self.graph_store)
            except Exception as error:  # pragma: no cover - exposed as response data through HTTP
                neo4j_sync_error = str(error)

        return IngestResponse(
            ingestion=ingestion_report,
            enrichment=enrichment_report,
            graph_stats=self.graph_stats(),
            neo4j_sync=neo4j_sync,
            neo4j_sync_error=neo4j_sync_error,
        )

    def query(
        self,
        *,
        question: str,
        mode: str,
        top_k: int | None = None,
        max_hops: int | None = None,
        max_chunks: int | None = None,
    ) -> QueryResult:
        cleaned_question = question.strip()
        if not cleaned_question:
            raise ValueError("question must not be empty")

        try:
            if mode == "local":
                return self.query_engine.local_search(
                    graph_store=self.graph_store,
                    question=cleaned_question,
                    top_k=top_k,
                    max_hops=max_hops,
                    max_chunks=max_chunks,
                )

            return self.query_engine.global_search(
                graph_store=self.graph_store,
                question=cleaned_question,
                top_k_communities=top_k,
            )
        except ValueError as error:
            raise GraphNotReadyError(str(error)) from error

    def graph_stats(self) -> GraphStatsResponse:
        return GraphStatsResponse.model_validate(self.graph_store.stats())

    def entity_detail(self, *, name: str) -> EntityDetailResponse:
        canonical_name = self._resolve_entity_name(name)
        node_data = self.graph_store.graph.nodes[canonical_name]
        return EntityDetailResponse(
            canonical_name=canonical_name,
            aliases=[str(alias) for alias in node_data.get("aliases", [])],
            community_id=node_data.get("community_id"),
            source_ids=[str(source_id) for source_id in node_data.get("source_ids", [])],
            neighbors=self._collect_neighbors(canonical_name),
            chunk_excerpts=self._collect_chunk_excerpts(canonical_name),
        )

    def _source_exists(self, source_id: str) -> bool:
        chunks = self.graph_store.graph.graph.get("chunks", {})
        return any(
            isinstance(chunk_data, dict) and chunk_data.get("source_id") == source_id
            for chunk_data in chunks.values()
        )

    def _resolve_entity_name(self, name: str) -> str:
        normalized_name = name.strip().casefold()
        if not normalized_name:
            raise EntityNotFoundError("entity name must not be empty")

        for node_name, node_data in self.graph_store.graph.nodes(data=True):
            candidate_names = [node_name]
            aliases = node_data.get("aliases", [])
            if isinstance(aliases, list):
                candidate_names.extend(str(alias) for alias in aliases)

            if any(candidate.casefold() == normalized_name for candidate in candidate_names):
                return node_name

        raise EntityNotFoundError(f"entity '{name}' was not found in the graph")

    def _collect_neighbors(self, canonical_name: str) -> list[EntityNeighbor]:
        neighbor_records: dict[str, dict[str, object]] = {}

        for _, target, edge_data in self.graph_store.graph.out_edges(canonical_name, data=True):
            relation = str(edge_data.get("relation", "related_to"))
            record = neighbor_records.setdefault(
                target,
                {"relations": set(), "directions": set()},
            )
            record["relations"].add(relation)
            record["directions"].add("outgoing")

        for source, _, edge_data in self.graph_store.graph.in_edges(canonical_name, data=True):
            relation = str(edge_data.get("relation", "related_to"))
            record = neighbor_records.setdefault(
                source,
                {"relations": set(), "directions": set()},
            )
            record["relations"].add(relation)
            record["directions"].add("incoming")

        neighbors: list[EntityNeighbor] = []
        for neighbor_name in sorted(neighbor_records, key=str.casefold):
            record = neighbor_records[neighbor_name]
            directions = record["directions"]
            if directions == {"incoming", "outgoing"}:
                direction = "both"
            elif "incoming" in directions:
                direction = "incoming"
            else:
                direction = "outgoing"

            neighbors.append(
                EntityNeighbor(
                    neighbor_name=neighbor_name,
                    relations=sorted(record["relations"], key=str.casefold),
                    direction=direction,
                )
            )

        return neighbors

    def _collect_chunk_excerpts(self, canonical_name: str) -> list[RetrievedChunk]:
        mentions = self.graph_store.graph.nodes[canonical_name].get("mentions", [])
        if not isinstance(mentions, list):
            return []

        chunk_excerpts: list[RetrievedChunk] = []
        seen_chunk_ids: set[str] = set()
        for mention in mentions:
            if not isinstance(mention, dict):
                continue

            chunk_id = str(mention.get("chunk_id", "")).strip()
            source_id = str(mention.get("source_id", "")).strip()
            text = str(mention.get("text", "")).strip()
            if not chunk_id or not source_id or not text or chunk_id in seen_chunk_ids:
                continue

            chunk_excerpts.append(
                RetrievedChunk(chunk_id=chunk_id, source_id=source_id, text=text)
            )
            seen_chunk_ids.add(chunk_id)

        return chunk_excerpts