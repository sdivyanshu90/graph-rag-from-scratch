from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .config import Settings, get_settings
from .models import (
    CommunityRecord,
    EntitySearchRecord,
    ExtractionResult,
    LocalSearchBundle,
    NeighborhoodRelationship,
    Neo4jSyncReport,
    NodeHop,
    RetrievedChunk,
    TextChunk,
)

if TYPE_CHECKING:
    from .graph_store import NetworkXKnowledgeGraph

CHUNK_SYNC_QUERY = """
UNWIND $chunks AS chunk
MERGE (chunk_node:Chunk {chunk_id: chunk.chunk_id})
SET chunk_node.source_id = chunk.source_id,
    chunk_node.text = chunk.text,
    chunk_node.chunk_index = chunk.chunk_index,
    chunk_node.token_start = chunk.token_start,
    chunk_node.token_end = chunk.token_end
""".strip()

ENTITY_SYNC_QUERY = """
UNWIND $entities AS entity
MERGE (entity_node:Entity {canonical_name: entity.canonical_name})
SET entity_node.name = entity.name,
    entity_node.aliases = entity.aliases,
    entity_node.source_ids = entity.source_ids,
    entity_node.embedding_text = entity.embedding_text,
    entity_node.embedding = entity.embedding,
    entity_node.community_id = entity.community_id
WITH entity_node, entity
UNWIND entity.mentions AS mention
MATCH (chunk_node:Chunk {chunk_id: mention.chunk_id})
MERGE (entity_node)-[mention_rel:MENTIONED_IN {chunk_id: mention.chunk_id}]->(chunk_node)
SET mention_rel.source_id = mention.source_id,
    mention_rel.text = mention.text
""".strip()

RELATIONSHIP_SYNC_QUERY = """
UNWIND $relationships AS relationship
MATCH (source:Entity {canonical_name: relationship.source})
MATCH (target:Entity {canonical_name: relationship.target})
MERGE (source)-[rel:RELATES_TO {
    relation: relationship.relation,
    chunk_id: relationship.chunk_id,
    source_id: relationship.source_id
}]->(target)
SET rel.chunk_text = relationship.chunk_text
""".strip()

DELETE_COMMUNITIES_QUERY = """
MATCH (community:Community)
DETACH DELETE community
""".strip()

COMMUNITY_SYNC_QUERY = """
UNWIND $communities AS community
MERGE (community_node:Community {community_id: community.community_id})
SET community_node.summary = community.summary,
    community_node.node_names = community.node_names
WITH community_node, community
UNWIND community.node_names AS node_name
MATCH (entity_node:Entity {canonical_name: node_name})
MERGE (entity_node)-[:IN_COMMUNITY]->(community_node)
""".strip()

RESET_QUERY = """
MATCH (node)
DETACH DELETE node
""".strip()

STATS_QUERY = """
MATCH (entity:Entity)
WITH count(entity) AS node_count
MATCH ()-[rel:RELATES_TO]->()
WITH node_count, count(rel) AS edge_count
MATCH (chunk:Chunk)
WITH node_count, edge_count, count(chunk) AS chunk_count
OPTIONAL MATCH (community:Community)
RETURN node_count, edge_count, chunk_count, count(community) AS community_count
""".strip()

ENTITY_RECORDS_QUERY = """
MATCH (entity:Entity)
WHERE entity.embedding IS NOT NULL
RETURN entity.canonical_name AS node_name,
       coalesce(entity.aliases, []) AS aliases,
       entity.embedding AS embedding
ORDER BY entity.canonical_name
""".strip()

COMMUNITY_RECORDS_QUERY = """
MATCH (community:Community)
WHERE community.summary IS NOT NULL
RETURN community.community_id AS community_id,
       community.summary AS summary,
       coalesce(community.node_names, []) AS node_names
ORDER BY community.community_id
""".strip()

LOCAL_RELATIONSHIPS_QUERY = """
MATCH (source:Entity)-[rel:RELATES_TO]->(target:Entity)
WHERE source.canonical_name IN $node_names
  AND target.canonical_name IN $node_names
RETURN source.canonical_name AS source,
       rel.relation AS relation,
       target.canonical_name AS target,
       rel.chunk_id AS chunk_id,
       rel.source_id AS source_id,
       rel.chunk_text AS chunk_text
ORDER BY source, relation, target, rel.chunk_id
""".strip()

LOCAL_CHUNKS_QUERY = """
MATCH (entity:Entity)-[:MENTIONED_IN]->(chunk:Chunk)
WHERE entity.canonical_name IN $node_names
RETURN DISTINCT chunk.chunk_id AS chunk_id,
       chunk.source_id AS source_id,
       chunk.text AS text
ORDER BY chunk.chunk_id
LIMIT $max_chunks
""".strip()


class Neo4jKnowledgeGraph:
    def __init__(self, *, driver: Any, database: str = "neo4j") -> None:
        self.driver = driver
        self.database = database

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> "Neo4jKnowledgeGraph":
        resolved_settings = settings or get_settings()
        if not resolved_settings.neo4j_uri or not resolved_settings.neo4j_uri.strip():
            raise ValueError("NEO4J_URI is required before using the Neo4j graph store")
        if not resolved_settings.neo4j_username or not resolved_settings.neo4j_username.strip():
            raise ValueError("NEO4J_USERNAME is required before using the Neo4j graph store")
        if (
            resolved_settings.neo4j_password is None
            or not resolved_settings.neo4j_password.get_secret_value().strip()
        ):
            raise ValueError("NEO4J_PASSWORD is required before using the Neo4j graph store")

        from neo4j import GraphDatabase

        driver = GraphDatabase.driver(
            resolved_settings.neo4j_uri,
            auth=(
                resolved_settings.neo4j_username,
                resolved_settings.neo4j_password.get_secret_value(),
            ),
        )
        return cls(driver=driver, database=resolved_settings.neo4j_database)

    def close(self) -> None:
        close = getattr(self.driver, "close", None)
        if callable(close):
            close()

    def reset(self) -> None:
        with self.driver.session(database=self.database) as session:
            session.run(RESET_QUERY)

    def add_extraction(self, *, chunk: TextChunk, extraction: ExtractionResult) -> None:
        chunk_payload = [chunk.model_dump()]
        entity_payload = [
            {
                "canonical_name": entity_name,
                "name": entity_name,
                "aliases": [],
                "source_ids": [chunk.source_id],
                "embedding_text": None,
                "embedding": None,
                "community_id": None,
                "mentions": [
                    {
                        "chunk_id": chunk.chunk_id,
                        "source_id": chunk.source_id,
                        "text": chunk.text,
                    }
                ],
            }
            for entity_name in extraction.entities
        ]
        relationship_payload = [
            {
                "source": relationship.source,
                "relation": relationship.relation,
                "target": relationship.target,
                "chunk_id": chunk.chunk_id,
                "source_id": chunk.source_id,
                "chunk_text": chunk.text,
            }
            for relationship in extraction.relationships
        ]

        with self.driver.session(database=self.database) as session:
            session.run(CHUNK_SYNC_QUERY, {"chunks": chunk_payload})
            if entity_payload:
                session.run(ENTITY_SYNC_QUERY, {"entities": entity_payload})
            if relationship_payload:
                session.run(RELATIONSHIP_SYNC_QUERY, {"relationships": relationship_payload})

    def sync_from_networkx(
        self,
        graph_store: NetworkXKnowledgeGraph,
        *,
        clear_existing: bool = True,
    ) -> Neo4jSyncReport:
        graph = graph_store.graph
        chunk_payload = list(graph.graph.get("chunks", {}).values())
        entity_payload = [
            {
                "canonical_name": str(node_data.get("canonical_name", node_name)),
                "name": str(node_data.get("name", node_name)),
                "aliases": [str(alias) for alias in node_data.get("aliases", [])],
                "source_ids": [str(source_id) for source_id in node_data.get("source_ids", [])],
                "embedding_text": node_data.get("embedding_text"),
                "embedding": node_data.get("embedding"),
                "community_id": node_data.get("community_id"),
                "mentions": [dict(mention) for mention in node_data.get("mentions", [])],
            }
            for node_name, node_data in graph.nodes(data=True)
        ]
        relationship_payload = [
            {
                "source": source,
                "relation": str(edge_data.get("relation", "related_to")),
                "target": target,
                "chunk_id": str(edge_data.get("chunk_id", "")),
                "source_id": str(edge_data.get("source_id", "")),
                "chunk_text": str(edge_data.get("chunk_text", "")),
            }
            for source, target, edge_data in graph.edges(data=True)
        ]
        communities = [
            {
                "community_id": int(community_id),
                "summary": community_data.get("summary"),
                "node_names": [str(node_name) for node_name in community_data.get("node_names", [])],
            }
            for community_id, community_data in graph.graph.get("communities", {}).items()
        ]

        with self.driver.session(database=self.database) as session:
            if clear_existing:
                session.run(RESET_QUERY)
            if chunk_payload:
                session.run(CHUNK_SYNC_QUERY, {"chunks": chunk_payload})
            if entity_payload:
                session.run(ENTITY_SYNC_QUERY, {"entities": entity_payload})
            if relationship_payload:
                session.run(RELATIONSHIP_SYNC_QUERY, {"relationships": relationship_payload})
            session.run(DELETE_COMMUNITIES_QUERY)
            if communities:
                session.run(COMMUNITY_SYNC_QUERY, {"communities": communities})

        return Neo4jSyncReport(
            entity_count=len(entity_payload),
            relationship_count=len(relationship_payload),
            chunk_count=len(chunk_payload),
            community_count=len(communities),
        )

    def stats(self) -> dict[str, int]:
        with self.driver.session(database=self.database) as session:
            records = session.run(STATS_QUERY).data()

        if not records:
            return {
                "node_count": 0,
                "edge_count": 0,
                "chunk_count": 0,
                "community_count": 0,
            }

        record = records[0]
        return {
            "node_count": int(record.get("node_count", 0)),
            "edge_count": int(record.get("edge_count", 0)),
            "chunk_count": int(record.get("chunk_count", 0)),
            "community_count": int(record.get("community_count", 0)),
        }

    def list_entity_records(self) -> list[EntitySearchRecord]:
        with self.driver.session(database=self.database) as session:
            records = session.run(ENTITY_RECORDS_QUERY).data()
        return [EntitySearchRecord.model_validate(record) for record in records]

    def list_community_records(self) -> list[CommunityRecord]:
        with self.driver.session(database=self.database) as session:
            records = session.run(COMMUNITY_RECORDS_QUERY).data()
        return [CommunityRecord.model_validate(record) for record in records]

    def get_local_search_bundle(
        self,
        *,
        seed_node_names: list[str],
        max_hops: int,
        max_chunks: int,
    ) -> LocalSearchBundle:
        if not seed_node_names:
            return LocalSearchBundle()

        neighborhood_query = self._local_neighborhood_query(max_hops=max_hops)
        with self.driver.session(database=self.database) as session:
            node_records = session.run(
                neighborhood_query,
                {"seed_names": seed_node_names},
            ).data()

            node_hops = [NodeHop.model_validate(record) for record in node_records]
            node_names = [node_hop.node_name for node_hop in node_hops]
            if not node_names:
                return LocalSearchBundle()

            relationship_records = session.run(
                LOCAL_RELATIONSHIPS_QUERY,
                {"node_names": node_names},
            ).data()
            chunk_records = session.run(
                LOCAL_CHUNKS_QUERY,
                {"node_names": node_names, "max_chunks": max_chunks},
            ).data()

        return LocalSearchBundle(
            node_hops=node_hops,
            relationships=[
                NeighborhoodRelationship.model_validate(record)
                for record in relationship_records
            ],
            retrieved_chunks=[RetrievedChunk.model_validate(record) for record in chunk_records],
        )

    @staticmethod
    def _local_neighborhood_query(*, max_hops: int) -> str:
        return f"""
UNWIND $seed_names AS seed_name
MATCH (seed:Entity {{canonical_name: seed_name}})
MATCH path = (seed)-[:RELATES_TO*0..{max_hops}]-(neighbor:Entity)
WITH neighbor.canonical_name AS node_name, min(length(path)) AS hop_distance
RETURN node_name, hop_distance
ORDER BY hop_distance, node_name
""".strip()