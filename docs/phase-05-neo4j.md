# Phase 5 - Upgrade to Neo4j

Phase 4 proved that the Graph RAG logic works. Phase 5 changes where the graph lives.

Why this phase exists:

- NetworkX is excellent for learning and prototyping.
- But it keeps the whole graph in one Python process.
- Real systems usually need persistence, indexing, and multi-user access.

That is where Neo4j comes in.

Analogy first:

Think of NetworkX like a whiteboard in your room.

- It is fast to sketch on.
- You can erase and redraw things easily.
- But only you can really use it, and it disappears if the room closes.

Think of Neo4j like a shared filing room built specifically for graphs.

- the graph stays there after your script exits
- many queries can hit it safely
- the query language is designed for graph-shaped questions

## 1. The learning design choice in this phase

Before the code, one non-obvious decision:

We are **not** rewriting all enrichment logic directly inside Neo4j yet.

Instead, for this learning phase, we:

1. build and enrich the graph the way we already understand in Python
2. sync that enriched graph into Neo4j
3. run local and global retrieval from Neo4j using Cypher

What just happened?

We isolated what Neo4j is buying us: persistence and graph querying. If we also moved fuzzy merging, community detection, and embedding generation into the database right now, the learning signal would get muddy.

Design note:

This is a deliberate teaching trade-off. It is not the only architecture, but it is the clearest one for seeing how Neo4j changes storage and query behavior without hiding earlier phases behind database complexity.

## 2. Add Neo4j configuration

We add connection settings to `.env`:

```python
class Settings(BaseSettings):
    neo4j_uri: str | None = Field(default=None, alias="NEO4J_URI")
    neo4j_username: str | None = Field(default=None, alias="NEO4J_USERNAME")
    neo4j_password: SecretStr | None = Field(default=None, alias="NEO4J_PASSWORD")
    neo4j_database: str = Field(default="neo4j", alias="NEO4J_DATABASE")
```

What just happened?

We added the connection details needed by the official Neo4j Python driver. They live in `.env` for the same reason the OpenAI key lives there: secrets and deployment settings should not be hardcoded into source files.

## 3. The Neo4j graph store

Here is the new storage class:

```python
class Neo4jKnowledgeGraph:
    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> "Neo4jKnowledgeGraph":
        from neo4j import GraphDatabase

        driver = GraphDatabase.driver(
            resolved_settings.neo4j_uri,
            auth=(resolved_settings.neo4j_username, resolved_settings.neo4j_password.get_secret_value()),
        )
        return cls(driver=driver, database=resolved_settings.neo4j_database)
```

What just happened?

We created a small wrapper around the official Neo4j driver. The wrapper owns Cypher queries and connection details so the rest of the project can ask for graph operations instead of hand-rolling database calls everywhere.

Design note:

The import of `neo4j` happens inside `from_settings()`, not at module import time. That keeps tests lightweight and lets us use fake drivers without requiring a live Neo4j installation just to import the code.

## 4. Sync the enriched NetworkX graph into Neo4j

The key upgrade step is `sync_from_networkx()`.

```python
def sync_from_networkx(self, graph_store: NetworkXKnowledgeGraph, *, clear_existing: bool = True) -> Neo4jSyncReport:
    graph = graph_store.graph
    chunk_payload = list(graph.graph.get("chunks", {}).values())
    entity_payload = [...]
    relationship_payload = [...]
    communities = [...]
```

What just happened?

We walked the enriched in-memory graph, converted it into plain payloads, and then used Cypher `MERGE` statements to recreate the graph inside Neo4j.

What gets stored in Neo4j:

- `(:Chunk)` nodes for chunk provenance
- `(:Entity)` nodes with aliases, embeddings, and source IDs
- `[:MENTIONED_IN]` relationships from entities to chunks
- `[:RELATES_TO]` relationships between entities with `relation`, `chunk_id`, and `chunk_text`
- `(:Community)` nodes plus `[:IN_COMMUNITY]` membership relationships

Design note:

We store raw chunk provenance in Neo4j too, not just entity relationships. That is crucial because local search still needs to reconstruct human-readable evidence, not only graph structure.

Reasonable alternative:

You could ingest directly into Neo4j from Phase 2 onward. We support that with `add_extraction()`, but the sync path is more useful for learning because it lets you compare the pre- and post-database graph explicitly.

## 5. Same queries in NetworkX vs Cypher

This is the heart of the phase. Here are the same ideas written both ways.

### Graph stats

NetworkX:

```python
{
    "node_count": graph.number_of_nodes(),
    "edge_count": graph.number_of_edges(),
    "chunk_count": len(graph.graph["chunks"]),
}
```

Cypher:

```cypher
MATCH (entity:Entity)
WITH count(entity) AS node_count
MATCH ()-[rel:RELATES_TO]->()
WITH node_count, count(rel) AS edge_count
MATCH (chunk:Chunk)
WITH node_count, edge_count, count(chunk) AS chunk_count
OPTIONAL MATCH (community:Community)
RETURN node_count, edge_count, chunk_count, count(community) AS community_count
```

What just happened?

In NetworkX, the graph is already in memory, so counting is direct. In Neo4j, you ask the database to compute the counts and return them.

### One-hop neighborhood around one entity

NetworkX:

```python
projection = graph.to_undirected()
nx.single_source_shortest_path_length(projection, "Alice", cutoff=1)
```

Cypher:

```cypher
MATCH path = (:Entity {canonical_name: $name})-[:RELATES_TO*0..1]-(:Entity)
RETURN path
```

What just happened?

Both versions are doing hop expansion. The NetworkX version calls a graph algorithm directly. The Neo4j version uses Cypher path syntax to ask the database for the neighborhood.

### Community summaries for global search

NetworkX:

```python
graph.graph["community_summaries"]
```

Cypher:

```cypher
MATCH (community:Community)
RETURN community.community_id, community.summary
ORDER BY community.community_id
```

What just happened?

In NetworkX, community summaries are Python-side metadata. In Neo4j, they become first-class nodes you can query, filter, and join with the rest of the graph.

Design note:

This is one of the clearest examples of what Neo4j buys you. Metadata that used to live in a Python dictionary now becomes queryable database state.

## 6. Neo4j-backed retrieval

The new `Neo4jQueryEngine` mirrors the two retrieval modes from Phase 4.

### Local search

```python
entity_records = graph_store.list_entity_records()
node_matches = self._rank_nodes(question, query_embedding, entity_records, resolved_top_k)
bundle = graph_store.get_local_search_bundle(
    seed_node_names=[match.node_name for match in node_matches],
    max_hops=resolved_max_hops,
    max_chunks=resolved_max_chunks,
)
```

What just happened?

We still do the semantic ranking client-side using stored embeddings, but the graph neighborhood and chunk evidence now come from Neo4j instead of NetworkX.

Design note:

This split is intentional. For a learning build, it is easier to keep embedding math explicit in Python and use Cypher for graph traversal and storage. Later, if you wanted, you could move vector search deeper into the database stack.

### Global search

```python
communities = [community for community in graph_store.list_community_records() if community.summary is not None]
community_matches = self._rank_communities(question, communities, resolved_top_k)
```

What just happened?

We fetched community summaries from Neo4j, ranked them against the question, and then answered from those selected reports the same way we did in the NetworkX version.

## 7. When to keep NetworkX vs when to use Neo4j

Keep NetworkX when:

- you are prototyping or teaching yourself graph ideas
- the graph is small enough to fit comfortably in memory
- you want fast iteration inside one Python process
- you do not need persistence or concurrent access

Use Neo4j when:

- the graph must survive process restarts
- you want multiple clients or services to query the same graph
- graph queries are becoming more complex and you want Cypher
- graph size is pushing beyond what feels comfortable in one in-memory object

Design note:

Neo4j is not "better" in every sense. It is better at persistence, operational scale, and graph querying across sessions. NetworkX is often better for quick experimentation and graph algorithms inside a notebook or script.

## Experiment prompts

1. Read the Cypher and NetworkX versions of the one-hop neighborhood query side by side. Which one feels easier to reason about right now, and why?
2. Sync the graph into a live Neo4j instance and inspect it in Neo4j Browser. Does seeing the stored `Chunk`, `Entity`, and `Community` nodes change your mental model of the system?
3. Add one more query to the Neo4j store, such as "show me all chunk excerpts mentioning River County." How much easier is that query in Cypher than it would be against ad-hoc Python dictionaries?

## Checkpoint

By the end of this phase, you should be able to explain:

- what Neo4j buys over NetworkX
- how Cypher expresses graph patterns differently from Python graph traversal
- why provenance still matters after moving to a database
- when you would keep NetworkX anyway

What questions do you have before we move on?
