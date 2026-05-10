# Phase 3 - Graph Enrichment

Phase 2 gave us a raw graph. Phase 3 makes that graph more useful.

Why this phase exists:

- Real documents mention the same entity in slightly different ways.
- Raw graphs are often too noisy to search directly.
- Global questions need higher-level structure, not just isolated triples.

Think of Phase 2 as pinning facts on a wall. Phase 3 is where we tidy the wall, group related cards into clusters, and add summary labels so we can answer both detailed and big-picture questions later.

If we skip this phase, local retrieval will be noisy, global retrieval will be weak, and the graph will keep duplicate entities that should really be one thing.

## 1. Merge duplicate entity nodes with fuzzy matching

Before the code, the key idea:

**Levenshtein distance** is a very simple way to think about string similarity. It asks: "How many tiny edits would I need to turn one name into another?"

- `Acme Corp` to `Acme Corp.` is just one tiny edit.
- `Alice` to `Alicia` is a few edits.
- `Alice` to `River County` is many edits.

That makes it useful for catching small naming differences like punctuation, spacing, or singular/plural drift.

Here is the merger:

```python
class EntityMerger:
    def merge_graph(self, graph: nx.MultiDiGraph) -> list[EntityMergeGroup]:
        original_node_names = sorted(graph.nodes(), key=str.casefold)

        for node_name in original_node_names:
            cluster = [node_name]
            for other_name in original_node_names:
                similarity = self._similarity_score(node_name, other_name)
                if similarity >= self.threshold:
                    cluster.append(other_name)

            canonical_name = self._choose_canonical_name(graph, cluster)
            self._merge_into_canonical(graph, canonical_name=canonical_name, merged_names=merged_names)
```

What just happened?

We scanned the entity labels, found names that look similar enough, picked one canonical name to keep, and folded the duplicates into it.

Design note:

This is a deliberately simple entity-resolution strategy for learning. It is easy to inspect, but it is not a full production entity linker. That is a trade-off worth making early because you can see exactly why a merge happened.

How canonical naming works here:

- prefer the version with more mentions
- break ties in favor of the cleaner-looking surface form
- keep the discarded variants in an `aliases` list

That last step matters because we do not want to lose provenance just because we merged labels.

Reasonable alternatives:

- Exact lowercase match only: safer, but misses punctuation and spacing variants.
- Embedding-based entity matching: more flexible, but easier to over-merge early on.
- A full entity-resolution model: stronger, but much more complex than this project needs right now.

Common beginner mistakes:

- Setting the threshold too low and merging unrelated entities.
- Assuming fuzzy matching understands meaning. It only sees character-level similarity.
- Dropping aliases and then losing track of how the original documents referred to the entity.

## 2. Detect communities with the Leiden algorithm

Analogy first:

Imagine a school cafeteria. Students naturally cluster into friend groups. People in the same group talk to each other a lot, and people in different groups talk less often. A community detection algorithm looks for that same pattern in a graph.

In graph terms, a **community** is a set of nodes that are more densely connected to each other than to the rest of the graph.

Why this matters for Graph RAG:

- local search wants nearby facts around one entity
- global search wants thematic structure across many facts

Communities are one of the bridges between those two needs.

Here is the detector:

```python
class LeidenCommunityDetector:
    def detect(self, graph: nx.MultiDiGraph) -> list[list[str]]:
        projection = self._build_projection(graph)
        if projection.number_of_edges() == 0:
            return [[node_name] for node_name in sorted(projection.nodes(), key=str.casefold)]

        from cdlib import algorithms

        community_partition = algorithms.leiden(projection)
        return [sorted(community, key=str.casefold) for community in community_partition.communities]
```

What just happened?

We converted the directed multigraph into a simpler undirected projection, then asked Leiden to split it into densely connected groups.

Design note:

We detect communities after merging duplicates because duplicate nodes can distort the graph structure. If `Acme Corp` and `Acme Corp.` stay separate, the algorithm may think there are two clusters when there is really one.

Reasonable alternatives:

- Louvain: also common and often good enough.
- Connected components: too crude, because any weak link can collapse everything into one giant group.
- No communities at all: simpler, but then global question answering has no higher-level organization.

Common beginner mistakes:

- Treating community labels like human-certified truth. They are algorithmic guesses.
- Expecting perfect groupings on tiny graphs.
- Running community detection before cleaning obvious duplicates.

## 3. Compute node embeddings separately from chunk embeddings

Why do this at all if we already embedded chunks?

Because chunks and nodes serve different jobs.

- Chunk embeddings help you find supporting passages.
- Node embeddings help you find the _entity anchor_ that the question is really about.

That means node embeddings are not a replacement for chunk embeddings. They are another retrieval handle.

Here is the node embedder integration:

```python
class SentenceTransformerNodeEmbedder:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        self.model = SentenceTransformer(model_name)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        embeddings = self.model.encode(texts, normalize_embeddings=True)
        return embeddings.tolist()
```

And here is how the enrichment pipeline builds the text to embed:

```python
def _build_node_embedding_text(node_name: str, node_data: dict[str, object]) -> str:
    lines = [f"Entity: {node_name}"]
    aliases = node_data.get("aliases", [])
    mentions = node_data.get("mentions", [])
```

What just happened?

We represented each node as a compact text record built from its canonical name, aliases, and a few mention excerpts, then embedded that record. This gives the node a semantic signature that is richer than the bare entity label.

Design note:

We embed nodes separately from chunks because later local search needs to answer: "Which entity should I start traversing from?" That is a different question from: "Which passage sounds similar to the query?"

Reasonable alternatives:

- Embed only the node name: simple, but often too little context.
- Average all chunk embeddings for a node: useful, but less transparent for a learning build.
- Use a graph neural network: powerful, but much too opaque for this project.

Common beginner mistakes:

- Thinking node embeddings make graph traversal unnecessary.
- Embedding only labels like `Project X` and then wondering why search is weak.
- Storing embeddings nowhere and then having to recompute them every query.

## 4. Generate community summaries ahead of time

Now we add one more layer: a short paragraph for each community.

Why summaries matter:

- Global questions often ask about themes, not individual entities.
- Ranking summaries is cheaper than re-reading every raw chunk at query time.
- Precomputed summaries give the future query engine a compact map of the graph.

Here is the summarizer:

```python
class LLMCommunitySummarizer:
    def summarize(self, community: CommunityRecord) -> str:
        entity_lines = "\n".join(f"- {node_name}" for node_name in community.node_names)
        return self.llm_client.complete_text(
            system_prompt=COMMUNITY_SUMMARY_SYSTEM_PROMPT,
            user_prompt=f"Entities:\n{entity_lines}",
        ).strip()
```

What just happened?

We took the entity names in one community, prompted the LLM to write one short paragraph about the likely shared theme, and stored that paragraph for later retrieval.

Design note:

We generate community summaries ahead of time instead of at query time because global search should be fast and stable. If we summarize on every question, we pay extra latency every time and risk slightly different summaries for the same community.

Reasonable alternatives:

- Summarize at query time: more flexible, but slower and less predictable.
- Do not summarize at all: cheaper, but then global search has to inspect raw nodes or chunks directly.
- Include edge labels and chunk excerpts in the summary prompt: stronger, but more prompt complexity than we need right now.

Common beginner mistakes:

- Letting the model invent a theme that the entity list does not really support.
- Making summaries too long, which defeats their role as a compact retrieval layer.
- Forgetting to store the summaries once generated.

## 5. Put the enrichment steps together

Here is the orchestration layer:

```python
class GraphEnrichmentPipeline:
    def enrich(self, graph_store: NetworkXKnowledgeGraph) -> EnrichmentReport:
        merge_groups = self.merger.merge_graph(graph)
        communities = self._assign_communities(graph)
        embedded_node_count = self._embed_nodes(graph)
        communities, summarized_community_count = self._summarize_communities(graph, communities)
```

What just happened?

We created one explicit enrichment pipeline that runs in a sensible order: merge first, then group the cleaned graph, then embed the nodes, then summarize the communities.

Design note:

This order matters. If you embed and summarize before merging, you waste work on duplicates and carry avoidable noise into later retrieval.

Here is the mocked demo path in this repo:

```bash
python3 examples/phase3_enrichment_demo.py
```

What just happened?

That demo enriches a small hand-built graph with fake community detection, fake embeddings, and fake summaries so you can inspect the data flow without downloading models or making API calls.

## Experiment prompts

1. Print the community assignments. Do the groupings make intuitive sense? If not, is the graph too small, too noisy, or too sparsely connected?
2. Lower the fuzzy-match threshold and watch what gets merged. At what point do clearly different entities start collapsing together?
3. Change the node embedding text so it uses only the canonical name and not the mention excerpts. How much weaker would later node retrieval likely become?

## Checkpoint

By the end of this phase, you should be able to explain:

- why fuzzy string matching helps clean a raw graph
- what a community is in graph terms
- why node embeddings and chunk embeddings solve different retrieval problems
- why precomputed community summaries help answer global questions

What questions do you have before we move on?
