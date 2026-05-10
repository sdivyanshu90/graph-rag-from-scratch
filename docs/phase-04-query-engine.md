# Phase 4 - Query Engine

Phase 3 gave us an enriched graph. Phase 4 is where the graph starts answering questions.

Why this phase exists:

- A graph is only useful if we can retrieve the right context from it.
- Not all questions want the same kind of context.
- Some questions are about one entity and its neighborhood, while others ask for themes across the whole corpus.

So in this phase we build **two retrieval modes**, not one.

Analogy first:

Imagine a city map.

- **Local search** is like dropping a pin on one address, then walking a block or two around it.
- **Global search** is like stepping back and asking which neighborhoods matter most for your question.

That is the intuition behind the formal graph ideas:

- a **hop** means following one graph edge
- **hop expansion** means walking outward from one node to nearby nodes
- a **community** is a cluster of nodes that are more connected to each other than to the rest of the graph

## 1. Local vs global search, side by side

Before writing code, here is the contrast we are building:

| Mode          | Best for                            | Retrieval anchor                 | Context fed to LLM           |
| ------------- | ----------------------------------- | -------------------------------- | ---------------------------- |
| Local search  | specific entity questions           | top matching nodes               | nearby chunks plus relations |
| Global search | thematic or summarization questions | top matching community summaries | community reports            |

Examples:

- `What did Alice do in River County?` is local.
- `What are the main themes across these documents?` is global.

Why keep both modes?

- Embedding similarity is good at finding semantically related starting points.
- Graph traversal is good at following explicit relationships once you know where to start.
- Community summaries are good at compressing big-picture structure.

If we only used local search, thematic questions would feel narrow and myopic. If we only used global search, entity-specific questions would feel vague and over-abstracted.

Design note:

This is one of the core Graph RAG ideas to internalize: **embedding similarity and graph traversal are not competitors**. Similarity helps choose an anchor. Traversal helps gather the relational neighborhood around that anchor.

Common beginner mistakes:

- Expecting one retrieval strategy to win on every question.
- Treating graph traversal as a replacement for semantic search.
- Using global summaries when the question is clearly asking about a specific entity.

## 2. Typed query results and provenance

Before implementing retrieval, we define the shape of what a query returns.

```python
class RetrievedChunk(BaseModel):
    chunk_id: str
    source_id: str
    text: str


class QueryProvenance(BaseModel):
    node_names: list[str] = Field(default_factory=list)
    community_ids: list[int] = Field(default_factory=list)
    chunk_ids: list[str] = Field(default_factory=list)
    source_ids: list[str] = Field(default_factory=list)


class QueryResult(BaseModel):
    mode: Literal["local", "global"]
    question: str
    answer: str
    context_text: str
```

What just happened?

We defined a typed return shape for the query engine. That means the engine returns not just an answer string, but also the exact context and provenance that produced that answer.

Design note:

**Provenance** means "where did this answer come from?" In RAG systems, provenance matters because trust depends on being able to inspect the supporting evidence. If the system gives an answer but cannot show which nodes, chunks, or communities supported it, it becomes much harder to verify.

Reasonable alternative:

You could return only `answer` and `context_text`. That is simpler, but it throws away structured provenance that becomes very useful in APIs and debugging.

## 3. Local search

Now we implement the local mode.

Why this mode exists:

- Some questions are about one person, company, place, or event.
- Those questions usually need nearby facts, not the whole graph.
- The graph helps because once we find the anchor node, we can expand outward by hops.

Here is the local search flow:

```python
def local_search(self, *, graph_store: NetworkXKnowledgeGraph, question: str, ...) -> QueryResult:
    query_embedding = self._embed_query(question)
    node_matches = self._rank_nodes(graph, query_embedding, resolved_top_k)
    neighborhood_distances = self._expand_neighborhood(graph, [match.node_name for match in node_matches], resolved_max_hops)
    retrieved_chunks = self._collect_local_chunks(graph, neighborhood_distances=neighborhood_distances, max_chunks=resolved_max_chunks)
```

What just happened?

We took the question, embedded it, found the most similar nodes, walked outward by 1 or 2 hops, and then reconstructed a chunk-level context from the neighborhood.

Design note:

In the implementation for this repo, local ranking adds a small bonus when the question explicitly names a node or alias. Pure cosine similarity can sometimes rank a highly connected related entity above the entity the user clearly asked about, because the node embedding text contains lots of surrounding terms. The small lexical bonus helps specific questions feel more intuitive without replacing semantic similarity.

Design note:

We expand over an **undirected projection** of the graph during retrieval. That may sound odd because the stored graph is directed, but for context gathering we usually care that two entities are connected at all, not only whether the edge points in one direction. This is a retrieval choice, not a modeling change.

How local context gets reconstructed:

- start with chunk mentions attached to the matched and expanded nodes
- deduplicate chunk IDs so repeated mentions do not spam the prompt
- include observed relations inside the neighborhood so the LLM sees structure as well as raw text

Common beginner mistakes:

- Expanding too many hops and flooding the prompt with unrelated context.
- Using only node names and forgetting the original chunk text.
- Confusing "top matching node" with "final answer". It is only the starting point.

Here is the prompt-building part:

```python
def _build_local_context(...):
    lines = [
        f"Question: {question}",
        "Seed node matches:",
        "Expanded neighborhood:",
        "Observed relationships:",
        "Supporting chunk excerpts:",
    ]
```

What just happened?

We turned the retrieved graph neighborhood into a readable prompt that the answer model can use. This is where retrieval becomes RAG: the graph and chunk evidence are now explicitly fed into the generator.

Design note:

The answer model is not traversing the graph itself. Our code does the retrieval and context assembly first, then hands the model a structured evidence packet.

## 4. Global search

Now we implement the thematic mode.

Why this mode exists:

- Some questions ask for the big picture.
- The whole graph is too large to read directly at query time.
- Community summaries act like precomputed neighborhood reports.

This is the map-reduce intuition:

- **map step**: score each community report against the question
- **reduce step**: keep only the top communities and synthesize over them

Here is the global search flow:

```python
def global_search(self, *, graph_store: NetworkXKnowledgeGraph, question: str, ...) -> QueryResult:
    community_matches = self._rank_communities(graph, question, resolved_top_k)
    context_text = self._build_global_context(question=question, community_matches=community_matches)
    answer = self._answer_question(...)
```

What just happened?

We ranked the community summaries against the question, selected the best ones, and used those summaries as the answer context instead of raw chunks.

Design note:

We embed the question and the community summaries in the same vector space so that community selection is still semantic, even though the final context is higher-level than local chunk retrieval.

Reasonable alternatives:

- score communities with an LLM instead of embeddings: more expressive, but slower and more expensive
- use every community summary: simpler, but wastes context window on irrelevant regions of the graph
- skip summaries and read all nodes in each community: possible, but much noisier

Common beginner mistakes:

- Asking the global mode for a precise entity action and expecting crisp details.
- Forgetting that summary quality limits global answer quality.
- Treating map-reduce as magical when it is really just rank, then synthesize.

## 5. Answer generation and optional streaming

Both retrieval modes eventually do the same last step: pass retrieved context to the LLM.

```python
def _answer_question(..., on_token: Callable[[str], None] | None) -> str:
    if on_token is not None and isinstance(self.answer_llm, StreamingAnswerGenerationModel):
        for token in self.answer_llm.stream_text(...):
            on_token(token)
```

What just happened?

We added one optional streaming path. If the model supports token streaming, the query engine can forward tokens to a callback while still assembling the final answer text.

Design note:

The callback keeps the orchestration explicit. The query engine still controls retrieval and answer assembly; streaming is just an output behavior layered on top.

## 6. Same question through both modes

In [examples/phase4_query_demo.py](examples/phase4_query_demo.py), we run the same question through both retrieval modes:

```text
What did Alice do in River County?
```

Local mode retrieves:

- Alice as the top node anchor
- nearby nodes like Acme Corp and River County
- raw chunk excerpts such as "Alice from Acme Corp worked with River County on flood sensors."

Global mode retrieves:

- the highest-scoring community report
- a summary like "Acme Corp, Alice, and River County revolve around flood-sensor work and county resilience."

What just happened?

The same question produced two very different contexts. Local mode pulled concrete evidence. Global mode pulled an abstract summary. That is exactly the contrast you want to understand.

Which mode should you choose?

- Choose local when the question names a person, place, company, or event.
- Choose local when verbs matter, like `did`, `worked`, `founded`, `acquired`, or `managed`.
- Choose global when the question asks for themes, trends, patterns, or high-level summaries.
- Choose global when you see cues like `main themes`, `overall`, `across the documents`, or `big picture`.

## Experiment prompts

1. Ask a very specific entity question through global search. How does the retrieved context become more abstract than the question really needs?
2. Increase local hop expansion from `1` to `3`. Which extra chunks start showing up, and when does the context become noisy?
3. Lower `GLOBAL_SEARCH_TOP_K` to `1` and then raise it to `3`. Does the answer become sharper or more diluted?

## Checkpoint

By the end of this phase, you should be able to explain:

- when embedding similarity wins and when graph traversal wins
- why local search reconstructs raw chunk context
- why global search uses community summaries
- how provenance supports trust in a RAG answer

What questions do you have before we move on?
