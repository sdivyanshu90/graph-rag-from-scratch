# Phase 1 - Foundations

Before we touch Retrieval-Augmented Generation, we need the parts that make Graph RAG different from plain vector search.

Why this phase exists:

- A flat vector store is good at finding text that _sounds semantically similar_.
- A graph is good at representing _who is connected to what_.
- Graph RAG works because it combines both ideas instead of pretending they are the same thing.

We are intentionally _not_ introducing `.env`, Pydantic, or LLM calls yet. Those matter in Phase 2, but they would distract from the core ideas here.

## 1. What is a knowledge graph?

Concrete analogy:

Imagine a newsroom wall covered with index cards.

- One card says `Alice`.
- One card says `Project Atlas`.
- One card says `Bob`.
- A string connects `Alice` to `Project Atlas` with the label `worked_on`.
- Another string connects `Bob` to `Project Atlas` with the label `managed`.

That wall is already a tiny knowledge graph.

- The cards are **nodes**.
- The strings are **edges**.
- The edge labels tell you **what kind of relationship** exists.
- A **1-hop neighbour** is anything reachable by following one edge.

Without this structure, you just have text blobs. You can search the words, but you cannot easily ask relational questions like "Who worked on the same project as Alice?"

Here is a 10-line NetworkX example:

```python
import networkx as nx
graph = nx.Graph()
graph.add_node("Alice", kind="person")
graph.add_node("Project Atlas", kind="project")
graph.add_node("Bob", kind="person")
graph.add_edge("Alice", "Project Atlas", relation="worked_on")
graph.add_edge("Bob", "Project Atlas", relation="managed")
print("Nodes:", list(graph.nodes(data=True)))
print("1-hop from Alice:", list(graph.neighbors("Alice")))
print("Alice -> Project Atlas:", graph["Alice"]["Project Atlas"]["relation"])
```

What just happened?

We created three things, connected them with labeled relationships, and then asked the graph for Alice's direct neighbours. In graph language, we just performed a 1-hop traversal.

Design note:

We use an undirected `Graph` here because it keeps the first example simple. In the real Graph RAG pipeline, relationships often behave more naturally as directed triples like `Alice -> worked_on -> Project Atlas`. We are postponing that extra detail until the ingestion phase.

Reasonable alternative:

You could start with a plain Python dictionary such as `{"Alice": ["Project Atlas"]}`. That is fine for a toy example, but NetworkX gives you graph operations and attributes immediately, which is more useful for learning the real workflow.

Common beginner mistakes:

- Thinking the graph "understands" the world automatically. It only knows the nodes and edges you explicitly add.
- Confusing a node label like `Alice` with a full entity record. Later we will attach more metadata.
- Forgetting that a graph answers relationship questions best, not every question.

## 2. What are embeddings and cosine similarity?

Analogy first:

Imagine every sentence is dropped onto a map where nearby points mean "similar meaning". Two sentences that say nearly the same thing land close together, even if they use different words.

That map is what an embedding model gives us: a vector, or list of numbers, that captures semantic meaning.

We can compare two vectors with **cosine similarity**.

- `1.0` means almost identical direction, so very similar meaning.
- `0.0` means mostly unrelated.
- negative values mean strongly different directions.

Example:

```python
from sentence_transformers import SentenceTransformer

sentences = [
    "Alice leads the climate research project.",
    "Alice is in charge of the climate study.",
    "Bananas are rich in potassium.",
]

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
embeddings = model.encode(sentences, normalize_embeddings=True)
similarity = embeddings @ embeddings.T

print("      |   s1 |   s2 |   s3")
for index, row in enumerate(similarity, start=1):
    print(f"s{index:<5} | " + " | ".join(f"{value:0.3f}" for value in row))
```

What just happened?

The model turned each sentence into a vector. Because we asked for normalized vectors, the dot product `embeddings @ embeddings.T` becomes cosine similarity. The two Alice sentences should score much higher against each other than either one scores against the banana sentence.

Design note:

We use cosine similarity because we care about **direction** of meaning more than raw vector magnitude. For normalized embeddings, cosine similarity is a simple and strong default for retrieval.

Reasonable alternatives:

- Euclidean distance: sometimes useful, but less standard for semantic retrieval.
- Dot product without normalization: faster in some systems, but magnitude can distort intuition for beginners.
- A different embedding model: larger models can be better, but `all-MiniLM-L6-v2` is fast and easy to learn with.

Common beginner mistakes:

- Expecting the exact same numbers on every machine. The pattern matters more than the last decimal.
- Thinking high similarity means "factually identical". It only means semantically close.
- Comparing sentence text directly and assuming paraphrases will fail. Embeddings are specifically meant to help with paraphrases.

## 3. What is chunking, and why do we overlap chunks?

Chunking means splitting a long document into smaller pieces before embedding or sending them to an LLM.

Why chunk at all?

- Embedding models and LLMs have context limits.
- Retrieval works better on focused passages than giant documents.
- Smaller chunks make provenance easier later because you can point to the exact supporting text.

Now the key idea: **overlap**.

If one fact sits near the boundary between two chunks, zero overlap can split the fact awkwardly. A small overlap repeats some boundary tokens so important context is less likely to be lost.

```python
def chunk_words(text: str, size: int, overlap: int) -> list[str]:
    if overlap >= size:
        raise ValueError("overlap must be smaller than size")
    words = text.split()
    step = size - overlap
    return [
        " ".join(words[start : start + size])
        for start in range(0, len(words), step)
        if words[start : start + size]
    ]

sample = (
    "Alice joined the city climate lab in 2021 and later led the flood-risk "
    "project with Bob. Together they published a report on river safety."
)

print("No overlap:", chunk_words(sample, size=8, overlap=0))
print("With overlap:", chunk_words(sample, size=8, overlap=2))
```

What just happened?

We split the same paragraph two ways. With no overlap, each chunk starts exactly where the previous one ended. With overlap, a few words are intentionally repeated, so details that cross a boundary survive in both neighboring chunks.

Design note:

Overlap is a recall-friendly trade-off: you pay a little extra storage and some duplicate text to reduce the chance of chopping a useful fact in half. For learning, this is one of the most important retrieval quality ideas to internalize.

Reasonable alternatives:

- Sentence-based chunking: cleaner boundaries, but sentence lengths vary a lot.
- Character-based chunking: easy to implement, but it ignores token boundaries.
- Semantic chunking: often better, but it adds complexity too early for a first build.

Common beginner mistakes:

- Setting overlap equal to chunk size, which makes the step size zero.
- Treating characters, words, and tokens as if they were the same unit.
- Making chunks so small that each chunk loses the context needed to answer a question.

Why `~512` tokens with `~64` overlap is a common starting point:

- `512` tokens is usually big enough to hold a coherent passage.
- `64` tokens is usually enough to protect boundary facts without creating too much duplication.
- They are defaults, not laws. The right values depend on document style, model context size, and the kinds of questions you care about.

## Experiment prompts

1. Change the graph example so Alice and Bob are connected directly. What new 1-hop answers become possible?
2. Replace the banana sentence with another climate sentence. How does the similarity table change?
3. Set chunk overlap to `0`, then to `4`, then to `7`. Where does the trade-off start to feel wasteful?

## Checkpoint

If this phase landed, you should already be able to explain:

- why a graph is better than raw text for relationship questions
- why embeddings help with paraphrases
- why overlap protects context at chunk boundaries

What questions do you have before we move on?
