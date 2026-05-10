# Phase 2 - Document Ingestion Pipeline

Phase 1 gave us the building blocks. Phase 2 is where raw text first turns into a graph.

Why this phase exists:

- A graph cannot answer anything until we put structure into it.
- Raw documents are too large to hand directly to a graph builder in one shot.
- We need a repeatable pipeline: split text, extract facts, store them, and keep provenance.

Think of this phase like turning a long witness statement into evidence cards.

- The document is the full witness statement.
- The chunks are smaller evidence snippets.
- The triples are the facts written onto index cards.
- The graph is the board where those cards get connected.

If we skip this phase, the later retrieval system has nothing trustworthy to search or traverse.

## 1. Configuration and typed models

Before implementing the pipeline, we need two boring-looking pieces that actually prevent a lot of pain:

- `.env` configuration, so secrets and tuning knobs live outside the code
- Pydantic models, so our pipeline passes around validated objects instead of loose dictionaries

Why `.env` first?

Hardcoding API keys is dangerous because keys leak into git history, screenshots, logs, and shared snippets. Once a secret is committed, you should assume it is compromised.

Here is the configuration code:

```python
from functools import lru_cache

from dotenv import load_dotenv
from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

load_dotenv()


class Settings(BaseSettings):
    openai_api_key: SecretStr | None = Field(default=None, alias="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-4.1-mini", alias="OPENAI_MODEL")
    openai_base_url: str | None = Field(default=None, alias="OPENAI_BASE_URL")
    token_encoding: str = Field(default="cl100k_base", alias="TOKEN_ENCODING")
    chunk_size: int = Field(default=512, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(default=64, alias="CHUNK_OVERLAP")
```

What just happened?

We created one typed settings object that pulls values from `.env`. That means the pipeline code can ask for `settings.chunk_size` or `settings.openai_model` without manually reading environment variables everywhere.

Design note:

Pydantic buys us validation and clear shapes. A plain dictionary would work at first, but it makes bugs easier to hide because every lookup is just a string key and missing fields fail later and more mysteriously.

Now the core data models:

```python
class RelationshipTriple(BaseModel):
    source: str
    relation: str
    target: str


class ExtractionResult(BaseModel):
    entities: list[str] = Field(default_factory=list)
    relationships: list[RelationshipTriple] = Field(default_factory=list)


class TextChunk(BaseModel):
    chunk_id: str
    source_id: str
    text: str
    chunk_index: int
    token_start: int
    token_end: int
```

What just happened?

We gave names to the objects that move through the pipeline. Instead of passing anonymous JSON blobs around, we now have explicit types for chunks, extracted triples, and relationship records.

Design note:

A **triple** is just `subject -> predicate -> object`.

- `Alice -> worked_on -> Project Atlas`
- `Bob -> managed -> Project Atlas`

Triples are the lingua franca of knowledge graphs because they are the smallest useful unit of relational meaning. Almost any graph fact can be reduced to this shape.

Reasonable alternative:

You could make `entities` a list of richer objects with `name`, `type`, and `description`. That is a good future improvement, but for a learning build it is better to keep entity extraction simple and make relationships the star of the phase.

Common beginner mistakes:

- Using plain dicts everywhere and losing track of what keys are required.
- Letting empty entity names or relation labels slip through.
- Treating triples as a database requirement instead of a modeling convenience.

## 2. Chunking the raw document

Before the LLM can extract relationships, we have to feed it pieces that fit comfortably into context and still preserve local meaning.

Why `~512` tokens with `~64` overlap is a common starting point:

- `512` tokens is usually large enough to hold a coherent passage.
- `64` tokens is usually enough to protect facts that sit near a boundary.
- It is large enough to reduce fragmentation, but small enough to avoid huge prompts.

Here is the chunker:

```python
import tiktoken


class TokenChunker:
    def __init__(self, config: ChunkingConfig) -> None:
        self.config = config
        self.encoder = tiktoken.get_encoding(config.encoding_name)

    def chunk_text(self, *, text: str, source_id: str) -> list[TextChunk]:
        token_ids = self.encoder.encode(text.strip())
        step_size = self.config.chunk_size - self.config.chunk_overlap

        chunks: list[TextChunk] = []
        for chunk_index, token_start in enumerate(range(0, len(token_ids), step_size)):
            token_end = min(token_start + self.config.chunk_size, len(token_ids))
            chunk_text = self.encoder.decode(token_ids[token_start:token_end]).strip()
            chunks.append(
                TextChunk(
                    chunk_id=f"{source_id}:{chunk_index}",
                    source_id=source_id,
                    text=chunk_text,
                    chunk_index=chunk_index,
                    token_start=token_start,
                    token_end=token_end,
                )
            )
        return chunks
```

What just happened?

We turned a long string into overlapping token windows and gave each window its own ID. That ID matters because later every extracted fact can point back to the chunk it came from.

Design note:

We chunk by tokens rather than characters because model limits are measured in tokens, not letters. Character counts are a rough shortcut, but they drift from what the model actually sees.

How to tune these numbers:

- If chunks are too small, facts get split and retrieval becomes brittle.
- If chunks are too large, the extractor sees too much mixed context and may produce noisy triples.
- If overlap is too small, boundary facts disappear.
- If overlap is too large, you pay for a lot of duplicated text.

Reasonable alternatives:

- Sentence-based chunking: better readability, but sentence lengths vary a lot.
- Semantic chunking: often stronger, but harder to reason about early on.
- No overlap: simpler, but much easier to break at boundaries.

Common beginner mistakes:

- Forgetting to enforce `overlap < chunk_size`.
- Assuming tokens are the same as words.
- Using one giant chunk because the document is "not that long".

## 3. Extracting entities and relationships with an LLM

Now we reach the first true Graph RAG move: turning text into graph facts.

Why use an LLM here?

- It can generalize across writing styles without us writing lots of rules.
- It can recover implicit phrasing like paraphrases better than a naive regex approach.
- It keeps the orchestration explicit: the model extracts, but our code still owns validation and storage.

Here is the extractor interface and prompt:

```python
EXTRACTION_SYSTEM_PROMPT = """
You extract a small knowledge graph from plain text.

Return JSON only with this exact shape:
{
  "entities": ["entity name"],
  "relationships": [
    {"source": "entity name", "relation": "relation_name", "target": "entity name"}
  ]
}
""".strip()


class EntityRelationshipExtractor:
    def __init__(self, llm_client: ChatModel) -> None:
        self.llm_client = llm_client

    def extract(self, chunk: TextChunk) -> ExtractionResult:
        raw_response = self.llm_client.complete_json(
            system_prompt=EXTRACTION_SYSTEM_PROMPT,
            user_prompt=self._build_user_prompt(chunk),
        )
        return ExtractionResult.model_validate_json(raw_response)
```

What just happened?

We made the extractor depend on a tiny interface, not directly on the OpenAI client. That is important because it lets us swap in a fake client in tests and keep the workflow explicit.

Design note:

The LLM is not the source of truth. The schema is. We let the model propose entities and relationships, then force the output through Pydantic. That is a good habit because model outputs are probabilistic and your graph storage should not be.

Reasonable alternatives:

- spaCy or a rule-based extractor: cheaper and deterministic, but less flexible for relationship extraction.
- end-to-end agents: more automated, but harder to inspect and learn from.

Common beginner mistakes:

- Trusting model output without validation.
- Accepting prose around the JSON and then fighting parser errors.
- Forgetting the edge case where a paragraph genuinely has no useful entities.

Here is the OpenAI-compatible transport:

```python
class OpenAIChatClient:
    def complete_json(self, *, system_prompt: str, user_prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=0.0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return response.choices[0].message.content
```

What just happened?

We wrapped the model call in one tiny class whose whole job is: send prompt, request JSON, return text. That keeps the LLM boundary obvious instead of spreading API calls throughout the project.

Design note:

We keep the LLM client thin on purpose. The extractor owns prompting, the models own validation, and the graph store owns persistence. Separating those concerns makes later debugging much easier.

## 4. Storing triples in NetworkX

Once we have triples, we need a graph structure that preserves both relationships and provenance.

Why NetworkX for now?

- It is lightweight and easy to inspect in Python.
- It keeps the graph in memory, which is fine for a learning project.
- It lets us focus on graph ideas before introducing database operational complexity.

Here is the graph store:

```python
class NetworkXKnowledgeGraph:
    def __init__(self) -> None:
        self.graph = nx.MultiDiGraph()
        self.graph.graph["chunks"] = {}

    def add_extraction(self, *, chunk: TextChunk, extraction: ExtractionResult) -> None:
        self.graph.graph["chunks"][chunk.chunk_id] = chunk.model_dump()

        for entity_name in extraction.entities:
            self._upsert_entity(entity_name=entity_name, chunk=chunk)

        for relationship_index, relationship in enumerate(extraction.relationships):
            self.graph.add_edge(
                relationship.source,
                relationship.target,
                key=f"{chunk.chunk_id}:{relationship_index}",
                relation=relationship.relation,
                chunk_id=chunk.chunk_id,
                chunk_text=chunk.text,
            )
```

What just happened?

We stored the chunk itself, added entity nodes, and created directed edges for each triple. The graph now knows both the relational structure and where each relation came from.

Design note:

We store the raw chunk text alongside each node via `mentions` metadata because later retrieval needs to reconstruct human-readable context, not just bare node names. Without that provenance, you can answer "Alice is connected to Project Atlas," but you cannot easily show the original supporting text that builds trust.

Why `MultiDiGraph` instead of a plain `Graph`?

- `DiGraph` keeps direction, which matches triples naturally.
- `Multi` lets us keep multiple edges between the same pair of entities if they come from different chunks or use different relation labels.

Reasonable alternative:

You could attach chunk IDs only to edges and not to nodes. That saves duplication, but it makes later context reconstruction clumsier because nodes would not know which source text mentioned them.

Common beginner mistakes:

- Using an undirected graph for directed facts.
- Overwriting one relationship with another between the same entity pair.
- Storing only the entity names and forgetting the supporting text.

## 5. Putting the pipeline together

Now we assemble the three moving parts: chunker, extractor, and graph store.

```python
class IngestionPipeline:
    def __init__(
        self,
        *,
        chunker: TokenChunker,
        extractor: EntityRelationshipExtractor,
        graph_store: NetworkXKnowledgeGraph,
    ) -> None:
        self.chunker = chunker
        self.extractor = extractor
        self.graph_store = graph_store

    def ingest_text(self, *, source_id: str, text: str) -> IngestionReport:
        chunks = self.chunker.chunk_text(text=text, source_id=source_id)
        for chunk in chunks:
            extraction = self.extractor.extract(chunk)
            self.graph_store.add_extraction(chunk=chunk, extraction=extraction)
```

What just happened?

We created one explicit orchestration layer. It does not hide anything magical. It just runs the steps in order: split text, extract facts, store them.

Design note:

Keeping orchestration explicit is the whole point of this learning project. You should be able to read the pipeline top to bottom and see exactly where each fact came from.

Here is the mocked demo path in this repo:

```bash
python3 examples/phase2_ingest_demo.py
```

What just happened?

That example runs the full ingestion flow with a scripted fake LLM response, so you can inspect the graph structure without paying for API calls.

## 6. Why the mocked tests matter

The extractor and pipeline are tested with fake LLM responses.

Why mock the LLM?

- Unit tests should be fast.
- Unit tests should be deterministic.
- Unit tests should not spend money or depend on network access.

Mocking means replacing the real API call with a pretend object that returns known data. This lets us test our own logic instead of the provider's uptime.

One important test in this repo checks a subtle normalization step: if the model forgets to list `Project Atlas` under `entities` but still uses it in a relationship, we add it back in from the triple. Otherwise the graph would miss a node that clearly exists in the relation.

## Experiment prompts

1. Feed in two documents about the same topic. What happens to shared entities? For now, they will be duplicated if the names differ even slightly, because merging comes in Phase 3.
2. Deliberately give the extractor a paragraph with no meaningful entities. Does it return empty arrays? Does the pipeline stay stable instead of crashing?
3. Change `CHUNK_OVERLAP` from `64` to `0`. Which facts become more vulnerable to being split across chunks?

## Checkpoint

By the end of this phase, you should be able to explain:

- why chunking happens before extraction
- what a triple is and why it is useful
- why provenance matters for trust
- why we keep raw chunk text attached to nodes and edges

What questions do you have before we move on?
