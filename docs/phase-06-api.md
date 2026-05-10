# Phase 6 - REST API With FastAPI

Phase 5 gave us a working Graph RAG system plus an optional Neo4j upgrade. Phase 6 makes that system callable over HTTP.

Why this phase exists:

- a graph pipeline is much easier to reuse once it is behind stable endpoints
- an API forces us to define clear request and response shapes
- provenance becomes even more important once another client is consuming answers instead of reading Python objects directly

Think of this phase like putting a front desk in front of the graph system.

- `POST /ingest` hands the front desk a new document to process
- `POST /query` asks a question and gets back an answer plus evidence
- `GET /graph/stats` asks for a quick dashboard snapshot
- `GET /graph/entity/{name}` asks for one entity card and its nearby evidence

If we skip this phase, the project still works, but only as code you call manually. The API is what turns the learning project into a reusable service.

## 1. Keep the API layer thin

Before the code, the main design decision:

We keep FastAPI thin and push most logic into one explicit service class.

Why?

- endpoints should be easy to scan
- graph orchestration should live in normal Python methods, not be buried inside route functions
- tests become simpler when the API can be given a fake service

Here is the service boundary:

```python
class GraphRAGAPIService:
    def ingest_text(self, *, source_id: str, text: str) -> IngestResponse:
        ...

    def query(self, *, question: str, mode: Literal["local", "global"], ...) -> QueryResult:
        ...

    def graph_stats(self) -> GraphStatsResponse:
        ...

    def entity_detail(self, *, name: str) -> EntityDetailResponse:
        ...
```

What just happened?

We created one application service that owns all HTTP-facing operations. FastAPI becomes a transport layer, not the place where graph logic lives.

Design note:

In this phase, the API still serves queries from the in-memory NetworkX graph because that keeps the request path easiest to understand. If Neo4j sync is enabled, ingest also performs a best-effort sync after enrichment so the database stays available for Phase 5 comparisons and persistence experiments.

## 2. Request and response models

Before implementing the endpoints, we define the HTTP payloads with Pydantic.

```python
class IngestRequest(BaseModel):
    source_id: str
    text: str


class QueryRequest(BaseModel):
    question: str
    mode: Literal["local", "global"] = "local"
    top_k: int | None = None
    max_hops: int | None = None
    max_chunks: int | None = None
```

What just happened?

We made the request shapes explicit and validated them before any graph code runs.

Design note:

This is one of the reasons Pydantic is worth using. Plain dictionaries would work, but you would be hand-checking fields, default values, and type errors inside each endpoint. With Pydantic, the schema lives in one clear place.

We also define API-specific response models, including neighbors and chunk excerpts for the entity detail endpoint.

Common beginner mistakes:

- letting empty strings through for required fields like `question` or `source_id`
- putting transport-specific shapes directly into graph logic classes
- returning unstructured dicts that slowly drift over time

## 3. Build the FastAPI app

Here is the app factory:

```python
def create_app(service: GraphRAGAPIService | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.graph_rag_service = service or GraphRAGAPIService.from_settings()
        try:
            yield
        finally:
            app.state.graph_rag_service.close()
```

What just happened?

We used a lifespan hook to create the service once at startup and clean it up at shutdown. The app can also accept an injected fake service, which is exactly what we use in tests.

Design note:

The app factory keeps the runtime explicit. You can see exactly where the service is built, where config is read, and where external resources such as the optional Neo4j driver get closed.

## 4. Endpoint by endpoint

### `POST /ingest`

Why `POST`?

Because ingest changes server state. It creates chunks, entities, relationships, embeddings, communities, and possibly a Neo4j sync.

Why this path?

`/ingest` is short and literal. It describes the action clearly without pretending the client is already managing low-level chunk or triple objects.

Status codes we use:

- `201 Created` when a new source is accepted and processed
- `409 Conflict` when the same `source_id` is submitted again in this learning build
- `422 Unprocessable Entity` when the JSON body fails schema validation

Here is the endpoint:

```python
@app.post("/ingest", response_model=IngestResponse, status_code=status.HTTP_201_CREATED)
def ingest_document(payload: IngestRequest, request: Request) -> IngestResponse:
    return graph_rag_service.ingest_text(source_id=payload.source_id, text=payload.text)
```

What just happened?

The API accepted raw text, passed it into the pipeline, and returned both the ingestion report and the follow-up enrichment report.

Worked curl example:

```bash
curl -X POST http://localhost:8000/ingest \
  -H 'Content-Type: application/json' \
  -d '{
    "source_id": "city-report-1",
    "text": "Alice from Acme Corp worked with River County on flood sensors. Bob later reviewed the deployment plan."
  }'
```

### `POST /query`

Why `POST` instead of `GET`?

- the request body contains more than a simple URL parameter
- query settings like `mode`, `top_k`, and `max_hops` fit naturally in JSON
- some questions may be long, which is awkward in URLs

Why one path for both local and global modes?

Because the client is asking one conceptual thing: answer a question. The `mode` field makes the retrieval strategy explicit without splitting the public surface too early.

Status codes we use:

- `200 OK` when the question is answered successfully
- `409 Conflict` when the graph is not ready yet, such as querying before ingest or before enrichment data exists
- `422 Unprocessable Entity` for invalid request shapes

Here is the endpoint:

```python
@app.post("/query", response_model=QueryResult)
def query_graph(payload: QueryRequest, request: Request) -> QueryResult:
    return graph_rag_service.query(
        question=payload.question,
        mode=payload.mode,
        top_k=payload.top_k,
        max_hops=payload.max_hops,
        max_chunks=payload.max_chunks,
    )
```

What just happened?

The API forwarded the question to the explicit query engine and returned the answer plus structured provenance.

Worked curl example:

```bash
curl -X POST http://localhost:8000/query \
  -H 'Content-Type: application/json' \
  -d '{
    "question": "What did Alice do in River County?",
    "mode": "local",
    "top_k": 2,
    "max_hops": 1,
    "max_chunks": 3
  }'
```

### `GET /graph/stats`

Why `GET`?

Because this endpoint is read-only. It just reports the current graph size.

Why this path?

`/graph/stats` tells the client this is metadata about the graph as a whole, not a query answer.

Status codes we use:

- `200 OK` on success

Here is the endpoint:

```python
@app.get("/graph/stats", response_model=GraphStatsResponse)
def graph_stats(request: Request) -> GraphStatsResponse:
    return graph_rag_service.graph_stats()
```

What just happened?

We exposed the current node, edge, chunk, and community counts as a tiny monitoring endpoint.

Worked curl example:

```bash
curl http://localhost:8000/graph/stats
```

### `GET /graph/entity/{name}`

Why `GET`?

Because the client is fetching one existing graph view.

Why put the entity name in the path?

Because the name identifies the resource being inspected. This endpoint is conceptually "give me the entity card for this node."

Status codes we use:

- `200 OK` when the entity exists
- `404 Not Found` when the canonical name or alias is not present in the graph

Here is the endpoint:

```python
@app.get("/graph/entity/{name}", response_model=EntityDetailResponse)
def graph_entity(name: str, request: Request) -> EntityDetailResponse:
    return graph_rag_service.entity_detail(name=name)
```

What just happened?

We exposed a debugging and trust-building endpoint that returns one entity's aliases, neighbors, and chunk excerpts.

Worked curl example:

```bash
curl http://localhost:8000/graph/entity/Alice
```

## 5. Common API design questions

### Why reject duplicate `source_id`s with `409`?

Because this learning build does not implement full upsert semantics. If we silently accepted duplicate source IDs, the behavior would be harder to reason about. `409 Conflict` makes the limitation explicit.

### Why return provenance in `POST /query`?

Because trust matters. The client should know which nodes, chunks, and communities were used to produce the answer.

### Why keep Neo4j sync best-effort here?

Because the API's main job in this phase is to make the request flow easy to read. If Neo4j is available, we sync after ingest. If it is down, the in-memory learning pipeline still works and the response reports the sync error instead of hiding it.

## 6. Docker Compose

This repo now includes `docker-compose.yml` plus a `Dockerfile`.

Why add them here?

- the API is easier to try when one command starts both Neo4j and FastAPI
- environment-driven configuration stays visible
- it makes the Phase 5 plus Phase 6 integration concrete

Worked command:

```bash
docker compose up --build
```

What just happened?

That command starts Neo4j on `7474` and `7687`, builds the API container, and runs FastAPI on `8000` with Neo4j sync enabled inside the compose environment.

## 7. Testing the API

The API tests use `fastapi.testclient.TestClient` with a fake injected service.

Why mock the service here?

- endpoint tests should focus on HTTP behavior, status codes, and response shapes
- they should not need a real LLM, sentence-transformer model, or Neo4j server
- it keeps failures local and easy to understand

This is the same testing principle from earlier phases: mock external dependencies so you can test your own orchestration clearly.

## Experiment prompts

1. Change `POST /query` from `mode: "local"` to `mode: "global"` in the same curl command. How does the answer and provenance change?
2. Try ingesting the same `source_id` twice and watch the `409` response. How would you redesign this if you wanted true upserts?
3. Call `GET /graph/entity/{name}` with an alias instead of the canonical name. Does it still resolve correctly? If so, why is that useful?

## Checkpoint

By the end of this phase, you should be able to explain:

- why `POST /ingest` and `POST /query` are `POST` instead of `GET`
- why `GET /graph/stats` and `GET /graph/entity/{name}` are read-only endpoints
- why provenance is still crucial once the system is behind HTTP
- why a thin API layer is easier to maintain than putting graph logic directly in route functions

What questions do you have before we move on?
