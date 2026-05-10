# graph-rag-from-scratch

This repository is a learning-first build of Graph RAG in Python. The system stays explicit on purpose: chunking, extraction, enrichment, retrieval, Neo4j sync, the REST API, and evaluation are all readable as ordinary Python instead of being hidden inside opaque agent frameworks.

## Quickstart

Local setup:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# fill in OPENAI_API_KEY before using the real extractor, summaries, answering, or eval judge
```

Run the API locally:

```bash
python3 -m uvicorn graph_rag.api:create_app --factory --reload --host 0.0.0.0 --port 8000
```

Run the Docker stack with Neo4j + API:

```bash
docker compose up --build
```

## Architecture

```text
plain text document
				|
				v
TokenChunker
				|
				v
EntityRelationshipExtractor (LLM)
				|
				v
NetworkXKnowledgeGraph
				|
				v
GraphEnrichmentPipeline
	- fuzzy merge
	- communities
	- node embeddings
	- community summaries
				|
				+-------------------------------+
				|                               |
				v                               v
GraphQueryEngine                  Neo4jKnowledgeGraph
	- local search                    - optional sync for persistence
	- global search                   - Cypher retrieval from Phase 5
				|
				+-------------------------------+
				|                               |
				v                               v
FastAPI                         EvaluationHarness
	- POST /ingest                  - exact match
	- POST /query                   - LLM-as-judge
	- GET  /graph/stats             - JSON report output
	- GET  /graph/entity/{name}
```

## Phase Guide

Phase 1: [docs/phase-01-foundations.md](docs/phase-01-foundations.md)

- what a knowledge graph is
- a tiny NetworkX graph example
- sentence embeddings plus cosine similarity
- chunking with and without overlap

Phase 2: [docs/phase-02-ingestion.md](docs/phase-02-ingestion.md)

- `.env`-driven configuration
- typed Pydantic models for chunks, triples, and ingestion reports
- token-based chunking with overlap
- LLM-based entity and relationship extraction
- a NetworkX knowledge graph with chunk provenance attached to nodes and edges
- mocked pytest tests for the extractor and ingestion pipeline

Phase 3: [docs/phase-03-enrichment.md](docs/phase-03-enrichment.md)

- fuzzy entity merging with RapidFuzz-style similarity
- Leiden-based community assignment hooks
- node embeddings for entity-centric retrieval
- precomputed community summaries for later global search
- mocked pytest tests for graph enrichment

Phase 4: [docs/phase-04-query-engine.md](docs/phase-04-query-engine.md)

- side-by-side local vs global retrieval
- node-embedding local search with hop expansion and chunk reconstruction
- community-summary global search with map-reduce style ranking
- answer generation with optional token streaming callbacks
- typed provenance for nodes, chunks, and communities
- mocked pytest tests for both retrieval modes

Phase 5: [docs/phase-05-neo4j.md](docs/phase-05-neo4j.md)

- an official-driver Neo4j store and query engine
- a sync path from the enriched NetworkX graph into Neo4j
- Cypher-based local and global retrieval equivalents
- side-by-side NetworkX vs Cypher query comparisons
- mocked pytest tests for the Neo4j store and retrieval path

Phase 6: [docs/phase-06-api.md](docs/phase-06-api.md)

- a FastAPI layer over ingest, query, stats, and entity inspection
- HTTP request and response models with Pydantic v2
- optional best-effort Neo4j sync after ingest
- pytest coverage for the API surface
- worked curl examples and Docker support

Phase 7: [docs/phase-07-evaluation.md](docs/phase-07-evaluation.md)

- typed evaluation fixtures and per-case report records
- normalized exact-match scoring
- LLM-as-judge scoring for faithfulness and relevance on a 1-5 scale
- a JSON report script that reuses the real query path
- mocked pytest coverage for the evaluation harness

## Demos And Tests

Run the Phase 1 demos:

```bash
python3 examples/phase1_networkx_demo.py
python3 examples/phase1_embedding_demo.py
python3 examples/phase1_chunking_demo.py
```

Run the Phase 2 mocked demo and tests:

```bash
python3 examples/phase2_ingest_demo.py
python3 -m pytest tests/test_entity_extractor.py tests/test_ingest_pipeline.py
```

Run the Phase 3 mocked demo and tests:

```bash
python3 examples/phase3_enrichment_demo.py
python3 -m pytest tests/test_graph_enrichment.py
```

Run the Phase 4 mocked demo and tests:

```bash
python3 examples/phase4_query_demo.py
python3 -m pytest tests/test_query_engine.py
```

Run the Phase 5 comparison demo and Neo4j tests:

```bash
python3 examples/phase5_neo4j_demo.py
python3 -m pytest tests/test_neo4j_store.py tests/test_neo4j_query_engine.py
```

Run the Phase 6 API tests:

```bash
python3 -m pytest tests/test_api.py
```

Run the Phase 7 mocked demo and evaluation tests:

```bash
python3 examples/phase7_eval_demo.py
python3 -m pytest tests/test_evaluation.py
```

Run the full mocked regression suite:

```bash
python3 -m pytest
```

## Evaluation Script

Run the real evaluation harness with a JSON fixture and write a JSON report:

```bash
python3 scripts/run_eval.py \
	--fixture evals/sample_eval_fixture.json \
	--output evals/report.json
```

The sample fixture includes both source documents and evaluation cases. The script builds the normal Graph RAG service from `.env`, ingests the documents, runs the cases through the existing query path, and writes a machine-readable report.

## API Curls

Ingest one document:

```bash
curl -X POST http://localhost:8000/ingest \
	-H 'Content-Type: application/json' \
	-d '{
		"source_id": "city-report-1",
		"text": "Alice from Acme Corp worked with River County on flood sensors. Bob later reviewed the deployment plan."
	}'
```

Ask a local question:

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

Get graph stats:

```bash
curl http://localhost:8000/graph/stats
```

Inspect one entity:

```bash
curl http://localhost:8000/graph/entity/Alice
```

Core modules:

- extractor transport: [graph_rag/llm.py](graph_rag/llm.py) and [graph_rag/extractor.py](graph_rag/extractor.py)
- enrichment: [graph_rag/enrichment.py](graph_rag/enrichment.py)
- in-memory retrieval: [graph_rag/query.py](graph_rag/query.py)
- Neo4j sync and retrieval: [graph_rag/neo4j_store.py](graph_rag/neo4j_store.py) and [graph_rag/neo4j_query.py](graph_rag/neo4j_query.py)
- API layer: [graph_rag/api.py](graph_rag/api.py) and [graph_rag/api_service.py](graph_rag/api_service.py)
- evaluation harness: [graph_rag/evaluation.py](graph_rag/evaluation.py) and [scripts/run_eval.py](scripts/run_eval.py)
