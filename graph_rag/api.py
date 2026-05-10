from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, status

from .api_models import EntityDetailResponse, GraphStatsResponse, IngestRequest, IngestResponse, QueryRequest
from .api_service import DuplicateSourceError, EntityNotFoundError, GraphNotReadyError, GraphRAGAPIService
from .models import QueryResult


def create_app(service: GraphRAGAPIService | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if not hasattr(app.state, "graph_rag_service"):
            app.state.graph_rag_service = service or GraphRAGAPIService.from_settings()
        try:
            yield
        finally:
            app.state.graph_rag_service.close()

    app = FastAPI(
        title="Graph RAG From Scratch",
        version="0.6.0",
        summary="Learning-focused Graph RAG API with explicit ingestion, enrichment, and retrieval steps.",
        lifespan=lifespan,
    )

    if service is not None:
        app.state.graph_rag_service = service

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post(
        "/ingest",
        response_model=IngestResponse,
        status_code=status.HTTP_201_CREATED,
    )
    def ingest_document(payload: IngestRequest, request: Request) -> IngestResponse:
        graph_rag_service: GraphRAGAPIService = request.app.state.graph_rag_service
        try:
            return graph_rag_service.ingest_text(
                source_id=payload.source_id,
                text=payload.text,
            )
        except DuplicateSourceError as error:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(error)) from error

    @app.post("/query", response_model=QueryResult)
    def query_graph(payload: QueryRequest, request: Request) -> QueryResult:
        graph_rag_service: GraphRAGAPIService = request.app.state.graph_rag_service
        try:
            return graph_rag_service.query(
                question=payload.question,
                mode=payload.mode,
                top_k=payload.top_k,
                max_hops=payload.max_hops,
                max_chunks=payload.max_chunks,
            )
        except GraphNotReadyError as error:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(error)) from error

    @app.get("/graph/stats", response_model=GraphStatsResponse)
    def graph_stats(request: Request) -> GraphStatsResponse:
        graph_rag_service: GraphRAGAPIService = request.app.state.graph_rag_service
        return graph_rag_service.graph_stats()

    @app.get("/graph/entity/{name}", response_model=EntityDetailResponse)
    def graph_entity(name: str, request: Request) -> EntityDetailResponse:
        graph_rag_service: GraphRAGAPIService = request.app.state.graph_rag_service
        try:
            return graph_rag_service.entity_detail(name=name)
        except EntityNotFoundError as error:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(error)) from error

    return app