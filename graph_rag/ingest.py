from __future__ import annotations

from typing import Protocol

from .chunking import TokenChunker
from .extractor import EntityRelationshipExtractor
from .models import ExtractionResult, IngestionReport, TextChunk


class ExtractionGraphStore(Protocol):
    def add_extraction(self, *, chunk: TextChunk, extraction: ExtractionResult) -> None:
        ...


class IngestionPipeline:
    def __init__(
        self,
        *,
        chunker: TokenChunker,
        extractor: EntityRelationshipExtractor,
        graph_store: ExtractionGraphStore,
    ) -> None:
        self.chunker = chunker
        self.extractor = extractor
        self.graph_store = graph_store

    def ingest_text(self, *, source_id: str, text: str) -> IngestionReport:
        chunks = self.chunker.chunk_text(text=text, source_id=source_id)
        entity_count = 0
        relationship_count = 0

        for chunk in chunks:
            extraction = self.extractor.extract(chunk)
            entity_count += len(extraction.entities)
            relationship_count += len(extraction.relationships)
            self.graph_store.add_extraction(chunk=chunk, extraction=extraction)

        return IngestionReport(
            source_id=source_id,
            chunk_count=len(chunks),
            entity_count=entity_count,
            relationship_count=relationship_count,
        )