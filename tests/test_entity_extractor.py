from __future__ import annotations

import json

from graph_rag.extractor import EntityRelationshipExtractor
from graph_rag.models import TextChunk


class FakeLLMClient:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload

    def complete_json(self, *, system_prompt: str, user_prompt: str) -> str:
        del system_prompt
        del user_prompt
        return json.dumps(self.payload)


def test_extractor_adds_entities_referenced_inside_relationships() -> None:
    extractor = EntityRelationshipExtractor(
        FakeLLMClient(
            {
                "entities": ["Alice"],
                "relationships": [
                    {
                        "source": "Alice",
                        "relation": "worked_on",
                        "target": "Project Atlas",
                    }
                ],
            }
        )
    )
    chunk = TextChunk(
        chunk_id="doc-1:0",
        source_id="doc-1",
        text="Alice worked on Project Atlas.",
        chunk_index=0,
        token_start=0,
        token_end=6,
    )

    result = extractor.extract(chunk)

    assert result.entities == ["Alice", "Project Atlas"]
    assert result.relationships[0].relation == "worked_on"


def test_extractor_accepts_empty_results_for_entity_free_text() -> None:
    extractor = EntityRelationshipExtractor(
        FakeLLMClient({"entities": [], "relationships": []})
    )
    chunk = TextChunk(
        chunk_id="doc-2:0",
        source_id="doc-2",
        text="It rained yesterday.",
        chunk_index=0,
        token_start=0,
        token_end=4,
    )

    result = extractor.extract(chunk)

    assert result.entities == []
    assert result.relationships == []