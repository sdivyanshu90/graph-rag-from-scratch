from __future__ import annotations

from pydantic import ValidationError

from .llm import ChatModel
from .models import ExtractionResult, TextChunk

EXTRACTION_SYSTEM_PROMPT = """
You extract a small knowledge graph from plain text.

Return JSON only with this exact shape:
{
  "entities": ["entity name"],
  "relationships": [
    {"source": "entity name", "relation": "relation_name", "target": "entity name"}
  ]
}

Rules:
- Include only entities explicitly supported by the text.
- Use short relation labels such as founded, worked_on, located_in, acquired.
- If no entities or relationships are present, return empty arrays.
- Do not add any explanatory text outside the JSON.
""".strip()


class EntityRelationshipExtractor:
    def __init__(self, llm_client: ChatModel) -> None:
        self.llm_client = llm_client

    def extract(self, chunk: TextChunk) -> ExtractionResult:
        raw_response = self.llm_client.complete_json(
            system_prompt=EXTRACTION_SYSTEM_PROMPT,
            user_prompt=self._build_user_prompt(chunk),
        )

        try:
            return ExtractionResult.model_validate_json(raw_response)
        except ValidationError as error:
            raise ValueError("LLM returned invalid extraction JSON") from error

    @staticmethod
    def _build_user_prompt(chunk: TextChunk) -> str:
        return (
            "Extract named entities and explicit relationships from the following chunk.\n\n"
            f"Source ID: {chunk.source_id}\n"
            f"Chunk ID: {chunk.chunk_id}\n"
            f"Chunk text:\n{chunk.text}"
        )