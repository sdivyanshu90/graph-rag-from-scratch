from __future__ import annotations

import tiktoken

from .models import ChunkingConfig, TextChunk


class TokenChunker:
    def __init__(self, config: ChunkingConfig) -> None:
        self.config = config
        self.encoder = tiktoken.get_encoding(config.encoding_name)

    def chunk_text(self, *, text: str, source_id: str) -> list[TextChunk]:
        stripped_text = text.strip()
        if not stripped_text:
            return []

        token_ids = self.encoder.encode(stripped_text)
        step_size = self.config.chunk_size - self.config.chunk_overlap
        chunks: list[TextChunk] = []

        for chunk_index, token_start in enumerate(range(0, len(token_ids), step_size)):
            token_end = min(token_start + self.config.chunk_size, len(token_ids))
            chunk_token_ids = token_ids[token_start:token_end]
            if not chunk_token_ids:
                continue

            chunk_text = self.encoder.decode(chunk_token_ids).strip()
            if not chunk_text:
                continue

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