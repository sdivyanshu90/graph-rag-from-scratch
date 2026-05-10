from __future__ import annotations

from collections.abc import Iterator
from typing import Protocol

from openai import OpenAI

from .config import Settings, get_settings


class ChatModel(Protocol):
    def complete_json(self, *, system_prompt: str, user_prompt: str) -> str:
        ...


class OpenAIChatClient:
    def __init__(self, *, model: str, api_key: str, base_url: str | None = None) -> None:
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> "OpenAIChatClient":
        resolved_settings = settings or get_settings()
        if (
            resolved_settings.openai_api_key is None
            or not resolved_settings.openai_api_key.get_secret_value().strip()
        ):
            raise ValueError("OPENAI_API_KEY is required before calling the real extractor")

        return cls(
            model=resolved_settings.openai_model,
            api_key=resolved_settings.openai_api_key.get_secret_value(),
            base_url=resolved_settings.openai_base_url,
        )

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

        return self._extract_text(response.choices[0].message.content)

    def complete_text(self, *, system_prompt: str, user_prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=0.0,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return self._extract_text(response.choices[0].message.content)

    def stream_text(self, *, system_prompt: str, user_prompt: str) -> Iterator[str]:
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=0.0,
            stream=True,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        for chunk in response:
            if not chunk.choices:
                continue

            delta = chunk.choices[0].delta.content
            if delta:
                yield delta

    @staticmethod
    def _extract_text(content: str | None) -> str:
        if content is None:
            raise ValueError("LLM returned an empty response")
        return content