from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class LLMResponse:
    content: str
    raw: object | None = None


class LLMClient:
    """Small OpenAI-compatible chat client.

    Credentials and endpoint are read from environment variables. This keeps
    the reproduction code free of hard-coded API keys and model names.
    """

    def __init__(self, api_key: str | None = None, base_url: str | None = None) -> None:
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL")
        self._client = None

    def chat(self, model: str, messages: list[dict[str, str]], temperature: float) -> LLMResponse:
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is not set.")

        try:
            from openai import OpenAI
        except Exception as exc:
            raise RuntimeError("The openai package is required for LLM calls.") from exc

        if self._client is None:
            kwargs = {"api_key": self.api_key}
            if self.base_url:
                kwargs["base_url"] = self.base_url
            self._client = OpenAI(**kwargs)

        response = self._client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
        )
        message = response.choices[0].message
        return LLMResponse(content=message.content or "", raw=response)
