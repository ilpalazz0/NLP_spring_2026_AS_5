from __future__ import annotations

from openai import OpenAI

from rag_system.llm.base import BaseGenerator


class OpenAIGenerator(BaseGenerator):
    def __init__(self, model_name: str, api_key: str, base_url: str | None = None) -> None:
        self.model_name = model_name
        client_kwargs = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url
        self.client = OpenAI(**client_kwargs)

    def generate(self, prompt: str) -> str:
        response = self.client.responses.create(
            model=self.model_name,
            input=prompt,
        )
        return response.output_text.strip()
