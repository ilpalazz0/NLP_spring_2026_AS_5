from __future__ import annotations

import json
from dataclasses import dataclass
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


@dataclass
class OllamaGenerator:
    model_name: str = "qwen2.5:7b-instruct"
    base_url: str = "http://localhost:11434"
    temperature: float = 0.2
    num_predict: int = 512
    timeout: float = 120.0
    system_prompt: str | None = None

    def generate(self, prompt: str) -> str:
        prompt = (prompt or "").strip()
        if not prompt:
            return ""

        base = self.base_url.rstrip("/")
        url = f"{base}/api/generate"

        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.num_predict,
            },
        }

        if self.system_prompt:
            payload["system"] = self.system_prompt

        data = json.dumps(payload).encode("utf-8")
        request = Request(
            url=url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urlopen(request, timeout=self.timeout) as response:
                raw = response.read().decode("utf-8")
        except HTTPError as exc:
            try:
                detail = exc.read().decode("utf-8", errors="replace")
            except Exception:
                detail = str(exc)
            raise RuntimeError(f"Ollama HTTP error {exc.code}: {detail}") from exc
        except URLError as exc:
            raise RuntimeError(
                "Could not connect to Ollama at "
                f"{self.base_url}. Make sure Ollama is running and the model is pulled. "
                f"Original error: {exc}"
            ) from exc
        except Exception as exc:
            raise RuntimeError(f"Ollama request failed: {exc}") from exc

        try:
            parsed = json.loads(raw)
        except Exception as exc:
            raise RuntimeError(f"Invalid JSON from Ollama: {raw[:500]}") from exc

        text = parsed.get("response", "")
        if not isinstance(text, str):
            text = str(text)

        return text.strip()