"""Client for any OpenAI-compatible local LLM server (Ollama, LMStudio, vLLM, llama.cpp)."""
from __future__ import annotations

import json
import urllib.error
import urllib.request

from .config import Config


class LLMError(RuntimeError):
    pass


class LocalLLM:
    def __init__(self, cfg: Config):
        self._cfg = cfg

    def _post(self, path: str, payload: dict) -> dict:
        url = self._cfg.llm_base_url.rstrip("/") + path
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode(),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._cfg.llm_api_key}",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=self._cfg.request_timeout_s) as resp:
                return json.loads(resp.read())
        except urllib.error.URLError as e:
            raise LLMError(f"cannot reach LLM server at {url}: {e}") from e

    def chat(self, system: str, user: str, temperature: float = 0.2) -> str:
        data = self._post("/chat/completions", {
            "model": self._cfg.llm_model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "temperature": temperature,
        })
        return data["choices"][0]["message"]["content"]

    def embed(self, texts: list[str]) -> list[list[float]]:
        data = self._post("/embeddings", {
            "model": self._cfg.embed_model,
            "input": texts,
        })
        return [row["embedding"] for row in data["data"]]
