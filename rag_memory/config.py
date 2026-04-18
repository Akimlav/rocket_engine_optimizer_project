"""Environment-driven configuration for the RAG memory system."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Config:
    project_root: Path = _PROJECT_ROOT
    results_dir: Path = Path(os.environ.get("RAG_RESULTS_DIR", _PROJECT_ROOT / "results"))
    store_dir: Path = Path(os.environ.get("RAG_STORE_DIR", _PROJECT_ROOT / ".rag_memory"))
    llm_base_url: str = os.environ.get("RAG_LLM_URL", "http://localhost:11434/v1")
    llm_api_key: str = os.environ.get("RAG_LLM_KEY", "ollama")
    llm_model: str = os.environ.get("RAG_LLM_MODEL", "llama3.1:8b")
    embed_model: str = os.environ.get("RAG_EMBED_MODEL", "nomic-embed-text")
    top_k: int = int(os.environ.get("RAG_TOP_K", "5"))
    request_timeout_s: float = float(os.environ.get("RAG_TIMEOUT", "120"))


def load() -> Config:
    cfg = Config()
    cfg.store_dir.mkdir(parents=True, exist_ok=True)
    (cfg.store_dir / "chroma").mkdir(exist_ok=True)
    return cfg
