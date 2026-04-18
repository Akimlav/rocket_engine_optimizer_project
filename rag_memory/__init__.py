"""RAG memory for the rocket-engine nozzle optimizer.

Indexes every run in results/ into a local vector store and answers questions
about them using a locally hosted LLM (Ollama, LMStudio, llama.cpp, vLLM — any
OpenAI-compatible endpoint).
"""
from __future__ import annotations

from .config import Config, load
from .hook import index_run
from .indexer import Indexer
from .llm import LocalLLM, LLMError
from .rag import RAG, Answer
from .store import Store

__all__ = [
    "Config",
    "load",
    "Indexer",
    "index_run",
    "LocalLLM",
    "LLMError",
    "RAG",
    "Answer",
    "Store",
]
