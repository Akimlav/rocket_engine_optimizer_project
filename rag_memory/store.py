"""Vector store wrapper (ChromaDB, persistent on disk)."""
from __future__ import annotations

from typing import Any

import chromadb

from .config import Config
from .llm import LocalLLM


class Store:
    COLLECTION = "engine_runs"

    def __init__(self, cfg: Config, llm: LocalLLM | None = None):
        self._cfg = cfg
        self._llm = llm or LocalLLM(cfg)
        self._client = chromadb.PersistentClient(path=str(cfg.store_dir / "chroma"))
        self._col = self._client.get_or_create_collection(self.COLLECTION)

    def upsert(self, run_id: str, document: str, metadata: dict[str, Any]) -> None:
        emb = self._llm.embed([document])[0]
        self._col.upsert(
            ids=[run_id],
            documents=[document],
            metadatas=[_sanitize_metadata(metadata)],
            embeddings=[emb],
        )

    def query(self, text: str, top_k: int | None = None) -> list[dict[str, Any]]:
        k = top_k or self._cfg.top_k
        emb = self._llm.embed([text])[0]
        raw = self._col.query(query_embeddings=[emb], n_results=k)
        out: list[dict[str, Any]] = []
        ids = raw.get("ids", [[]])[0]
        for i, rid in enumerate(ids):
            out.append({
                "run_id": rid,
                "document": raw["documents"][0][i],
                "metadata": raw["metadatas"][0][i],
                "distance": (raw.get("distances") or [[None] * len(ids)])[0][i],
            })
        return out

    def list_all(self) -> list[dict[str, Any]]:
        raw = self._col.get()
        return [
            {"run_id": rid, "metadata": m}
            for rid, m in zip(raw.get("ids", []), raw.get("metadatas", []))
        ]

    def delete(self, run_id: str) -> None:
        self._col.delete(ids=[run_id])

    def count(self) -> int:
        return self._col.count()


def _sanitize_metadata(meta: dict[str, Any]) -> dict[str, Any]:
    """Chroma only accepts str/int/float/bool scalars in metadata."""
    clean: dict[str, Any] = {}
    for k, v in meta.items():
        if v is None:
            continue
        if isinstance(v, (str, int, float, bool)):
            clean[k] = v
        else:
            clean[k] = str(v)
    return clean
