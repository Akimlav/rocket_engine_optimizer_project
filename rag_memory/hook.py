"""One-liner hook for engine_system.py: auto-index a run after it is saved.

Usage in engine_system.py, right after `print(f"[SAVED] {out_dir}")`:

    try:
        from rag_memory.hook import index_run
        index_run(name)
    except Exception as e:
        print(f"[rag-memory] skipped indexing: {e}")

Silent-by-default: if Chroma or the LLM server is unavailable, the simulation
still succeeds. Indexing is a best-effort side-effect, never a blocker.
"""
from __future__ import annotations


def index_run(run_name: str) -> bool:
    """Index a freshly-saved run. Returns True if the run was re-embedded."""
    from .indexer import Indexer
    return Indexer().index_one(run_name)
