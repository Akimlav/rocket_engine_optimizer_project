"""Scan results/ and upsert each run into the vector store.

Idempotent: a manifest maps run_name -> content hash. A run is only re-embedded
when its hash changes, so repeated `rag index` calls are cheap.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .config import Config, load
from .store import Store

_ARTIFACT_EXTS = {".png", ".pdf", ".svg", ".csv", ".dat", ".npy", ".npz"}
_MANIFEST_NAME = "index_manifest.json"


class Indexer:
    def __init__(self, cfg: Config | None = None, store: Store | None = None):
        self.cfg = cfg or load()
        self.store = store or Store(self.cfg)
        self._manifest_path = self.cfg.store_dir / _MANIFEST_NAME
        self._manifest = self._load_manifest()

    def index_all(self, force: bool = False) -> dict[str, int]:
        stats = {"scanned": 0, "updated": 0, "skipped": 0, "removed": 0}
        if not self.cfg.results_dir.is_dir():
            return stats
        seen: set[str] = set()
        for run_dir in sorted(p for p in self.cfg.results_dir.iterdir() if p.is_dir()):
            stats["scanned"] += 1
            seen.add(run_dir.name)
            changed = self.index_one(run_dir.name, force=force)
            stats["updated" if changed else "skipped"] += 1
        for stale in list(self._manifest.keys()):
            if stale not in seen:
                self.store.delete(stale)
                del self._manifest[stale]
                stats["removed"] += 1
        self._save_manifest()
        return stats

    def index_one(self, run_name: str, force: bool = False) -> bool:
        run_dir = self.cfg.results_dir / run_name
        if not run_dir.is_dir():
            return False
        payload = _read_run(run_dir)
        if payload is None:
            return False
        doc, meta = _build_document(run_name, run_dir, payload)
        digest = hashlib.sha1(doc.encode()).hexdigest()
        if not force and self._manifest.get(run_name) == digest:
            return False
        self.store.upsert(run_name, doc, meta)
        self._manifest[run_name] = digest
        self._save_manifest()
        return True

    def _load_manifest(self) -> dict[str, str]:
        if self._manifest_path.exists():
            try:
                return json.loads(self._manifest_path.read_text())
            except json.JSONDecodeError:
                pass
        return {}

    def _save_manifest(self) -> None:
        self._manifest_path.write_text(json.dumps(self._manifest, indent=2, sort_keys=True))


def _read_run(run_dir: Path) -> dict[str, Any] | None:
    results = run_dir / "results.json"
    if not results.exists():
        return None
    try:
        return json.loads(results.read_text())
    except json.JSONDecodeError:
        return None


def _build_document(run_name: str, run_dir: Path, payload: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    inputs = payload.get("inputs") or {}
    performance = payload.get("performance") or payload.get("perf") or {}
    optimization = payload.get("optimization") or payload.get("optimizer") or {}
    timestamp = payload.get("timestamp", "")
    artifacts = sorted(
        p.name for p in run_dir.iterdir()
        if p.is_file() and p.suffix.lower() in _ARTIFACT_EXTS
    )

    lines = [f"Run: {run_name}"]
    if timestamp:
        lines.append(f"Timestamp: {timestamp}")
    if inputs:
        lines.append("Inputs:")
        lines.extend(f"  {k} = {v}" for k, v in inputs.items())
    if performance:
        lines.append("Performance:")
        lines.extend(f"  {k} = {v}" for k, v in performance.items())
    if optimization:
        lines.append("Optimization:")
        lines.extend(f"  {k} = {v}" for k, v in optimization.items())
    if artifacts:
        lines.append(f"Artifacts: {', '.join(artifacts)}")

    log_excerpt = _log_excerpt(run_dir, run_name)
    if log_excerpt:
        lines.append("Log excerpt:")
        lines.append(log_excerpt)

    meta: dict[str, Any] = {
        "run_name": run_name,
        "timestamp": timestamp,
        "fuel": inputs.get("fuel"),
        "oxidizer": inputs.get("oxidizer"),
        "throat_radius": inputs.get("throat_radius"),
        "p_tank_f": inputs.get("p_tank_f"),
        "p_tank_o": inputs.get("p_tank_o"),
        "mode": inputs.get("mode") or payload.get("mode"),
        "thrust_N": performance.get("thrust_N"),
        "Isp_s": performance.get("Isp_s"),
        "exit_mach": performance.get("exit_mach"),
        "area_ratio": performance.get("area_ratio_Ae_At"),
        "artifacts": ",".join(artifacts),
        "path": str(run_dir),
    }
    return "\n".join(lines), meta


def _log_excerpt(run_dir: Path, run_name: str, max_chars: int = 2000) -> str:
    for pattern in (f"log.nozzle_{run_name}", "log.*", "*.log"):
        for log_file in run_dir.glob(pattern):
            try:
                text = log_file.read_text(errors="replace")
            except OSError:
                continue
            if len(text) <= max_chars:
                return text.strip()
            head = text[: max_chars // 2]
            tail = text[-max_chars // 2 :]
            return f"{head}\n...\n{tail}".strip()
    return ""
