"""CLI entry point.  Run as `python -m rag_memory.cli <subcommand>`."""
from __future__ import annotations

import argparse
import json
import sys
import time

from .config import load
from .indexer import Indexer
from .rag import RAG
from .store import Store


def _cmd_index(args: argparse.Namespace) -> int:
    stats = Indexer().index_all(force=args.force)
    print(json.dumps(stats, indent=2))
    return 0


def _cmd_query(args: argparse.Namespace) -> int:
    if args.auto_index:
        Indexer().index_all()
    rag = RAG()
    result = rag.ask(args.question, top_k=args.top_k)
    print(result.answer)
    if result.sources:
        print("\nSources:", ", ".join(result.sources))
    return 0


def _cmd_list(_: argparse.Namespace) -> int:
    store = Store(load())
    rows = store.list_all()
    if not rows:
        print("(no indexed runs)")
        return 0
    width = max(len(r["run_id"]) for r in rows)
    for r in rows:
        meta = r["metadata"] or {}
        tags = " ".join(
            f"{k}={meta[k]}" for k in ("fuel", "oxidizer", "thrust_N", "Isp_s")
            if meta.get(k) is not None
        )
        print(f"{r['run_id']:<{width}}  {meta.get('timestamp','')}  {tags}")
    return 0


def _cmd_delete(args: argparse.Namespace) -> int:
    Store(load()).delete(args.run_id)
    print(f"deleted {args.run_id}")
    return 0


def _cmd_watch(args: argparse.Namespace) -> int:
    indexer = Indexer()
    print(f"watching {indexer.cfg.results_dir} (every {args.interval}s; Ctrl-C to stop)")
    try:
        while True:
            stats = indexer.index_all()
            if stats["updated"] or stats["removed"]:
                print(time.strftime("%H:%M:%S"), json.dumps(stats))
            time.sleep(args.interval)
    except KeyboardInterrupt:
        return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="rag-memory")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_index = sub.add_parser("index", help="Scan results/ and (re)embed any changed runs")
    p_index.add_argument("--force", action="store_true", help="Re-embed every run even if unchanged")
    p_index.set_defaults(func=_cmd_index)

    p_query = sub.add_parser("query", help="Ask a natural-language question over indexed runs")
    p_query.add_argument("question")
    p_query.add_argument("-k", "--top-k", type=int, default=None)
    p_query.add_argument("--no-auto-index", dest="auto_index", action="store_false",
                         help="Skip the pre-query index refresh")
    p_query.set_defaults(func=_cmd_query, auto_index=True)

    p_list = sub.add_parser("list", help="List indexed runs with key metadata")
    p_list.set_defaults(func=_cmd_list)

    p_del = sub.add_parser("delete", help="Remove a run from the index")
    p_del.add_argument("run_id")
    p_del.set_defaults(func=_cmd_delete)

    p_watch = sub.add_parser("watch", help="Poll results/ and index new runs on the fly")
    p_watch.add_argument("--interval", type=float, default=5.0)
    p_watch.set_defaults(func=_cmd_watch)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
