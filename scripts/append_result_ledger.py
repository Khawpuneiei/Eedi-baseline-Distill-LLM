"""Append a dated results entry for a finished queue run to DEVLOG.md."""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--devlog", type=Path, required=True)
    parser.add_argument("--gpu-hours", type=float, required=True)
    parser.add_argument("--teacher", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = json.loads(args.metrics.read_text(encoding="utf-8"))
    k = payload["k"]
    lines = [
        "",
        f"## {dt.date.today().isoformat()} — Scale-B validation run (queued)",
        "",
        f"- Retriever BAAI/bge-small-en-v1.5 (2 epochs, batch 32, max length 256); reranker",
        f"  cross-encoder/ms-marco-MiniLM-L-6-v2 (1 epoch, batch 32); rationale teacher {args.teacher}",
        "  (downscaled from Qwen2.5-7B to fit the 6 h / 8 GiB budget). Split seed 42, 20% grouped validation.",
        f"- GPU wall time for training, rationale generation, and evaluation: {args.gpu_hours} h.",
        "- Full metrics, data hashes, and step logs are in `results/`.",
        "",
        f"| Condition | Queries | MAP@{k} | Recall@{k} |",
        "|---|---:|---:|---:|",
    ]
    for condition, row in payload["results"].items():
        lines.append(
            f"| {condition} | {row['query_count']} | {row['map_at_k']:.4f} | {row['recall_at_k']:.4f} |"
        )
    with args.devlog.open("a", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
