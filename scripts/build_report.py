"""Regenerate a readable Markdown report from saved evaluation metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from eedi_baseline.reporting import render_markdown_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, default=Path("outputs/validation/metrics.json"))
    parser.add_argument("--output", type=Path, default=Path("outputs/validation/report.md"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload: Any = json.loads(args.metrics.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("results"), dict):
        raise ValueError("metrics file must contain a results object")
    metadata = payload.get("metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError("metrics metadata must be an object")
    k = payload.get("k", 25)
    if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
        raise ValueError("metrics k must be a positive integer")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        render_markdown_report(payload["results"], k=k, metadata=metadata),
        encoding="utf-8",
    )
    print(f"Markdown report written to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
