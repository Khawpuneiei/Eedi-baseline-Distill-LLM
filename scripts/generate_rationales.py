"""Generate label-blind rationale-cache rows for a prepared query split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from eedi_baseline.io import read_jsonl
from eedi_baseline.rationale import DEFAULT_TEACHER_MODEL, generate_rationales


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Prepared train, validation, or test JSONL")
    parser.add_argument("--output", type=Path, required=True, help="Rationale cache JSONL path")
    parser.add_argument("--model", default=DEFAULT_TEACHER_MODEL)
    parser.add_argument("--revision", help="Optional immutable Hugging Face revision")
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--device", help="Torch device such as cuda or cpu; defaults to auto")
    parser.add_argument("--batch-size", type=int, default=1, help="Prompts per left-padded generate call")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    records = read_jsonl(args.input)
    cache = generate_rationales(
        records,
        args.output,
        model_name=args.model,
        revision=args.revision,
        max_new_tokens=args.max_new_tokens,
        device=args.device,
        batch_size=args.batch_size,
        progress=True,
    )
    print(json.dumps({"cached_rationales": len(cache), "output": str(args.output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
