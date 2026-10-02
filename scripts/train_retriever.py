"""Fine-tune and save the dense misconception retriever."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

from eedi_baseline.io import read_jsonl
from eedi_baseline.retrieval import DEFAULT_RETRIEVER_MODEL, train_retriever


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=Path("data/processed/train.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("models/retriever"))
    parser.add_argument("--model", default=DEFAULT_RETRIEVER_MODEL)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--device", help="Torch device such as cuda or cpu; defaults to auto")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    records = read_jsonl(args.train)
    model = train_retriever(
        records,
        args.output_dir,
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
        max_length=args.max_length,
        device=args.device,
    )
    digest = hashlib.sha256(args.train.read_bytes()).hexdigest()
    try:
        import torch

        runtime = {
            "torch": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        }
    except ImportError:
        runtime = {"torch": "unavailable", "cuda_available": False, "gpu_name": None}
    run = {
        "schema_version": 1,
        "model_id": model.model_id,
        "model_revision": model.model_revision,
        "training_data_sha256": digest,
        "training_record_count": len(records),
        "parameters": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "seed": args.seed,
            "max_length": args.max_length,
            "device_requested": args.device or "auto",
        },
        "runtime": {"python": platform.python_version(), **runtime},
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "training_run.json").write_text(
        json.dumps(run, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(run, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
