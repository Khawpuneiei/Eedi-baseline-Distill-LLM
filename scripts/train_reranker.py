"""Train a small cross-encoder on the retriever's candidate pools."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

from eedi_baseline.io import read_jsonl
from eedi_baseline.reranking import (
    DEFAULT_RERANKER_MODEL,
    build_reranker_pairs,
    train_reranker,
)
from eedi_baseline.retrieval import load_retriever, rank_candidates
from scripts.evaluate import load_rationales_for_records


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=Path("data/processed/train.jsonl"))
    parser.add_argument("--catalog", type=Path, default=Path("data/processed/catalog.json"))
    parser.add_argument("--retriever", type=Path, default=Path("models/retriever"))
    parser.add_argument("--output-dir", type=Path, default=Path("models/reranker"))
    parser.add_argument("--rationales", type=Path, help="Optional cache for the rationale ablation")
    parser.add_argument("--model", default=DEFAULT_RERANKER_MODEL)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--top-k", type=int, default=25)
    parser.add_argument("--random-negatives", type=int, default=4)
    parser.add_argument("--device", help="Torch device such as cuda or cpu; defaults to auto")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    records = read_jsonl(args.train)
    catalog = json.loads(args.catalog.read_text(encoding="utf-8"))
    if not isinstance(catalog, dict):
        raise ValueError("catalog must be a JSON object mapping IDs to descriptions")
    retriever = load_retriever(args.retriever, device=args.device)
    candidate_rows = rank_candidates(
        records,
        retriever,
        catalog,
        k=args.top_k,
        batch_size=args.batch_size,
    )
    if len(candidate_rows) != len(records):
        raise ValueError("retriever candidate output does not match training query count")
    candidates_by_query = {
        str(record["query_id"]): [str(item["misconception_id"]) for item in candidates]
        for record, candidates in zip(records, candidate_rows)
    }
    rationales = (
        load_rationales_for_records(args.rationales, records)
        if args.rationales is not None
        else None
    )
    pairs = build_reranker_pairs(
        records,
        candidates_by_query,
        catalog,
        n_random_negatives=args.random_negatives,
        seed=args.seed,
        rationales=rationales,
    )
    reranker = train_reranker(
        pairs,
        args.output_dir,
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_length=args.max_length,
        device=args.device,
    )
    digest = hashlib.sha256(args.train.read_bytes()).hexdigest()
    catalog_hash = hashlib.sha256(args.catalog.read_bytes()).hexdigest()
    rationale_hash = hashlib.sha256(args.rationales.read_bytes()).hexdigest() if args.rationales else None
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
        "model_id": reranker.model_id,
        "model_revision": reranker.model_revision,
        "training_data_sha256": digest,
        "catalog_sha256": catalog_hash,
        "rationale_cache_sha256": rationale_hash,
        "training_record_count": len(records),
        "retrieved_query_count": len(candidates_by_query),
        "training_pair_count": len(pairs),
        "retriever_model_id": retriever.model_id,
        "retriever_model_revision": retriever.model_revision,
        "parameters": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "seed": args.seed,
            "max_length": args.max_length,
            "top_k": args.top_k,
            "random_negatives": args.random_negatives,
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
