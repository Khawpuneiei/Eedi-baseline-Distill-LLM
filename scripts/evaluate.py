"""Score matched retrieval, reranking, and optional rationale conditions."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from eedi_baseline.io import load_rationales, read_jsonl
from eedi_baseline.rationale import build_rationale_prompt
from eedi_baseline.reporting import write_evaluation_artifacts
from eedi_baseline.reranking import (
    compose_reranker_query,
    load_reranker,
    rank_scored_candidates,
    score_reranker,
)
from eedi_baseline.retrieval import load_retriever, rank_candidates


def build_scoring_pairs(
    records: Sequence[Mapping[str, Any]],
    candidate_ids_by_query: Mapping[str, Sequence[str]],
    catalog: Mapping[str, str],
    *,
    rationales: Mapping[str, str] | None = None,
) -> list[dict[str, str]]:
    """Build inference pairs from the fixed retrieved pool without reading labels."""
    rows = list(records)
    query_ids: list[str] = []
    seen: set[str] = set()
    for position, record in enumerate(rows, start=1):
        if not isinstance(record, Mapping):
            raise ValueError(f"query record {position} must be a mapping")
        query_id = record.get("query_id")
        query = record.get("query")
        if not isinstance(query_id, str) or not query_id.strip():
            raise ValueError(f"query record {position} has an invalid query_id")
        if query_id in seen:
            raise ValueError(f"duplicate query_id {query_id!r}")
        if not isinstance(query, str) or not query.strip():
            raise ValueError(f"query record {position} has an empty query")
        query_ids.append(query_id)
        seen.add(query_id)

    expected = set(query_ids)
    actual = set(candidate_ids_by_query)
    missing = expected - actual
    extra = actual - expected
    if missing or extra:
        raise ValueError(
            f"candidate pool query coverage mismatch; missing={sorted(missing)}, extra={sorted(extra)}"
        )
    if rationales is not None:
        rationale_ids = set(rationales)
        missing_rationales = expected - rationale_ids
        extra_rationales = rationale_ids - expected
        if missing_rationales or extra_rationales:
            raise ValueError(
                "rationale query coverage mismatch; "
                f"missing={sorted(missing_rationales)}, extra={sorted(extra_rationales)}"
            )

    pairs: list[dict[str, str]] = []
    for record, query_id in zip(rows, query_ids):
        candidate_ids = candidate_ids_by_query[query_id]
        if isinstance(candidate_ids, (str, bytes)) or not isinstance(candidate_ids, Sequence):
            raise ValueError(f"candidate pool for {query_id!r} must be a sequence of IDs")
        query = str(record["query"])
        rationale: str | None = None
        if rationales is not None:
            rationale = rationales[query_id]
            if not isinstance(rationale, str) or not rationale.strip():
                raise ValueError(f"rationale for {query_id!r} must be non-empty text")
            query = compose_reranker_query(query, rationale)

        seen_candidates: set[str] = set()
        for candidate_id in candidate_ids:
            if not isinstance(candidate_id, str) or candidate_id not in catalog:
                raise ValueError(f"candidate pool for {query_id!r} contains an unknown ID")
            if candidate_id in seen_candidates:
                raise ValueError(f"candidate pool for {query_id!r} contains duplicate ID {candidate_id!r}")
            seen_candidates.add(candidate_id)
            candidate_text = catalog[candidate_id]
            if not isinstance(candidate_text, str) or not candidate_text.strip():
                raise ValueError(f"catalog entry {candidate_id!r} has an empty description")
            pairs.append(
                {
                    "query_id": query_id,
                    "query": query,
                    "candidate_id": candidate_id,
                    "candidate_text": candidate_text,
                }
            )
    return pairs


def load_rationales_for_records(
    path: str | Path,
    records: Sequence[Mapping[str, Any]],
) -> dict[str, str]:
    """Load a complete rationale cache and bind each row to its source text."""
    expected_hashes: dict[str, str] = {}
    for position, record in enumerate(records, start=1):
        if not isinstance(record, Mapping):
            raise ValueError(f"validation record {position} must be a mapping")
        query_id = record.get("query_id")
        if not isinstance(query_id, str) or not query_id.strip():
            raise ValueError(f"validation record {position} has an invalid query_id")
        if query_id in expected_hashes:
            raise ValueError(f"duplicate query_id {query_id!r}")
        prompt = build_rationale_prompt(record)
        expected_hashes[query_id] = hashlib.sha256(prompt.encode("utf-8")).hexdigest()

    cache_path = Path(path)
    cache_rows = read_jsonl(cache_path)
    rationales = load_rationales(cache_path, expected_query_ids=set(expected_hashes))
    cache_by_id = {str(row["query_id"]): row for row in cache_rows}
    for row in cache_rows:
        for field in ("teacher_model", "teacher_revision", "prompt_version"):
            value = row.get(field)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"rationale cache entry is missing {field}")
    provenance = {
        (
            row.get("teacher_model"),
            row.get("teacher_revision"),
            row.get("prompt_version"),
        )
        for row in cache_rows
    }
    if len(provenance) != 1:
        raise ValueError("rationale cache contains mixed teacher or prompt provenance")
    for query_id, expected_hash in expected_hashes.items():
        row = cache_by_id[query_id]
        if row.get("input_sha256") != expected_hash:
            raise ValueError(f"rationale cache input hash mismatch for {query_id!r}")
    return rationales


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_catalog(path: Path) -> dict[str, str]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"catalog file {path} is not valid JSON") from error
    if not isinstance(value, dict) or any(
        not isinstance(key, str) or not isinstance(name, str) or not name.strip()
        for key, name in value.items()
    ):
        raise ValueError(f"catalog file {path} must map string IDs to non-empty descriptions")
    return value


def _rerank_pairs(
    records: Sequence[Mapping[str, Any]],
    candidate_ids_by_query: Mapping[str, Sequence[str]],
    catalog: Mapping[str, str],
    model: Any,
    *,
    k: int,
    batch_size: int,
    rationales: Mapping[str, str] | None = None,
) -> dict[str, list[dict[str, float | str]]]:
    pairs = build_scoring_pairs(
        records,
        candidate_ids_by_query,
        catalog,
        rationales=rationales,
    )
    scores = score_reranker(model, pairs, batch_size=batch_size)
    if len(scores) != len(pairs):
        raise ValueError(f"reranker returned {len(scores)} scores for {len(pairs)} candidate pairs")
    scores_by_query: dict[str, list[tuple[str, float]]] = {
        str(record["query_id"]): [] for record in records
    }
    for pair, score in zip(pairs, scores):
        scores_by_query[pair["query_id"]].append((pair["candidate_id"], float(score)))

    return {
        query_id: rank_scored_candidates(
            [candidate_id for candidate_id, _ in query_scores],
            [score for _, score in query_scores],
            k=k,
        )
        for query_id, query_scores in scores_by_query.items()
    }


def run_evaluation(
    *,
    validation_path: str | Path,
    catalog_path: str | Path,
    retriever_dir: str | Path,
    reranker_dir: str | Path,
    output_dir: str | Path,
    rationale_reranker_dir: str | Path | None = None,
    rationales_path: str | Path | None = None,
    k: int = 25,
    batch_size: int = 32,
    device: str | None = None,
) -> dict[str, dict[str, float | int]]:
    """Run matched validation conditions and persist metrics and predictions."""
    if (rationale_reranker_dir is None) != (rationales_path is None):
        raise ValueError("--rationale-reranker and --rationales must be supplied together")
    validation_file = Path(validation_path)
    catalog_file = Path(catalog_path)
    records = read_jsonl(validation_file)
    if not records:
        raise ValueError("validation data is empty")
    catalog = _load_catalog(catalog_file)
    query_ids = [str(record.get("query_id", "")) for record in records]
    if len(set(query_ids)) != len(query_ids) or any(not query_id for query_id in query_ids):
        raise ValueError("validation data has missing or duplicate query IDs")

    retriever = load_retriever(retriever_dir, device=device)
    reranker = load_reranker(reranker_dir, device=device)
    retrieved = rank_candidates(records, retriever, catalog, k=k, batch_size=batch_size)
    if len(retrieved) != len(records):
        raise ValueError(f"retriever returned {len(retrieved)} pools for {len(records)} validation queries")
    candidates_by_query: dict[str, list[str]] = {}
    retrieval_rankings: dict[str, list[dict[str, float | str]]] = {}
    for record, candidates in zip(records, retrieved):
        query_id = str(record["query_id"])
        candidates_by_query[query_id] = [str(item["misconception_id"]) for item in candidates]
        retrieval_rankings[query_id] = candidates

    rankings_by_condition: dict[str, dict[str, Sequence[Any]]] = {
        "retriever": retrieval_rankings,
        "reranker": _rerank_pairs(
            records,
            candidates_by_query,
            catalog,
            reranker,
            k=k,
            batch_size=batch_size,
        ),
    }

    rationale_model_id: str | None = None
    rationale_model_revision: str | None = None
    rationale_cache_hash: str | None = None
    rationale_provenance: dict[str, str] | None = None
    if rationale_reranker_dir is not None and rationales_path is not None:
        rationale_file = Path(rationales_path)
        rationales = load_rationales_for_records(rationale_file, records)
        cache_record = read_jsonl(rationale_file)[0]
        rationale_provenance = {
            "teacher_model": str(cache_record["teacher_model"]),
            "teacher_revision": str(cache_record["teacher_revision"]),
            "prompt_version": str(cache_record["prompt_version"]),
        }
        rationale_reranker = load_reranker(rationale_reranker_dir, device=device)
        rankings_by_condition["rationale"] = _rerank_pairs(
            records,
            candidates_by_query,
            catalog,
            rationale_reranker,
            k=k,
            batch_size=batch_size,
            rationales=rationales,
        )
        rationale_model_id = str(getattr(rationale_reranker, "model_id", rationale_reranker_dir))
        rationale_model_revision = str(getattr(rationale_reranker, "model_revision", "unknown"))
        rationale_cache_hash = _sha256_file(rationale_file)

    try:
        import torch

        torch_version = torch.__version__
        cuda_available = bool(torch.cuda.is_available())
        gpu_name = torch.cuda.get_device_name(0) if cuda_available else None
    except ImportError:
        torch_version = "unavailable"
        cuda_available = False
        gpu_name = None
    try:
        import transformers

        transformers_version = transformers.__version__
    except ImportError:
        transformers_version = "unavailable"

    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "validation_sha256": _sha256_file(validation_file),
        "catalog_sha256": _sha256_file(catalog_file),
        "validation_query_count": len(records),
        "k": k,
        "batch_size": batch_size,
        "python": platform.python_version(),
        "torch": torch_version,
        "transformers": transformers_version,
        "cuda_available": cuda_available,
        "gpu_name": gpu_name,
        "device_requested": device or "auto",
        "retriever_model_id": str(getattr(retriever, "model_id", retriever_dir)),
        "retriever_model_revision": str(getattr(retriever, "model_revision", "unknown")),
        "reranker_model_id": str(getattr(reranker, "model_id", reranker_dir)),
        "reranker_model_revision": str(getattr(reranker, "model_revision", "unknown")),
        "rationale_reranker_model_id": rationale_model_id,
        "rationale_reranker_model_revision": rationale_model_revision,
        "rationale_provenance": rationale_provenance,
        "rationale_cache_sha256": rationale_cache_hash,
    }
    return write_evaluation_artifacts(
        output_dir,
        records,
        rankings_by_condition,
        k=k,
        metadata=metadata,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation", type=Path, default=Path("data/processed/validation.jsonl"))
    parser.add_argument("--catalog", type=Path, default=Path("data/processed/catalog.json"))
    parser.add_argument("--retriever", type=Path, default=Path("models/retriever"))
    parser.add_argument("--reranker", type=Path, default=Path("models/reranker"))
    parser.add_argument("--rationale-reranker", type=Path)
    parser.add_argument("--rationales", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/validation"))
    parser.add_argument("--k", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", help="Torch device such as cuda or cpu; defaults to auto")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    results = run_evaluation(
        validation_path=args.validation,
        catalog_path=args.catalog,
        retriever_dir=args.retriever,
        reranker_dir=args.reranker,
        rationale_reranker_dir=args.rationale_reranker,
        rationales_path=args.rationales,
        output_dir=args.output_dir,
        k=args.k,
        batch_size=args.batch_size,
        device=args.device,
    )
    print(json.dumps(results, indent=2, sort_keys=True))
    print(f"Evaluation artifacts written to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
