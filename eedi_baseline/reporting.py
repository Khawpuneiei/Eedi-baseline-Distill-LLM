"""Matched evaluation summaries and a small, provenance-aware Markdown report."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .io import write_jsonl
from .metrics import mean_average_precision_at_k, mean_recall_at_k


def _query_id_set(records: Sequence[Mapping[str, Any]]) -> tuple[list[str], list[str]]:
    query_ids: list[str] = []
    relevant: list[str] = []
    seen: set[str] = set()
    for position, record in enumerate(records):
        query_id = str(record.get("query_id", "")).strip()
        label = str(record.get("misconception_id", "")).strip()
        if not query_id:
            raise ValueError(f"record at position {position} has no query_id")
        if query_id in seen:
            raise ValueError(f"duplicate query_id {query_id!r}")
        if not label:
            raise ValueError(f"record {query_id!r} has no misconception_id")
        seen.add(query_id)
        query_ids.append(query_id)
        relevant.append(label)
    if not query_ids:
        raise ValueError("cannot evaluate an empty query set")
    return query_ids, relevant


def _prediction_ids(ranking: Sequence[Any], query_id: str) -> list[str]:
    if isinstance(ranking, (str, bytes)):
        raise ValueError(f"ranking for query {query_id!r} must be a sequence of IDs")
    result: list[str] = []
    for item in ranking:
        if isinstance(item, Mapping):
            candidate_id = item.get("misconception_id", item.get("candidate_id", item.get("id")))
        else:
            candidate_id = item
        if candidate_id is None or not str(candidate_id).strip():
            raise ValueError(f"ranking for query {query_id!r} contains an empty candidate ID")
        result.append(str(candidate_id))
    return result


def _ranked_candidates(ranking: Sequence[Any], query_id: str) -> list[dict[str, Any]]:
    ids = _prediction_ids(ranking, query_id)
    candidates: list[dict[str, Any]] = []
    for item, candidate_id in zip(ranking, ids):
        candidate = {"candidate_id": candidate_id}
        if isinstance(item, Mapping) and "score" in item:
            score = item["score"]
            if isinstance(score, bool) or not isinstance(score, (int, float)):
                raise ValueError(f"ranking for query {query_id!r} has a non-numeric candidate score")
            candidate["score"] = float(score)
        candidates.append(candidate)
    return candidates


def evaluate_conditions(
    records: Sequence[Mapping[str, Any]],
    rankings_by_condition: Mapping[str, Mapping[str, Sequence[Any]]],
    *,
    k: int = 25,
) -> dict[str, dict[str, float | int]]:
    """Score several matched ranking conditions over the same labeled rows.

    Every condition must contain exactly one ranking for every query. This
    prevents a missing prediction from silently shrinking the evaluation set.
    """
    query_ids, relevant = _query_id_set(records)
    expected = set(query_ids)
    summaries: dict[str, dict[str, float | int]] = {}
    for condition, rankings in rankings_by_condition.items():
        actual = set(rankings)
        missing = expected - actual
        extra = actual - expected
        if missing or extra:
            detail = []
            if missing:
                detail.append(f"missing query rankings: {sorted(missing)}")
            if extra:
                detail.append(f"unexpected query rankings: {sorted(extra)}")
            raise ValueError(f"condition {condition!r} query coverage mismatch ({'; '.join(detail)})")
        ordered_predictions = [
            _prediction_ids(rankings[query_id], query_id)
            for query_id in query_ids
        ]
        summaries[str(condition)] = {
            "query_count": len(query_ids),
            "map_at_k": mean_average_precision_at_k(ordered_predictions, relevant, k=k),
            "recall_at_k": mean_recall_at_k(ordered_predictions, relevant, k=k),
        }
    return summaries


def render_markdown_report(
    results: Mapping[str, Mapping[str, Any]],
    *,
    k: int = 25,
    metadata: Mapping[str, Any] | None = None,
) -> str:
    """Render observed conditions; absent conditions are labeled ``Not run``."""
    display_names = {
        "retriever": "Retrieval baseline",
        "retrieval": "Retrieval baseline",
        "reranker": "Reranker",
        "rationale": "Rationale reranker",
        "rationale_reranker": "Rationale reranker",
    }
    canonical_order = ["retriever", "reranker", "rationale"]
    ordered_keys = [key for key in canonical_order if key in results]
    ordered_keys.extend(key for key in results if key not in ordered_keys)
    if "rationale" not in results and "rationale_reranker" not in results:
        ordered_keys.append("rationale")

    def score(value: Any) -> str:
        if value is None:
            return "Not run"
        try:
            return f"{float(value):.4f}"
        except (TypeError, ValueError):
            return "Not run"

    lines = [
        "# Eedi validation results",
        "",
        f"Primary metric: MAP@{k}. All conditions must use the same question-grouped validation split.",
        "",
        "| Condition | Queries | MAP@" + str(k) + " | Recall@" + str(k) + " |",
        "|---|---:|---:|---:|",
    ]
    for key in ordered_keys:
        values = results.get(key, {})
        name = display_names.get(key, key.replace("_", " ").title())
        count = values.get("query_count", "Not run")
        lines.append(
            f"| {name} | {count} | {score(values.get('map_at_k'))} | {score(values.get('recall_at_k'))} |"
        )
    if metadata:
        lines.extend(["", "## Run metadata", ""])
        for key in sorted(metadata):
            lines.append(f"- **{key}:** {metadata[key]}")
    lines.extend(["", "Scores are generated from saved validation predictions; no score is inferred for a condition that was not run.", ""])
    return "\n".join(lines)


def write_evaluation_artifacts(
    output_dir: str | Path,
    records: Sequence[Mapping[str, Any]],
    rankings_by_condition: Mapping[str, Mapping[str, Sequence[Any]]],
    *,
    k: int = 25,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, dict[str, float | int]]:
    """Validate matched rankings and save metrics, per-query IDs, and Markdown."""
    # Evaluate before touching the destination: incomplete output must not look
    # like a finished run.
    results = evaluate_conditions(records, rankings_by_condition, k=k)
    for condition in rankings_by_condition:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", condition):
            raise ValueError(f"invalid condition name for artifact paths: {condition!r}")

    query_ids, _ = _query_id_set(records)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    saved_metadata = dict(metadata or {})
    metrics_payload = {
        "schema_version": 1,
        "k": k,
        "primary_metric": "map_at_k",
        "results": results,
        "metadata": saved_metadata,
    }
    (output / "metrics.json").write_text(
        json.dumps(metrics_payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    records_by_id = {str(record["query_id"]): record for record in records}
    for condition, rankings in rankings_by_condition.items():
        prediction_rows = []
        for query_id in query_ids:
            source = records_by_id[query_id]
            ranked_candidates = _ranked_candidates(rankings[query_id][:k], query_id)
            prediction_rows.append(
                {
                    "query_id": query_id,
                    "question_id": source.get("question_id"),
                    "answer_key": source.get("answer_key"),
                    "gold_misconception_id": str(source["misconception_id"]),
                    "ranked_misconception_ids": [item["candidate_id"] for item in ranked_candidates],
                    "ranked_candidates": ranked_candidates,
                }
            )
        write_jsonl(output / f"predictions_{condition}.jsonl", prediction_rows)

    (output / "report.md").write_text(
        render_markdown_report(results, k=k, metadata=saved_metadata),
        encoding="utf-8",
    )
    return results
