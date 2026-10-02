"""Compact binary cross-encoder training and scoring utilities."""

from __future__ import annotations

import json
import math
import random
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
from typing import Any


DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
DEFAULT_MAX_LENGTH = 256


@dataclass
class CrossEncoderReranker:
    """A locally saved sequence classifier and its tokenizer settings."""

    model: Any
    tokenizer: Any
    model_id: str
    model_revision: str
    max_length: int
    device: str


def _positive_integer(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _validated_catalog(catalog: Mapping[str, str]) -> dict[str, str]:
    if not isinstance(catalog, Mapping):
        raise ValueError("catalog must map misconception IDs to names")
    validated: dict[str, str] = {}
    for candidate_id, name in catalog.items():
        if not isinstance(candidate_id, str) or not candidate_id.strip():
            raise ValueError("catalog IDs must be non-empty strings")
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"catalog name for {candidate_id!r} must be non-empty")
        validated[candidate_id] = name.strip()
    return validated


def compose_reranker_query(query: str, rationale: str | None = None) -> str:
    """Append a clearly delimited rationale while preserving control inputs."""

    if not isinstance(query, str):
        raise ValueError("query must be a string")
    if rationale is None or not isinstance(rationale, str) or not rationale.strip():
        return query
    return f"{query}\n\n[RATIONALE]\n{rationale.strip()}\n[/RATIONALE]"


def rank_scored_candidates(
    candidate_ids: Sequence[str], scores: Sequence[float], k: int | None = None
) -> list[dict[str, float | str]]:
    """Order and deduplicate only the candidates passed to the reranker."""

    if len(candidate_ids) != len(scores):
        raise ValueError("candidate_ids and scores must have the same length")
    if k is not None and (isinstance(k, bool) or not isinstance(k, int) or k < 0):
        raise ValueError("k must be a non-negative integer or None")
    best_scores: dict[str, float] = {}
    for candidate_id, raw_score in zip(candidate_ids, scores):
        if not isinstance(candidate_id, str) or not candidate_id.strip():
            raise ValueError("candidate IDs must be non-empty strings")
        try:
            score = float(raw_score)
        except (TypeError, ValueError) as error:
            raise ValueError(f"candidate score for {candidate_id!r} is not numeric") from error
        if not math.isfinite(score):
            raise ValueError(f"candidate score for {candidate_id!r} must be finite")
        previous = best_scores.get(candidate_id)
        if previous is None or score > previous:
            best_scores[candidate_id] = score
    ranked = sorted(best_scores.items(), key=lambda item: (-item[1], item[0]))
    if k is not None:
        ranked = ranked[:k]
    return [{"candidate_id": candidate_id, "score": score} for candidate_id, score in ranked]


def build_reranker_pairs(
    records: Iterable[Mapping[str, Any]],
    candidate_ids_by_query: Mapping[str, Sequence[str]],
    catalog: Mapping[str, str],
    n_random_negatives: int = 4,
    seed: int = 13,
    rationales: Mapping[str, str] | None = None,
) -> list[dict[str, str | int]]:
    """Build one gold positive and unique retrieved/random negatives per query."""

    if not isinstance(candidate_ids_by_query, Mapping):
        raise ValueError("candidate_ids_by_query must map query IDs to candidate IDs")
    if isinstance(n_random_negatives, bool) or not isinstance(n_random_negatives, int):
        raise ValueError("n_random_negatives must be a non-negative integer")
    if n_random_negatives < 0:
        raise ValueError("n_random_negatives must be a non-negative integer")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    if rationales is not None and not isinstance(rationales, Mapping):
        raise ValueError("rationales must map query IDs to rationale text")
    validated_catalog = _validated_catalog(catalog)
    rows = list(records)
    pairs: list[dict[str, str | int]] = []
    seen_query_ids: set[str] = set()

    for row_number, record in enumerate(rows, start=1):
        if not isinstance(record, Mapping):
            raise ValueError(f"training record {row_number} must be a mapping")
        query_id = record.get("query_id")
        query = record.get("query")
        gold_id = record.get("misconception_id")
        if not isinstance(query_id, str) or not query_id.strip():
            raise ValueError(f"training record {row_number} has an invalid query_id")
        if query_id in seen_query_ids:
            raise ValueError(f"training query_id {query_id!r} occurs more than once")
        seen_query_ids.add(query_id)
        if not isinstance(query, str) or not query.strip():
            raise ValueError(f"training record {row_number} has an empty query")
        if not isinstance(gold_id, str) or gold_id not in validated_catalog:
            raise ValueError(f"training record {row_number} has an unknown misconception_id")

        rationale: str | None = None
        if rationales is not None:
            rationale = rationales.get(query_id)
            if not isinstance(rationale, str) or not rationale.strip():
                raise ValueError(f"rationales has no non-empty entry for {query_id!r}")
        composed_query = compose_reranker_query(query, rationale)
        retrieved_ids = candidate_ids_by_query.get(query_id, ())
        if isinstance(retrieved_ids, (str, bytes)) or not isinstance(retrieved_ids, Sequence):
            raise ValueError(f"candidates for {query_id!r} must be a sequence of IDs")
        unique_retrieved: list[str] = []
        for candidate_id in retrieved_ids:
            if not isinstance(candidate_id, str) or candidate_id not in validated_catalog:
                raise ValueError(f"query {query_id!r} contains an unknown candidate ID")
            if candidate_id not in unique_retrieved:
                unique_retrieved.append(candidate_id)

        negative_ids = [candidate_id for candidate_id in unique_retrieved if candidate_id != gold_id]
        random_pool = sorted(
            candidate_id
            for candidate_id in validated_catalog
            if candidate_id != gold_id and candidate_id not in negative_ids
        )
        random_count = min(n_random_negatives, len(random_pool))
        random_negatives = random.Random(seed + row_number).sample(random_pool, random_count)
        candidate_labels = [(gold_id, 1)]
        candidate_labels.extend((candidate_id, 0) for candidate_id in negative_ids)
        candidate_labels.extend((candidate_id, 0) for candidate_id in random_negatives)

        for candidate_id, label in candidate_labels:
            pairs.append(
                {
                    "query_id": query_id,
                    "query": composed_query,
                    "candidate_id": candidate_id,
                    "candidate_text": validated_catalog[candidate_id],
                    "label": label,
                }
            )
    return pairs


def _validated_training_pairs(
    pairs: Iterable[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[str], list[str], list[int]]:
    rows = list(pairs)
    if not rows:
        raise ValueError("at least one reranker training pair is required")
    normalized: list[dict[str, Any]] = []
    queries: list[str] = []
    candidates: list[str] = []
    labels: list[int] = []
    for row_number, pair in enumerate(rows, start=1):
        if not isinstance(pair, Mapping):
            raise ValueError(f"training pair {row_number} must be a mapping")
        query = pair.get("query")
        candidate_text = pair.get("candidate_text")
        label = pair.get("label")
        rationale = pair.get("rationale")
        if not isinstance(query, str) or not query.strip():
            raise ValueError(f"training pair {row_number} has an empty query")
        if not isinstance(candidate_text, str) or not candidate_text.strip():
            raise ValueError(f"training pair {row_number} has empty candidate_text")
        if isinstance(label, bool) or not isinstance(label, Integral) or label not in (0, 1):
            raise ValueError(f"training pair {row_number} label must be 0 or 1")
        if rationale is not None and not isinstance(rationale, str):
            raise ValueError(f"training pair {row_number} rationale must be a string")
        composed_query = compose_reranker_query(query, rationale)
        normalized.append({**pair, "query": composed_query, "label": int(label)})
        queries.append(composed_query)
        candidates.append(candidate_text.strip())
        labels.append(int(label))
    if set(labels) != {0, 1}:
        raise ValueError("reranker training requires both positive and negative pairs")
    return normalized, queries, candidates, labels


def _resolve_torch_device(torch: Any, device: str | None) -> Any:
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def train_reranker(
    pairs: Iterable[Mapping[str, Any]],
    output_dir: str | Path,
    model_name: str = DEFAULT_RERANKER_MODEL,
    epochs: int = 1,
    batch_size: int = 8,
    learning_rate: float = 2e-5,
    max_length: int = DEFAULT_MAX_LENGTH,
    device: str | None = None,
) -> CrossEncoderReranker:
    """Fine-tune a one-logit cross-encoder with binary relevance labels."""

    normalized_pairs, queries, candidates, labels = _validated_training_pairs(pairs)
    epochs = _positive_integer("epochs", epochs)
    batch_size = _positive_integer("batch_size", batch_size)
    max_length = _positive_integer("max_length", max_length)
    if isinstance(learning_rate, bool) or not isinstance(learning_rate, (int, float)):
        raise ValueError("learning_rate must be positive and finite")
    if learning_rate <= 0 or not math.isfinite(float(learning_rate)):
        raise ValueError("learning_rate must be positive and finite")
    if not isinstance(model_name, str) or not model_name.strip():
        raise ValueError("model_name must be a non-empty string")

    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as error:
        raise RuntimeError("reranker training requires torch and transformers") from error

    torch.manual_seed(13)
    target_device = _resolve_torch_device(torch, device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)
    model.to(target_device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(learning_rate))
    binary_loss = torch.nn.BCEWithLogitsLoss()

    for epoch in range(epochs):
        order = list(range(len(normalized_pairs)))
        random.Random(13 + epoch).shuffle(order)
        for start in range(0, len(order), batch_size):
            indices = order[start : start + batch_size]
            encoded = tokenizer(
                [queries[index] for index in indices],
                [candidates[index] for index in indices],
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(target_device) for key, value in encoded.items()}
            logits = model(**encoded).logits.reshape(-1).float()
            target_labels = torch.tensor(
                [labels[index] for index in indices], dtype=torch.float32, device=target_device
            )
            loss = binary_loss(logits, target_labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

    model.eval()
    model_revision = getattr(model.config, "_commit_hash", None) or "unresolved-at-load"
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    metadata = {
        "model_id": model_name,
        "model_revision": model_revision,
        "max_length": max_length,
        "num_labels": 1,
        "objective": "binary_relevance_bce_with_logits",
    }
    (output_path / "reranker_config.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return CrossEncoderReranker(
        model=model,
        tokenizer=tokenizer,
        model_id=model_name,
        model_revision=model_revision,
        max_length=max_length,
        device=str(target_device),
    )


def load_reranker(model_dir: str | Path, device: str | None = None) -> CrossEncoderReranker:
    """Load a saved cross-encoder from local files only."""

    model_path = Path(model_dir)
    config_path = model_path / "reranker_config.json"
    if not config_path.is_file():
        raise ValueError(f"{model_path} is not a saved reranker directory")
    metadata = json.loads(config_path.read_text(encoding="utf-8"))
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as error:
        raise RuntimeError("loading a reranker requires torch and transformers") from error
    target_device = _resolve_torch_device(torch, device)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_path, local_files_only=True)
    model.to(target_device)
    model.eval()
    return CrossEncoderReranker(
        model=model,
        tokenizer=tokenizer,
        model_id=metadata["model_id"],
        model_revision=metadata["model_revision"],
        max_length=_positive_integer("max_length", metadata["max_length"]),
        device=str(target_device),
    )


def score_reranker(
    model: CrossEncoderReranker,
    pairs: Iterable[Mapping[str, Any]],
    batch_size: int = 32,
) -> list[float]:
    """Return aligned relevance logits for only the supplied query-candidate pairs."""

    batch_size = _positive_integer("batch_size", batch_size)
    rows = list(pairs)
    normalized_queries: list[str] = []
    candidate_texts: list[str] = []
    for row_number, pair in enumerate(rows, start=1):
        if not isinstance(pair, Mapping):
            raise ValueError(f"reranker pair {row_number} must be a mapping")
        query = pair.get("query")
        candidate_text = pair.get("candidate_text")
        if not isinstance(query, str) or not query.strip():
            raise ValueError(f"reranker pair {row_number} has an empty query")
        if not isinstance(candidate_text, str) or not candidate_text.strip():
            raise ValueError(f"reranker pair {row_number} has empty candidate_text")
        rationale = pair.get("rationale")
        normalized_queries.append(compose_reranker_query(query, rationale))
        candidate_texts.append(candidate_text.strip())
    if not rows:
        return []

    try:
        import torch
    except ImportError as error:
        raise RuntimeError("reranker scoring requires torch and transformers") from error
    target_device = torch.device(model.device)
    model.model.eval()
    scores: list[float] = []
    with torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            batch_queries = normalized_queries[start : start + batch_size]
            batch_candidates = candidate_texts[start : start + batch_size]
            encoded = model.tokenizer(
                batch_queries,
                batch_candidates,
                padding=True,
                truncation=True,
                max_length=model.max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(target_device) for key, value in encoded.items()}
            logits = model.model(**encoded).logits.reshape(-1).float().detach().cpu().tolist()
            scores.extend(float(score) for score in logits)
    return scores
