"""Compact supervised dense retrieval for Eedi misconception candidates."""

from __future__ import annotations

import json
import math
import random
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_RETRIEVER_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_MAX_LENGTH = 512
# The BGE model card recommends this prefix for short-query-to-passage search.
DEFAULT_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


@dataclass
class DenseRetriever:
    """A locally reloadable encoder and the metadata needed to reproduce it."""

    encoder: Any
    tokenizer: Any
    model_id: str
    model_revision: str
    max_length: int
    query_instruction: str
    device: str


def _positive_integer(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _validated_catalog(catalog: Mapping[str, str]) -> dict[str, str]:
    if not isinstance(catalog, Mapping):
        raise ValueError("catalog must map misconception IDs to names")
    validated: dict[str, str] = {}
    for misconception_id, name in catalog.items():
        if not isinstance(misconception_id, str) or not misconception_id.strip():
            raise ValueError("catalog IDs must be non-empty strings")
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"catalog name for {misconception_id!r} must be non-empty")
        validated[misconception_id] = name.strip()
    return validated


def rank_top_k(
    scored_candidates: Iterable[tuple[str, float]],
    catalog: Mapping[str, str],
    k: int = 25,
) -> list[dict[str, float | str]]:
    """Return unique catalog IDs ordered by score, then lexicographic ID."""

    if isinstance(k, bool) or not isinstance(k, int) or k < 0:
        raise ValueError("k must be a non-negative integer")
    known_ids = set(_validated_catalog(catalog))
    best_scores: dict[str, float] = {}
    for candidate in scored_candidates:
        try:
            misconception_id, raw_score = candidate
        except (TypeError, ValueError) as error:
            raise ValueError("each scored candidate must be an (ID, score) pair") from error
        if not isinstance(misconception_id, str) or not misconception_id.strip():
            raise ValueError("candidate IDs must be non-empty strings")
        if misconception_id not in known_ids:
            continue
        try:
            score = float(raw_score)
        except (TypeError, ValueError) as error:
            raise ValueError(f"candidate score for {misconception_id!r} is not numeric") from error
        if not math.isfinite(score):
            raise ValueError(f"candidate score for {misconception_id!r} must be finite")
        previous = best_scores.get(misconception_id)
        if previous is None or score > previous:
            best_scores[misconception_id] = score

    ranked = sorted(best_scores.items(), key=lambda item: (-item[1], item[0]))[:k]
    return [
        {"misconception_id": misconception_id, "score": score}
        for misconception_id, score in ranked
    ]


def _training_examples(records: Iterable[Mapping[str, Any]]) -> tuple[list[str], list[str], list[str]]:
    examples = list(records)
    if not examples:
        raise ValueError("at least one labeled training record is required")
    queries: list[str] = []
    labels: list[str] = []
    label_names: dict[str, str] = {}
    for row_number, record in enumerate(examples, start=1):
        if not isinstance(record, Mapping):
            raise ValueError(f"training record {row_number} must be a mapping")
        query = record.get("query")
        misconception_id = record.get("misconception_id")
        misconception_name = record.get("misconception_name")
        if not isinstance(query, str) or not query.strip():
            raise ValueError(f"training record {row_number} has an empty query")
        if not isinstance(misconception_id, str) or not misconception_id.strip():
            raise ValueError(f"training record {row_number} has an invalid misconception_id")
        if not isinstance(misconception_name, str) or not misconception_name.strip():
            raise ValueError(f"training record {row_number} has an invalid misconception_name")
        normalized_id = misconception_id.strip()
        normalized_name = misconception_name.strip()
        existing_name = label_names.setdefault(normalized_id, normalized_name)
        if existing_name != normalized_name:
            raise ValueError(f"misconception {normalized_id!r} has conflicting names")
        queries.append(query.strip())
        labels.append(normalized_id)
    return queries, labels, [label_names[key] for key in label_names]


def _resolve_torch_device(torch: Any, device: str | None) -> Any:
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _format_query(query: str, instruction: str = DEFAULT_QUERY_INSTRUCTION) -> str:
    return f"{instruction}{query}" if instruction else query


def _concept_text(name: str) -> str:
    return f"Misconception: {name}"


def _embed_texts(
    encoder: Any,
    tokenizer: Any,
    texts: Sequence[str],
    *,
    max_length: int,
    device: Any,
    torch: Any,
) -> Any:
    encoded = tokenizer(
        list(texts),
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    encoded = {key: value.to(device) for key, value in encoded.items()}
    output = encoder(**encoded)
    embeddings = output.last_hidden_state[:, 0]
    return torch.nn.functional.normalize(embeddings, p=2, dim=1)


def train_retriever(
    records: Iterable[Mapping[str, Any]],
    output_dir: str | Path,
    model_name: str = DEFAULT_RETRIEVER_MODEL,
    epochs: int = 1,
    batch_size: int = 8,
    learning_rate: float = 2e-5,
    seed: int = 13,
    max_length: int = DEFAULT_MAX_LENGTH,
    device: str | None = None,
) -> DenseRetriever:
    """Fine-tune BGE with a batch multi-positive contrastive objective.

    Misconception IDs are unique columns in each contrastive batch. Every query
    with the same target ID marks that shared column positive, so repeated
    examples of a misconception cannot become false negatives.
    """

    queries, labels, label_names_in_order = _training_examples(records)
    epochs = _positive_integer("epochs", epochs)
    batch_size = _positive_integer("batch_size", batch_size)
    max_length = _positive_integer("max_length", max_length)
    if isinstance(learning_rate, bool) or not isinstance(learning_rate, (int, float)):
        raise ValueError("learning_rate must be positive and finite")
    if learning_rate <= 0 or not math.isfinite(float(learning_rate)):
        raise ValueError("learning_rate must be positive and finite")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    if not isinstance(model_name, str) or not model_name.strip():
        raise ValueError("model_name must be a non-empty string")

    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
    except ImportError as error:
        raise RuntimeError("retriever training requires torch and transformers") from error

    torch.manual_seed(seed)
    target_device = _resolve_torch_device(torch, device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    encoder = AutoModel.from_pretrained(model_name)
    encoder.to(target_device)
    encoder.train()
    optimizer = torch.optim.AdamW(encoder.parameters(), lr=float(learning_rate))
    label_to_name = dict(zip(dict.fromkeys(labels), label_names_in_order))
    all_label_ids = list(label_to_name)
    temperature = 0.05

    for epoch in range(epochs):
        order = list(range(len(queries)))
        random.Random(seed + epoch).shuffle(order)
        for batch_number, start in enumerate(range(0, len(order), batch_size)):
            indices = order[start : start + batch_size]
            query_ids = [labels[index] for index in indices]
            positive_ids = list(dict.fromkeys(query_ids))
            negative_pool = [
                misconception_id
                for misconception_id in all_label_ids
                if misconception_id not in set(positive_ids)
            ]
            if negative_pool:
                negative_count = min(max(1, len(positive_ids)), len(negative_pool))
                negatives = random.Random(seed + epoch * 100_003 + batch_number).sample(
                    negative_pool, negative_count
                )
            else:
                negatives = []
            candidate_ids = positive_ids + negatives
            candidate_names = [label_to_name[misconception_id] for misconception_id in candidate_ids]

            query_embeddings = _embed_texts(
                encoder,
                tokenizer,
                [_format_query(queries[index]) for index in indices],
                max_length=max_length,
                device=target_device,
                torch=torch,
            )
            concept_embeddings = _embed_texts(
                encoder,
                tokenizer,
                [_concept_text(name) for name in candidate_names],
                max_length=max_length,
                device=target_device,
                torch=torch,
            )
            logits = query_embeddings @ concept_embeddings.T / temperature
            positive_mask = torch.tensor(
                [[query_id == candidate_id for candidate_id in candidate_ids] for query_id in query_ids],
                dtype=torch.bool,
                device=target_device,
            )
            positive_logits = logits.masked_fill(~positive_mask, float("-inf"))
            loss = (torch.logsumexp(logits, dim=1) - torch.logsumexp(positive_logits, dim=1)).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

    encoder.eval()
    model_revision = getattr(encoder.config, "_commit_hash", None) or "unresolved-at-load"
    output_path = Path(output_dir)
    encoder_path = output_path / "encoder"
    output_path.mkdir(parents=True, exist_ok=True)
    encoder.save_pretrained(encoder_path)
    tokenizer.save_pretrained(encoder_path)
    metadata = {
        "model_id": model_name,
        "model_revision": model_revision,
        "max_length": max_length,
        "query_instruction": DEFAULT_QUERY_INSTRUCTION,
        "pooling": "normalized_cls",
        "training_objective": "multi_positive_in_batch_contrastive",
        "temperature": temperature,
    }
    (output_path / "retriever_config.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return DenseRetriever(
        encoder=encoder,
        tokenizer=tokenizer,
        model_id=model_name,
        model_revision=model_revision,
        max_length=max_length,
        query_instruction=DEFAULT_QUERY_INSTRUCTION,
        device=str(target_device),
    )


def load_retriever(model_dir: str | Path, device: str | None = None) -> DenseRetriever:
    """Load a previously saved encoder without consulting the model hub."""

    model_path = Path(model_dir)
    config_path = model_path / "retriever_config.json"
    encoder_path = model_path / "encoder"
    if not config_path.is_file() or not encoder_path.is_dir():
        raise ValueError(f"{model_path} is not a saved retriever directory")
    metadata = json.loads(config_path.read_text(encoding="utf-8"))
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
    except ImportError as error:
        raise RuntimeError("loading a retriever requires torch and transformers") from error
    target_device = _resolve_torch_device(torch, device)
    tokenizer = AutoTokenizer.from_pretrained(encoder_path, local_files_only=True)
    encoder = AutoModel.from_pretrained(encoder_path, local_files_only=True)
    encoder.to(target_device)
    encoder.eval()
    return DenseRetriever(
        encoder=encoder,
        tokenizer=tokenizer,
        model_id=metadata["model_id"],
        model_revision=metadata["model_revision"],
        max_length=_positive_integer("max_length", metadata["max_length"]),
        query_instruction=metadata.get("query_instruction", ""),
        device=str(target_device),
    )


def rank_candidates(
    queries: Iterable[Mapping[str, Any]],
    model: DenseRetriever,
    catalog: Mapping[str, str],
    k: int = 25,
    batch_size: int = 32,
) -> list[list[dict[str, float | str]]]:
    """Return per-query top-k catalog IDs and cosine scores."""

    if isinstance(k, bool) or not isinstance(k, int) or k < 0:
        raise ValueError("k must be a non-negative integer")
    batch_size = _positive_integer("batch_size", batch_size)
    validated_catalog = _validated_catalog(catalog)
    rows = list(queries)
    query_texts: list[str] = []
    for row_number, record in enumerate(rows, start=1):
        if not isinstance(record, Mapping):
            raise ValueError(f"query record {row_number} must be a mapping")
        query = record.get("query")
        if not isinstance(query, str) or not query.strip():
            raise ValueError(f"query record {row_number} has an empty query")
        query_texts.append(query.strip())
    if not rows or not validated_catalog or k == 0:
        return [[] for _ in rows]

    try:
        import torch
    except ImportError as error:
        raise RuntimeError("retrieval inference requires torch and transformers") from error

    target_device = torch.device(model.device)
    candidate_ids = sorted(validated_catalog)
    model.encoder.eval()
    with torch.inference_mode():
        candidate_embedding_batches = [
            _embed_texts(
                model.encoder,
                model.tokenizer,
                [
                    _concept_text(validated_catalog[item])
                    for item in candidate_ids[start : start + batch_size]
                ],
                max_length=model.max_length,
                device=target_device,
                torch=torch,
            )
            for start in range(0, len(candidate_ids), batch_size)
        ]
        candidate_embeddings = torch.cat(candidate_embedding_batches, dim=0)
        results: list[list[dict[str, float | str]]] = []
        for start in range(0, len(query_texts), batch_size):
            batch_queries = query_texts[start : start + batch_size]
            query_embeddings = _embed_texts(
                model.encoder,
                model.tokenizer,
                [_format_query(query, model.query_instruction) for query in batch_queries],
                max_length=model.max_length,
                device=target_device,
                torch=torch,
            )
            scores = (query_embeddings @ candidate_embeddings.T).detach().cpu().tolist()
            results.extend(
                rank_top_k(zip(candidate_ids, row_scores), validated_catalog, k=k)
                for row_scores in scores
            )
    return results
