"""Hand-checkable ranking metrics for misconception candidate lists."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any


def _validate_k(k: int) -> None:
    if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
        raise ValueError("k must be a positive integer")


def _relevant_set(relevant_ids: Any) -> set[Any]:
    if isinstance(relevant_ids, (str, bytes)):
        values = (relevant_ids,)
    else:
        try:
            values = tuple(relevant_ids)
        except TypeError as error:
            raise ValueError("relevant_ids must be an iterable of IDs or a single ID") from error
    try:
        return set(values)
    except TypeError as error:
        raise ValueError("relevant IDs must be hashable") from error


def _prediction_list(predicted_ids: Any) -> list[Any]:
    if isinstance(predicted_ids, (str, bytes)):
        return [predicted_ids]
    try:
        return list(predicted_ids)
    except TypeError as error:
        raise ValueError("predicted_ids must be an iterable of ranked IDs") from error


def average_precision_at_k(predicted_ids: Iterable[Any], relevant_ids: Any, k: int = 25) -> float:
    """Compute AP@k, counting each relevant ID only at its first rank.

    Duplicate predictions occupy their ranked position but cannot contribute a
    second hit. The score is divided by ``min(k, number of relevant IDs)``.
    """
    _validate_k(k)
    relevant = _relevant_set(relevant_ids)
    if not relevant:
        return 0.0

    hits = 0
    precision_sum = 0.0
    seen_relevant: set[Any] = set()
    for rank, predicted_id in enumerate(_prediction_list(predicted_ids)[:k], start=1):
        try:
            is_relevant = predicted_id in relevant
        except TypeError as error:
            raise ValueError("predicted IDs must be hashable") from error
        if is_relevant and predicted_id not in seen_relevant:
            seen_relevant.add(predicted_id)
            hits += 1
            precision_sum += hits / rank

    return precision_sum / min(k, len(relevant))


def recall_at_k(predicted_ids: Iterable[Any], relevant_ids: Any, k: int = 25) -> float:
    """Compute the fraction of distinct relevant IDs present in the top k."""
    _validate_k(k)
    relevant = _relevant_set(relevant_ids)
    if not relevant:
        return 0.0

    found: set[Any] = set()
    for predicted_id in _prediction_list(predicted_ids)[:k]:
        try:
            if predicted_id in relevant:
                found.add(predicted_id)
        except TypeError as error:
            raise ValueError("predicted IDs must be hashable") from error
    return len(found) / len(relevant)


def _paired_queries(predictions: Iterable[Any], relevant: Iterable[Any]) -> tuple[list[Any], list[Any]]:
    try:
        prediction_rows = list(predictions)
        relevant_rows = list(relevant)
    except TypeError as error:
        raise ValueError("predictions and relevant must be iterables of per-query IDs") from error
    if len(prediction_rows) != len(relevant_rows):
        raise ValueError(
            f"predictions and relevant must contain the same number of queries; "
            f"got {len(prediction_rows)} and {len(relevant_rows)}"
        )
    return prediction_rows, relevant_rows


def mean_average_precision_at_k(predictions: Iterable[Any], relevant: Iterable[Any], k: int = 25) -> float:
    """Average AP@k over paired queries; an empty query set scores zero."""
    _validate_k(k)
    prediction_rows, relevant_rows = _paired_queries(predictions, relevant)
    if not prediction_rows:
        return 0.0
    return sum(
        average_precision_at_k(prediction_row, relevant_row, k=k)
        for prediction_row, relevant_row in zip(prediction_rows, relevant_rows)
    ) / len(prediction_rows)


def mean_recall_at_k(predictions: Iterable[Any], relevant: Iterable[Any], k: int = 25) -> float:
    """Average Recall@k over paired queries; an empty query set scores zero."""
    _validate_k(k)
    prediction_rows, relevant_rows = _paired_queries(predictions, relevant)
    if not prediction_rows:
        return 0.0
    return sum(
        recall_at_k(prediction_row, relevant_row, k=k)
        for prediction_row, relevant_row in zip(prediction_rows, relevant_rows)
    ) / len(prediction_rows)
