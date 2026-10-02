"""Schema validation and long-format conversion for Eedi competition data."""

from __future__ import annotations

import csv
import math
import random
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


ANSWER_KEYS = ("A", "B", "C", "D")
BASE_COLUMNS = (
    "QuestionId",
    "CorrectAnswer",
    "QuestionText",
    "AnswerAText",
    "AnswerBText",
    "AnswerCText",
    "AnswerDText",
)
MISCONCEPTION_COLUMNS = tuple(f"Misconception{key}" for key in ANSWER_KEYS)
KAGGLE_MISCONCEPTION_COLUMNS = frozenset(f"{column}Id" for column in MISCONCEPTION_COLUMNS)
CATALOG_COLUMNS = ("MisconceptionId", "MisconceptionName")


class DataSchemaError(ValueError):
    """Raised when a source file or row does not satisfy its expected schema."""


class DataValidationError(ValueError):
    """Raised when source values violate the Eedi data contract."""


def _check_csv_headers(fieldnames: Sequence[str] | None, required: Sequence[str], source: str) -> list[str]:
    if not fieldnames:
        raise DataSchemaError(f"{source} is missing a CSV header row")
    names = list(fieldnames)
    if any(name is None or not name.strip() for name in names):
        raise DataSchemaError(f"{source} contains an empty CSV column name")
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise DataSchemaError(f"{source} contains duplicate CSV columns: {', '.join(duplicates)}")
    missing = [name for name in required if name not in names]
    if missing:
        raise DataSchemaError(f"{source} is missing required columns: {', '.join(missing)}")
    return names


def read_csv_rows(path: str | Path) -> list[dict[str, str | None]]:
    """Read and validate an Eedi wide CSV, returning its source rows.

    The required misconception columns are treated as a group: they must all be
    present for labeled data or all be absent for unlabeled test data. Extra
    upstream columns are retained on each source row and ignored by expansion.
    """
    source_path = Path(path)
    with source_path.open("r", newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        # The Kaggle files name the label columns MisconceptionAId..MisconceptionDId.
        if reader.fieldnames is not None:
            reader.fieldnames = [
                name[:-2] if name in KAGGLE_MISCONCEPTION_COLUMNS else name for name in reader.fieldnames
            ]
        fieldnames = _check_csv_headers(reader.fieldnames, BASE_COLUMNS, str(source_path))
        label_columns = [column for column in MISCONCEPTION_COLUMNS if column in fieldnames]
        if label_columns and len(label_columns) != len(MISCONCEPTION_COLUMNS):
            missing = [column for column in MISCONCEPTION_COLUMNS if column not in fieldnames]
            raise DataSchemaError(
                f"{source_path} has an incomplete misconception schema; missing columns: {', '.join(missing)}"
            )

        rows: list[dict[str, str | None]] = []
        for line_number, row in enumerate(reader, start=2):
            if None in row:
                raise DataSchemaError(f"{source_path} row {line_number} has more values than its header")
            missing_values = [column for column, value in row.items() if value is None]
            if missing_values:
                raise DataSchemaError(
                    f"{source_path} row {line_number} is missing values for columns: {', '.join(missing_values)}"
                )
            rows.append(dict(row))
    return rows


def load_misconception_mapping(path: str | Path) -> dict[str, str]:
    """Load misconception IDs and names, rejecting malformed and duplicate IDs."""
    source_path = Path(path)
    mapping: dict[str, str] = {}
    with source_path.open("r", newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        _check_csv_headers(reader.fieldnames, CATALOG_COLUMNS, str(source_path))
        for line_number, row in enumerate(reader, start=2):
            if None in row:
                raise DataSchemaError(f"{source_path} row {line_number} has more values than its header")
            misconception_id = row.get("MisconceptionId")
            name = row.get("MisconceptionName")
            if not isinstance(misconception_id, str) or not misconception_id.strip():
                raise DataValidationError(f"{source_path} row {line_number} has an empty MisconceptionId")
            if not isinstance(name, str) or not name.strip():
                raise DataValidationError(f"{source_path} row {line_number} has an empty MisconceptionName")
            if misconception_id in mapping:
                raise DataValidationError(
                    f"{source_path} contains duplicate misconception ID {misconception_id!r}"
                )
            mapping[misconception_id] = name
    return mapping


def _validate_catalog(catalog: Mapping[str, str]) -> None:
    if not isinstance(catalog, Mapping):
        raise DataSchemaError("misconception catalog must be a mapping from string IDs to names")
    for misconception_id, name in catalog.items():
        if not isinstance(misconception_id, str) or not misconception_id.strip():
            raise DataValidationError("misconception catalog IDs must be non-empty strings")
        if not isinstance(name, str) or not name.strip():
            raise DataValidationError(f"misconception catalog entry {misconception_id!r} has an empty name")


def _validate_source_row(row: Mapping[str, Any], index: int, *, labeled: bool) -> tuple[str, str, str, dict[str, str]]:
    if not isinstance(row, Mapping):
        raise DataSchemaError(f"source row {index} must be a mapping of column names to values")

    required_columns = list(BASE_COLUMNS)
    if labeled:
        required_columns.extend(MISCONCEPTION_COLUMNS)
    missing_columns = [column for column in required_columns if column not in row]
    if missing_columns:
        raise DataSchemaError(f"source row {index} is missing required columns: {', '.join(missing_columns)}")

    for column, value in row.items():
        if value is None and column not in MISCONCEPTION_COLUMNS:
            raise DataSchemaError(f"source row {index} has a missing value in column {column}")

    question_id = row["QuestionId"]
    correct_key = row["CorrectAnswer"]
    question = row["QuestionText"]
    if not isinstance(question_id, str) or not question_id.strip():
        raise DataValidationError(f"source row {index} has an empty QuestionId")
    if not isinstance(correct_key, str) or correct_key not in ANSWER_KEYS:
        raise DataValidationError(
            f"question {question_id!r} has invalid CorrectAnswer {correct_key!r}; expected one of A, B, C, D"
        )
    if not isinstance(question, str) or not question.strip():
        raise DataValidationError(f"question {question_id!r} has an empty QuestionText")

    answer_texts: dict[str, str] = {}
    for answer_key in ANSWER_KEYS:
        column = f"Answer{answer_key}Text"
        answer_text = row[column]
        if not isinstance(answer_text, str) or not answer_text.strip():
            raise DataValidationError(f"question {question_id!r} has an empty {column}; no distractor may be dropped")
        answer_texts[answer_key] = answer_text
    return question_id, correct_key, question, answer_texts


def _expand_unlabeled_record(question_id: str, answer_key: str, question: str, correct_answer: str, distractor: str) -> dict[str, str]:
    return {
        "query_id": f"{question_id}_{answer_key}",
        "question_id": question_id,
        "answer_key": answer_key,
        "question": question,
        "correct_answer": correct_answer,
        "distractor": distractor,
        "query": f"Question: {question}\nCorrect answer: {correct_answer}\nDistractor: {distractor}",
    }


def expand_training_rows(rows: Sequence[Mapping[str, Any]], catalog: Mapping[str, str]) -> list[dict[str, str]]:
    """Expand labeled wide rows into one validated example per incorrect answer."""
    _validate_catalog(catalog)
    records: list[dict[str, str]] = []
    seen_question_ids: set[str] = set()
    for index, row in enumerate(rows, start=1):
        question_id, correct_key, question, answer_texts = _validate_source_row(row, index, labeled=True)
        if question_id in seen_question_ids:
            raise DataValidationError(f"duplicate QuestionId {question_id!r} in training rows")
        seen_question_ids.add(question_id)

        correct_label = row[f"Misconception{correct_key}"]
        if correct_label is not None and str(correct_label).strip():
            raise DataValidationError(
                f"question {question_id!r} has a misconception label on its correct answer {correct_key}"
            )

        correct_answer = answer_texts[correct_key]
        for answer_key in ANSWER_KEYS:
            if answer_key == correct_key:
                continue
            query_id = f"{question_id}_{answer_key}"
            label = row[f"Misconception{answer_key}"]
            if not isinstance(label, str) or not label.strip():
                raise DataValidationError(
                    f"question {question_id!r} has an empty misconception label for incorrect answer {answer_key} ({query_id})"
                )
            if label not in catalog:
                raise DataValidationError(
                    f"question {question_id!r} answer {answer_key} ({query_id}) references unknown misconception ID {label!r}"
                )
            record = _expand_unlabeled_record(
                question_id,
                answer_key,
                question,
                correct_answer,
                answer_texts[answer_key],
            )
            record["misconception_id"] = label
            record["misconception_name"] = catalog[label]
            records.append(record)
    return records


def expand_test_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, str]]:
    """Expand unlabeled wide rows into one label-free record per wrong answer."""
    records: list[dict[str, str]] = []
    seen_question_ids: set[str] = set()
    for index, row in enumerate(rows, start=1):
        question_id, correct_key, question, answer_texts = _validate_source_row(row, index, labeled=False)
        if question_id in seen_question_ids:
            raise DataValidationError(f"duplicate QuestionId {question_id!r} in test rows")
        seen_question_ids.add(question_id)

        present_label_columns = [column for column in MISCONCEPTION_COLUMNS if column in row]
        if present_label_columns and len(present_label_columns) != len(MISCONCEPTION_COLUMNS):
            missing = [column for column in MISCONCEPTION_COLUMNS if column not in row]
            raise DataSchemaError(
                f"question {question_id!r} has an incomplete misconception schema; missing columns: {', '.join(missing)}"
            )
        if present_label_columns:
            for column in present_label_columns:
                value = row[column]
                if value is not None and str(value).strip():
                    raise DataValidationError(
                        f"test question {question_id!r} unexpectedly contains a misconception label in {column}"
                    )

        correct_answer = answer_texts[correct_key]
        for answer_key in ANSWER_KEYS:
            if answer_key == correct_key:
                continue
            records.append(
                _expand_unlabeled_record(
                    question_id,
                    answer_key,
                    question,
                    correct_answer,
                    answer_texts[answer_key],
                )
            )
    return records


def split_by_question(
    records: Sequence[Mapping[str, Any]],
    validation_fraction: float = 0.2,
    seed: int = 42,
    min_groups: int = 2,
) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]]]:
    """Split records reproducibly by question ID without splitting a group."""
    if isinstance(validation_fraction, bool) or not isinstance(validation_fraction, (int, float)):
        raise ValueError("validation_fraction must be a finite number strictly between 0 and 1")
    if not math.isfinite(validation_fraction) or not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be a finite number strictly between 0 and 1")
    if isinstance(min_groups, bool) or not isinstance(min_groups, int) or min_groups < 2:
        raise ValueError("min_groups must be an integer of at least 2")

    rows = list(records)
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for index, row in enumerate(rows, start=1):
        if not isinstance(row, Mapping):
            raise DataSchemaError(f"record {index} must be a mapping containing question_id")
        question_id = row.get("question_id")
        if not isinstance(question_id, str) or not question_id.strip():
            raise DataValidationError(f"record {index} has an empty or invalid question_id")
        grouped.setdefault(question_id, []).append(row)

    group_ids = list(grouped)
    if len(group_ids) < min_groups:
        raise ValueError(f"question-grouped split requires at least {min_groups} groups; found {len(group_ids)}")

    shuffled_group_ids = group_ids[:]
    random.Random(seed).shuffle(shuffled_group_ids)
    validation_count = min(len(group_ids) - 1, max(1, math.ceil(len(group_ids) * validation_fraction)))
    validation_groups = set(shuffled_group_ids[:validation_count])

    train_records = [row for row in rows if row["question_id"] not in validation_groups]
    validation_records = [row for row in rows if row["question_id"] in validation_groups]
    return train_records, validation_records
