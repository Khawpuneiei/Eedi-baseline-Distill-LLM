"""Small, strict helpers for experiment artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read JSON objects from a JSONL file, naming malformed input lines."""
    source = Path(path)
    rows: list[dict[str, Any]] = []
    with source.open("r", encoding="utf-8") as stream:
        for line_number, raw_line in enumerate(stream, start=1):
            if not raw_line.strip():
                continue
            try:
                value = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON in {source} at line {line_number}: {exc.msg}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"expected a JSON object in {source} at line {line_number}")
            rows.append(value)
    return rows


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    """Write JSON objects one per line while preserving readable Unicode."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            if not isinstance(row, dict):
                raise TypeError("JSONL rows must be dictionaries")
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            stream.write("\n")


def load_rationales(
    path: str | Path,
    expected_query_ids: set[str] | None = None,
) -> dict[str, str]:
    """Load a rationale cache and validate IDs and optional query coverage."""
    source = Path(path)
    rationales: dict[str, str] = {}
    for line_number, row in enumerate(read_jsonl(source), start=1):
        query_id = row.get("query_id")
        rationale = row.get("rationale")
        if not isinstance(query_id, str) or not query_id.strip():
            raise ValueError(f"missing query_id in {source} at line {line_number}")
        if query_id in rationales:
            raise ValueError(f"duplicate query_id {query_id!r} in {source}")
        if not isinstance(rationale, str) or not rationale.strip():
            raise ValueError(f"missing rationale for {query_id!r} in {source}")
        rationales[query_id] = rationale.strip()

    if expected_query_ids is not None:
        actual_ids = set(rationales)
        missing = expected_query_ids - actual_ids
        extra = actual_ids - expected_query_ids
        if missing:
            raise ValueError(f"missing rationale query IDs: {sorted(missing)}")
        if extra:
            raise ValueError(f"unexpected rationale query IDs: {sorted(extra)}")
    return rationales
