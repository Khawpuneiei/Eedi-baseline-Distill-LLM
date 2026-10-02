"""Validate competition CSVs and build reproducible long-format splits."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from eedi_baseline.data import (
    expand_test_rows,
    expand_training_rows,
    load_misconception_mapping,
    read_csv_rows,
    split_by_question,
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True))
            stream.write("\n")
            count += 1
    return count


def prepare_dataset(
    *,
    train_path: str | Path,
    test_path: str | Path,
    mapping_path: str | Path,
    output_dir: str | Path,
    validation_fraction: float = 0.2,
    seed: int = 42,
) -> dict[str, Any]:
    """Create question-grouped JSONL splits and a provenance manifest.

    Training rows are expanded into one record per incorrect answer, then
    split by question ID. Test rows are expanded separately and retain no
    target labels.
    """
    train_file = Path(train_path)
    test_file = Path(test_path)
    mapping_file = Path(mapping_path)
    destination = Path(output_dir)

    mapping = load_misconception_mapping(mapping_file)
    labeled = expand_training_rows(read_csv_rows(train_file), mapping)
    train_rows, validation_rows = split_by_question(
        labeled,
        validation_fraction=validation_fraction,
        seed=seed,
    )
    test_rows = expand_test_rows(read_csv_rows(test_file))

    destination.mkdir(parents=True, exist_ok=True)
    counts = {
        "train": _write_jsonl(destination / "train.jsonl", train_rows),
        "validation": _write_jsonl(destination / "validation.jsonl", validation_rows),
        "test": _write_jsonl(destination / "test.jsonl", test_rows),
    }
    catalog_path = destination / "catalog.json"
    catalog_path.write_text(
        json.dumps(dict(sorted(mapping.items())), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "seed": seed,
        "validation_fraction": validation_fraction,
        "source_files": {
            "train.csv": {"sha256": _sha256_file(train_file)},
            "test.csv": {"sha256": _sha256_file(test_file)},
            "misconception_mapping.csv": {"sha256": _sha256_file(mapping_file)},
        },
        "counts": {
            **counts,
            "misconceptions": len(mapping),
            "train_questions": len({str(row["question_id"]) for row in train_rows}),
            "validation_questions": len({str(row["question_id"]) for row in validation_rows}),
        },
        "outputs": [
            "train.jsonl",
            "validation.jsonl",
            "test.jsonl",
            "catalog.json",
        ],
    }
    (destination / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True, help="Competition train.csv")
    parser.add_argument("--test", type=Path, required=True, help="Competition test.csv")
    parser.add_argument("--mapping", type=Path, required=True, help="misconception_mapping.csv")
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = prepare_dataset(
        train_path=args.train,
        test_path=args.test,
        mapping_path=args.mapping,
        output_dir=args.output_dir,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
    )
    print(json.dumps(manifest["counts"], indent=2, sort_keys=True))
    print(f"Prepared dataset written to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
