"""Observable contract for the end-to-end CSV preparation entry point."""

from __future__ import annotations

import csv
import importlib
import json
import tempfile
import unittest
from pathlib import Path


class PrepareDatasetWorkflowTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.mapping_path = self.root / "misconception_mapping.csv"
        self.train_path = self.root / "train.csv"
        self.test_path = self.root / "test.csv"
        self.output_dir = self.root / "processed"

        with self.mapping_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=["MisconceptionId", "MisconceptionName"])
            writer.writeheader()
            writer.writerows([
                {"MisconceptionId": "10", "MisconceptionName": "adds denominators"},
                {"MisconceptionId": "20", "MisconceptionName": "subtracts the wrong value"},
                {"MisconceptionId": "30", "MisconceptionName": "uses a wrong sign"},
            ])

        fields = [
            "QuestionId", "CorrectAnswer", "QuestionText",
            "AnswerAText", "AnswerBText", "AnswerCText", "AnswerDText",
            "MisconceptionA", "MisconceptionB", "MisconceptionC", "MisconceptionD",
        ]
        train_rows = [
            {
                "QuestionId": "q1", "CorrectAnswer": "A", "QuestionText": "Add one half and one third.",
                "AnswerAText": "5/6", "AnswerBText": "2/5", "AnswerCText": "1/6", "AnswerDText": "1/5",
                "MisconceptionA": "", "MisconceptionB": "10", "MisconceptionC": "20", "MisconceptionD": "30",
            },
            {
                "QuestionId": "q2", "CorrectAnswer": "D", "QuestionText": "Subtract three from eight.",
                "AnswerAText": "11", "AnswerBText": "-5", "AnswerCText": "4", "AnswerDText": "5",
                "MisconceptionA": "10", "MisconceptionB": "20", "MisconceptionC": "30", "MisconceptionD": "",
            },
        ]
        with self.train_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(train_rows)

        test_fields = fields[:7]
        with self.test_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=test_fields)
            writer.writeheader()
            writer.writerow({
                "QuestionId": "q3", "CorrectAnswer": "B", "QuestionText": "Multiply two thirds by three.",
                "AnswerAText": "2/9", "AnswerBText": "2", "AnswerCText": "6", "AnswerDText": "1",
            })

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_preparation_writes_question_grouped_splits_and_unlabeled_test_rows(self) -> None:
        try:
            module = importlib.import_module("scripts.prepare_data")
        except ModuleNotFoundError:
            self.fail("scripts.prepare_data.prepare_dataset is missing")
        prepare_dataset = getattr(module, "prepare_dataset", None)
        self.assertTrue(callable(prepare_dataset), "prepare_dataset is missing")

        prepare_dataset(
            train_path=self.train_path,
            test_path=self.test_path,
            mapping_path=self.mapping_path,
            output_dir=self.output_dir,
            validation_fraction=0.5,
            seed=7,
        )

        def read_jsonl(path: Path) -> list[dict[str, object]]:
            return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

        train = read_jsonl(self.output_dir / "train.jsonl")
        validation = read_jsonl(self.output_dir / "validation.jsonl")
        test = read_jsonl(self.output_dir / "test.jsonl")
        catalog = json.loads((self.output_dir / "catalog.json").read_text(encoding="utf-8"))
        manifest = json.loads((self.output_dir / "manifest.json").read_text(encoding="utf-8"))

        self.assertEqual(6, len(train) + len(validation))
        self.assertEqual(3, len(test))
        self.assertFalse({row["question_id"] for row in train} & {row["question_id"] for row in validation})
        self.assertEqual({"10", "20", "30"}, set(catalog))
        self.assertEqual("adds denominators", catalog["10"])
        self.assertEqual(7, manifest["seed"])
        self.assertEqual(0.5, manifest["validation_fraction"])

        labeled = train + validation
        self.assertEqual({"q1_B", "q1_C", "q1_D", "q2_A", "q2_B", "q2_C"}, {row["query_id"] for row in labeled})
        self.assertEqual({"q3_A", "q3_C", "q3_D"}, {row["query_id"] for row in test})
        q1_b = next(row for row in labeled if row["query_id"] == "q1_B")
        self.assertEqual("5/6", q1_b["correct_answer"])
        self.assertEqual("2/5", q1_b["distractor"])
        self.assertIn("Add one half and one third.", q1_b["query"])
        self.assertIn("5/6", q1_b["query"])
        self.assertIn("2/5", q1_b["query"])
        self.assertEqual("10", q1_b["misconception_id"])
        self.assertTrue(all("misconception_id" not in row for row in test))


if __name__ == "__main__":
    unittest.main()
