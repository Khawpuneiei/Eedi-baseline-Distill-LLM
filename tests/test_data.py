import csv
import importlib
import math
import tempfile
import unittest
from pathlib import Path


QUESTION_ROW = {
    "QuestionId": "Q42",
    "CorrectAnswer": "B",
    "QuestionText": "How many are in each group?",
    "AnswerAText": "12",
    "AnswerBText": "6",
    "AnswerCText": "3",
    "AnswerDText": "2",
    "MisconceptionA": "M7",
    "MisconceptionB": "",
    "MisconceptionC": "M8",
    "MisconceptionD": "M7",
}

UNLABELED_QUESTION_ROW = {
    key: value
    for key, value in QUESTION_ROW.items()
    if not key.startswith("Misconception")
}


def _api(name):
    try:
        module = importlib.import_module("eedi_baseline.data")
    except ModuleNotFoundError as error:
        if error.name in {"eedi_baseline", "eedi_baseline.data"}:
            return None
        raise
    return getattr(module, name, None)


class DataPreparationTests(unittest.TestCase):
    def require_api(self, name):
        function = _api(name)
        self.assertTrue(callable(function), f"eedi_baseline.data.{name} must be implemented")
        return function

    def test_training_expansion_emits_each_wrong_answer_with_stable_query_fields(self):
        expand_training_rows = self.require_api("expand_training_rows")
        records = expand_training_rows(
            [QUESTION_ROW],
            {
                "M7": "Divides the total incorrectly",
                "M8": "Uses the wrong operation",
            },
        )

        self.assertEqual(
            records,
            [
                {
                    "query_id": "Q42_A",
                    "question_id": "Q42",
                    "answer_key": "A",
                    "question": "How many are in each group?",
                    "correct_answer": "6",
                    "distractor": "12",
                    "query": "Question: How many are in each group?\nCorrect answer: 6\nDistractor: 12",
                    "misconception_id": "M7",
                    "misconception_name": "Divides the total incorrectly",
                },
                {
                    "query_id": "Q42_C",
                    "question_id": "Q42",
                    "answer_key": "C",
                    "question": "How many are in each group?",
                    "correct_answer": "6",
                    "distractor": "3",
                    "query": "Question: How many are in each group?\nCorrect answer: 6\nDistractor: 3",
                    "misconception_id": "M8",
                    "misconception_name": "Uses the wrong operation",
                },
                {
                    "query_id": "Q42_D",
                    "question_id": "Q42",
                    "answer_key": "D",
                    "question": "How many are in each group?",
                    "correct_answer": "6",
                    "distractor": "2",
                    "query": "Question: How many are in each group?\nCorrect answer: 6\nDistractor: 2",
                    "misconception_id": "M7",
                    "misconception_name": "Divides the total incorrectly",
                },
            ],
        )
        self.assertNotIn("Q42_B", [record["query_id"] for record in records])

    def test_test_expansion_omits_misconception_fields(self):
        expand_test_rows = self.require_api("expand_test_rows")
        records = expand_test_rows([UNLABELED_QUESTION_ROW])

        self.assertEqual([record["query_id"] for record in records], ["Q42_A", "Q42_C", "Q42_D"])
        self.assertEqual(
            set(records[0]),
            {
                "query_id",
                "question_id",
                "answer_key",
                "question",
                "correct_answer",
                "distractor",
                "query",
            },
        )
        self.assertEqual(records[0]["query"], "Question: How many are in each group?\nCorrect answer: 6\nDistractor: 12")

    def test_csv_reader_rejects_missing_required_headers(self):
        read_csv_rows = self.require_api("read_csv_rows")
        fieldnames = [
            "QuestionId",
            "CorrectAnswer",
            "QuestionText",
            "AnswerAText",
            "AnswerBText",
            "AnswerCText",
            "MisconceptionA",
            "MisconceptionB",
            "MisconceptionC",
            "MisconceptionD",
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.csv"
            with path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerow({})

            with self.assertRaisesRegex(ValueError, "AnswerDText"):
                read_csv_rows(path)

    def test_csv_reader_accepts_kaggle_misconception_id_headers(self):
        read_csv_rows = self.require_api("read_csv_rows")
        fieldnames = [
            "QuestionId", "CorrectAnswer", "QuestionText",
            "AnswerAText", "AnswerBText", "AnswerCText", "AnswerDText",
            "MisconceptionAId", "MisconceptionBId", "MisconceptionCId", "MisconceptionDId",
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train.csv"
            with path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerow({name: "" for name in fieldnames} | {"MisconceptionBId": "7"})

            rows = read_csv_rows(path)
        self.assertEqual(rows[0]["MisconceptionB"], "7")
        self.assertNotIn("MisconceptionBId", rows[0])

    def test_training_expansion_rejects_invalid_correct_answer_key(self):
        expand_training_rows = self.require_api("expand_training_rows")
        row = dict(QUESTION_ROW, CorrectAnswer="E")

        with self.assertRaisesRegex(ValueError, "CorrectAnswer"):
            expand_training_rows([row], {"M7": "A misconception", "M8": "Another misconception"})

    def test_training_expansion_skips_unlabeled_wrong_answers(self):
        expand_training_rows = self.require_api("expand_training_rows")
        for missing_label in (None, "", "  "):
            with self.subTest(label=missing_label):
                row = dict(QUESTION_ROW, MisconceptionA=missing_label)
                records = expand_training_rows([row], {"M7": "A misconception", "M8": "Another misconception"})
                self.assertNotIn("Q42_A", [record["query_id"] for record in records])
                self.assertEqual(len(records), 2)

    def test_training_expansion_maps_float_formatted_kaggle_ids(self):
        expand_training_rows = self.require_api("expand_training_rows")
        row = dict(QUESTION_ROW, MisconceptionA="7.0", MisconceptionC="8.00", MisconceptionD="7")
        records = expand_training_rows([row], {"7": "A misconception", "8": "Another misconception"})
        self.assertEqual([record["misconception_id"] for record in records], ["7", "8", "7"])

    def test_training_expansion_rejects_label_on_correct_answer(self):
        expand_training_rows = self.require_api("expand_training_rows")
        row = dict(QUESTION_ROW, MisconceptionB="M7")

        with self.assertRaisesRegex(ValueError, "correct answer"):
            expand_training_rows([row], {"M7": "A misconception", "M8": "Another misconception"})

    def test_training_expansion_rejects_unknown_misconception_id(self):
        expand_training_rows = self.require_api("expand_training_rows")
        row = dict(QUESTION_ROW, MisconceptionC="M999")

        with self.assertRaisesRegex(ValueError, "M999"):
            expand_training_rows([row], {"M7": "A misconception", "M8": "Another misconception"})

    def test_expansion_rejects_a_blank_distractor_instead_of_dropping_it(self):
        expand_test_rows = self.require_api("expand_test_rows")
        row = dict(UNLABELED_QUESTION_ROW, AnswerCText="")

        with self.assertRaisesRegex(ValueError, "AnswerCText"):
            expand_test_rows([row])

    def test_catalog_loader_rejects_duplicate_misconception_ids(self):
        load_misconception_mapping = self.require_api("load_misconception_mapping")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "catalog.csv"
            path.write_text(
                "MisconceptionId,MisconceptionName\nM7,First name\nM7,Second name\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "M7"):
                load_misconception_mapping(path)

    def test_catalog_loader_rejects_bad_headers(self):
        load_misconception_mapping = self.require_api("load_misconception_mapping")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "catalog.csv"
            path.write_text("MisconceptionId,Description\nM7,Name\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "MisconceptionName"):
                load_misconception_mapping(path)

    def test_question_split_is_deterministic_and_keeps_groups_together(self):
        split_by_question = self.require_api("split_by_question")
        records = [
            {"query_id": "Q1_A", "question_id": "Q1"},
            {"query_id": "Q1_C", "question_id": "Q1"},
            {"query_id": "Q2_A", "question_id": "Q2"},
            {"query_id": "Q3_A", "question_id": "Q3"},
            {"query_id": "Q4_A", "question_id": "Q4"},
        ]

        train, validation = split_by_question(records, validation_fraction=0.25, seed=17)
        train_again, validation_again = split_by_question(records, validation_fraction=0.25, seed=17)

        self.assertEqual([row["query_id"] for row in train], [row["query_id"] for row in train_again])
        self.assertEqual([row["query_id"] for row in validation], [row["query_id"] for row in validation_again])
        self.assertFalse({row["question_id"] for row in train} & {row["question_id"] for row in validation})
        self.assertEqual(
            {row["query_id"] for row in train + validation},
            {"Q1_A", "Q1_C", "Q2_A", "Q3_A", "Q4_A"},
        )

    def test_question_split_rejects_invalid_fraction_and_minimum_groups(self):
        split_by_question = self.require_api("split_by_question")
        records = [{"question_id": "Q1"}, {"question_id": "Q2"}]
        for fraction in (0, 1, -0.1, math.nan):
            with self.subTest(fraction=fraction):
                with self.assertRaisesRegex(ValueError, "validation_fraction"):
                    split_by_question(records, validation_fraction=fraction)

        with self.assertRaisesRegex(ValueError, "min_groups"):
            split_by_question(records, min_groups=1)
        with self.assertRaisesRegex(ValueError, "groups"):
            split_by_question(records[:1], min_groups=2)


if __name__ == "__main__":
    unittest.main()
