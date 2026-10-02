"""Behavior tests for label-free, fixed-pool reranker evaluation pairs."""

from __future__ import annotations

import importlib
import json
import tempfile
import unittest
from pathlib import Path

from eedi_baseline.rationale import build_rationale_cache_record


class EvaluationPairTests(unittest.TestCase):
    def _builder(self):
        try:
            module = importlib.import_module("scripts.evaluate")
        except ModuleNotFoundError:
            self.fail("scripts.evaluate is missing")
        builder = getattr(module, "build_scoring_pairs", None)
        self.assertTrue(callable(builder), "build_scoring_pairs is missing")
        return builder

    def test_pairs_use_only_retrieved_candidates_even_when_gold_was_not_retrieved(self) -> None:
        build_scoring_pairs = self._builder()
        records = [
            {
                "query_id": "q1_B",
                "query": "Question: 2 + 2; correct: 4; distractor: 5",
                "misconception_id": "gold-omitted-by-retrieval",
            }
        ]

        pairs = build_scoring_pairs(
            records,
            {"q1_B": ["a", "b"]},
            {"a": "Adds one twice", "b": "Subtracts the wrong value"},
        )

        self.assertEqual(["a", "b"], [pair["candidate_id"] for pair in pairs])
        self.assertEqual(["Adds one twice", "Subtracts the wrong value"], [pair["candidate_text"] for pair in pairs])
        self.assertTrue(all("misconception_id" not in pair and "label" not in pair for pair in pairs))

    def test_rationale_pairs_keep_the_same_candidates_and_reject_missing_cache_rows(self) -> None:
        build_scoring_pairs = self._builder()
        records = [{"query_id": "q1_B", "query": "Question: 2 + 2; correct: 4; distractor: 5"}]

        baseline = build_scoring_pairs(records, {"q1_B": ["a", "b"]}, {"a": "Adds", "b": "Subtracts"})
        rationale = build_scoring_pairs(
            records,
            {"q1_B": ["a", "b"]},
            {"a": "Adds", "b": "Subtracts"},
            rationales={"q1_B": "Counts one extra item."},
        )

        self.assertEqual([pair["candidate_id"] for pair in baseline], [pair["candidate_id"] for pair in rationale])
        self.assertTrue(all("[RATIONALE]" in pair["query"] for pair in rationale))
        with self.assertRaisesRegex(ValueError, "rationale"):
            build_scoring_pairs(records, {"q1_B": ["a", "b"]}, {"a": "Adds", "b": "Subtracts"}, rationales={})

    def test_rationale_cache_rejects_same_query_ids_when_source_text_changed(self) -> None:
        module = importlib.import_module("scripts.evaluate")
        loader = getattr(module, "load_rationales_for_records", None)
        self.assertTrue(callable(loader), "load_rationales_for_records is missing")
        record = {
            "query_id": "q1_B",
            "question": "What is 2 + 2?",
            "correct_answer": "4",
            "distractor": "5",
            "query": "Question: What is 2 + 2? Correct: 4. Distractor: 5.",
        }
        cache_row = build_rationale_cache_record(record, "Adds an extra unit.", teacher_revision="rev-123")

        with tempfile.TemporaryDirectory() as directory:
            cache_path = Path(directory) / "rationales.jsonl"
            cache_path.write_text(json.dumps(cache_row) + "\n", encoding="utf-8")
            self.assertEqual({"q1_B": "Adds an extra unit."}, loader(cache_path, [record]))

            changed = {**record, "distractor": "6"}
            with self.assertRaisesRegex(ValueError, "input hash"):
                loader(cache_path, [changed])

    def test_rationale_cache_rejects_mixed_teacher_revisions(self) -> None:
        module = importlib.import_module("scripts.evaluate")
        loader = getattr(module, "load_rationales_for_records", None)
        self.assertTrue(callable(loader), "load_rationales_for_records is missing")
        first = {
            "query_id": "q1_B", "question": "What is 2 + 2?", "correct_answer": "4",
            "distractor": "5", "query": "q1",
        }
        second = {
            "query_id": "q2_C", "question": "What is 3 + 3?", "correct_answer": "6",
            "distractor": "7", "query": "q2",
        }
        rows = [
            build_rationale_cache_record(first, "Adds a unit.", teacher_revision="rev-123"),
            build_rationale_cache_record(second, "Adds a unit.", teacher_revision="rev-456"),
        ]

        with tempfile.TemporaryDirectory() as directory:
            cache_path = Path(directory) / "rationales.jsonl"
            cache_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "mixed"):
                loader(cache_path, [first, second])


if __name__ == "__main__":
    unittest.main()
