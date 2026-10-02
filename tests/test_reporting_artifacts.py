"""Behavior tests for the saved evaluation artifact boundary."""

from __future__ import annotations

import importlib
import json
import tempfile
import unittest
from pathlib import Path


class EvaluationArtifactTests(unittest.TestCase):
    def _api(self):
        module = importlib.import_module("eedi_baseline.reporting")
        writer = getattr(module, "write_evaluation_artifacts", None)
        self.assertTrue(callable(writer), "write_evaluation_artifacts is missing")
        return writer

    def test_writer_persists_hand_checked_metrics_predictions_and_not_run_condition(self) -> None:
        write_evaluation_artifacts = self._api()
        records = [
            {"query_id": "q1_B", "question_id": "q1", "answer_key": "B", "misconception_id": "10"},
            {"query_id": "q2_A", "question_id": "q2", "answer_key": "A", "misconception_id": "20"},
        ]
        rankings = {
            "retriever": {"q1_B": ["10", "30"], "q2_A": ["30", "20"]},
            "reranker": {"q1_B": ["30", "10"], "q2_A": ["20", "30"]},
        }

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            metrics = write_evaluation_artifacts(
                output,
                records,
                rankings,
                k=2,
                metadata={"seed": 42},
            )
            saved = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
            predicted = [
                json.loads(line)
                for line in (output / "predictions_retriever.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            report = (output / "report.md").read_text(encoding="utf-8")

        self.assertEqual(0.75, metrics["retriever"]["map_at_k"])
        self.assertEqual(0.75, saved["results"]["retriever"]["map_at_k"])
        self.assertEqual(42, saved["metadata"]["seed"])
        self.assertEqual(["10", "30"], predicted[0]["ranked_misconception_ids"])
        self.assertEqual(["q1_B", "q2_A"], [row["query_id"] for row in predicted])
        self.assertIn("Not run", report)

    def test_writer_rejects_incomplete_rankings_before_creating_artifacts(self) -> None:
        write_evaluation_artifacts = self._api()
        records = [{"query_id": "q1_B", "misconception_id": "10"}]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "results"
            with self.assertRaisesRegex(ValueError, "coverage mismatch"):
                write_evaluation_artifacts(output, records, {"retriever": {}}, k=25)
            self.assertFalse(output.exists())

    def test_prediction_artifact_keeps_scores_for_candidate_level_error_analysis(self) -> None:
        write_evaluation_artifacts = self._api()
        records = [{"query_id": "q1_B", "misconception_id": "10"}]
        rankings = {
            "retriever": {
                "q1_B": [
                    {"candidate_id": "10", "score": 0.8},
                    {"candidate_id": "20", "score": 0.2},
                ]
            }
        }

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            write_evaluation_artifacts(output, records, rankings, k=2)
            row = json.loads((output / "predictions_retriever.jsonl").read_text(encoding="utf-8"))

        self.assertEqual(["10", "20"], row["ranked_misconception_ids"])
        self.assertEqual(
            [{"candidate_id": "10", "score": 0.8}, {"candidate_id": "20", "score": 0.2}],
            row["ranked_candidates"],
        )


if __name__ == "__main__":
    unittest.main()
