"""Observable contracts for matched condition scoring and result reports."""

from __future__ import annotations

import importlib
import unittest


class ReportingTests(unittest.TestCase):
    def _api(self):
        try:
            module = importlib.import_module("eedi_baseline.reporting")
        except ModuleNotFoundError:
            self.fail("eedi_baseline.reporting is missing")
        evaluate_conditions = getattr(module, "evaluate_conditions", None)
        render_markdown_report = getattr(module, "render_markdown_report", None)
        self.assertTrue(callable(evaluate_conditions), "evaluate_conditions is missing")
        self.assertTrue(callable(render_markdown_report), "render_markdown_report is missing")
        return evaluate_conditions, render_markdown_report

    def test_matched_conditions_include_misses_in_map_and_recall(self) -> None:
        evaluate_conditions, _ = self._api()
        rows = [
            {"query_id": "q1_A", "misconception_id": "m1"},
            {"query_id": "q2_C", "misconception_id": "m2"},
        ]
        conditions = {
            "retriever": {
                "q1_A": ["m1", "m2"],
                "q2_C": ["m3", "m2"],
            },
            "reranker": {
                "q1_A": ["m2", "m1"],
                "q2_C": ["m3", "m4"],
            },
        }

        actual = evaluate_conditions(rows, conditions, k=25)

        self.assertEqual(2, actual["retriever"]["query_count"])
        self.assertEqual(0.75, actual["retriever"]["map_at_k"])
        self.assertEqual(1.0, actual["retriever"]["recall_at_k"])
        self.assertEqual(0.25, actual["reranker"]["map_at_k"])
        self.assertEqual(0.5, actual["reranker"]["recall_at_k"])

    def test_evaluation_rejects_missing_or_extra_query_rankings(self) -> None:
        evaluate_conditions, _ = self._api()
        rows = [{"query_id": "q1_A", "misconception_id": "m1"}]

        with self.assertRaisesRegex(ValueError, "query"):
            evaluate_conditions(rows, {"retriever": {}}, k=25)
        with self.assertRaisesRegex(ValueError, "query"):
            evaluate_conditions(rows, {"retriever": {"q1_A": ["m1"], "extra": []}}, k=25)

    def test_markdown_report_marks_unrun_rationale_condition_without_inventing_a_score(self) -> None:
        _, render_markdown_report = self._api()
        markdown = render_markdown_report({
            "retriever": {"query_count": 2, "map_at_k": 0.75, "recall_at_k": 1.0},
            "reranker": {"query_count": 2, "map_at_k": 0.5, "recall_at_k": 1.0},
        }, k=25)

        self.assertIn("0.7500", markdown)
        self.assertIn("0.5000", markdown)
        self.assertIn("Rationale", markdown)
        self.assertIn("Not run", markdown)
        self.assertNotIn("0.0000", markdown)


if __name__ == "__main__":
    unittest.main()
