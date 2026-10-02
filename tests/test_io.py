"""Behavior tests for the JSONL and rationale-cache artifact boundary."""

from __future__ import annotations

import importlib
import tempfile
import unittest
from pathlib import Path


class ArtifactIOTests(unittest.TestCase):
    def _api(self):
        try:
            module = importlib.import_module("eedi_baseline.io")
        except ModuleNotFoundError:
            self.fail("eedi_baseline.io is missing")
        for name in ("read_jsonl", "write_jsonl", "load_rationales"):
            self.assertTrue(callable(getattr(module, name, None)), f"{name} is missing")
        return module

    def test_jsonl_round_trip_preserves_unicode_and_nested_values(self) -> None:
        module = self._api()
        rows = [
            {"query_id": "q1_A", "query": "François adds the fractions", "scores": [0.25, 0.75]},
            {"query_id": "q2_C", "query": "What is ⅓ × 6?", "metadata": {"seed": 42}},
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "predictions.jsonl"
            module.write_jsonl(path, rows)
            actual = module.read_jsonl(path)
        self.assertEqual(rows, actual)

    def test_jsonl_parser_reports_the_line_of_malformed_json(self) -> None:
        module = self._api()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.jsonl"
            path.write_text('{"query_id":"q1"}\n{bad json}\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "line 2"):
                module.read_jsonl(path)

    def test_rationale_cache_rejects_duplicate_and_missing_query_ids(self) -> None:
        module = self._api()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rationales.jsonl"
            path.write_text(
                '{"query_id":"q1_A","rationale":"adds denominators"}\n'
                '{"query_id":"q1_A","rationale":"multiplies numerators"}\n',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "duplicate"):
                module.load_rationales(path)

            path.write_text('{"query_id":"q1_A","rationale":"adds denominators"}\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "missing"):
                module.load_rationales(path, expected_query_ids={"q1_A", "q2_C"})


if __name__ == "__main__":
    unittest.main()
