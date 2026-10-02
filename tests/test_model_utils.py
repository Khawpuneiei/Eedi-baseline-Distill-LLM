"""Pure behavior tests for the retrieve, rerank, and rationale utilities."""

from __future__ import annotations

import hashlib
import importlib
import unittest
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any


class ModelUtilityTests(unittest.TestCase):
    def require_function(self, module_name: str, function_name: str) -> Callable[..., Any]:
        try:
            module = importlib.import_module(f"eedi_baseline.{module_name}")
        except ImportError as error:
            self.fail(f"{module_name} utility module is missing: {error}")
        function = getattr(module, function_name, None)
        self.assertTrue(callable(function), f"{module_name}.{function_name} is missing")
        return function

    def test_retrieval_top_k_is_deterministic_and_deduplicates_catalog_ids(self) -> None:
        rank_top_k = self.require_function("retrieval", "rank_top_k")
        catalog = {"a": "Alpha", "b": "Beta", "c": "Gamma"}

        ranked = rank_top_k(
            [("b", 0.9), ("a", 0.9), ("b", 0.9), ("c", 0.7), ("outside", 1.0)],
            catalog,
            k=3,
        )

        self.assertEqual(
            ranked,
            [
                {"misconception_id": "a", "score": 0.9},
                {"misconception_id": "b", "score": 0.9},
                {"misconception_id": "c", "score": 0.7},
            ],
        )

    def test_retrieval_encodes_catalog_in_bounded_batches(self) -> None:
        import torch

        retrieval_module = importlib.import_module("eedi_baseline.retrieval")
        vectors = {
            "query": [1.0, 0.0],
            "Misconception: Alpha": [1.0, 0.0],
            "Misconception: Beta": [0.0, 1.0],
            "Misconception: Gamma": [-1.0, 0.0],
            "Misconception: Delta": [0.5, 0.5],
            "Misconception: Epsilon": [-0.5, 0.5],
        }

        class FakeTokenizer:
            def __init__(self) -> None:
                self.batch_sizes: list[int] = []

            def __call__(self, texts, **_kwargs):
                self.batch_sizes.append(len(texts))
                token_ids = [list(vectors).index(text.replace(
                    "Represent this sentence for searching relevant passages: ", ""
                )) for text in texts]
                return {"input_ids": torch.tensor(token_ids, dtype=torch.long).unsqueeze(1)}

        class FakeEncoder:
            def eval(self) -> None:
                return None

            def __call__(self, input_ids, **_kwargs):
                token_vectors = torch.tensor(list(vectors.values()), dtype=torch.float32)
                return SimpleNamespace(
                    last_hidden_state=token_vectors[input_ids.squeeze(1)].unsqueeze(1)
                )

        tokenizer = FakeTokenizer()
        model = retrieval_module.DenseRetriever(
            encoder=FakeEncoder(),
            tokenizer=tokenizer,
            model_id="fixture",
            model_revision="fixture-revision",
            max_length=8,
            query_instruction=(
                "Represent this sentence for searching relevant passages: "
            ),
            device="cpu",
        )
        catalog = {
            "a": "Alpha",
            "b": "Beta",
            "c": "Gamma",
            "d": "Delta",
            "e": "Epsilon",
        }

        ranked = retrieval_module.rank_candidates(
            [{"query_id": "q1", "query": "query"}], model, catalog, k=3, batch_size=2
        )

        self.assertLessEqual(max(tokenizer.batch_sizes), 2)
        self.assertEqual(
            [item["misconception_id"] for item in ranked[0]], ["a", "d", "b"]
        )

    def test_reranker_pairs_include_gold_positive_and_exclude_gold_from_negatives(self) -> None:
        build_reranker_pairs = self.require_function("reranking", "build_reranker_pairs")
        records = [
            {
                "query_id": "q-1",
                "question_id": "question-1",
                "answer_key": "B",
                "question": "What is 2 + 2?",
                "correct_answer": "4",
                "distractor": "5",
                "query": "Question: What is 2 + 2?\nCorrect: 4\nDistractor: 5",
                "misconception_id": "2",
                "misconception_name": "Adds one too many",
            }
        ]
        catalog = {
            "1": "Adds the wrong terms",
            "2": "Adds one too many",
            "3": "Subtracts instead of adding",
            "4": "Confuses digits with place value",
        }

        pairs = build_reranker_pairs(
            records,
            {"q-1": ["3", "2", "1", "3"]},
            catalog,
            n_random_negatives=3,
            seed=19,
        )

        self.assertEqual(
            [(pair["candidate_id"], pair["label"]) for pair in pairs],
            [("2", 1), ("3", 0), ("1", 0), ("4", 0)],
        )
        self.assertEqual(pairs[0]["query_id"], "q-1")
        self.assertEqual(pairs[0]["query"], records[0]["query"])
        self.assertEqual(pairs[0]["candidate_text"], "Adds one too many")
        self.assertEqual(len({pair["candidate_id"] for pair in pairs}), len(pairs))
        self.assertFalse(
            any(pair["candidate_id"] == "2" and pair["label"] == 0 for pair in pairs)
        )

    def test_reranker_pairs_compose_cached_rationale_only_when_provided(self) -> None:
        build_reranker_pairs = self.require_function("reranking", "build_reranker_pairs")
        record = {
            "query_id": "q-1",
            "query": "Question: What is 2 + 2? Correct: 4. Distractor: 5.",
            "misconception_id": "2",
            "misconception_name": "Adds one too many",
        }

        pairs = build_reranker_pairs(
            [record],
            {"q-1": ["2"]},
            {"2": "Adds one too many"},
            rationales={"q-1": "Adds an extra tile while counting."},
        )

        self.assertEqual(
            pairs[0]["query"],
            "Question: What is 2 + 2? Correct: 4. Distractor: 5.\n\n"
            "[RATIONALE]\nAdds an extra tile while counting.\n[/RATIONALE]",
        )

    def test_reranker_ordering_cannot_add_candidates_and_breaks_ties_by_id(self) -> None:
        rank_scored_candidates = self.require_function("reranking", "rank_scored_candidates")

        ranked = rank_scored_candidates(["z", "a", "z", "m"], [0.8, 0.8, 0.9, 0.3])

        self.assertEqual(
            ranked,
            [
                {"candidate_id": "z", "score": 0.9},
                {"candidate_id": "a", "score": 0.8},
                {"candidate_id": "m", "score": 0.3},
            ],
        )
        self.assertEqual({item["candidate_id"] for item in ranked}, {"z", "a", "m"})

    def test_reranker_training_rejects_float_relevance_labels(self) -> None:
        validate_pairs = self.require_function("reranking", "_validated_training_pairs")
        malformed_pairs = [
            {"query": "Question", "candidate_text": "Positive", "label": 1.0},
            {"query": "Question", "candidate_text": "Negative", "label": 0},
        ]

        with self.assertRaisesRegex(ValueError, "label must be 0 or 1"):
            validate_pairs(malformed_pairs)

    def test_no_rationale_keeps_the_reranker_query_byte_for_byte(self) -> None:
        compose_reranker_query = self.require_function("reranking", "compose_reranker_query")
        query = "  Question: 2 + 2?\nAnswer: 4; distractor: 5  "

        self.assertEqual(compose_reranker_query(query), query)

    def test_rationale_is_appended_in_a_delimited_section(self) -> None:
        compose_reranker_query = self.require_function("reranking", "compose_reranker_query")

        composed = compose_reranker_query("Question text", "Adds the same number twice")

        self.assertEqual(
            composed,
            "Question text\n\n[RATIONALE]\nAdds the same number twice\n[/RATIONALE]",
        )

    def test_rationale_prompt_uses_only_question_answer_and_distractor(self) -> None:
        build_rationale_prompt = self.require_function("rationale", "build_rationale_prompt")
        record = {
            "query_id": "q-8",
            "question": "A box has 3 red and 2 blue tiles. How many tiles?",
            "correct_answer": "5",
            "distractor": "6",
            "misconception_id": "gold-17",
            "misconception_name": "Gold-only private label",
        }

        prompt = build_rationale_prompt(record)

        self.assertEqual(
            prompt,
            "You are a math tutor. Infer the likely reasoning behind the incorrect answer using only the question, correct answer, and distractor. Do not name a misconception category. Return one concise sentence describing the reasoning error.\n\n"
            "Question: A box has 3 red and 2 blue tiles. How many tiles?\n"
            "Correct answer: 5\n"
            "Distractor: 6\n\n"
            "Reasoning error:",
        )
        self.assertNotIn("gold-17", prompt)
        self.assertNotIn("Gold-only private label", prompt)

    def test_rationale_cache_record_binds_input_hash_and_prompt_identity(self) -> None:
        build_rationale_prompt = self.require_function("rationale", "build_rationale_prompt")
        build_rationale_cache_record = self.require_function(
            "rationale", "build_rationale_cache_record"
        )
        query = {
            "query_id": "q-8",
            "question": "A box has 3 red and 2 blue tiles. How many tiles?",
            "correct_answer": "5",
            "distractor": "6",
        }
        prompt = build_rationale_prompt(query)
        expected_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()

        record = build_rationale_cache_record(
            query,
            "Adds an extra tile while counting.",
            teacher_model="Qwen/Qwen2.5-7B-Instruct",
            teacher_revision="revision-sha",
        )

        self.assertEqual(
            record,
            {
                "query_id": "q-8",
                "rationale": "Adds an extra tile while counting.",
                "teacher_model": "Qwen/Qwen2.5-7B-Instruct",
                "teacher_revision": "revision-sha",
                "prompt_version": "eedi-error-rationale-v1",
                "input_sha256": expected_hash,
            },
        )
        self.assertEqual(record, build_rationale_cache_record(
            query,
            "Adds an extra tile while counting.",
            teacher_model="Qwen/Qwen2.5-7B-Instruct",
            teacher_revision="revision-sha",
        ))


if __name__ == "__main__":
    unittest.main()
