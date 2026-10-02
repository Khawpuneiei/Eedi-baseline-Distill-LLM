import importlib
import unittest


def _metric_api(name):
    try:
        module = importlib.import_module("eedi_baseline.metrics")
    except ModuleNotFoundError as error:
        if error.name in {"eedi_baseline", "eedi_baseline.metrics"}:
            return None
        raise
    return getattr(module, name, None)


class RankingMetricTests(unittest.TestCase):
    def require_api(self, name):
        function = _metric_api(name)
        self.assertTrue(callable(function), f"eedi_baseline.metrics.{name} must be implemented")
        return function

    def test_single_label_average_precision_at_25_uses_rank_25(self):
        average_precision_at_k = self.require_api("average_precision_at_k")
        predictions = [f"noise-{rank}" for rank in range(1, 25)] + ["M7"]

        self.assertEqual(average_precision_at_k(predictions, "M7", k=25), 1 / 25)

    def test_average_precision_uses_multiple_relevant_ids_and_expected_denominator(self):
        average_precision_at_k = self.require_api("average_precision_at_k")

        # Hits at ranks 2 and 3 contribute 1/2 and 2/3; AP@4 divides by min(4, 3).
        self.assertAlmostEqual(
            average_precision_at_k(["noise", "M1", "M2", "M1"], {"M1", "M2", "M3"}, k=4),
            7 / 18,
        )

    def test_average_precision_ignores_hits_beyond_k(self):
        average_precision_at_k = self.require_api("average_precision_at_k")

        self.assertEqual(average_precision_at_k(["noise", "M7"], {"M7"}, k=1), 0)

    def test_duplicate_predictions_never_count_as_second_relevant_hit(self):
        average_precision_at_k = self.require_api("average_precision_at_k")
        recall_at_k = self.require_api("recall_at_k")
        predictions = ["M1", "M1", "M2"]

        # The repeated M1 occupies rank 2 but contributes no second hit.
        self.assertAlmostEqual(average_precision_at_k(predictions, {"M1", "M2"}, k=3), 5 / 6)
        self.assertEqual(recall_at_k(predictions, {"M1", "M2"}, k=2), 1 / 2)

    def test_empty_relevance_and_empty_query_collections_return_zero(self):
        average_precision_at_k = self.require_api("average_precision_at_k")
        mean_average_precision_at_k = self.require_api("mean_average_precision_at_k")
        recall_at_k = self.require_api("recall_at_k")
        mean_recall_at_k = self.require_api("mean_recall_at_k")

        self.assertEqual(average_precision_at_k([], ["M1"], k=25), 0)
        self.assertEqual(average_precision_at_k("M1", [], k=25), 0)
        self.assertEqual(recall_at_k(["M1"], [], k=25), 0)
        self.assertEqual(mean_average_precision_at_k([], [], k=25), 0)
        self.assertEqual(mean_recall_at_k([], [], k=25), 0)
        self.assertEqual(mean_average_precision_at_k([["M1"], []], [["M1"], ["M2"]]), 0.5)
        self.assertEqual(mean_recall_at_k([["M1"], []], [["M1"], ["M2"]]), 0.5)

    def test_recall_at_k_counts_distinct_relevant_ids(self):
        recall_at_k = self.require_api("recall_at_k")

        self.assertEqual(recall_at_k(["noise", "M1", "M2"], {"M1", "M2", "M3"}, k=3), 2 / 3)

    def test_metric_functions_do_not_mutate_prediction_or_relevance_lists(self):
        average_precision_at_k = self.require_api("average_precision_at_k")
        recall_at_k = self.require_api("recall_at_k")
        mean_average_precision_at_k = self.require_api("mean_average_precision_at_k")
        mean_recall_at_k = self.require_api("mean_recall_at_k")
        predictions = ["M1", "M1", "M2"]
        relevant = ["M1", "M2"]
        batch_predictions = [["M1", "M2"], ["M3"]]
        batch_relevant = [["M1"], ["M3"]]
        snapshots = (predictions[:], relevant[:], [row[:] for row in batch_predictions], [row[:] for row in batch_relevant])

        average_precision_at_k(predictions, relevant)
        recall_at_k(predictions, relevant)
        mean_average_precision_at_k(batch_predictions, batch_relevant)
        mean_recall_at_k(batch_predictions, batch_relevant)

        self.assertEqual((predictions, relevant, batch_predictions, batch_relevant), snapshots)

    def test_all_metrics_reject_nonpositive_or_noninteger_k(self):
        metric_names = (
            "average_precision_at_k",
            "mean_average_precision_at_k",
            "recall_at_k",
            "mean_recall_at_k",
        )
        for name in metric_names:
            function = self.require_api(name)
            for invalid_k in (0, -1, 1.5, True):
                with self.subTest(metric=name, k=invalid_k):
                    with self.assertRaisesRegex(ValueError, "k"):
                        if name.startswith("mean_"):
                            function([[]], [[]], k=invalid_k)
                        else:
                            function([], [], k=invalid_k)


if __name__ == "__main__":
    unittest.main()
