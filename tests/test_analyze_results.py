import unittest

from scripts.analyze_results import question_bootstrap


class QuestionBootstrapTests(unittest.TestCase):
    def test_constant_delta_has_a_degenerate_interval(self):
        ids = ["1_A", "1_B", "2_C"]
        question_of = {"1_A": "1", "1_B": "1", "2_C": "2"}
        stats = question_bootstrap(ids, question_of, {q: 0.25 for q in ids}, resamples=50, seed=0)
        self.assertAlmostEqual(stats["mean"], 0.25)
        self.assertEqual(stats["ci95"], [0.25, 0.25])
        self.assertEqual((stats["questions"], stats["queries"]), (2, 3))

    def test_resamples_whole_questions(self):
        # Question 1 always moves both of its queries together, so every draw is a mix of
        # whole questions: the mean of any draw is 1.0 (only q1), 0.0 (only q2), or 2/3.
        ids = ["1_A", "1_B", "2_C"]
        question_of = {"1_A": "1", "1_B": "1", "2_C": "2"}
        delta = {"1_A": 1.0, "1_B": 1.0, "2_C": 0.0}
        stats = question_bootstrap(ids, question_of, delta, resamples=200, seed=3)
        self.assertAlmostEqual(stats["mean"], 2 / 3)
        for bound in stats["ci95"]:
            self.assertTrue(any(abs(bound - v) < 1e-9 for v in (0.0, 2 / 3, 1.0)))


if __name__ == "__main__":
    unittest.main()
