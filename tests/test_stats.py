import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from oevl.stats import bootstrap_diff, bootstrap_diff_runs  # noqa: E402


class TestBootstrap(unittest.TestCase):
    def test_resamples_sentences_not_tokens(self):
        # one sentence: every resample is the whole set, so the interval collapses onto the difference
        r = bootstrap_diff(["s"] * 4, [1, 1, 1, 0], [1, 0, 0, 0])
        self.assertEqual(r, {"diff": 0.5, "lo": 0.5, "hi": 0.5})

    def test_interval(self):
        sents = [f"s{i // 3}" for i in range(60)]
        a = [1 if i % 4 else 0 for i in range(60)]
        b = [1 if i % 3 else 0 for i in range(60)]
        r = bootstrap_diff(sents, a, b)
        self.assertAlmostEqual(r["diff"], sum(a) / 60 - sum(b) / 60)
        self.assertLess(r["lo"], r["diff"])
        self.assertGreater(r["hi"], r["diff"])
        self.assertEqual(r, bootstrap_diff(sents, a, b))  # seed 0: the same interval every time
        self.assertEqual(bootstrap_diff(sents, a, a), {"diff": 0.0, "lo": 0.0, "hi": 0.0})
        self.assertIsNone(bootstrap_diff([], [], []))

    def test_runs_widen_the_interval(self):
        sents = [f"s{i // 3}" for i in range(60)]
        steady = [[1 if i % 4 else 0 for i in range(60)]] * 3  # three identical runs
        shaky = [[1] * 60, [0] * 60, [1 if i % 2 else 0 for i in range(60)]]  # runs that disagree
        mean = [sum(r[i] for r in shaky) / 3 for i in range(60)]
        one, two = bootstrap_diff(sents, mean, steady[0]), bootstrap_diff_runs(sents, shaky, steady)
        self.assertAlmostEqual(two["diff"], one["diff"])
        self.assertGreater(two["hi"] - two["lo"], 2 * (one["hi"] - one["lo"]))  # run variance now counts
        self.assertEqual(two, bootstrap_diff_runs(sents, shaky, steady))
        self.assertEqual(bootstrap_diff_runs(["s"] * 3, [[1, 1, 0]], [[1, 0, 0]]),
                         {"diff": 1 / 3, "lo": 1 / 3, "hi": 1 / 3})
        self.assertIsNone(bootstrap_diff_runs([], [[]], [[]]))


if __name__ == "__main__":
    unittest.main()
