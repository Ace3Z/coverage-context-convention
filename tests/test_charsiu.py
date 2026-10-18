import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import oevl.charsiu_g2p as charsiu  # noqa: E402
from oevl.charsiu_g2p import PREFIX, error_type, lev, score  # noqa: E402
from oevl.experiment import TESTS  # noqa: E402


class TestCharsiu(unittest.TestCase):
    def test_prefix(self):
        self.assertEqual(PREFIX, "<ang>:")  # as data_utils.py, no space

    def test_lev(self):
        self.assertEqual(lev("kitten", "sitting"), 3)
        self.assertEqual(lev("", "ab"), 2)
        self.assertEqual(lev(["a", "b"], ["a"]), 1)

    def test_error_type(self):
        self.assertEqual(error_type("ɡoːd", "ɡoːd", False), "correct")
        self.assertEqual(error_type("ɡoːd", "ɡod", False), "length_only")
        self.assertEqual(error_type("ɡod", "ɡoːd", False), "length_only")
        self.assertEqual(error_type("ɡoːd", "ɡuːd", False), "segment")
        self.assertEqual(error_type("ɡoːdes", "ɡoːd", True), "truncation")
        self.assertEqual(error_type("kɑrful", "kɑːrful", True), "length_only")  # complete output, no room for EOS
        self.assertEqual(error_type("ɡoːd", "ɡoːd", True), "correct")  # a cut-off output can still match

    def test_score(self):
        s = score(["ɡoːd", "ɑb", "kyn"], ["ɡod", "ɑb", "kin"], ["length_only", "correct", "segment"])
        self.assertEqual((s["word_errors"], s["length_only"], s["segment"], s["truncation"]), (2, 1, 1, 0))
        self.assertAlmostEqual(s["wer"], 2 / 3)
        self.assertAlmostEqual(s["per"], 2 / 9)
        self.assertEqual(s["length_only_share"], 0.5)

    @unittest.skipUnless((ROOT / "data/brunetti-length.txt").exists(), "Real Tauber source not downloaded")
    def test_sets_are_the_experiment_sets(self):
        sets = charsiu.tests()
        self.assertEqual(list(sets), TESTS)  # the dev set joins under the experiment's name
        self.assertTrue(all(sets.values()))


if __name__ == "__main__":
    unittest.main()
