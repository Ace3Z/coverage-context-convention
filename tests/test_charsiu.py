import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import oevl.charsiu_g2p as charsiu  # noqa: E402
from oevl.charsiu_g2p import PREFIX, error_type, lev, score  # noqa: E402


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

    def test_dev_set_joins_when_present(self):
        with tempfile.TemporaryDirectory() as d:
            dev = Path(d) / "john_ws.jsonl"
            row = {"id": "john:0", "source": "john_ws", "tokens": ["gōd", "is"], "tokens_bare": ["god", "is"]}
            dev.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf8")
            old, charsiu.DEV = charsiu.DEV, dev
            try:
                sets = charsiu.tests()
            finally:
                charsiu.DEV = old
        self.assertEqual([r["gold_pattern"] for r in sets["john_ws"]], ["1", "0"])


if __name__ == "__main__":
    unittest.main()
