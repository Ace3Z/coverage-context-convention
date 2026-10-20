import contextlib
import csv
import io
import re
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from oevl import experiment as E  # noqa: E402
from oevl import paper_numbers as PN  # noqa: E402
from oevl.encoder_summary import read_tables  # noqa: E402


class TestCore(unittest.TestCase):
    def test_reachable(self):
        lex = {"god": {"0": set(), "1": set()}, "man": {"1": set()}}
        for k, gold, want in (("god", "1", True), ("man", "0", False), ("wæter", "00", True), ("wæter", "10", False)):
            self.assertEqual(PN.reachable({"key": k, "gold_pattern": gold}, lex), want, (k, gold))

    def test_wilson(self):
        lo, hi = PN.wilson(324, 415)  # Wyatt and Klaeber on class B
        self.assertEqual((round(100 * lo, 2), round(100 * hi, 2)), (73.84, 81.79))
        self.assertAlmostEqual(PN.wilson(0, 10)[0], 0)

    def test_loose_and_longest_run(self):
        self.assertEqual([PN.loose(k) for k in ("feor", "oþþe", "neah", "meahte", "hyra")],
                         ["fer", "oþe", "neh", "mehte", "hira"])
        self.assertEqual(PN.longest_run([list("abcx"), list("qbcd")], list("zabcd")), (3, tuple("abc")))
        self.assertEqual(PN.longest_run([list("xy")], list("ab")), (0, ()))

    def test_counts_from_pcts(self):
        self.assertEqual(PN.counts_from_pcts([96.14, 95.85], 7639), [7344, 7322])
        with self.assertRaises(SystemExit):
            PN.counts_from_pcts([96.145], 7639)

    def test_fixed_rule(self):
        item = {"gold_in_candidates": "yes", "set": "soliloquies", "hs_form": "", "wyatt_form": "", "system_form": "þē",
                "class": ""}
        self.assertIsNone(PN.fixed_rule(item))
        self.assertEqual(PN.fixed_rule({**item, "gold_in_candidates": "no", "class": "B"}), "ceiling")
        self.assertEqual(PN.fixed_rule({**item, "class": "B"}), "grammatical")
        beo = {**item, "set": "tauber_beowulf", "hs_form": "Þē", "class": "B"}
        self.assertEqual(PN.fixed_rule(beo), "gold_in_doubt")
        self.assertEqual(PN.fixed_rule({**beo, "hs_form": "þe"}), "grammatical")
        self.assertEqual(PN.fixed_rule({**beo, "hs_form": "not_aligned", "system_form": "æþelingas"}), "grammatical")

    def test_numbers_refuses_bad_names(self):
        put = PN.Numbers()
        put("CorpusSentences", "7,901")
        for name in ("CorpusSentences", "Corpus2", "Þe"):
            with self.assertRaises(SystemExit):
                put(name, "1")


class TestWorkedExamples(unittest.TestCase):
    """The examples the paper quotes hold on the committed data, and a changed prediction stops the run."""

    @classmethod
    def setUpClass(cls):
        with contextlib.redirect_stdout(io.StringIO()):
            train = E.build_data(ROOT / "data", E.EXCLUDE_D3)[0]
            cls.lex, cls.counts = E.lexicon(train, str(ROOT / "data/lex_wikipron.csv"))
            cls.tb = E.heldout(ROOT / "data", "tauber_beowulf")
        cls.preds = {s: PN.jsonl(ROOT / f"results/wikipron/{s}_predictions.jsonl")
                     for s in ("soliloquies", "tauber_beowulf")}
        cls.enc = {s: {t["id"]: t for t in table} for s, table in read_tables(ROOT / "results/encoder").items()}
        with open(ROOT / "results/error_typing/labels.tsv", encoding="utf8") as f:
            cls.labels = list(csv.DictReader(f, delimiter="\t"))

    def test_examples_hold(self):
        PN.example_soliloquies_00026(self.preds["soliloquies"], self.enc["soliloquies"])
        self.assertEqual(len(PN.example_geata(self.preds["tauber_beowulf"], self.enc["tauber_beowulf"], self.lex,
                                              self.counts)), 39)
        se = PN.example_se(self.tb, PN.tauber_columns(ROOT / "data/brunetti-length.txt"), self.preds["tauber_beowulf"],
                           self.enc["tauber_beowulf"])
        self.assertEqual((len(se), sum(gloss == "the" for _, gloss in se)), (175, 105))
        self.assertEqual(PN.example_2796(self.enc["tauber_beowulf"], self.labels), 1)

    def test_a_changed_prediction_fails(self):
        enc = dict(self.enc["soliloquies"])
        enc["soliloquies:00026:3"] = {**enc["soliloquies:00026:3"], "capfull/local2_s3": "0"}  # one local2 run misses þē
        with self.assertRaises(SystemExit):
            PN.example_soliloquies_00026(self.preds["soliloquies"], enc)

    def test_no_macro_is_defined_twice(self):
        """The paper inputs all three generated number files, so no name may repeat across them."""
        names = [n for f in ("numbers.tex", "numbers_encoder.tex", "numbers_paper.tex") for n in
                 re.findall(r"\\newcommand\{\\(\w+)\}", (ROOT / "report/generated" / f).read_text(encoding="utf8"))]
        self.assertGreater(len(names), 1000)
        self.assertEqual(len(names), len(set(names)))


if __name__ == "__main__":
    unittest.main()
