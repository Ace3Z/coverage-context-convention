import collections
import csv
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from oevl.agreement import align, classes, compare, disagreements, gate, hand_check, kappa  # noqa: E402
from oevl.experiment import bare, tokens  # noqa: E402


def edition(name, words):
    return list(tokens([{"id": name, "source": name, "tokens": words, "tokens_bare": [bare(w) for w in words]}]))


A = edition("a", ["þā", "cōm", "of", "mōre"])
B = edition("b", ["ða", "com", "Fol", "of", "more"])  # ð spelling and an inserted folio mark
LEX = {"þa": {"1"}, "com": {"0", "1"}, "of": {"0"}, "more": {"00", "10"}}
COUNTS = collections.defaultdict(collections.Counter, {"com": collections.Counter({"1": 3}),
                                                       "more": collections.Counter({"00": 2, "10": 2})})


class TestAgreement(unittest.TestCase):
    def test_align(self):
        self.assertEqual([(x["gold"], y["gold"]) for x, y in align(A, B)],
                         [("þā", "ða"), ("cōm", "com"), ("of", "of"), ("mōre", "more")])

    def test_kappa(self):
        self.assertAlmostEqual(kappa(["0", "0", "1", "1"], ["0", "1", "1", "1"]), 0.5)  # po .75, pe .5
        self.assertIsNone(kappa(["0", "0"], ["0", "0"]))
        self.assertIsNone(kappa([], []))

    def test_compare_and_classes(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "ambiguity_classes.csv"
            p.write_text("skeleton,n,final_label\ncom,3,A\nmore,4,C\nof,1,\n", encoding="utf8")
            cls = classes(p)
            self.assertEqual(classes(Path(d) / "missing.csv"), {})
        self.assertEqual(cls, {"com": "A", "more": "C"})
        _, res = compare(A, B, "a", "b", LEX, COUNTS, cls)
        self.assertEqual((res["aligned"], res["all"]["n"], res["all"]["agreement"]), (4, 4, 0.25))
        amb = res["lexicon_ambiguous"]  # com and more; the prior picks 1 for com and 0 for more (tie goes short)
        self.assertEqual((amb["n"], amb["agreement"], amb["prior_vs_a"], amb["prior_vs_b"]), (2, 0.0, 0.5, 0.5))
        self.assertEqual(res["attested_ambiguous"]["n"], 1)
        self.assertEqual((res["class_A"]["n"], res["class_B"]["n"], res["class_C"]["n"]), (1, 0, 1))
        self.assertNotIn("class_A", compare(A, B, "a", "b", LEX, COUNTS, {})[1])

    def test_disagreements(self):
        pairs, _ = compare(A, B, "a", "b", LEX, COUNTS, {})
        rows = disagreements(pairs, A, B, "a", "b", LEX, COUNTS)
        self.assertEqual([r["key"] for r in rows], ["þa", "com", "more"])
        self.assertEqual(rows[1]["b_context"], "ða [com] Fol of more")

    def test_gate(self):
        res = {"lexicon_ambiguous": {"n": 500, "agreement": .95, "prior_vs_a": .85, "prior_vs_b": .86},
               "attested_ambiguous": {"n": 499, "agreement": .99, "prior_vs_a": .5, "prior_vs_b": .5}}
        g = gate(res, "a", "b")
        self.assertTrue(g["lexicon_ambiguous_floor_vs_a"]["clears_bar"])       # exactly 10 points on 500 tokens
        self.assertFalse(g["lexicon_ambiguous_floor_vs_b"]["clears_bar"])      # 9 points
        self.assertFalse(g["attested_ambiguous_floor_vs_a"]["clears_bar"])     # 49 points but only 499 tokens

    def test_hand_check(self):
        rows = [{"a_id": f"a:{i}", "b_id": f"b:{i}"} for i in range(3)]
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "checked.tsv"
            with open(p, "w", encoding="utf8", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["a_id", "b_id", "label", "labelled_by"], delimiter="\t")
                w.writeheader()
                w.writerows({**r, "label": "other", "labelled_by": "x"} for r in rows)
            self.assertEqual(hand_check(p, rows)["labels"], {"other": 3})
            self.assertTrue(hand_check(p, rows[:2])["stale"])


if __name__ == "__main__":
    unittest.main()
