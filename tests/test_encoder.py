import collections
import contextlib
import csv
import gzip
import importlib.util
import io
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from oevl import encoder_summary  # noqa: E402
from oevl.encoder_experiment import limit, make_input, score  # noqa: E402

SENT = "ic clypie to þe drihten þu þe eall geworhtest and þu ana eart".split()


def tok(i, sent=SENT):
    return {"sentence": sent, "index": i}


class TestMakeInput(unittest.TestCase):
    def test_original_arms(self):
        self.assertEqual(make_input(tok(3), "word", 128), "þe")
        self.assertEqual(make_input(tok(3), "left", 128), "ic clypie to [[þe]]")
        self.assertEqual(make_input(tok(3), "sentence", 128),
                         "ic clypie to [[þe]] drihten þu þe eall geworhtest and þu ana eart")

    def test_local_windows(self):
        self.assertEqual(make_input(tok(3), "local2", 128), "clypie to [[þe]] drihten þu")
        self.assertEqual(make_input(tok(0), "local2", 128), "[[ic]] clypie to")
        self.assertEqual(make_input(tok(len(SENT) - 1), "local5", 128), "eall geworhtest and þu ana [[eart]]")

    def test_byte_cap_keeps_target(self):
        long_sent = ("þæt wæs gōd cyning " * 40).split()
        for arm in ("left", "sentence", "local2", "local5"):
            for i in (0, 37, 80, len(long_sent) - 1):
                for cap in (128, 256):
                    x = make_input(tok(i, long_sent), arm, cap)
                    self.assertLessEqual(len(x.encode()), cap, (arm, i, cap))
                    self.assertIn(f"[[{long_sent[i].lower()}]]", x)


class TestLocalInputs(unittest.TestCase):
    def test_local_arms_never_reach_a_cap(self):
        """encoder_summary.REPLICATES rests on this: local2 and local5 see the same input at every cap of the grid."""
        from oevl.encoder_experiment import candidates
        from oevl.experiment import EXCLUDE_D3, TESTS, build_data, heldout, lexicon
        with contextlib.redirect_stdout(io.StringIO()):
            train, _, _, _ = build_data(ROOT / "data", EXCLUDE_D3)
            lex, _ = lexicon(train, str(ROOT / "data/lex_wikipron.csv"))
            rows = [r for r in train if r["gold_pattern"] and len(lex.get(r["key"], set())) >= 2]
            rows += [r for t in TESTS for r in heldout(ROOT / "data", t) if len(candidates(r, lex)[0]) >= 2]
        for arm, most in (("local2", 57), ("local5", 103)):
            self.assertEqual(max(len(make_input(r, arm, 10 ** 6).encode()) for r in rows), most)
            self.assertTrue(all(make_input(r, arm, 128) == make_input(r, arm, 2560) for r in rows), arm)


class TestLimit(unittest.TestCase):
    def test_fixed_subset(self):
        ex = [{"id": str(j)} for j in range(1000)]
        a, b = limit(ex, 100), limit(list(ex), 100)
        self.assertEqual(a, b)                                   # same subset on every call
        self.assertEqual(len(a), 100)
        self.assertEqual([int(r["id"]) for r in a], sorted(int(r["id"]) for r in a))  # corpus order kept
        self.assertIs(limit(ex, None), ex)
        self.assertIs(limit(ex, 5000), ex)


class TestScore(unittest.TestCase):
    def test_classes_and_collapse_checks(self):
        lex = {"þe": {"0": set(), "1": set()}, "ge": {"0": set(), "1": set()}, "god": {"1": set()}}
        counts = collections.defaultdict(collections.Counter, {"þe": collections.Counter({"0": 3, "1": 1})})
        rows = [{"key": k, "gold": k, "gold_pattern": g} for k, g in (("þe", "1"), ("þe", "0"), ("ge", "0"), ("god", "1"))]
        prior = ["0", "0", "0", "1"]
        out = score(rows, ["1"] * 4, prior, lex, counts, {"þe": "C"})  # always the most long marks
        self.assertEqual(out["accuracy"], {"n": 4, "accuracy": 0.5})
        self.assertEqual(out["lexicon_ambiguous"], {"n": 3, "accuracy": 0.333333})
        self.assertEqual(out["class_C"], {"n": 2, "accuracy": 0.5})
        self.assertEqual(out["class_unlisted"], {"n": 1, "accuracy": 0.0})
        self.assertEqual(out["class_A"], {"n": 0, "accuracy": None})
        self.assertEqual((out["most_long_share"], out["prior_agreement"]), (1.0, 0.0))
        out = score(rows, prior, prior, lex, counts, {"þe": "C"})
        self.assertEqual((out["most_long_share"], out["prior_agreement"]), (0.0, 1.0))


def run(main, *args):
    with mock.patch.object(sys, "argv", ["prog", *args]), contextlib.redirect_stdout(io.StringIO()):
        main()


GOLD, AMB, CLS = ["1", "0", "0", "1"], [True, True, True, False], ["B", "B", "C", None]
POPS = {"accuracy": [0, 1, 2, 3], "lexicon_ambiguous": [0, 1, 2], "attested_ambiguous": [], "minority_reading": [],
        "class_A": [], "class_B": [0, 1], "class_C": [2], "class_unlisted": []}


def fake_run(folder, arm, seed, preds, max_bytes=128):
    """A finished encoder run on four soliloquies tokens in two sentences, as encoder_experiment writes it."""
    folder.mkdir(parents=True)
    rows = [{"id": f"sol:{i}", "sentence_id": f"sol:{i // 2}", "gold_pattern": g, "frequency": "0",
             "lexicon_ambiguous": a, "attested_ambiguous": False, "minority_reading": False, "class": c,
             f"encoder_{arm}": p} for i, (g, a, c, p) in enumerate(zip(GOLD, AMB, CLS, preds))]
    (folder / "soliloquies_predictions.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    sol = {k: {"n": len(ix), "accuracy": sum(preds[i] == GOLD[i] for i in ix) / len(ix) if ix else None}
           for k, ix in POPS.items()}
    arm_summary = {"train_loss_per_epoch": [0.5], "final_train_loss": 0.5, "collapsed": seed == 2,
                   "soliloquies": {**sol, "most_long_share": 0.5, "prior_agreement": 0.5}}
    (folder / "metrics.json").write_text(json.dumps({"seed": seed, "max_bytes": max_bytes, "exclude_train": [],
                                                     "arms": {arm: arm_summary}}))


@mock.patch.object(encoder_summary, "minority", lambda tables, metrics: set())  # the fixture ids are not in data/
class TestSummary(unittest.TestCase):
    def test_pools_seeds_and_pairs_systems(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            fake_run(d / "runs/cap128/word_s1", "word", 1, ["0", "0", "0", "1"])
            fake_run(d / "runs/cap128/word_s2", "word", 2, ["1", "1", "0", "1"])
            for seed in (1, 2):
                fake_run(d / f"runs/cap128/sentence_s{seed}", "sentence", seed, GOLD)
            run(encoder_summary.main, "--runs", str(d / "runs"), "--out", str(d / "enc"), "--report", str(d / "gen"))
            s = json.loads((d / "enc/summary.json").read_text())["cap128"]
            word = s["arms"]["word"]
            self.assertEqual(word["sets"]["soliloquies"]["lexicon_ambiguous"],
                             {"n": 3, "values": [2 / 3, 2 / 3], "mean": 2 / 3, "sd": 0.0})
            self.assertEqual((word["collapsed"], word["n_collapsed"]), ([False, True], 1))
            self.assertEqual(s["frequency"]["soliloquies"]["accuracy"], {"n": 4, "accuracy": 0.5})
            paired = s["paired"]["soliloquies"]
            self.assertAlmostEqual(paired["lexicon_ambiguous"]["sentence-word"]["diff"], 1 / 3)
            self.assertAlmostEqual(paired["class_B"]["sentence-word"]["diff"], 0.5)  # word: 1 of 2 right per seed
            self.assertEqual([r["a_right_b_wrong"] for r in paired["lexicon_ambiguous"]["sentence-word"]["mcnemar"]],
                             [1, 1])
            self.assertEqual(set(paired["lexicon_ambiguous"]),
                             {"sentence-word", "word-frequency", "sentence-frequency"})
            with gzip.open(d / "enc/soliloquies_predictions.tsv.gz", "rt", encoding="utf8") as f:
                rows = list(csv.DictReader(f, delimiter="\t"))
            self.assertEqual([r["cap128/word_s2"] for r in rows], ["1", "1", "0", "1"])
            self.assertEqual([r["cap128:class"] for r in rows], ["B", "B", "C", ""])
            self.assertIn("\\newcommand{\\NEncCapOneTwoEightSolLexBilVsWord}{+33.33}",
                          (d / "gen/numbers_encoder.tex").read_text())
            self.assertIn("Bilateral & 2 & 1 & 100.00", (d / "gen/tab_encoder.tex").read_text())
            self.assertNotIn("local_replicates", json.loads((d / "enc/summary.json").read_text()))
            self.check_from_tsv(d)

    def test_local_replicates_are_pooled(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            fake_run(d / "runs/cap128/local2_s1", "local2", 1, ["0", "0", "0", "1"])
            fake_run(d / "runs/cap256/local2_s1", "local2", 1, GOLD, 256)
            fake_run(d / "runs/cap256/left_s1", "left", 1, GOLD, 256)
            run(encoder_summary.main, "--runs", str(d / "runs"), "--out", str(d / "enc"), "--report", str(d / "gen"))
            s = json.loads((d / "enc/summary.json").read_text())
            self.assertEqual(s["cap128"]["arms"]["local2"]["identical_inputs_at"], ["cap128", "cap256"])
            self.assertNotIn("identical_inputs_at", s["cap256"]["arms"]["left"])
            pooled = s["local_replicates"]["arms"]["local2"]
            self.assertEqual(pooled["runs"], ["cap128/local2_s1", "cap256/local2_s1"])
            self.assertAlmostEqual(pooled["sets"]["soliloquies"]["lexicon_ambiguous"]["mean"], 5 / 6)
            self.assertEqual(s["replicate_spread"]["local2"]["s1"]["soliloquies"]["class_B"],
                             {"min": 0.5, "max": 1.0, "by_config": {"cap128": 0.5, "cap256": 1.0}})
            # prior 2/3 right on the lexicon-ambiguous tokens, local2 5/6 averaged over its two runs
            self.assertAlmostEqual(s["local_replicates"]["paired"]["soliloquies"]["lexicon_ambiguous"]
                                   ["local2-frequency"]["diff"], 1 / 6)
            tab, num = (d / "gen/tab_encoder.tex").read_text(), (d / "gen/numbers_encoder.tex").read_text()
            self.assertNotIn("replicates", tab)
            self.assertIn(" & Local 2 & 1 & 0 & 100.00", tab)  # only at its full cap, 256 bytes, like the numbers
            self.assertEqual(tab.count("Local 2"), 1)
            self.assertIn("\\newcommand{\\NEncRepSolLexLocalTwoVsFreq}{+16.67}", num)
            self.assertIn("\\newcommand{\\NEncRepSolClsBLocalTwoSpread}{50.00}", num)
            self.assertNotIn("NEncCapOneTwoEightSolLexLocalTwo", num)  # a replicate arm only at its full cap
            self.assertIn("\\newcommand{\\NEncCapTwoFiveSixSolLexLocalTwo}{100.00}", num)
            self.assertIn("\\newcommand{\\NEncCapTwoFiveSixSolLexLocalTwoVsFreq}{+33.33}", num)
            self.assertIn("\\newcommand{\\NEncRunsSolClsBLocalTwoMin}{50.00}", num)  # over every run of the arm
            self.assertIn("\\newcommand{\\NEncRunsSolClsBLocalTwoMax}{100.00}", num)
            self.assertEqual(s["identical_runs"],
                             {"runs": 3, "distinct": 2, "groups": [["cap256/left_s1", "cap256/local2_s1"]]})
            self.assertIn("\\newcommand{\\NEncRunsDistinct}{2}", num)
            self.check_from_tsv(d)

    def check_from_tsv(self, d):
        """--from-tsv gives the same three files from the tables and the run records alone (derived fields blanked)."""
        s = json.loads((d / "enc/summary.json").read_text())
        for k in encoder_summary.BLOCKS:
            s.pop(k, None)
        for C in s.values():
            C["paired"], C["frequency"] = {}, {}
            for A in C["arms"].values():
                A.pop("identical_inputs_at", None)
                for v in (v for S in A["sets"].values() for v in S.values() if isinstance(v, dict)):
                    v.update(n=None, mean=None, sd=None)
        (d / "tsv").mkdir()
        for f in (d / "enc").glob("*.tsv.gz"):
            shutil.copy(f, d / "tsv")
        (d / "tsv/summary.json").write_text(json.dumps(s))
        run(encoder_summary.main, "--from-tsv", "--out", str(d / "tsv"), "--report", str(d / "gen2"))
        for a, b in (("enc/summary.json", "tsv/summary.json"), ("gen/tab_encoder.tex", "gen2/tab_encoder.tex"),
                     ("gen/numbers_encoder.tex", "gen2/numbers_encoder.tex")):
            self.assertEqual((d / a).read_text(), (d / b).read_text(), b)

    def test_refuses_predictions_that_disagree_with_metrics(self):
        with tempfile.TemporaryDirectory() as d:
            fake_run(Path(d) / "runs/cap128/word_s1", "word", 1, GOLD)
            m = Path(d) / "runs/cap128/word_s1/metrics.json"
            m.write_text(m.read_text().replace('"accuracy": 1.0', '"accuracy": 0.9', 1))
            with self.assertRaises(SystemExit):
                run(encoder_summary.main, "--runs", str(Path(d) / "runs"), "--out", str(Path(d) / "enc"),
                    "--report", str(Path(d) / "gen"))


class TestMinority(unittest.TestCase):
    def test_recomputed_from_the_data(self):
        """The summary counts strict minority readings of the committed tables, on which the prior is never right."""
        out = ROOT / "results/encoder"
        tables = encoder_summary.read_tables(out)
        metrics = encoder_summary.runs_from_summary(json.loads((out / "summary.json").read_text(encoding="utf8")))
        with contextlib.redirect_stdout(io.StringIO()):
            stale = encoder_summary.minority(tables, metrics)
        self.assertEqual(stale, set(metrics))  # the committed tables keep the flag that counted ties
        for s, n in (("john_ws", 278), ("soliloquies", 607), ("tauber_beowulf", 325)):
            mino = [t for t in tables[s] if t["capfull:minority_reading"] == "1"]
            self.assertEqual(len(mino), n, s)
            self.assertFalse(any(t["capfull:frequency"] == t["gold"] for t in mino), s)


@unittest.skipUnless(importlib.util.find_spec("transformers"), "needs torch and transformers (requirements.txt)")
class TestTinySmoke(unittest.TestCase):
    def test_two_seeds_then_summary(self):
        """The CPU smoke test: tiny random encoder, all arms on two seeds (D3 default) and one run on all five works, then encoder_summary."""
        from oevl import encoder_experiment
        arms = ("word", "left", "sentence", "local2", "local5")
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            for cfg, seed, which, extra in (("cap128", 1, arms, []), ("cap128", 2, arms, []),
                                            ("cap128_zup", 1, ("sentence",), ["--exclude-train", ""])):
                out = d / f"{cfg}_s{seed}"
                run(encoder_experiment.main, "--tiny", "--max-bytes", "128", "--seed", str(seed),
                    "--arms", ",".join(which), "--out", str(out), *extra)
                (d / "runs" / cfg).mkdir(parents=True, exist_ok=True)
                for arm in which:  # one folder per arm and seed, as the GPU runs are laid out
                    (d / "runs" / cfg / f"{arm}_s{seed}").symlink_to(out)
            run(encoder_summary.main, "--runs", str(d / "runs"), "--out", str(d / "enc"), "--report", str(d / "gen"))
            s = json.loads((d / "enc/summary.json").read_text())
            self.assertEqual(list(s["cap128"]["arms"]), list(arms))
            self.assertEqual(s["cap128"]["arms"]["local5"]["seeds"], [1, 2])
            self.assertEqual(s["cap128_zup"]["settings"]["exclude_train"], [])
            self.assertIsNone(s["cap128_zup"]["arms"]["sentence"]["sets"]["soliloquies"]["accuracy"]["sd"])
            self.assertLessEqual({"sentence-word", "left-word", "sentence-left", "local2-word", "local5-word",
                                  "word-frequency"}, set(s["cap128"]["paired"]["tauber_beowulf"]["class_B"]))
            run1 = json.loads((d / "cap128_s1/metrics.json").read_text())
            self.assertEqual(run1["exclude_train"], ["zupitza_grammar"])
            self.assertEqual(len(run1["arms"]["left"]["train_loss_per_epoch"]), 1)
            self.assertIn(run1["arms"]["left"]["collapsed"], (True, False))
            # a number recomputed from the committed table alone
            with gzip.open(d / "enc/soliloquies_predictions.tsv.gz", "rt", encoding="utf8") as f:
                rows = list(csv.DictReader(f, delimiter="\t"))
            amb = [r for r in rows if r["cap128:lexicon_ambiguous"] == "1"]
            self.assertAlmostEqual(sum(r["cap128/sentence_s2"] == r["gold"] for r in amb) / len(amb),
                                   s["cap128"]["arms"]["sentence"]["sets"]["soliloquies"]["lexicon_ambiguous"]["values"][1])
            # by default (D3) the encoder's prior is the one run_all.sh wrote
            with open(ROOT / "results/wikipron/soliloquies_predictions.jsonl", encoding="utf8") as f:
                ref = {x["id"]: x["frequency_prediction"] for x in map(json.loads, f)}
            self.assertEqual([r["cap128:frequency"] for r in rows], [ref[r["id"]] for r in rows])
            self.assertTrue((d / "gen/tab_encoder.tex").exists() and (d / "gen/numbers_encoder.tex").exists())


if __name__ == "__main__":
    unittest.main()
