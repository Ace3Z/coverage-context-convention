import collections
import copy
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from oevl.audit_splits import grams  # noqa: E402
from oevl.clean_corpus import check, clean_record  # noqa: E402
from oevl.experiment import TRAIN, bare, build_data, heldout, load, tokens  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def run(*args):
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    return subprocess.run([sys.executable, "-m", *args], env=env, capture_output=True, text=True)


class TestCorpus(unittest.TestCase):
    def test_files_clean(self):
        """Every by_source file: inventory only, bare(marked) == bare form, no sentence-level split field."""
        for path in sorted((DATA / "by_source").glob("*.jsonl")):
            rows = load(path)
            self.assertEqual(check(rows), [], path.name)
            self.assertEqual([r["id"] for r in rows if "split" in r], [], path.name)

    def test_clean_record(self):
        r = {"id": "x", "split": "train", "text": "Đā сlēne, þˉa Hr¯wðel.", "text_bare": "Đa сlene, þˉa Hr¯wðel.",
             "tokens": ["Đā", "сlēne", "þˉa", "Hr", "wðel"], "tokens_bare": ["Đa", "сlene", "þˉa", "Hr", "wðel"]}
        self.assertTrue(check([r]))
        log = collections.Counter()
        clean_record(r, log)
        self.assertEqual(r, {"id": "x", "text": "Ðā clēne, þā Hrēðel.", "text_bare": "Ða clene, þa Hreðel.",
                             "tokens": ["Ðā", "clēne", "þā", "Hrēðel"], "tokens_bare": ["Ða", "clene", "þa", "Hreðel"]})
        self.assertEqual(log, {"removed_field_split": 1, "mapped_U+0110_to_U+00D0": 1, "mapped_U+0441_to_U+0063": 1,
                               "repaired_word_þˉa": 1, "repaired_word_Hr¯wðel": 1})
        self.assertEqual(check([r]), [])
        again, log2 = copy.deepcopy(r), collections.Counter()
        self.assertEqual((clean_record(again, log2), log2), (r, {}))  # a second run changes nothing


class TestSplits(unittest.TestCase):
    @unittest.skipUnless((DATA / "brunetti-length.txt").exists(), "Real Tauber source not downloaded")
    def test_no_overlap_after_exclusion(self):
        """What the loaders return for dev and test shares no 6-gram with train."""
        seen = set().union(*(g for t in TRAIN for g in grams(tokens(load(DATA / "by_source" / f"{t}.jsonl"))).values()))
        audited = json.loads((DATA / "leakage_excluded.json").read_text(encoding="utf8"))["excluded"]
        self.assertTrue({"soliloquies", "tauber_beowulf"} <= set(audited))
        for name in audited:
            self.assertEqual(set().union(*grams(heldout(DATA, name)).values()) & seen, set(), name)

    def test_dev_fixture(self):
        """A small fake John: never trained on, overlapping sentence excluded, scored like the Soliloquies."""
        old = "þæt wæs gōd cyning and hē wæs swīðe gōd".split()
        sents = {"train": [old, "se god is gōd".split()], "john_ws": [old, "hē cwæð tō him god is gōd".split()],
                 "soliloquies": ["þū eart gōd".split()], "hs_beowulf": ["gōd cyning".split()],
                 "wyatt_beowulf": ["gōd cyning".split()]}
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            (d / "by_source").mkdir()
            for name in TRAIN + ["john_ws", "soliloquies", "hs_beowulf", "wyatt_beowulf"]:
                with open(d / "by_source" / f"{name}.jsonl", "w", encoding="utf8") as f:
                    for i, ws in enumerate(sents.get(name, sents["train"])):
                        f.write(json.dumps({"id": f"{name}:{i}", "source": name, "tokens": ws,
                                            "tokens_bare": [bare(w) for w in ws]}, ensure_ascii=False) + "\n")
            cols = [""] * 18
            cols[9], cols[10], cols[-1] = "god", ".", "gōd"
            (d / "brunetti-length.txt").write_text("|".join(cols) + "\n", encoding="utf8")
            tauber = str(d / "brunetti-length.txt")

            self.assertEqual(run("oevl.audit_splits", "--data", str(d), "--tauber", tauber).returncode, 0)
            excluded = json.loads((d / "leakage_excluded.json").read_text(encoding="utf8"))["excluded"]
            self.assertEqual(excluded, {"john_ws": ["john_ws:0"], "soliloquies": [], "tauber_beowulf": []})
            train = build_data(d)[0]
            self.assertNotIn("john_ws", {r["source"] for r in train})
            dev = heldout(d, "john_ws")
            self.assertEqual({r["sentence_id"] for r in dev}, {"john_ws:1"})

            p = run("oevl.experiment", "--data", str(d), "--tauber", tauber, "--out", str(d / "out"), "--tests", "john_ws")
            self.assertEqual(p.returncode, 0, p.stderr)
            metrics = json.loads((d / "out" / "metrics.json").read_text(encoding="utf8"))
            self.assertEqual(metrics["john_ws"]["frequency"]["n"], len(dev))
            self.assertTrue((d / "out" / "john_ws_predictions.jsonl").exists())
            self.assertEqual(run("oevl.experiment", "--data", str(d), "--tests", "john").returncode, 2)


if __name__ == "__main__":
    unittest.main()
