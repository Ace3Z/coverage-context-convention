import json
import os
import sys
import tempfile
import unittest
import zipfile
from argparse import Namespace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from oevl.pilot import pack, restore, validate  # noqa: E402

ARGS = Namespace(epochs=1, max_bytes=128, seed=1, batch=32, lr=3e-4, train_limit=10000, tiny=True)


def first_ids(test, n=5):
    with open(ROOT / "results" / "wikipron" / f"{test}_predictions.jsonl", encoding="utf8") as f:
        return [json.loads(next(f))["id"] for _ in range(n)]


def fake_arm(out, arm="word", **override):
    out.mkdir(parents=True, exist_ok=True)
    arms = {arm: {t: {"accuracy": {"n": 5, "accuracy": 1.0}} for t in ("soliloquies", "tauber_beowulf")}}
    metrics = {"epochs": 1, "max_bytes": 128, "seed": 1, "batch": 32, "lr": 0.0003, "train_limit": 10000,
               "train_examples": 256, "train_subset_sha256": "abc", "arms": arms, **override}
    (out / "metrics.json").write_text(json.dumps(metrics))
    for t in ("soliloquies", "tauber_beowulf"):
        (out / f"{t}_predictions.jsonl").write_text(
            "".join(json.dumps({"id": i, f"encoder_{arm}": "0"}) + "\n" for i in first_ids(t)))


class TestValidate(unittest.TestCase):
    def test_accepts_a_good_arm(self):
        with tempfile.TemporaryDirectory() as d:
            fake_arm(Path(d) / "P2")
            self.assertEqual(validate(Path(d) / "P2", "word", ARGS), "abc")

    def test_rejects(self):
        for override, arm in (({"max_bytes": 256}, "word"), ({"train_limit": None}, "word"), ({}, "left")):
            with self.subTest(override=override, arm=arm), tempfile.TemporaryDirectory() as d:
                fake_arm(Path(d) / "P2", **override)
                with self.assertRaises((AssertionError, KeyError)):
                    validate(Path(d) / "P2", arm, ARGS)

    def test_rejects_full_run_with_wrong_size(self):
        with tempfile.TemporaryDirectory() as d:
            fake_arm(Path(d) / "P2")
            with self.assertRaises(AssertionError):
                validate(Path(d) / "P2", "word", Namespace(**{**vars(ARGS), "tiny": False}))

    def test_rejects_duplicate_ids(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "P2"
            fake_arm(out)
            p = out / "soliloquies_predictions.jsonl"
            p.write_text(p.read_text() + p.read_text().splitlines()[0] + "\n")
            with self.assertRaises(AssertionError):
                validate(out, "word", ARGS)


class TestValidateMore(unittest.TestCase):
    def test_full_run_rejects_a_subset_of_ids(self):
        with tempfile.TemporaryDirectory() as d:
            fake_arm(Path(d) / "P2", train_examples=10000)
            with self.assertRaises(AssertionError):  # only 5 of the test ids are present
                validate(Path(d) / "P2", "word", Namespace(**{**vars(ARGS), "tiny": False}))

    def test_rejects_wrong_n(self):
        with tempfile.TemporaryDirectory() as d:
            fake_arm(Path(d) / "P2")
            m = json.loads((Path(d) / "P2" / "metrics.json").read_text())
            m["arms"]["word"]["soliloquies"]["accuracy"]["n"] = 6
            (Path(d) / "P2" / "metrics.json").write_text(json.dumps(m))
            with self.assertRaises(AssertionError):
                validate(Path(d) / "P2", "word", ARGS)


def finished(base, arm, tag="x"):
    fake_arm(base / arm)
    (base / arm / "COMPLETE.json").write_text(json.dumps({"id": arm, "tag": tag}))


class TestArchive(unittest.TestCase):
    def test_pack_then_restore(self):
        with tempfile.TemporaryDirectory() as d:
            base = Path(d) / "run" / "encoder_pilot_128_results"
            finished(base, "P2")
            archive = pack(base)
            fresh = Path(d) / "new" / "encoder_pilot_128_results"
            fresh.mkdir(parents=True)
            self.assertEqual(restore(archive, fresh), ["P2"])
            self.assertEqual((fresh / "P2" / "metrics.json").read_text(), (base / "P2" / "metrics.json").read_text())

    def test_restore_merges_and_keeps_local_finished_arms(self):
        with tempfile.TemporaryDirectory() as d:
            old = Path(d) / "old" / "encoder_pilot_128_results"
            finished(old, "P1", "old")
            finished(old, "P2", "old")
            fake_arm(old / "P3")                          # unfinished in the archive: never restored
            archive = pack(old)
            base = Path(d) / "new" / "encoder_pilot_128_results"
            finished(base, "P1", "local")                 # finished locally: kept
            fake_arm(base / "P2")                         # interrupted locally: replaced by the finished copy
            self.assertEqual(restore(archive, base), ["P2"])
            self.assertIn("local", (base / "P1" / "COMPLETE.json").read_text())
            self.assertIn("old", (base / "P2" / "COMPLETE.json").read_text())
            self.assertFalse((base / "P3").exists())

    def test_restore_from_an_extracted_folder(self):
        with tempfile.TemporaryDirectory() as d:
            old = Path(d) / "input" / "encoder_pilot_128_results"
            finished(old, "P4")
            base = Path(d) / "work" / "encoder_pilot_128_results"
            base.mkdir(parents=True)
            self.assertEqual(restore(old, base), ["P4"])
            self.assertTrue((base / "P4" / "COMPLETE.json").exists())

    def test_restore_refuses_paths_outside_the_folder(self):
        with tempfile.TemporaryDirectory() as d:
            archive = Path(d) / "evil.zip"
            with zipfile.ZipFile(archive, "w") as z:
                z.writestr("encoder_pilot_128_results/../escaped.txt", "x")
                z.writestr("other_folder/P1/COMPLETE.json", "{}")
                z.writestr("encoder_pilot_128_results/P1/COMPLETE.json", "{}")
                z.writestr("encoder_pilot_128_results/P1/ok.txt", "x")
            base = Path(d) / "out" / "encoder_pilot_128_results"
            base.mkdir(parents=True)
            restore(archive, base)
            self.assertTrue((base / "P1" / "ok.txt").exists())
            self.assertFalse((Path(d) / "out" / "escaped.txt").exists())
            self.assertFalse((Path(d) / "out" / "other_folder").exists())


class TestCli(unittest.TestCase):
    def test_encoder_rejects_bad_arguments(self):
        import subprocess
        env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
        for args in (["--arms", "sentense"], ["--train-limit", "0"]):
            p = subprocess.run([sys.executable, "-m", "oevl.encoder_experiment", *args], cwd=ROOT, env=env,
                               capture_output=True, text=True)
            self.assertEqual(p.returncode, 2, (args, p.stderr))


if __name__ == "__main__":
    unittest.main()
