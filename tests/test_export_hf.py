import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from oevl.experiment import bare, build_data, heldout, load, norm, pattern  # noqa: E402
from oevl.export_hf import SPLITS, export  # noqa: E402

DATA = ROOT / "data"
TAUBER = DATA / "brunetti-length.txt"


@unittest.skipUnless(TAUBER.exists(), "Real Tauber source not downloaded")
class TestExportHF(unittest.TestCase):
    def test_export(self):
        """Every sentence is exported once, token counts match the manifest and the experiment loaders."""
        manifest = json.loads((DATA / "manifest.json").read_text(encoding="utf8"))["sources"]
        audit = json.loads((DATA / "leakage_excluded.json").read_text(encoding="utf8"))["excluded"]
        train, _, hs, wy = build_data(DATA)
        with tempfile.TemporaryDirectory() as d:
            counts = export(DATA, TAUBER, d)
            files = {s: load(Path(d) / f"{s}.jsonl") for s in SPLITS}
            card = (Path(d) / "README.md").read_text(encoding="utf8")
        fields = set(files["train"][0])
        self.assertNotIn("split", fields)
        for split, rows in files.items():
            for r in rows:
                self.assertEqual(set(r), fields, r["id"])
                self.assertEqual([bare(w) for w in r["tokens"]], [norm(b) for b in r["tokens_bare"]], r["id"])
                self.assertEqual(r["length_patterns"], [pattern(w) for w in r["tokens"]], r["id"])
            corpus = [s for s in SPLITS[split] if s in manifest]
            if corpus:  # the corpus files: every token of every source the manifest counts
                self.assertEqual(counts[split]["tokens"], sum(manifest[s]["tokens"] for s in corpus), split)
                self.assertEqual(counts[split]["sentences"], sum(manifest[s]["units"] for s in corpus), split)
        ids = [r["id"] for rows in files.values() for r in rows]
        self.assertEqual(len(ids), len(set(ids)))
        # flagged = exactly the audit's exclusions, and the rest is what the experiments score
        self.assertEqual({r["id"] for rows in files.values() for r in rows if r["leakage_excluded"]},
                         {s for v in audit.values() for s in v})
        unflagged = {s: sum(len(r["tokens"]) for r in rows if not r["leakage_excluded"]) for s, rows in files.items()}
        self.assertEqual(unflagged, {"train": len(train), "dev": len(heldout(DATA, "john_ws")),
                                     "test_prose": len(heldout(DATA, "soliloquies")),
                                     "test_verse": len(heldout(DATA, "tauber_beowulf")), "agreement": len(hs) + len(wy)})
        self.assertIn("license: cc-by-sa-4.0", card)
        self.assertNotIn("$", card)
        self.assertIn(f"| {counts['test_verse']['tokens']:,} |", card)


if __name__ == "__main__":
    unittest.main()
