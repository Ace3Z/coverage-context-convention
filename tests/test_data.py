import collections
import copy
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from oevl.clean_corpus import check, clean_record  # noqa: E402
from oevl.experiment import load  # noqa: E402

DATA = Path(__file__).resolve().parents[1] / "data"


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


if __name__ == "__main__":
    unittest.main()
