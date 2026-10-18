import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import oevl.charsiu_g2p as charsiu  # noqa: E402
from oevl.charsiu_g2p import PREFIX  # noqa: E402


class TestCharsiu(unittest.TestCase):
    def test_prefix(self):
        self.assertEqual(PREFIX, "<ang>:")  # as data_utils.py, no space

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
