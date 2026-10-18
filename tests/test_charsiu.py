import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from oevl.charsiu_g2p import PREFIX  # noqa: E402


class TestCharsiu(unittest.TestCase):
    def test_prefix(self):
        self.assertEqual(PREFIX, "<ang>:")  # as data_utils.py, no space


if __name__ == "__main__":
    unittest.main()
