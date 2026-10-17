import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from build_lexicons import clean  # noqa: E402


class TestLexicons(unittest.TestCase):
    def test_clean(self):
        self.assertEqual(clean("Gōd"), "gōd")                    # macrons kept, lower-cased
        self.assertEqual(clean("a-nýdan", acute=True), "anȳdan")  # Bosworth-Toller acute -> macron, hyphen dropped
        self.assertEqual(clean("ǽtter", acute=True), "ǣtter")
        self.assertIsNone(clean("strong weak"))                   # multi-word forms are skipped
        self.assertIsNone(clean("ang-adecl2"))                    # template names with digits are skipped


if __name__ == "__main__":
    unittest.main()
