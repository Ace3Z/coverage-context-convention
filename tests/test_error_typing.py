import random
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from oevl.error_typing import draw  # noqa: E402


class TestErrorTyping(unittest.TestCase):
    def test_draw_pools_soliloquies_then_tauber(self):
        # given in the other order, the pool is still soliloquies first, then tauber_beowulf
        pool = {"tauber_beowulf": [f"t{i}" for i in range(300)], "soliloquies": [f"s{i}" for i in range(250)]}
        self.assertEqual(draw(pool), random.Random(1).sample(pool["soliloquies"] + pool["tauber_beowulf"], 200))


if __name__ == "__main__":
    unittest.main()
