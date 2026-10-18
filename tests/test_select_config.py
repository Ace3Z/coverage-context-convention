import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from oevl.select_config import OUT, select  # noqa: E402


def runs(z, lr, ep, acc):
    return [{"zupitza": z, "lr": lr, "epochs": ep, "seed": s, "dev_lexamb": acc, "dev_lexamb_n": 1,
             "dev_lexamb_shared": acc, "collapsed": False} for s in (1, 2)]


class TestSelect(unittest.TestCase):
    def test_ties_go_to_fewer_epochs_then_zupitza_then_lower_lr(self):
        rows = (runs(False, 3e-4, 5, 96.0) + runs(False, 3e-4, 3, 95.95) + runs(True, 3e-4, 3, 95.92)
                + runs(True, 1e-4, 3, 95.9) + runs(True, 1e-4, 1, 95.89))  # the last is 0.11 below the best
        scored, tied, choice = select(rows)
        self.assertEqual(scored[0]["mean"], 96.0)
        self.assertEqual(len(tied), 4)
        self.assertEqual((choice["zupitza"], choice["lr"], choice["epochs"]), (True, 1e-4, 3))

    def test_committed_choice(self):
        rec = json.loads(OUT.read_text(encoding="utf8"))
        scored, tied, choice = select(rec["runs"])
        self.assertEqual(scored, rec["configs"])
        self.assertEqual({k: choice[k] for k in rec["choice"]}, rec["choice"])
        self.assertEqual(len(tied), 1 + len(rec["tied_with_choice"]))


if __name__ == "__main__":
    unittest.main()
