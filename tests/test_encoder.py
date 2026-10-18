import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from oevl.encoder_experiment import limit, make_input  # noqa: E402

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


class TestLimit(unittest.TestCase):
    def test_fixed_subset(self):
        ex = [{"id": str(j)} for j in range(1000)]
        a, b = limit(ex, 100), limit(list(ex), 100)
        self.assertEqual(a, b)                                   # same subset on every call
        self.assertEqual(len(a), 100)
        self.assertEqual([int(r["id"]) for r in a], sorted(int(r["id"]) for r in a))  # corpus order kept
        self.assertIs(limit(ex, None), ex)
        self.assertIs(limit(ex, 5000), ex)


if __name__ == "__main__":
    unittest.main()
