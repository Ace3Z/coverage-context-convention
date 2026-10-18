import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from oevl.build_lexicons import clarkhall, clean, wiktionary, write  # noqa: E402


class TestLexicons(unittest.TestCase):
    def test_clean(self):
        self.assertEqual(clean("Gōd"), "gōd")                    # macrons kept, lower-cased
        self.assertEqual(clean("a-nýdan", acute=True), "anȳdan")  # Bosworth-Toller acute -> macron, hyphen dropped
        self.assertEqual(clean("ǽtter", acute=True), "ǣtter")
        self.assertIsNone(clean("strong weak"))                   # multi-word forms are skipped
        self.assertIsNone(clean("ang-adecl2"))                    # template names with digits are skipped

    def test_bt_digraphs(self):
        self.assertEqual(clean("ðeáh", acute=True), "ðēah")       # eá -> ēa, not eā
        self.assertEqual(clean("beón", acute=True), "bēon")       # eó -> ēo
        self.assertEqual(clean("a-fliéman", acute=True), "aflīeman")  # ié -> īe
        self.assertEqual(clean("a-driógan", acute=True), "adrīogan")  # ió -> īo
        self.assertEqual(clean("scéawere", acute=True), "scēawere")   # accent already on the first letter
        self.assertEqual(clean("ge-ánlǽcan", acute=True), "geānlǣcan")  # no digraph across a hyphen
        self.assertEqual(clean("diácon", acute=True), "diācon")   # iá is not a diphthong

    def test_teaching_marks(self):
        self.assertEqual(clean("sċearp"), "scearp")
        self.assertEqual(clean("rīċe"), "rīce")
        self.assertEqual(clean("ġēar"), "gēar")

    def test_junk(self):
        self.assertIsNone(clean("-līce"))                         # partial variant
        self.assertIsNone(clean("æn-"))                           # prefix
        self.assertIsNone(clean("hādianv", acute=True))           # j and v are not Old English letters
        self.assertIsNone(clean("(k)"))
        self.assertIsNone(clean("bœ̄n"))

    def test_clarkhall_ge_markers(self):
        xml = ('<tei:orth property="x">gōd</tei:orth>\n'
               '<tei:orth extent="full" expand="ge-āclian">\n  <tei:lbl expand="ge-">+</tei:lbl>\n'
               '  <tei:seg>āclian</tei:seg>\n</tei:orth>\n'
               '<tei:orth extent="full">\n  <tei:lbl expand="ge-_optional">±</tei:lbl>\n'
               '  <tei:seg>ādlian</tei:seg>\n</tei:orth>\n'
               '<tei:orth>\n  <tei:seg>bān</tei:seg>\n  <tei:lbl rend="sup">2</tei:lbl>\n</tei:orth>\n'
               '<tei:orth extent="part">\n  <tei:seg>(a,</tei:seg>\n</tei:orth>\n')
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "ch.xml"
            p.write_text(xml, encoding="utf8")
            self.assertEqual(clarkhall(p), {"gōd", "āclian", "ādlian", "bān"})

    def test_wiktionary_page_title(self):
        recs = [{"word": "stan", "forms": [{"form": "stān", "tags": ["canonical"]},
                                           {"form": "ang-decl", "tags": ["inflection-template"]},
                                           {"form": "stānas", "tags": ["nominative", "plural"], "source": "declension"}]},
                {"word": "sccarp", "forms": [{"form": "sċarp", "tags": ["canonical"]}]},
                {"word": "and", "forms": []}]
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "k.jsonl"
            p.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in recs), encoding="utf8")
            self.assertEqual(wiktionary(p), {"stān", "stānas", "scarp", "and"})  # no unmarked twin "stan"

    def test_union_sources(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "u.csv"
            write(p, {"gōd", "god"}, {"clarkhall": {"gōd"}, "bt": {"gōd", "god"}})
            self.assertEqual(p.read_text(encoding="utf8").splitlines(), ["marked_form,sources", "god,bt", "gōd,clarkhall bt"])


if __name__ == "__main__":
    unittest.main()
