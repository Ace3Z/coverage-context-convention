import collections
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from oevl import build_corpus as bc  # noqa: E402
from oevl.clean_corpus import clean_record  # noqa: E402


class TestBuildCorpus(unittest.TestCase):
    def test_tokens(self):
        self.assertEqual(bc.tokens("wē Gār-Dena, ācenned'sē 21 hI-e."), ["wē", "GārDena", "ācennedsē", "hIe"])

    def test_editorial(self):
        self.assertEqual(bc.editorial("fugla [wynn], hēt): gehwæþ7res †in ð_am *wine ġe·lēafa"),
                         "fugla wynn, hēt : gehwæþ res in ðam wine ġelēafa")

    def test_clean(self):
        log = collections.Counter()
        self.assertEqual(bc.clean("gebrõþru  āsęcgan ċ ȳ n̄", log), "gebroþru āsecgan c ȳ n")
        self.assertEqual(log, {"dropped_mark_U+0303": 1, "dropped_mark_U+0328": 1, "dropped_mark_U+0307": 1,
                               "dropped_mark_stray_macron": 1})
        self.assertEqual(bc.clean("þå weorc", log), "þå weorc")  # a ring is left for clean_corpus (þå -> þā)
        # Roman numerals go; a lone i (a letter name) and lone capitals stay
        self.assertEqual(bc.clean("ymb xxxi wintra, ān i oððe, Engliscra LXII, I.", log),
                         "ymb wintra, ān i oððe, Engliscra , I.")

    def test_sentences(self):
        log = collections.Counter()
        verses = "1. Æfter Matheus, hēr is bōc. 2. Sōðlīce Abraham gestrȳnde Isaac;\n"
        self.assertEqual(bc.sentences(verses, log), ["Æfter Matheus, hēr is bōc.", "Sōðlīce Abraham gestrȳnde Isaac;"])
        self.assertEqual(bc.sentences("þencest?' Abraham maðelode\nswā hine Drihten hēt):", log),
                         ["þencest?' Abraham maðelode swā hine Drihten hēt :"])
        # names without an Old English letter go when the paragraph is Old English (Matthew 22:32)
        self.assertEqual(bc.sentences("32. Ic eom Abrahames God, and Isaaces God, and Iacobes God? Nys God nā "
                                      "dēadra, ac lybbyndra.", log), ["Nys God nā dēadra, ac lybbyndra."])
        self.assertEqual(log["dropped_sentence_mixed"], 1)
        self.assertEqual(bc.sentences("[From the Parker MS. of the Chronicle (A).]\n", log), [])
        self.assertEqual(bc.sentences("G. Heald þonne hys bebodu.", log, speaker=r"^\s*[AGR]\.\s+"),
                         ["Heald þonne hys bebodu."])

    def test_line_lang(self):
        self.assertEqual(bc.line_lang("3. Philippus, and Bartholomeus; Thomas, and Matheus, publicanus; and Iacobus "
                                      "Alphei, and Taddeus."), "eng")
        self.assertIsNone(bc.line_lang("25. And him fyligdon mycele menigu fram Galilea, and fram Decapoli, and fram "
                                       "Hierusalem, and fram Iudea, and fram begeondan Iordanen."))
        self.assertEqual(bc.line_lang("lifgan in lisse  lucis et pacis,"), "lat")
        self.assertIsNone(bc.line_lang("19. And þæt is Iohannes gewitnes:"))  # thorn: Old English
        self.assertIsNone(bc.line_lang("of heofonum geseald."))  # too short to judge

    def test_blocks(self):
        html = ('<div class="mw-parser-output"><div class="prp-pages-output">'
                '<div class="wst-center"><p>CAPUT I</p></div>'
                '<p>1. On frymðe<sup class="reference">[1]</sup> wæs<br>\nWord.</p>'
                '25. and for þām him næs nān þearf.<div class="wst-center"><p>CAPUT III</p></div>'
                '<table><tr><td><p>Ic Ælfrīc</p></td><td><p>Ego Ælfricus</p></td></tr></table>'
                '<div class="ws-poem-stanza"><span>Hwæt! wē Gār-Dena<span class="wst-gap"></span>in gēar-dagum'
                '</span></div></div></div>')
        self.assertEqual(bc.blocks(html), [("center", "CAPUT I"), ("p", "1. On frymðe wæs\n\nWord."),
                                           ("p", "25. and for þām him næs nān þearf."), ("center", "CAPUT III"),
                                           ("p", "Ic Ælfrīc"), ("p", "Hwæt! wē Gār-Dena in gēar-dagum")])

    def test_section_and_roman(self):
        title = "Evangelium Secundum Mattheum: the Gospel of Saint Matthew in West-Saxon/Chapter 1"
        self.assertEqual(bc.section(title), "Evangelium_Secundum_Mattheum__the_Gospel_of_Saint_Matthew_in_West_Saxon_Chapter_1")
        self.assertEqual(bc.section("Bright's Anglo-Saxon Reader/Ælfric's Preface to Genesis"),
                         "Bright_s_Anglo_Saxon_Reader_AElfric_s_Preface_to_Genesis")
        self.assertEqual([bc.roman(x) for x in ("I", "IV", "XIV", "XXVIII")], [1, 4, 14, 28])

    @unittest.skipUnless((ROOT / "data/raw/corpus/matthew_ws/pages.json").exists(), "Wikisource download not cached")
    def test_matthew_rebuilds_byte_for_byte(self):
        rows, _, _ = bc.build("matthew_ws", ROOT / "data/raw/corpus")
        rows = [clean_record(r, collections.Counter()) for r in rows]  # the committed files are cleaned
        data = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows).encode("utf8")
        self.assertEqual(data, (ROOT / "data/by_source/matthew_ws.jsonl").read_bytes())


if __name__ == "__main__":
    unittest.main()
