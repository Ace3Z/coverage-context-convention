"""Build data/by_source/<source>.jsonl from the editions.

Each source is fetched from its original site (Wikisource MediaWiki API, Project Gutenberg HTML) into
data/raw/corpus/<source>/ (gitignored) and goes through the same steps:
  1. paragraphs from the HTML: <p> and verse stanzas on Wikisource, the elements named in SOURCES on Gutenberg.
     Centred paragraphs, footnotes, footnote calls and margin line numbers are not text
  2. a line-level language filter: a line of four or more words with no length mark and no thorn or eth is
     dropped when it reads as English or Latin by function words, and a bracketed editor's note is English
  3. digits, brackets, daggers, asterisks, underscores and middle dots dropped
  4. sentence split after . ? ! followed by white space, inside one paragraph
  5. NFC, combining marks other than the macron and the ring dropped (ę -> e, ċ -> c), Roman numerals dropped.
     A ring is left for clean_corpus, which reads John's þå as þā
  6. sentence filters: in a paragraph with Old English sentences, a sentence of three or more words with no Old
     English letter and two or more capitalised words is dropped ("mixed": names and editors' notes, such as
     'Ic eom Abrahames God, and Isaaces God, and Iacobes God?'), and so is any sentence with four or more Latin
     function words
  7. tokens: letter runs, joined over inner hyphens and apostrophes, which are then removed
The two Beowulf editions skip the filters of steps 2 and 6, as the committed files show.

The original extraction scripts are not in the repository. These rules were inferred from the committed files
and their cleaning counts in data/manifest.json. With them, after clean_corpus, matthew_ws, bright_reader,
physiologus, hs_beowulf and john_ws rebuild byte for byte. The other four (soliloquies, sweet_primer, zupitza_grammar
and wyatt_beowulf) differ. In the first three this is mostly where the original language filter dropped lines these
rules keep (and the reverse), and in sweet_primer also in the 12 sentences the original removed as overlapping
Matthew. wyatt_beowulf skips the filters, so its differences have other causes.

Usage: python -m oevl.build_corpus --source john_ws --out data/by_source [--raw data/raw/corpus]
then python -m oevl.clean_corpus, which removes the old sentence-level split field and applies the character fixes.
clean_corpus reads only the repository's data/by_source and data/manifest.json, so it sees the new file only when
--out is that data/by_source.
The manifest next to --out (<out>/../manifest.json) gets the source's entry, with URL, retrieval date and
SHA-256 of every input.
"""
import argparse
import collections
import datetime
import hashlib
import json
import re
import unicodedata
import urllib.parse
import urllib.request
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
API = "https://en.wikisource.org/w/api.php"
UA = {"User-Agent": "oevl-corpus/1.0 (UZH course project; mahbodtajdini@gmail.com)"}
WS_LICENCE = "CC BY-SA 4.0 (Wikisource transcription of a public-domain edition)"
SOURCES = {
    "matthew_ws": dict(index="Gospel of Saint Matthew in West-Saxon.djvu", pages=(14, 160),
                       title="Evangelium Secundum Mattheum: the Gospel of Saint Matthew in West-Saxon",
                       work="Gospel of Matthew (West Saxon)", editor="Bright", year=1904, genre="prose",
                       split="blocks", role="train"),
    # pages 158-161 are the editor's notes, so the text stops at 157
    "john_ws": dict(index="Gospel of Saint John in West-Saxon.djvu", pages=(45, 157),
                    title="The Gospel of Saint John in West-Saxon",
                    work="Gospel of John (West Saxon)", editor="Bright & Harris", year=1904, genre="prose",
                    split="dev", role="dev"),
    "bright_reader": dict(root="Bright's Anglo-Saxon Reader", subpages=[
        "From the Gospels: St. Mark, Chap. IV", "Orpheus and Eurydice", "Account of the Poet Cædmon",
        "Cynewulf and Cyneheard", "Wars of Alfred the Great", "Alfred's Preface to the Pastoral Care",
        "From the Pastoral Care", "The Voyages of Ohthere and Wulfstan",
        "It is better to suffer an Injury than to inflict one", "Providence and Fate", "The Nature of God",
        "The Conversion of Edwin", "A Blickling Homily", "Ælfric's Homily on the Assumption of St. John the Apostle",
        "Ælfric's Homily on St. Gregory the Great", "Ælfric's Life of King Oswald", "Ælfric's Preface to Genesis",
        "The Legend of St. Andrew", "The Harrowing of Hell", "Cædmon's Genesis: The Offering of Isaac",
        "The Battle of Brunanburh", "The Wanderer", "The Phœnix"],
        work="Anglo-Saxon Reader (selections)", editor="Bright", year=1913, genre="mixed", split="blocks",
        role="train"),
    "soliloquies": dict(root="King Alfred's Old English Version of St. Augustine's Soliloquies",
                        subpages=["King Alfred's Preface", "Book 1", "Book 2", "Book 3"],
                        work="Alfred's Soliloquies", editor="Hargrove", year=1902, genre="prose", split="blocks",
                        role="test", speaker=r"(?:^\s*|(?<=[.?!]\s))\[?[AGR]\.\]?\s+"),
    "wyatt_beowulf": dict(root="Beowulf (Wyatt)",
                          subpages=[f"Beowulf {i:02d}" for i in range(44) if i != 30] + ["The Fight at Finnsburg"],
                          work="Beowulf", editor="Wyatt", year=1894, genre="verse", split="beowulf_wyatt",
                          role="agreement", filters=False, fragment="Finnsburg"),
    # Latin is in italics or small capitals, Zupitza's own sub-headings are h3, transcriber's notes are corrections
    "zupitza_grammar": dict(ebook=45861, units=("div.primary",), heads=("h2",),
                            skip=(".ital", "i", ".smallroman", ".smallcaps", "h3", ".correction"),
                            start='<div class="holder header">', end="<h2>INCIPIVNT NOMINA",
                            circumflex=True, work="Ælfric's Grammar", editor="Zupitza", year=1880, genre="prose",
                            split="blocks", role="train"),
    "sweet_primer": dict(ebook=34316, units=("div.poem",), heads=("h3",), join=True,
                         start="<h2>TEXTS.</h2>", end="<h2>NOTES.</h2>", work="Anglo-Saxon Primer (texts)",
                         editor="Sweet", year=1905, genre="mixed", split="train", role="train"),
    "physiologus": dict(ebook=14529, units=("td.left",), heads=("h1",),
                        work="Old English Physiologus", editor="Cook & Pitman", year=1921, genre="verse",
                        split="train", role="train"),
    "hs_beowulf": dict(ebook=9701, units=("div.fitt", "div#back"), heads=("h1", "h2"), filters=False,
                       fragment="FINNSBURG", end="<h2>LIST OF NAMES; NOTES; AND GLOSSARY.</h2>",
                       work="Beowulf", editor="Harrison & Sharp", year=1883, genre="verse", split="beowulf_hs",
                       role="agreement"),
}
LONG, SHORT = "āēīōūȳǣĀĒĪŌŪȲǢ", "aeiouyæAEIOUYÆ"
BARE = str.maketrans(LONG, SHORT)
OE = set(LONG + "þðæÞÐÆĐđ")
WORD = re.compile(r"[^\W\d_]+(?:[-'][^\W\d_]+)*")
ROMAN = re.compile(r"\b(?:(?!i\b)[ivx]+|[IVXLC]{2,})\b")  # lone i and lone capitals stay: 'ān i oððe', 'I.'
# English function words. Not be, for, in, is, on, to, was: Old English spellings too ('And on hiera dagum')
# and, of and an stay although Old English writes them too
EN = set("the of and are were been from with by at as that this which it its an a or not but".split())
# Latin function words, plus forms of deus and the incipit/explicit of headings
# not ac, in, me, se, sit, te: Old English words too
LA = set("et est sunt non ad cum de quod qui quae quam ut sed erat apud ex ab per nec neque enim autem ergo igitur si "
         "nisi tamen vel aut quid quia quo sic ita iam jam nunc hoc haec id ego tu mihi tibi sibi nos vos esse deus "
         "deum dei deo atque etiam incipit explicit".split())
VOID = {"br", "hr", "img", "link", "meta", "wbr", "input", "area", "base", "col", "embed", "source", "track"}
BLOCK = {"p", "div", "ol", "ul", "li", "table", "dl", "blockquote", "h1", "h2", "h3", "h4", "h5", "h6"}
WS_UNITS = ("p", "div.ws-poem-stanza")
WS_SKIP = ("style", "script", "sup", ".ws-noexport", ".wst-pline", ".references")


def match(specs, tag, cls, id_):
    """True if the element fits one of the specs: 'tag', '.class', 'tag.class' or 'tag#id'.

    cls is the list of the element's classes and id_ its id or None."""
    for spec in specs:
        t, c, i = re.fullmatch(r"(\w*)(?:\.([\w-]+))?(?:#([\w-]+))?", spec).groups()
        if t in ("", tag) and (not c or c in cls) and (not i or i == id_):
            return True
    return False


class Blocks(HTMLParser):
    """HTML parser that collects paragraphs in document order as (kind, text) in self.out.

    kind is 'p', 'center' (a paragraph inside a wst-center element, which is a heading on Wikisource) or 'head'
    (an element whose tag is in heads, its text nodes stripped and joined by a space). A paragraph is the outermost
    element matching units (by default a <p> or a verse stanza). units and skip are specs for match(). A <br> or
    the end of a block inside a paragraph ends a line, and a wst-gap element (the caesura) adds a space. Elements
    matching skip (by default styles, scripts, footnote calls, footnotes, margin line numbers and ws-noexport
    elements) are left out with everything inside them.

    Text left outside any <p> directly in the page body (an element of class prp-pages-output or mw-parser-output)
    is a paragraph too: ProofreadPage leaves the last paragraph of a scan page unwrapped when the next page starts
    with a centred heading. Of a two-column table (the Soliloquies print Augustine's Latin beside Alfred's text)
    only the first column is read: every <td> after the first in its row is skipped.
    """

    def __init__(self, units=WS_UNITS, skip=WS_SKIP, heads=()):
        """Start an empty parse. units and skip are specs for match(), heads are bare tag names.

        State: out (the result), cur (text pieces of the open paragraph or None), head (text nodes of the open
        heading or None), stack (open elements as (tag, skipped, has class wst-center, is the page body, opened a
        paragraph, opened a heading)), loose (text pieces directly in the page body) and cell (the number of <td>
        opened so far in the current row)."""
        super().__init__(convert_charrefs=True)
        self.units, self.skip, self.heads = units, skip, heads
        self.out, self.cur, self.head, self.stack, self.loose, self.cell = [], None, None, [], [], 0

    def flush(self):
        """Append the loose body text gathered so far to out as a 'p' paragraph, unless it is blank, and reset it."""
        if "".join(self.loose).strip():
            self.out.append(("p", "".join(self.loose)))
        self.loose = []

    def handle_starttag(self, tag, attrs):
        """Open an element. A <br> adds a line break to the open paragraph, else to the loose text, even inside a
        skipped element. Void elements stop there. Any other element first flushes the loose text when it is a
        block and is pushed on the stack.

        An element is skipped when it matches skip, sits inside a skipped element or is a <td> after the first in
        its row. It opens a paragraph when it matches units outside any skipped element and open paragraph, and a
        heading whenever its tag is in heads (inside a skipped element that heading comes out empty). A wst-gap
        element adds a space to the open paragraph."""
        if tag == "br":
            (self.cur if self.cur is not None else self.loose).append("\n")
        if tag in VOID:
            return
        if tag in BLOCK:
            self.flush()
        self.cell = 0 if tag == "tr" else self.cell + (tag == "td")
        cls, id_ = (dict(attrs).get("class") or "").split(), dict(attrs).get("id")
        skip = match(self.skip, tag, cls, id_) or (tag == "td" and self.cell > 1)
        body = "prp-pages-output" in cls or "mw-parser-output" in cls
        skip = skip or any(s[1] for s in self.stack)
        unit, head = not skip and self.cur is None and match(self.units, tag, cls, id_), tag in self.heads
        self.stack.append((tag, skip, "wst-center" in cls, body, unit, head))
        if unit:
            self.cur = []
        if head:
            self.head = []
        if "wst-gap" in cls and self.cur is not None:  # the caesura of a verse line
            self.cur.append(" ")

    def handle_endtag(self, tag):
        """Close an element. End tags of void elements and of elements that are not open are ignored. A block
        element first flushes the loose text. The stack is popped down to the innermost open element with this tag,
        so unclosed children close with it: a closed heading or paragraph is appended to out, and a block element
        closed inside a paragraph ends a line."""
        if tag in VOID or all(s[0] != tag for s in self.stack):
            return
        if tag in BLOCK:
            self.flush()
        while self.stack:
            t, _, _, _, unit, head = self.stack.pop()
            if head:  # its text nodes joined by a space, as in "THE ATTACK IN FINNSBURG. ‡"
                self.out.append(("head", " ".join(x.strip() for x in self.head if x.strip())))
                self.head = None
            elif unit:
                self.out.append(("center" if any(s[2] for s in self.stack) else "p", "".join(self.cur)))
                self.cur = None
            elif self.cur is not None and t in BLOCK:
                self.cur.append("\n")
            if t == tag:
                break

    def handle_data(self, data):
        """Send a text node to the open heading, else the open paragraph, else the loose text when the nearest
        enclosing block element is the page body. Text inside a skipped element and any other text is dropped."""
        if any(s[1] for s in self.stack):
            return
        if self.head is not None:
            self.head.append(data)
        elif self.cur is not None:
            self.cur.append(data)
        elif next((s[3] for s in reversed(self.stack) if s[0] in BLOCK), False):
            self.loose.append(data)


def blocks(html, **kw):
    """Paragraphs of html as a list of (kind, text) pairs from Blocks, which takes the keyword arguments.

    Loose body text left at the end is flushed, and a paragraph still open at the end (cut off by a Gutenberg end
    marker) is kept as a 'p'. A heading still open at the end is lost."""
    parser = Blocks(**kw)
    parser.feed(html)
    parser.close()
    parser.flush()
    return parser.out + ([("p", "".join(parser.cur))] if parser.cur else [])  # a paragraph cut off by an end marker


def line_lang(line):
    """'eng' or 'lat' for a line of 4+ words with no length mark and no thorn or eth that reads as English or
    Latin, else None. Ash is no evidence: the English notes spell Cædmon and Ælfric.

    The four-word minimum counts letter runs (WORD). The shares are over the words split at white space, punctuation
    included, so 'et,' is no Latin word. The line is Latin when at least 15% are Latin function words (LA) and no
    fewer than the English ones, else English when at least 30% are English function words (EN)."""
    if set(line) & set(LONG + "þðÞÐĐđ") or len(WORD.findall(line)) < 4:
        return None
    words = line.split()
    la = sum(w.lower() in LA for w in words) / len(words)
    en = sum(w.lower() in EN for w in words) / len(words)
    return "lat" if la >= 0.15 and la >= en else "eng" if en >= 0.3 else None


def unmarked(text):
    """text with the macron of every long vowel removed (NFD, drop U+0304, NFC), so line_lang can read it."""
    return unicodedata.normalize("NFC", unicodedata.normalize("NFD", text).replace("\u0304", ""))


def editorial(text):
    """Drop digits, brackets, daggers, asterisks, underscores and Sweet's hyphen dots, before the sentence split.

    Square brackets go without a trace ('fugla [wynn],' -> 'fugla wynn,'), round ones and digits leave a space
    ('hēt):' -> 'hēt :', 'gehwæþ7res' -> 'gehwæþ res'), as in the committed files."""
    return re.sub(r"[\[\]†_*·]", "", re.sub(r"[\d()]", " ", text))


def clean(text, log):
    """Drop combining marks except a macron NFC joins to its letter (a vowel, æ or g) and any ring, then Roman
    numerals (ROMAN), and return the text in NFC. Runs after the split, so a stray mark after a full stop (a caron in
    Wyatt's 'geweorces.') keeps two sentences together, as in the files.

    The text is decomposed, every combining mark other than the macron and the ring is dropped, and the rest is
    recomposed. A macron NFC cannot join (as on n) is dropped too. A ring stays even where it does not join, and
    clean_corpus repairs John's þå to þā but refuses any other ring. Each dropped mark is counted in log, a Counter,
    under dropped_mark_U+XXXX or dropped_mark_stray_macron. White space is collapsed to single spaces and the ends
    stripped, so a removed numeral can leave a space before punctuation ('Engliscra ,')."""
    out = []
    for ch in unicodedata.normalize("NFD", text):
        if unicodedata.combining(ch) and ch not in "\u0304\u030a":
            log[f"dropped_mark_U+{ord(ch):04X}"] += 1
            continue
        out.append(ch)
    text = unicodedata.normalize("NFC", "".join(out))
    if "\u0304" in text:
        log["dropped_mark_stray_macron"] += text.count("\u0304")
        text = text.replace("\u0304", "")
    return re.sub(r"\s+", " ", ROMAN.sub("", text)).strip()


def sentences(par, log, speaker=None, filters=True):
    """The kept sentences of one paragraph, after the language filters (unless filters is False) and cleaning.
    speaker is a regex for the speaker label of a line of dialogue (A. or [G.]), at a line start or after a sentence end.

    The paragraph is NFC-normalised, zero-width spaces are removed and every speaker label is cut. A line
    that line_lang reads as English or Latin is dropped, and so is every line of a paragraph that is one bracketed
    note with no length mark, or one that line_lang reads as English once its macrons are removed (unmarked). Blank lines are skipped. The kept lines are joined by spaces, passed
    through editorial, split after . ? or ! followed by white space and cleaned, and a piece with no letter is no
    sentence. In a paragraph where some sentence has an Old English letter (OE), a sentence of three or more words
    with none and two or more capitalised words is dropped. In any paragraph a sentence with four or more Latin
    function words (WORD letter runs) is dropped. With filters False none of the line or sentence drops apply.
    Drops are counted in log: dropped_line_eng/lat and dropped_words_eng/lat (words split at white space),
    dropped_sentence_mixed and dropped_sentence_latin_words, next to the dropped_mark counts of clean."""
    par = unicodedata.normalize("NFC", par).replace("\u200b", "")
    if speaker:
        par = re.sub(speaker, "", par, flags=re.M)
    # editor's note, e.g. [From the MS.]; one that names a marked word (the Bēowulf MS.) counts if it reads as English
    note = re.fullmatch(r"\s*\[[^\]]*\]\s*", par) and (not set(par) & set(LONG) or line_lang(unmarked(par)) == "eng")
    lines = []
    for line in (x for x in par.split("\n") if x.strip()):
        lang = filters and ("eng" if note else line_lang(line))
        if lang:
            log[f"dropped_line_{lang}"] += 1
            log[f"dropped_words_{lang}"] += len(line.split())
        else:
            lines.append(line)
    sents = [s for s in (clean(x, log) for x in re.split(r"(?<=[.?!])\s+", editorial(" ".join(lines))))
             if WORD.search(s)]
    oe = filters and any(set(s) & OE for s in sents)
    kept = []
    for s in sents:
        words = WORD.findall(s)
        if oe and not set(s) & OE and len(words) >= 3 and sum(w[0].isupper() for w in words) >= 2:
            log["dropped_sentence_mixed"] += 1
        elif filters and sum(w.lower() in LA for w in words) >= 4:
            log["dropped_sentence_latin_words"] += 1
        else:
            kept.append(s)
    return kept


def tokens(text):
    """Letter runs, joined over inner hyphens and apostrophes, which are then removed: Gār-Dena -> GārDena."""
    return [re.sub(r"[-']", "", w) for w in WORD.findall(text)]


def section(title):
    """Section id of a Wikisource title: Æ, æ and œ written AE, ae and oe, then every character other than an ASCII
    letter, digit or full stop replaced by an underscore."""
    return re.sub(r"[^A-Za-z0-9.]", "_", title.replace("Æ", "AE").replace("æ", "ae").replace("œ", "oe"))


def roman(s):
    """Value of a Roman numeral written with I, V, X and L, subtractive pairs such as IV included."""
    v = [{"I": 1, "V": 5, "X": 10, "L": 50}[c] for c in s]
    return sum(-a if a < b else a for a, b in zip(v, v[1:] + [0]))


def gospel(html, title, log):
    """(section, sentence) pairs of a gospel, as a generator. Chapters start at the centred 'CAPUT n' headings.

    Other centred paragraphs are headings and are left out, as is all text before the first CAPUT. The section is
    section('<title>/Chapter <n>') with n in Arabic numerals."""
    chapter = None
    for kind, par in blocks(html):
        m = re.fullmatch(r"\s*CAPUT ([IVXL]+)\s*", par)
        if kind == "center":
            chapter = roman(m.group(1)) if m else chapter
        elif chapter:
            yield from ((section(f"{title}/Chapter {chapter}"), s) for s in sentences(par, log))


def fetch(url, path, data=None):
    """Download url (POST data if given) to path once and return the provenance record.

    The download also writes <path without suffix>.meta.json with the URL (with data appended as a query string)
    and the retrieval date. Later calls only read the cache, whatever url and data they pass, so the record keeps
    the first retrieval date, and a cached file without that sidecar raises FileNotFoundError. Returns {url,
    retrieved, file (the file name), sha256 (of the cached bytes)}."""
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        req = urllib.request.Request(url, data=data and urllib.parse.urlencode(data).encode(), headers=UA)
        with urllib.request.urlopen(req, timeout=300) as r:
            path.write_bytes(r.read())
        meta = {"url": url + ("?" + urllib.parse.urlencode(data) if data else ""),
                "retrieved": datetime.date.today().isoformat()}
        path.with_suffix(".meta.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf8")
    meta = json.loads(path.with_suffix(".meta.json").read_text(encoding="utf8"))
    return dict(meta, file=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def ws_index(name, src, raw, log):
    """A book from its Wikisource scan pages, rendered as one transclusion.

    The MediaWiki parse API renders <pages index=... from=a to=b /> for src["pages"] = (a, b), both included, and
    the response is cached as <raw>/<name>/pages.json. Returns the gospel() pairs and a one-item input list whose
    record also counts the scan pages by ProofreadPage quality level (page_quality)."""
    a, b = src["pages"]
    path = raw / name / "pages.json"
    inp = fetch(API, path, dict(action="parse", format="json", formatversion=2, prop="text", contentmodel="wikitext",
                                title=src["title"], disablelimitreport=1,
                                text=f'<pages index="{src["index"]}" from="{a}" to="{b}" />'))
    html = json.loads(path.read_text(encoding="utf8"))["parse"]["text"]
    inp["page_quality"] = dict(sorted(collections.Counter(re.findall(r'data-page-quality="(\d)"', html)).items()))
    return list(gospel(html, src["title"], log)), [inp]


def ws_main(name, src, raw, log):
    """A work from its Wikisource subpages, each subpage a section.

    Subpage i (<root>/<sub>) is fetched with the MediaWiki parse API and cached as <raw>/<name>/<ii>.json. Only
    'p' paragraphs go to sentences(), with the source's speaker and filters settings, so centred headings are not
    text. The section is section('<root>/<sub>'). Returns (section, sentence) pairs and one input record per
    subpage, with its revision id (revid)."""
    pairs, inputs = [], []
    for i, sub in enumerate(src["subpages"]):
        title = f'{src["root"]}/{sub}'
        path = raw / name / f"{i:02d}.json"
        inp = fetch(API, path, dict(action="parse", format="json", formatversion=2, prop="text|revid", page=title))
        page = json.loads(path.read_text(encoding="utf8"))["parse"]
        inputs.append(dict(inp, revid=page["revid"]))
        pairs += [(section(title), s) for kind, par in blocks(page["text"]) if kind == "p"
                  for s in sentences(par, log, src.get("speaker"), src.get("filters", True))]
    return pairs, inputs


def gutenberg(name, src, raw, log):
    """A book from its Project Gutenberg HTML edition. The elements named in heads give the sections.

    The HTML is cached as <raw>/<name>/pg<n>-images.html and cut from src["start"] (included) to the first
    src["end"] after it (excluded), when given. Paragraphs come from blocks() with the source's units and heads,
    skipping style, script, sup and .pagenum elements and the source's skip specs. A paragraph's section is the
    text of the last heading before it with macrons removed (not passed through section()), or None before the
    first heading. With join, consecutive paragraphs with the same section become one, joined by a line break. With
    circumflex, circumflexes become macrons and j becomes i before sentences(), which gets the source's filters
    setting. Returns (section, sentence) pairs and a one-item input list."""
    n = src["ebook"]
    path = raw / name / f"pg{n}-images.html"
    inp = fetch(f"https://www.gutenberg.org/cache/epub/{n}/pg{n}-images.html", path)
    html = path.read_text(encoding="utf8")
    a = html.index(src["start"]) if "start" in src else 0
    html = html[a:html.index(src["end"], a)] if "end" in src else html[a:]
    pars, sec = [], None
    for kind, par in blocks(html, units=src["units"], skip=WS_SKIP[:3] + (".pagenum",) + src.get("skip", ()),
                            heads=src["heads"]):
        if kind == "head":
            sec = par.translate(BARE)  # 'VI. BEOWULF'S SPEECH.' in the committed file
        elif src.get("join") and pars and pars[-1][0] == sec:  # Sweet: a whole section is one paragraph
            pars[-1] = (sec, pars[-1][1] + "\n" + par)
        else:
            pars.append((sec, par))
    pairs = []
    for sec, par in pars:
        if src.get("circumflex"):  # Zupitza marks length with a circumflex and writes j for consonantal i
            par = unicodedata.normalize("NFC", unicodedata.normalize("NFD", par).replace("\u0302", "\u0304"))
            par = par.replace("j", "i").replace("J", "I")
        pairs += [(sec, s) for s in sentences(par, log, filters=src.get("filters", True))]
    return pairs, [inp]


def build(name, raw):
    """Rows, cleaning counts and input records for one source, with raw as the download cache.

    Each row is a dict with id '<name>:<NNNNN>', source, work, editor, year, genre, section, split, text, text_bare,
    tokens and tokens_bare, the bare forms with macrons removed (BARE). NNNNN is the sentence's position in the
    source. work is 'Finnsburh Fragment' when the section contains src["fragment"]. split is the source's split,
    except that 'blocks' gives the stale sentence-level split of the original files (of every 500 sentences 400
    train, 50 dev and 50 test). clean_corpus removes the split field from every row, whatever its value. Returns
    (rows, cleaning counts as a dict, input records)."""
    src = SOURCES[name]
    log = collections.Counter()
    pairs, inputs = (ws_index if "index" in src else gutenberg if "ebook" in src else ws_main)(name, src, raw, log)
    rows = []
    for i, (sec, text) in enumerate(pairs):
        split = src["split"]
        if split == "blocks":  # the stale sentence-level split of the original files, which clean_corpus removes
            split = ("train", "dev", "test")[(i % 500 >= 400) + (i % 500 >= 450)]
        toks = tokens(text)
        work = "Finnsburh Fragment" if src.get("fragment") and src["fragment"] in sec else src["work"]
        rows.append({"id": f"{name}:{i:05d}", "source": name, "work": work, "editor": src["editor"],
                     "year": src["year"], "genre": src["genre"], "section": sec, "split": split, "text": text,
                     "text_bare": text.translate(BARE), "tokens": toks,
                     "tokens_bare": [t.translate(BARE) for t in toks]})
    return rows, dict(log), inputs


def main():
    """Command line: build one source, write <out>/<source>.jsonl and update its entry in <out>/../manifest.json.

    The manifest is created when missing. work, editor, year, genre, units, tokens, long_vowel_marks, sha256,
    cleaning, role and inputs are overwritten, length_convention_in_source, licence and origin are set only when
    missing, and other fields are kept. units, tokens, long_vowel_marks and sha256 are of the file before
    clean_corpus, which refreshes them and adds its own counts to cleaning. Prints a one-line summary."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--source", required=True, choices=sorted(SOURCES))
    ap.add_argument("--out", required=True,
                    help="directory for <source>.jsonl (the manifest is <out>/../manifest.json)")
    ap.add_argument("--raw", default=str(ROOT / "data/raw/corpus"), help="download cache (gitignored)")
    args = ap.parse_args()
    rows, cleaning, inputs = build(args.source, Path(args.raw))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    data = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows).encode("utf8")
    (out / f"{args.source}.jsonl").write_bytes(data)
    mpath = out.parent / "manifest.json"
    manifest = json.loads(mpath.read_text(encoding="utf8")) if mpath.exists() else {"sources": {}}
    src = SOURCES[args.source]
    entry = manifest["sources"].setdefault(args.source, {})
    entry.update(work=src["work"], editor=src["editor"], year=src["year"], genre=src["genre"])
    pg = "ebook" in src
    entry.setdefault("length_convention_in_source", "circumflex" if src.get("circumflex") else "macron")
    entry.setdefault("licence", "Public domain (Project Gutenberg; PG header and licence removed)" if pg
                     else WS_LICENCE)
    entry.setdefault("origin", f"https://www.gutenberg.org/ebooks/{src['ebook']}" if pg else
                     "https://en.wikisource.org/wiki/" + ("Index:" + src["index"] if "index" in src
                                                          else src["root"]).replace(" ", "_"))
    entry.update(units=len(rows), tokens=sum(len(r["tokens"]) for r in rows),
                 long_vowel_marks=sum(c in LONG for r in rows for c in r["text"]),
                 sha256=hashlib.sha256(data).hexdigest(), cleaning=cleaning, role=src["role"], inputs=inputs)
    mpath.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf8")
    print(f"{args.source}: {len(rows)} sentences, {entry['tokens']} tokens, "
          f"{entry['long_vowel_marks']} long vowels -> {out / (args.source + '.jsonl')}")


if __name__ == "__main__":
    main()
