"""Build data/by_source/<source>.jsonl from the editions (plan stages S1 and S3).

Each source is fetched from its original site (Wikisource MediaWiki API, Project Gutenberg HTML) into
data/raw/corpus/<source>/ (gitignored) and goes through the same steps:
  1. paragraphs from the HTML: <p> and verse stanzas on Wikisource, the elements named in SOURCES on Gutenberg.
     Centred paragraphs, footnotes, footnote calls and margin line numbers are not text
  2. a line-level language filter: a line of four or more words with no length mark and no thorn or eth is
     dropped when it reads as English or Latin by function words, and a bracketed editor's note is English
  3. digits, brackets, daggers, asterisks, underscores and middle dots dropped
  4. sentence split after . ? ! followed by white space, inside one paragraph
  5. NFC, combining marks other than the macron dropped (ę -> e, ċ -> c), Roman numerals dropped
  6. sentence filters: in a paragraph with Old English sentences, a sentence of three or more words with no Old
     English letter and two or more capitalised words is dropped ("mixed": names and editors' notes, such as
     'Ic eom Abrahames God, and Isaaces God, and Iacobes God?'), and so is any sentence with four or more Latin
     function words
  7. tokens: letter runs, joined over inner hyphens and apostrophes, which are then removed
The two Beowulf editions skip the filters of steps 2 and 6, as the committed files show.

The original extraction scripts are not in the repository. These rules were inferred from the committed files
and their cleaning counts in data/manifest.json. With them matthew_ws, bright_reader, physiologus and hs_beowulf
rebuild byte for byte. The other four differ where the original language filter dropped lines these rules keep
(and the reverse), and in the 12 Sweet sentences the original removed as overlapping Matthew.

Usage: python -m oevl.build_corpus --source john_ws --out data/by_source [--raw data/raw/corpus]
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
                        role="test", speaker=r"^\s*[AGR]\.\s+"),
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
ROMAN = re.compile(r"\b(?:(?!i\b)[ivx]+|[IVXLC]{2,})\b")  # lone i and capitals stay: 'ān i oððe', 'I.'
# English function words; not be, for, in, is, on, to, was: Old English spellings too ('And on hiera dagum')
EN = set("the of and are were been from with by at as that this which it its an a or not but".split())
# Latin function words, plus the incipit/explicit of headings; not ac, in, me, se, sit, te: Old English words too
LA = set("et est sunt non ad cum de quod qui quae quam ut sed erat apud ex ab per nec neque enim autem ergo igitur si "
         "nisi tamen vel aut quid quia quo sic ita iam jam nunc hoc haec id ego tu mihi tibi sibi nos vos esse deus "
         "deum dei deo atque etiam incipit explicit".split())
VOID = {"br", "hr", "img", "link", "meta", "wbr", "input", "area", "base", "col", "embed", "source", "track"}
BLOCK = {"p", "div", "ol", "ul", "li", "table", "dl", "blockquote", "h1", "h2", "h3", "h4", "h5", "h6"}
WS_UNITS = ("p", "div.ws-poem-stanza")
WS_SKIP = ("style", "script", "sup", ".ws-noexport", ".wst-pline", ".references")


def match(specs, tag, cls, id_):
    """True if the element fits one of the specs: 'tag', '.class', 'tag.class' or 'tag#id'."""
    for spec in specs:
        t, c, i = re.fullmatch(r"(\w*)(?:\.([\w-]+))?(?:#([\w-]+))?", spec).groups()
        if t in ("", tag) and (not c or c in cls) and (not i or i == id_):
            return True
    return False


class Blocks(HTMLParser):
    """Paragraphs in document order as (kind, text): kind is 'p', 'center' (a centred paragraph, which is a
    heading on Wikisource) or 'head' (an element named in heads). A paragraph is an element matching units
    (by default a <p> or a verse stanza). A <br> or the end of a block inside it ends a line. Elements matching
    skip (by default footnote calls, footnotes and margin line numbers) are left out.

    Text left outside any <p> directly in the page body is a paragraph too: ProofreadPage leaves the last
    paragraph of a scan page unwrapped when the next page starts with a centred heading. Of a two-column table
    (the Soliloquies print Augustine's Latin beside Alfred's text) only the first column is read.
    """

    def __init__(self, units=WS_UNITS, skip=WS_SKIP, heads=()):
        super().__init__(convert_charrefs=True)
        self.units, self.skip, self.heads = units, skip, heads
        self.out, self.cur, self.head, self.stack, self.loose, self.cell = [], None, None, [], [], 0

    def flush(self):
        if "".join(self.loose).strip():
            self.out.append(("p", "".join(self.loose)))
        self.loose = []

    def handle_starttag(self, tag, attrs):
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
        if any(s[1] for s in self.stack):
            return
        if self.head is not None:
            self.head.append(data)
        elif self.cur is not None:
            self.cur.append(data)
        elif next((s[3] for s in reversed(self.stack) if s[0] in BLOCK), False):
            self.loose.append(data)


def blocks(html, **kw):
    parser = Blocks(**kw)
    parser.feed(html)
    parser.close()
    parser.flush()
    return parser.out + ([("p", "".join(parser.cur))] if parser.cur else [])  # a paragraph cut off by an end marker


def line_lang(line):
    """'eng' or 'lat' for a line of 4+ words with no length mark and no thorn or eth that reads as English or
    Latin, else None. Ash is no evidence: the English notes spell Cædmon and Ælfric."""
    if set(line) & set(LONG + "þðÞÐĐđ") or len(WORD.findall(line)) < 4:
        return None
    words = line.split()
    la = sum(w.lower() in LA for w in words) / len(words)
    en = sum(w.lower() in EN for w in words) / len(words)
    return "lat" if la >= 0.15 and la >= en else "eng" if en >= 0.3 else None


def editorial(text):
    """Drop digits, brackets, daggers, asterisks, underscores and Sweet's hyphen dots, before the sentence split.

    Square brackets go without a trace ('fugla [wynn],' -> 'fugla wynn,'), round ones and digits leave a space
    ('hēt):' -> 'hēt :', 'gehwæþ7res' -> 'gehwæþ res'), as in the committed files."""
    return re.sub(r"[\[\]†_*·]", "", re.sub(r"[\d()]", " ", text))


def clean(text, log):
    """NFC, then drop combining marks except a macron on a vowel, and Roman numerals. Runs after the split, so
    a stray mark after a full stop (a caron in Wyatt's 'geweorces.') keeps two sentences together, as in
    the files."""
    out = []
    for ch in unicodedata.normalize("NFD", text):
        if unicodedata.combining(ch) and ch != "\u0304":
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
    speaker is a regex for the speaker label that starts a line of dialogue."""
    par = unicodedata.normalize("NFC", par).replace("\u200b", "")
    if speaker:
        par = re.sub(speaker, "", par, flags=re.M)
    note = re.fullmatch(r"\s*\[[^\]]*\]\s*", par) and not set(par) & set(LONG)  # editor's note, e.g. [From the MS.]
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
    return re.sub(r"[^A-Za-z0-9.]", "_", title.replace("Æ", "AE").replace("æ", "ae").replace("œ", "oe"))


def roman(s):
    v = [{"I": 1, "V": 5, "X": 10, "L": 50}[c] for c in s]
    return sum(-a if a < b else a for a, b in zip(v, v[1:] + [0]))


def gospel(html, title, log):
    """(section, sentence) pairs. Chapters start at the centred 'CAPUT n' headings."""
    chapter = None
    for kind, par in blocks(html):
        m = re.fullmatch(r"\s*CAPUT ([IVXL]+)\s*", par)
        if kind == "center":
            chapter = roman(m.group(1)) if m else chapter
        elif chapter:
            yield from ((section(f"{title}/Chapter {chapter}"), s) for s in sentences(par, log))


def fetch(url, path, data=None):
    """Download url (POST data if given) to path once and return the provenance record."""
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
    """A book from its Wikisource scan pages, rendered as one transclusion."""
    a, b = src["pages"]
    path = raw / name / "pages.json"
    inp = fetch(API, path, dict(action="parse", format="json", formatversion=2, prop="text", contentmodel="wikitext",
                                title=src["title"], disablelimitreport=1,
                                text=f'<pages index="{src["index"]}" from="{a}" to="{b}" />'))
    html = json.loads(path.read_text(encoding="utf8"))["parse"]["text"]
    inp["page_quality"] = dict(sorted(collections.Counter(re.findall(r'data-page-quality="(\d)"', html)).items()))
    return list(gospel(html, src["title"], log)), [inp]


def ws_main(name, src, raw, log):
    """A work from its Wikisource subpages, each subpage a section."""
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
    """A book from its Project Gutenberg HTML edition. The elements named in heads give the sections."""
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
    src = SOURCES[name]
    log = collections.Counter()
    pairs, inputs = (ws_index if "index" in src else gutenberg if "ebook" in src else ws_main)(name, src, raw, log)
    rows = []
    for i, (sec, text) in enumerate(pairs):
        split = src["split"]
        if split == "blocks":  # the stale sentence-level split of the original files (AUDIT issue 12)
            split = ("train", "dev", "test")[(i % 500 >= 400) + (i % 500 >= 450)]
        toks = tokens(text)
        work = "Finnsburh Fragment" if src.get("fragment") and src["fragment"] in sec else src["work"]
        rows.append({"id": f"{name}:{i:05d}", "source": name, "work": work, "editor": src["editor"],
                     "year": src["year"], "genre": src["genre"], "section": sec, "split": split, "text": text,
                     "text_bare": text.translate(BARE), "tokens": toks,
                     "tokens_bare": [t.translate(BARE) for t in toks]})
    return rows, dict(log), inputs


def main():
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
    manifest = json.loads(mpath.read_text(encoding="utf8")) if mpath.exists() else {"sources": {}, "splits": {}}
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
    by_split = collections.defaultdict(list)  # the 'splits' block hashes the lines of each split in source order
    for name in manifest["sources"]:
        if (out / f"{name}.jsonl").exists():
            for line in (out / f"{name}.jsonl").read_text(encoding="utf8").splitlines(keepends=True):
                r = json.loads(line)
                by_split[r["split"]].append((line, len(r["tokens"])))
    manifest["splits"] = {k: {"units": len(v), "tokens": sum(n for _, n in v),
                              "sha256": hashlib.sha256("".join(x for x, _ in v).encode("utf8")).hexdigest()}
                          for k, v in sorted(by_split.items())}
    mpath.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf8")
    print(f"{args.source}: {len(rows)} sentences, {entry['tokens']} tokens, "
          f"{entry['long_vowel_marks']} long vowels -> {out / (args.source + '.jsonl')}")


if __name__ == "__main__":
    main()
