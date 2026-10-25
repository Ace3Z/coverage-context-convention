"""Build the candidate-spelling lists named in the proposal: Clark Hall, Wiktionary tables, Bosworth-Toller.

Each output is a CSV with a marked_form column that experiment.py reads with --lexicon. The two unions add a
sources column (space-separated names) so the frequency prior can back off to the pattern most sources attest.
  data/lex_clarkhall.csv   headwords and variants of Clark Hall's Concise Anglo-Saxon Dictionary (TEI conversion)
  data/lex_wiktionary.csv  headwords, alternative and inflected forms of Old English Wiktionary entries (kaikki.org)
  data/lex_bt.csv          headwords of Bosworth-Toller (LINDAT dump, acute accents read as macrons)
  data/lex_proposal.csv    union of the three
  data/lex_all.csv         union of the three + data/lex_wikipron.csv (not written if that file is missing)
A full build also writes data/lexicons_provenance.json (URL, file, licence, SHA-256 and form count of each source).

Usage: python -m oevl.build_lexicons [--raw DIR]   (downloads missing sources into DIR, default data/raw)
       python -m oevl.build_lexicons --unions-only  (rebuild lex_proposal/lex_all from the three shipped CSVs)
"""
import argparse
import csv
import hashlib
import json
import re
import unicodedata
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCES = {
    "clarkhall": ("https://raw.githubusercontent.com/Serg078/Conversion-of-The-Clark-Hall-Dictionary-of-Old-English-to-TEI/"
                  "b53fc7db19eea8fa94ac74f00a7e39d14bdd1a98/clark_hall_tei_rdfa.xml", "clark_hall_tei_rdfa.xml", "Unlicense"),
    "wiktionary": ("https://kaikki.org/dictionary/Old%20English/kaikki.org-dictionary-OldEnglish.jsonl",
                   "kaikki.org-dictionary-OldEnglish.jsonl", "CC BY-SA 4.0 (Wiktionary)"),
    "bt": ("https://lindat.mff.cuni.cz/repository/server/api/core/bitstreams/3010b742-b2c4-4152-870a-716ce1652e7c/content",
           "bosworth_entries_export.csv", "CC BY 4.0"),
}
WORD = re.compile(r"[abcdefghiklmnopqrstuwxyzþðæāēīōūȳǣ]+")  # Old English letters only (no j, v)
ACUTE_TO_MACRON = str.maketrans("áéíóúýǽ", "āēīōūȳǣ")
BT_DIGRAPHS = (("eá", "ēa"), ("eó", "ēo"), ("ié", "īe"), ("ió", "īo"))  # long diphthongs, accent on the 2nd letter
TEACHING_MARKS = str.maketrans("ċġęǫ", "cgeo")  # palatal dots and hooks, only length is scored
SKIP_TAGS = {"table-tags", "inflection-template", "class"}  # kaikki table metadata, not word forms


def clean(form, acute=False):
    """Return a dictionary form lower-cased and normalised like the corpus, or None unless it is one Old English word.

    The form is NFC-normalised, lower-cased and stripped. A form that starts or ends with a hyphen (an affix or a
    partial variant) gives None. With acute=True (Bosworth-Toller), the long diphthongs eá, eó, ié, ió become ēa, ēo,
    īe, īo, then the remaining á é í ó ú ý ǽ become ā ē ī ō ū ȳ ǣ. Then, as in the corpus, ċ ġ ę ǫ lose their marks and
    hyphens and apostrophes (' and ’) are removed, joining the parts. The result must be made only of the letters in
    WORD (Latin letters without j and v, þ ð æ and the macron vowels), so spaces, digits, brackets and other accents
    (an acute when acute=False) give None."""
    s = unicodedata.normalize("NFC", form).lower().strip()
    if s.startswith("-") or s.endswith("-"):  # affixes and partial variants are not words
        return None
    if acute:  # digraphs first, and before hyphens are joined, so ge-ánlǽcan keeps its ā
        for a, b in BT_DIGRAPHS:
            s = s.replace(a, b)
        s = s.translate(ACUTE_TO_MACRON)
    s = s.translate(TEACHING_MARKS).replace("-", "").replace("'", "").replace("’", "")
    return s if WORD.fullmatch(s) else None


def clarkhall(path):
    """Return the set of cleaned forms in every <tei:orth> of the Clark Hall TEI file at path (a Path).

    These are the headwords, their full variant spellings and the words of collocations, each on its own (tō and ǣfenes
    from tō ǣfenes). The file is read as text with regular expressions, not parsed as XML. A ge- marker (+ or ±) or a
    homograph number sits in a <tei:lbl>, which is removed with its text, so a form marked + or ± is listed without
    ge-. Other tags are stripped and their text kept. No expand attribute is used, so partial variants such as "(a,"
    are not expanded and fail clean."""
    text = path.read_text(encoding="utf8")
    # A headword with a ge- marker (+ or ±) or a homograph number is a multi-line orth with <tei:lbl> inside.
    orths = re.findall(r"<tei:orth[^>]*>(.*?)</tei:orth>", text, re.S)
    orths = [re.sub(r"<[^>]+>", "", re.sub(r"<tei:lbl[^>]*>.*?</tei:lbl>", "", m, flags=re.S)) for m in orths]
    return {w for w in (clean(m) for m in orths) if w}


def wiktionary(path):
    """Return the set of cleaned forms of the Old English entries in the kaikki.org JSONL file at path.

    Each line is one entry. An entry whose lang_code is set and is not "ang" is skipped. Every item of the entry's
    forms list is used (the canonical headword, alternative spellings, inflected forms) except those with a tag in
    SKIP_TAGS. The page title (word) is added only when no form is tagged canonical, because page titles are normally
    written without macrons and would add an unmarked twin of the headword."""
    out = set()
    with open(path, encoding="utf8") as f:
        for line in f:
            d = json.loads(line)
            if d.get("lang_code") not in (None, "ang"):
                continue
            forms = [x["form"] for x in d.get("forms", []) if not SKIP_TAGS & set(x.get("tags", []))]
            # The page title normally has no macrons. It is the headword only when kaikki gives no canonical form.
            if not any("canonical" in x.get("tags", []) for x in d.get("forms", [])):
                forms.append(d["word"])
            out |= {w for w in (clean(x) for x in forms) if w}
    return out


def bosworth_toller(path):
    """Return the set of cleaned headwords of the Bosworth-Toller CSV dump at path, with acute accents read as length.

    The file is semicolon-separated, with a header row and the columns id, headword and body (the entry's XML).
    Each <orth> in body (the shipped dump has one per entry) is cleaned with acute=True. The headword column is used
    only when body has no <orth>. Side effect: raises the csv module's field size limit to 10**9 for the whole
    process, a precaution for long entry bodies."""
    csv.field_size_limit(10**9)
    out = set()
    with open(path, encoding="utf8", newline="") as f:
        rows = csv.reader(f, delimiter=";", quotechar='"')
        next(rows)
        for row in rows:
            for orth in re.findall(r"<orth>(.*?)</orth>", row[2]) or [row[1]]:
                w = clean(orth, acute=True)
                if w:
                    out.add(w)
    return out


def write(path, forms, lists=None):
    """Write forms, sorted by code point, to the CSV at path under a marked_form header, replacing the file.

    lists: optional dict mapping each source name to its set of forms. When it is given and not empty, a sources
    column holds, space-separated and in the dict's order, the names of the sets that contain each form (empty when
    none does)."""
    with open(path, "w", encoding="utf8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["marked_form"] + (["sources"] if lists else []))
        w.writerows([x] + ([" ".join(n for n in lists if x in lists[n])] if lists else []) for x in sorted(forms))


def read(path):
    """Return the set of values in the marked_form column of the CSV at path."""
    with open(path, encoding="utf8") as f:
        return {r["marked_form"] for r in csv.DictReader(f)}


def main():
    """Build the three dictionaries from their sources and write them, the unions and the provenance file.

    With --unions-only, read data/lex_clarkhall.csv, data/lex_wiktionary.csv and data/lex_bt.csv and only rewrite
    the unions. Otherwise download each source into --raw when its file is missing, parse the three, write
    data/lex_<name>.csv for each (no sources column) and print its size and how many forms have a macron, then write
    the unions and data/lexicons_provenance.json. Outputs always go to <repo>/data, whatever --raw is. A file
    already in --raw is used as it is: its SHA-256 is recorded, not checked against the shipped provenance."""
    p = argparse.ArgumentParser()
    p.add_argument("--raw", default=str(ROOT / "data/raw"))
    p.add_argument("--unions-only", action="store_true")
    a = p.parse_args()
    if a.unions_only:
        write_unions({name: read(ROOT / f"data/lex_{name}.csv") for name in SOURCES})
        return
    raw = Path(a.raw)
    raw.mkdir(parents=True, exist_ok=True)
    provenance, lists = {}, {}
    for name, (url, fname, licence) in SOURCES.items():
        path = raw / fname
        if not path.exists():
            print(f"downloading {name} ...")
            urllib.request.urlretrieve(url, path)
        provenance[name] = {"url": url, "file": fname, "license": licence,
                            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    lists["clarkhall"] = clarkhall(raw / SOURCES["clarkhall"][1])
    lists["wiktionary"] = wiktionary(raw / SOURCES["wiktionary"][1])
    lists["bt"] = bosworth_toller(raw / SOURCES["bt"][1])
    for name, forms in lists.items():
        write(ROOT / f"data/lex_{name}.csv", forms)
        provenance[name]["forms"] = len(forms)
        print(f"{name}: {len(forms)} forms, {sum(bool(re.search('[āēīōūȳǣ]', x)) for x in forms)} with a macron")
    write_unions(lists)
    (ROOT / "data/lexicons_provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf8")


def write_unions(lists):
    """Write the union of the three dictionaries to data/lex_proposal.csv, and with WikiPron to data/lex_all.csv.

    lists: dict mapping source name to set of forms, with at least the keys clarkhall, wiktionary and bt. Each key
    in lists can appear in the sources column of lex_proposal.csv, and lex_all.csv adds wikipron, read from
    data/lex_wikipron.csv. The caller's dict is not changed. When lex_wikipron.csv is missing, lex_all.csv is not
    written and an older copy is left as it was. Prints the size of each union."""
    proposal = lists["clarkhall"] | lists["wiktionary"] | lists["bt"]
    write(ROOT / "data/lex_proposal.csv", proposal, lists)
    print(f"proposal (union of the three): {len(proposal)} forms")
    wp = ROOT / "data/lex_wikipron.csv"
    if wp.exists():
        lists = {**lists, "wikipron": read(wp)}
        allf = proposal | lists["wikipron"]
        write(ROOT / "data/lex_all.csv", allf, lists)
        print(f"all (three + WikiPron): {len(allf)} forms")


if __name__ == "__main__":
    main()
