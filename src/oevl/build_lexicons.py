"""Build the candidate-spelling lists named in the proposal: Clark Hall, Wiktionary tables, Bosworth-Toller.

Each output is a one-column CSV (marked_form) that experiment.py reads with --lexicon.
  data/lex_clarkhall.csv   headwords of Clark Hall's Concise Anglo-Saxon Dictionary (TEI conversion)
  data/lex_wiktionary.csv  headwords + inflection-table forms of Old English Wiktionary entries (kaikki.org)
  data/lex_bt.csv          headwords of Bosworth-Toller (LINDAT dump; acute accents mark length -> macrons)
  data/lex_proposal.csv    union of the three
  data/lex_all.csv         union of the three + data/lex_wikipron.csv

Usage: python -m oevl.build_lexicons [--raw DIR]   (downloads the sources into DIR if missing; default data/raw)
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
WORD = re.compile(r"[a-zþðæāēīōūȳǣ]+")
ACUTE_TO_MACRON = str.maketrans("áéíóúýǽ", "āēīōūȳǣ")
SKIP_TAGS = {"table-tags", "inflection-template", "class"}


def clean(form, acute=False):
    """Lower-case, NFC, drop hyphens/apostrophes; return None unless it is one Old English word."""
    s = unicodedata.normalize("NFC", form).lower().replace("-", "").replace("'", "").replace("’", "").strip()
    if acute:
        s = s.translate(ACUTE_TO_MACRON)
    return s if WORD.fullmatch(s) else None


def clarkhall(path):
    text = path.read_text(encoding="utf8")
    return {w for w in (clean(m) for m in re.findall(r"<tei:orth[^>]*>(.*?)</tei:orth>", text)) if w}


def wiktionary(path):
    out = set()
    with open(path, encoding="utf8") as f:
        for line in f:
            d = json.loads(line)
            if d.get("lang_code") not in (None, "ang"):
                continue
            forms = [d["word"]] + [x["form"] for x in d.get("forms", []) if not SKIP_TAGS & set(x.get("tags", []))]
            out |= {w for w in (clean(x) for x in forms) if w}
    return out


def bosworth_toller(path):
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


def write(path, forms):
    with open(path, "w", encoding="utf8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["marked_form"])
        w.writerows([x] for x in sorted(forms))


def read(path):
    with open(path, encoding="utf8") as f:
        return {r["marked_form"] for r in csv.DictReader(f)}


def main():
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
    proposal = lists["clarkhall"] | lists["wiktionary"] | lists["bt"]
    write(ROOT / "data/lex_proposal.csv", proposal)
    print(f"proposal (union of the three): {len(proposal)} forms")
    wp = ROOT / "data/lex_wikipron.csv"
    if wp.exists():
        allf = proposal | read(wp)
        write(ROOT / "data/lex_all.csv", allf)
        print(f"all (three + WikiPron): {len(allf)} forms")


if __name__ == "__main__":
    main()
