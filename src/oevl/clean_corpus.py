"""Clean the corpus files in data/by_source in place and refresh their entries in data/manifest.json.

Every change is one of the explicit mappings below, and each is counted in the file's manifest "cleaning" record.
It also removes the sentence-level "split" field of an earlier build: the split is by work (experiment.build_data).
Running it again on clean files changes nothing, and a new file such as john_ws.jsonl is cleaned the same way.
It writes nothing if any character outside the canonical inventory would remain, or if a bare token would no
longer equal its marked token with the length marks removed.
Usage: python -m oevl.clean_corpus
"""
import collections
import hashlib
import json
import sys
import unicodedata
from pathlib import Path

from oevl.experiment import bare, norm, pattern

ROOT = Path(__file__).resolve().parents[2]
# Canonical inventory (PLAN S2): Latin letters, þ ð æ, and length shown only by ā ē ī ō ū ȳ ǣ.
LETTERS = set("abcdefghijklmnopqrstuvwxyzþðæāēīōūȳǣ")
INVENTORY = LETTERS | {c.upper() for c in LETTERS}
CHARS = {
    "Đ": "Ð", "đ": "ð",                      # D with stroke typed for eth (Matthew, Soliloquies, John)
    "о": "o", "с": "c",                      # Cyrillic o and es inside Latin words (Soliloquies weоrlde, сlēne)
    "ċ": "c", "ġ": "g", "ę": "e", "ǫ": "o",  # teaching notations, dropped because only length is scored (PLAN S2)
    "ḥ": "h",                                # stray dot below (John geseaḥ)
}
# Whole-word repairs: (spelling in the text, tokens the source builder made of it) -> (marked, bare).
WORDS = {
    # Sweet: modifier macron typed before the vowel. Sweet prints þā 322 times and þa 8 times.
    ("þˉa", ("þˉa",)): ("þā", "þa"),
    # Wyatt l. 2474: the Wikisource transcription reads Hr¯wðel, which the builder split in two.
    # Wyatt prints Hrēðel elsewhere, and Harrison and Sharp read Hrēðel in this line.
    ("Hr¯wðel", ("Hr", "wðel")): ("Hrēðel", "Hreðel"),
    # John: ring typed for macron in "þå weorc þe ic wyrce" (those works), where the demonstrative is þā.
    ("þå", ("þå",)): ("þā", "þa"),
}
TABLE = str.maketrans(CHARS)


def sub(tokens, old, new):
    """Replace each run of tokens equal to old by the one token new; return the list and the number of runs."""
    out, i, n = [], 0, 0
    while i < len(tokens):
        if tuple(tokens[i:i + len(old)]) == old:
            out.append(new)
            i, n = i + len(old), n + 1
        else:
            out.append(tokens[i])
            i += 1
    return out, n


def clean_record(r, log):
    """Apply every mapping to one record in place, counting the changes in log, and return it."""
    if r.pop("split", None) is not None:
        log["removed_field_split"] += 1
    for ch, to in CHARS.items():
        n = sum(t.count(ch) for t in r["tokens"])
        if n:
            log[f"mapped_U+{ord(ch):04X}_to_U+{ord(to):04X}"] += n
    for k in ("text", "text_bare"):
        r[k] = r[k].translate(TABLE)
    for k in ("tokens", "tokens_bare"):
        r[k] = [t.translate(TABLE) for t in r[k]]
    for (text, old), (marked, plain) in WORDS.items():
        r["text"], r["text_bare"] = r["text"].replace(text, marked), r["text_bare"].replace(text, plain)
        r["tokens"], n = sub(r["tokens"], old, marked)
        r["tokens_bare"], _ = sub(r["tokens_bare"], old, plain)
        if n:
            log[f"repaired_word_{text}"] += n
    return r


def check(rows):
    """Problems that must not survive cleaning: characters outside the inventory, bare forms that do not match."""
    bad = []
    for r in rows:
        if len(r["tokens"]) != len(r["tokens_bare"]):
            bad.append(f"{r['id']}: {len(r['tokens'])} tokens but {len(r['tokens_bare'])} bare tokens")
        for w, b in zip(r["tokens"], r["tokens_bare"]):
            if set(w + b) - INVENTORY:
                bad.append(f"{r['id']}: {w!r} / {b!r} has {sorted(set(w + b) - INVENTORY)} outside the inventory")
            elif bare(w) != norm(b):
                bad.append(f"{r['id']}: bare({w!r}) != {b!r}")
        for k in ("text", "text_bare"):  # punctuation is fine here, letters and marks must be in the inventory
            odd = {c for c in r[k] if unicodedata.category(c)[0] in "LM" or unicodedata.category(c) == "Sk"} - INVENTORY
            if odd:
                bad.append(f"{r['id']}: {k} has {sorted(odd)} outside the inventory")
    return bad


def main():
    mpath = ROOT / "data" / "manifest.json"
    manifest = json.loads(mpath.read_text(encoding="utf8"))
    manifest.pop("splits", None)  # sentence-level split counts of the same earlier build
    done, problems = [], []
    for path in sorted((ROOT / "data" / "by_source").glob("*.jsonl")):
        log = collections.Counter()
        rows = [clean_record(json.loads(x), log) for x in path.read_text(encoding="utf8").splitlines() if x.strip()]
        problems += [f"{path.name} {p}" for p in check(rows)]
        done.append((path, rows, log))
    if problems:
        sys.exit("clean_corpus: nothing written, add a mapping for:\n" + "\n".join(problems))
    for path, rows, log in done:
        data = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows).encode()
        path.write_bytes(data)
        s = manifest["sources"].setdefault(path.stem, {})
        s.update(units=len(rows), tokens=sum(len(r["tokens"]) for r in rows),
                 long_vowel_marks=sum(pattern(t).count("1") for r in rows for t in r["tokens"]),
                 sha256=hashlib.sha256(data).hexdigest())
        cleaning = s.setdefault("cleaning", {})
        for k, n in log.items():
            cleaning[k] = cleaning.get(k, 0) + n
        print(f"{path.name}: {dict(log) or 'already clean'}")
    mpath.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf8")


if __name__ == "__main__":
    main()
