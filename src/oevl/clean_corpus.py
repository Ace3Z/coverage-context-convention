"""Clean the corpus files in data/by_source in place and refresh their entries in data/manifest.json.

Every change to a record is one of the explicit mappings below (CHARS for characters, WORDS for whole words), and
each is counted over the marked tokens and added to the counts already in the file's manifest "cleaning" record.
It also removes the sentence-level "split" field of an earlier build, and the manifest's "splits" block with it:
the split is by work (experiment.build_data).
Running it again on clean files changes nothing, and a new file such as john_ws.jsonl is cleaned the same way.
It writes nothing if a token would keep a character outside the canonical inventory or a text field a letter or
mark outside it, if a sentence's marked and bare token lists would differ in length, or if a bare token would no
longer equal its marked token with the length marks removed (compared in lower case).
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
# Canonical inventory: Latin letters, þ ð æ, and length shown only by ā ē ī ō ū ȳ ǣ.
LETTERS = set("abcdefghijklmnopqrstuvwxyzþðæāēīōūȳǣ")
INVENTORY = LETTERS | {c.upper() for c in LETTERS}
# Character mappings. The committed files hold none of ċ ġ ę ǫ ḥ, whose marks build_corpus drops first (the
# "dropped_mark_" counts), so only Đ and the two Cyrillic letters have fired (the "mapped_" counts in
# data/manifest.json).
CHARS = {
    "Đ": "Ð", "đ": "ð",                      # D with stroke typed for eth (Matthew, Soliloquies, John)
    "о": "o", "с": "c",                      # Cyrillic o and es inside Latin words (Soliloquies weоrlde, сlēne)
    "ċ": "c", "ġ": "g", "ę": "e", "ǫ": "o",  # teaching notations, dropped because only length is scored
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
    """Replace each run of tokens equal to old by the one token new, and return the new list and the number of runs.

    old is a non-empty tuple of tokens (an empty one would never end the loop, a list would never match). Runs are
    matched left to right without overlap, and the tokens list passed in is not modified.
    """
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
    """Apply every mapping to one record in place, counting the changes in log, and return it.

    r is one corpus row with "text", "text_bare", "tokens" and "tokens_bare", and an old "split" field is dropped.
    log is a collections.Counter. It gains "removed_field_split", "mapped_U+XXXX_to_U+YYYY" with the occurrences of
    each CHARS character in the marked tokens, and "repaired_word_<spelling>" per WORDS run replaced in the marked
    tokens. The text fields get the same mappings by plain substring replacement, so a WORDS spelling inside a longer
    word would change the text but not the tokens. A clean record comes back unchanged with nothing logged.
    """
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
    """Problems that must not survive cleaning: characters outside the inventory, bare forms that do not match.

    rows are corpus records with "id", "text", "text_bare", "tokens" and "tokens_bare". It returns one message per
    problem, starting with the record id, or an empty list when the rows are clean. A record is flagged when its two
    token lists differ in length (pairs are then checked up to the shorter list), when a token pair has a character
    outside INVENTORY (so tokens hold only inventory letters, precomposed), or else when bare(marked) != norm(bare),
    a comparison in lower case after NFC. Each pair gives at most one message. In the text fields only letters,
    combining marks and modifier symbols such as ¯ are checked, so punctuation, digits and spaces pass. The text is
    never compared with the tokens.
    """
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
    """Clean every data/by_source/*.jsonl, rewrite the files and update their entries in data/manifest.json.

    Both paths are under the repository root found from this file, whatever the working directory, and blank input
    lines are dropped. All files are cleaned and checked in memory first. If any problem is found it exits with
    status 1, the problems on stderr, and writes nothing. Otherwise every file is rewritten, even when nothing changed
    (UTF-8, one JSON object per line, non-ASCII kept), and manifest["sources"][<file stem>] (created if missing) gets
    units (sentences), tokens, long_vowel_marks (long vowels of either case in the marked tokens) and the SHA-256 of
    the written bytes, with this run's counts added to its "cleaning" record. The top-level "splits" block is
    removed, the manifest is written with indent 2, and one line per file is printed.
    """
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
