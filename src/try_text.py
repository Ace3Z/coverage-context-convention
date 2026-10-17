"""Add Old English long-vowel marks (macrons) to your own sentences.

Usage (inside oe_project, with the venv active):
  python src/try_text.py my_sentences.txt
  python src/try_text.py my_sentences.txt --gold my_sentences_marked.txt --tsv words.tsv

my_sentences.txt: one sentence per line. Any macrons in it are removed first.
--gold: the same lines with correct macrons; then accuracy is printed.
--tsv:  writes one row per word (useful for counting wrong changes).
Trains the same system as the report (default: bilateral, seed 1, training texts + WikiPron).
Changes no existing file.
"""
import argparse
import csv
import re
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from experiment import LONG, SHORT, bare, build_data, key, lexicon, norm, pattern, predict  # noqa: E402

TOKEN = re.compile(r"[^\W\d_]+(?:[-'’][^\W\d_]+)*")  # same word pattern used to build the corpus
ARMS = {"frequency": "frequency", "word": "word", "left": "left", "bilateral": "sentence"}
TO_LONG = dict(zip(SHORT, LONG)) | dict(zip(SHORT.upper(), "ĀĒĪŌŪȲǢ"))


def words(line):
    return TOKEN.findall(unicodedata.normalize("NFC", line))


def clean(w):  # how the corpus stores a word: no hyphens or apostrophes
    return w.replace("-", "").replace("'", "").replace("’", "")


def strip_marks(text):
    return "".join(c if c.lower() not in LONG else (SHORT[LONG.index(c.lower())].upper() if c.isupper() else SHORT[LONG.index(c)]) for c in text)


def apply(word, pat):
    """Put macrons on the vowel letters of `word` where `pat` has a 1."""
    out, i = [], 0
    for c in word:
        if c.lower() in SHORT:
            out.append(TO_LONG[c] if i < len(pat) and pat[i] == "1" else c)
            i += 1
        else:
            out.append(c)
    return "".join(out)


def main():
    p = argparse.ArgumentParser(description="Add Old English macrons to your own sentences.")
    p.add_argument("input", help="text file, one sentence per line")
    p.add_argument("--gold", help="same lines with correct macrons (optional)")
    p.add_argument("--model", choices=list(ARMS), default="bilateral")
    p.add_argument("--lexicon", default=str(ROOT / "data/lex_wikipron.csv"), help='"none" = training texts only')
    p.add_argument("--tsv", help="write one row per word to this file (optional)")
    a = p.parse_args()

    lines = [unicodedata.normalize("NFC", strip_marks(unicodedata.normalize("NFC", x)))
             for x in Path(a.input).read_text(encoding="utf8").splitlines()]
    gold_lines = Path(a.gold).read_text(encoding="utf8").splitlines() if a.gold else None
    if gold_lines is not None and len(gold_lines) != len(lines):
        sys.exit(f"--gold has {len(gold_lines)} lines but the input has {len(lines)}")
    ext = None if a.lexicon == "none" else a.lexicon
    if ext and not Path(ext).exists():
        sys.exit(f"{ext} not found: run `bash run_all.sh` once first (or use --lexicon none)")

    print("training the model (takes under a minute) ...", file=sys.stderr)
    train, _, _, _ = build_data(ROOT / "data")
    lex, counts = lexicon(train, ext)

    test, skipped = [], []
    for n, line in enumerate(lines):
        ws = [clean(w) for w in words(line)]
        gs = [clean(w) for w in words(gold_lines[n])] if gold_lines else ws
        if [bare(g) for g in gs] != [norm(w) for w in ws]:
            skipped.append(n + 1)
            gs = ws
        for i, (w, g) in enumerate(zip(ws, gs)):
            test.append({"line": n, "index": i, "bare": w, "gold": g, "sentence": ws, "key": key(w),
                         "gold_pattern": pattern(g), "scored": gold_lines is not None and n + 1 not in skipped})
    preds = predict(train, test, lex, counts, ARMS[a.model], seed=1) if test else []

    by_line = {}
    for r, pr in zip(test, preds):
        r["pred"] = pr
        r["known"] = bool(lex.get(r["key"]))
        r["changed"] = "1" in pr
        by_line.setdefault(r["line"], []).append(r)

    for n, line in enumerate(lines):
        rs = iter(by_line.get(n, []))
        out = TOKEN.sub(lambda m: apply(m.group(0), next(rs)["pred"]), line)
        print(f"{n + 1:>3}  {line}\n     {out}")

    total = len(test)
    vowel = [r for r in test if r["gold_pattern"] or r["pred"]]
    print("\nsummary")
    print(f"  model: {a.model}, dictionary: {'training only' if ext is None else 'training + WikiPron'}")
    print(f"  lines: {len(lines)}   words: {total}   words with a, e, i, o, u, y or æ: {len(vowel)}")
    if total:
        unknown = sum(not r["known"] for r in test)
        changed = sum(r["changed"] for r in test)
        print(f"  unknown words (left unchanged): {unknown} ({100 * unknown / total:.1f}%)")
        print(f"  words given at least one macron: {changed} ({100 * changed / total:.1f}%)")
    scored = [r for r in test if r["scored"]]
    if gold_lines is not None:
        if skipped:
            print(f"  not scored, words differ from --gold: lines {', '.join(map(str, skipped))}")
        if scored:
            ok = sum(r["pred"] == r["gold_pattern"] for r in scored)
            print(f"  accuracy against --gold: {ok}/{len(scored)} = {100 * ok / len(scored):.2f}%")

    if a.tsv:
        with open(a.tsv, "w", encoding="utf8", newline="") as f:
            wr = csv.writer(f, delimiter="\t")
            wr.writerow(["line", "word", "output", "known", "changed", "gold", "correct"])
            for r in test:
                wr.writerow([r["line"] + 1, r["bare"], apply(r["bare"], r["pred"]), int(r["known"]), int(r["changed"]),
                             r["gold"] if r["scored"] else "", int(r["pred"] == r["gold_pattern"]) if r["scored"] else ""])
        print(f"  per-word table: {a.tsv}")


if __name__ == "__main__":
    main()
