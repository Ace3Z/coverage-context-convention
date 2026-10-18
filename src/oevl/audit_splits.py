"""Leakage audit (PLAN S4): word 6-grams shared by the training works and each held-out set.

Tokens are compared bare, lower-cased, with ð folded to þ (experiment.key), and 6-grams stay inside a sentence.
It prints the overlap matrix (distinct shared 6-grams per training work), writes data/leakage_excluded.json with
the dev and test sentences that share a 6-gram with train, and asserts that what remains shares none.
experiment.heldout leaves those sentences out of every evaluation. The two Beowulf agreement editions are audited
but not excluded, because they are only compared with each other.
Usage: python -m oevl.audit_splits [--data data] [--tauber data/brunetti-length.txt]
"""
import argparse
import collections
import json
from pathlib import Path

from oevl.experiment import TESTS, TRAIN, load, load_tauber, tokens

N = 6
AGREEMENT = ("hs_beowulf", "wyatt_beowulf")


def grams(rows):
    """sentence id -> set of word 6-grams of that sentence"""
    keys = collections.defaultdict(list)
    for r in rows:
        keys[r["sentence_id"]].append(r["key"])
    return {s: {tuple(k[i:i + N]) for i in range(len(k) - N + 1)} for s, k in keys.items()}


def matrix(held, train):
    """held-out set -> training work -> number of distinct 6-grams they share"""
    return {h: {t: len(set().union(*g.values()) & set().union(*tg.values())) for t, tg in train.items()}
            for h, g in held.items()}


def show(title, m, train):
    print(f"{title}\n{'':16}" + "".join(f"{t[:12]:>13}" for t in train))
    for h, row in m.items():
        print(f"{h:16}" + "".join(f"{row[t]:>13}" for t in train))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--tauber", default="data/brunetti-length.txt")
    a = ap.parse_args()
    src = Path(a.data) / "by_source"
    train = {t: grams(tokens(load(src / f"{t}.jsonl"))) for t in TRAIN}
    seen = set().union(*(g for tg in train.values() for g in tg.values()))
    rows, held = {}, {}
    for name in TESTS + list(AGREEMENT):
        if name == "tauber_beowulf":
            rows[name] = load_tauber(a.tauber)
        elif (src / f"{name}.jsonl").exists():
            rows[name] = list(tokens(load(src / f"{name}.jsonl")))
        else:
            print(f"{name}: no file in {src}, not audited")
            continue
        held[name] = grams(rows[name])
    show("Distinct 6-grams shared with each training work", matrix(held, train), train)

    out = {"ngram": N, "text": "bare tokens, lower-cased, ð folded to þ, 6-grams within a sentence",
           "train": TRAIN, "excluded": {}, "counts": {}}
    for name in held:
        hit = sorted(s for s, g in held[name].items() if g & seen)
        n_tok = sum(r["sentence_id"] in set(hit) for r in rows[name])
        c = {"sentences": len(hit), "of_sentences": len(held[name]), "tokens": n_tok, "of_tokens": len(rows[name])}
        print(f"{name}: {c['sentences']} of {c['of_sentences']} sentences, {c['tokens']} of {c['of_tokens']} tokens "
              f"share a 6-gram with train" + (" (not excluded: agreement only)" if name in AGREEMENT else ""))
        if name not in AGREEMENT:
            out["excluded"][name], out["counts"][name] = hit, c
            held[name] = {s: g for s, g in held[name].items() if s not in set(hit)}
    after = matrix({h: held[h] for h in out["excluded"]}, train)
    show("After exclusion", after, train)
    assert not any(v for row in after.values() for v in row.values()), "a dev or test sentence still shares a 6-gram"
    path = Path(a.data) / "leakage_excluded.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf8")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
