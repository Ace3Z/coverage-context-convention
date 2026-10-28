"""Leakage audit: word 6-grams shared by the training works and each held-out set.

Tokens are compared as experiment.key gives them (bare, lower-cased, ð folded to þ, hyphens and apostrophes dropped),
and 6-grams stay inside a sentence. Train is all five works in experiment.TRAIN, Zupitza included although decision D3
leaves it out of the final training. It prints the overlap matrix (distinct shared 6-grams per training work), writes
leakage_excluded.json in the --data folder with the dev and test sentences that share a 6-gram with train, and
asserts that what remains shares none. experiment.heldout reads that file and leaves those sentences out of the dev
and test sets it loads for scoring. The two Beowulf agreement editions are audited but not excluded, because they
are not dev or test sets: oevl.agreement uses them whole, for editor agreement and the prior's floor on each edition.
Usage: python -m oevl.audit_splits [--data data] [--tauber data/brunetti-length.txt]
"""
import argparse
import collections
import json
from collections.abc import Collection, Iterable, Mapping
from pathlib import Path
from typing import Any

from oevl.experiment import TESTS, TRAIN, load, load_tauber, tokens

N = 6
AGREEMENT = ("hs_beowulf", "wyatt_beowulf")


def grams(rows: Iterable[Mapping[str, Any]]) -> dict[str, set[tuple[str, ...]]]:
    """Map each sentence id to the set of word 6-grams in that sentence, each a tuple of six experiment.key strings.

    rows: an iterable of token dicts as experiment.tokens yields them or load_tauber returns them. Only "sentence_id"
    and "key" are read, and each sentence's tokens are taken in the order they come. A sentence with fewer than six
    tokens gets an empty set, so it can never share a 6-gram."""
    keys: collections.defaultdict[str, list[str]] = collections.defaultdict(list)
    for r in rows:
        keys[r["sentence_id"]].append(r["key"])
    return {s: {tuple(k[i:i + N]) for i in range(len(k) - N + 1)} for s, k in keys.items()}


def matrix(held: Mapping[str, Mapping[str, set[tuple[str, ...]]]],
           train: Mapping[str, Mapping[str, set[tuple[str, ...]]]]) -> dict[str, dict[str, int]]:
    """Count the distinct 6-grams that each held-out set shares with each training work.

    held, train: set or work name -> grams() output. The 6-grams of all sentences on each side are pooled before
    intersecting. Returns held-out name -> training work name -> count."""
    return {h: {t: len(set().union(*g.values()) & set().union(*tg.values())) for t, tg in train.items()}
            for h, g in held.items()}


def show(title: str, m: Mapping[str, Mapping[str, int]], train: Collection[str]) -> None:
    """Print a matrix() result under a title, one row per held-out set and one column per training work.

    Only the keys of train are used, for the column order. Column headers are cut to 12 characters."""
    print(f"{title}\n{'':16}" + "".join(f"{t[:12]:>13}" for t in train))
    for h, row in m.items():
        print(f"{h:16}" + "".join(f"{row[t]:>13}" for t in train))


def main() -> None:
    """Run the audit, print the overlap matrices before and after exclusion and write leakage_excluded.json.

    Reads <data>/by_source/<work>.jsonl for the training works, john_ws, soliloquies and the two agreement editions,
    and Tauber's Beowulf from --tauber, whose default does not follow --data. A missing training or Tauber file raises
    FileNotFoundError. A missing john_ws, soliloquies or agreement file is reported and skipped, and then has no entry
    in the output. The JSON holds "ngram" (6), "text" (a short note on the comparison), "train" (experiment.TRAIN)
    and, for each dev and test set, "excluded" (sorted ids of the sentences sharing any 6-gram with any training work)
    and "counts" (sentences and tokens excluded, out of of_sentences and of_tokens in the whole set). The agreement
    editions appear only in the first matrix and the printed counts. The file is written to
    <data>/leakage_excluded.json, replacing any earlier one, after the assert, so a failed check writes nothing."""
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

    out: dict[str, Any]
    out = {"ngram": N,
           "text": "bare tokens, lower-cased, ð folded to þ, hyphens and apostrophes dropped, 6-grams within a "
                   "sentence",
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
