"""Editor agreement on Beowulf and the go/no-go gate.

Three length-marked editions of the poem: Harrison and Sharp (1883), Wyatt (1894) and the Tauber gold (marks
from the Perseus text of Klaeber 1922, the set the models are scored against). The Harrison and Sharp and Wyatt files
also hold the Finnsburh Fragment. Each pair is aligned token by token on keys (lower-cased forms without macrons,
with ð written þ and hyphens and apostrophes dropped), as in experiment.agreement, and aligned tokens are compared on
their vowel-length patterns. Ambiguity and the floor (the prior) use the headline training set and lexicon, as
results/wikipron: training forms without Zupitza (decision D3) + WikiPron.

  python -m oevl.agreement   -> results/agreement/metrics.json and disagreements.tsv (run_all.sh saves the printout
                                as agreement.log)

If data/ambiguity_classes.csv exists (columns skeleton and a final A/B/C label), agreement is also given per class.
If results/agreement/disagreements_checked.tsv exists (the seed-1 sample of 100 Harrison and Sharp vs Wyatt
disagreements), its label counts are added to metrics.json, or a stale note when the sample no longer matches the
current disagreements. One run of an LLM, Claude Opus 5.5 (model id claude-opus-5-5), labelled the sample. A blind
review by further runs of the same model on 2026-10-19 changed four labels (see the labelled_by column). The labels
were then manually checked, and none was changed.
"""
import collections
import csv
import difflib
import json
import random
from pathlib import Path

from oevl.experiment import EXCLUDE_D3, build_data, flags, key, lexicon, load_tauber, predict

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "agreement"
PAIRS = (("hs", "wyatt"), ("hs", "tauber"), ("wyatt", "tauber"))
GATE_POINTS, GATE_N = 10, 500
GATE_RULE = (f"ceiling = editor agreement on ambiguous aligned tokens; floor = most-frequent-reading prior scored on "
             f"the same tokens against one editor of the pair; continue to modelling if ceiling minus floor is at least "
             f"{GATE_POINTS} points on at least {GATE_N} tokens")
GATE_NOTE = ("Applied after the fact: the rule was not written down before the agreement numbers were measured. "
             "These fields say which side of the bar each pair lands on. They do not decide the outcome.")


def align(a, b):
    """Token pairs (row of a, row of b) from the equal runs of the two key sequences.

    a, b: token rows of two editions in text order (dicts with a key field, as made by experiment.tokens).
    autojunk is off so that frequent keys are not treated as junk. Tokens in replaced, inserted or deleted runs are
    left out."""
    sm = difflib.SequenceMatcher(None, [r["key"] for r in a], [r["key"] for r in b], autojunk=False)
    return [p for op, i, j, k, m in sm.get_opcodes() if op == "equal" for p in zip(a[i:j], b[k:m])]


def rate(x, y):
    """Share of positions where the two equal-length lists hold the same value, or None when x is empty."""
    return sum(p == q for p, q in zip(x, y)) / len(x) if x else None


def kappa(x, y):
    """Cohen's kappa of two equal-length label lists, or None when they are empty or hold only one label in all."""
    from sklearn.metrics import cohen_kappa_score
    return float(cohen_kappa_score(x, y)) if x and len(set(x + y)) > 1 else None


def classes(path):
    """skeleton -> A, B or C from the class file (LLM labels, manually checked), or empty when the file is absent.

    Skeletons are normalised with experiment.key. The first letter of the label, upper-cased, is the class, and rows
    whose label does not start with A, B or C are left out. When two skeletons share a key, the later row wins.
    Raises SystemExit when no label column (final_label, final, class or label, in that order) is found, which includes
    a file with a header but no rows. A missing skeleton column raises KeyError instead."""
    if not Path(path).exists():
        return {}
    with open(path, encoding="utf8") as f:
        rows = list(csv.DictReader(f))
    # the committed class file names its label column 'final'; the other names are accepted for older copies
    col = next((c for c in ("final_label", "final", "class", "label") if rows and c in rows[0]), None)
    if col is None:
        raise SystemExit(f"{path}: needs a skeleton column and a final label column (final_label, final, class or label)")
    return {key(r["skeleton"]): r[col].strip()[:1].upper() for r in rows if r[col].strip()[:1].upper() in ("A", "B", "C")}


def compare(a, b, na, nb, lex, counts, cls):
    """Aligned pairs and, per population, n, raw agreement, kappa and the prior's accuracy against each side.

    a, b: token rows of two editions. na, nb: their short names, used in the result keys. lex, counts: the lexicon and
    training pattern counts from experiment.lexicon. cls: skeleton -> class letter from classes(), or empty.
    The prior is the frequency arm of experiment.predict, run on the a-side tokens. It depends only on the key, which
    both sides share. Populations: all, lexicon_ambiguous and attested_ambiguous (flags of the a-side token, which also
    depend only on its key) and, when cls is not empty, class_A, class_B and class_C. A class population takes every
    aligned token of its skeletons, whether or not the skeleton is lexicon-ambiguous.
    Returns (pairs, res), with pairs from align(a, b). res holds aligned, {na}_total, {nb}_total, alignment_rate_{na}
    and alignment_rate_{nb} (aligned over each edition's token count) and, per population, a dict of n, agreement,
    cohen_kappa, prior_vs_{na} and prior_vs_{nb}. agreement and the two prior scores are shares in [0, 1] of tokens
    whose whole patterns match, and cohen_kappa is kappa() of the two editions' patterns. All four are None for an
    empty population, and cohen_kappa also when the population holds a single pattern."""
    pairs = align(a, b)
    prior = dict(zip((x["id"] for x, _ in pairs), predict([], [x for x, _ in pairs], lex, counts, "frequency")))
    pops = {"all": lambda r: True,
            "lexicon_ambiguous": lambda r: flags(r, lex, counts)["lexicon_ambiguous"],
            "attested_ambiguous": lambda r: flags(r, lex, counts)["attested_ambiguous"]}
    pops.update({"class_" + c: (lambda r, c=c: cls.get(r["key"]) == c) for c in ("ABC" if cls else "")})
    res = {"aligned": len(pairs), f"{na}_total": len(a), f"{nb}_total": len(b),
           f"alignment_rate_{na}": len(pairs) / len(a), f"alignment_rate_{nb}": len(pairs) / len(b)}
    for pop, keep in pops.items():
        sub = [(x, y) for x, y in pairs if keep(x)]
        ga, gb, pr = [x["gold_pattern"] for x, _ in sub], [y["gold_pattern"] for _, y in sub], [prior[x["id"]] for x, _ in sub]
        res[pop] = {"n": len(sub), "agreement": rate(ga, gb), "cohen_kappa": kappa(ga, gb),
                    f"prior_vs_{na}": rate(pr, ga), f"prior_vs_{nb}": rate(pr, gb)}
    return pairs, res


def gate(res, na, nb):
    """Ceiling, floor and gap of one editor pair on the two ambiguous populations, the floor taken against each editor.

    res is the result dict of compare() and na, nb are the pair's names. The populations are lexicon_ambiguous and
    attested_ambiguous. Returns four entries, {population}_floor_vs_{editor}, each a dict of n, ceiling (the editors'
    agreement, a share in [0, 1]), floor (the prior's agreement with that editor, also a share), gap_points (100 times
    ceiling minus floor, rounded to two decimals) and clears_bar (a gap of at least GATE_POINTS on at least GATE_N
    tokens, judged on the gap rounded to nine decimals so that exactly GATE_POINTS clears). Raises TypeError for an
    empty population, whose agreement is None."""
    out = {}
    for pop in ("lexicon_ambiguous", "attested_ambiguous"):
        s = res[pop]
        for ref in (na, nb):
            gap = round(100 * (s["agreement"] - s[f"prior_vs_{ref}"]), 9)  # rounded so that exactly 10 points clears
            out[f"{pop}_floor_vs_{ref}"] = {"n": s["n"], "ceiling": s["agreement"], "floor": s[f"prior_vs_{ref}"],
                                             "gap_points": round(gap, 2), "clears_bar": gap >= GATE_POINTS and s["n"] >= GATE_N}
    return out


def context(rows, i, w=4):
    """The marked forms of rows i - w to i + w joined by spaces, row i in square brackets.

    rows: one edition's token rows in text order, i: the index of the focal row in rows, w: tokens on each side.
    Returns one string. The window crosses sentence boundaries and is cut at the ends of the edition."""
    return " ".join(f"[{r['gold']}]" if j == i else r["gold"] for j, r in enumerate(rows[max(0, i - w):i + w + 1], max(0, i - w)))


def disagreements(pairs, a, b, na, nb, lex, counts):
    """One row for disagreements.tsv per aligned pair whose patterns differ, in alignment order.

    pairs: aligned pairs from compare(). a, b: the full editions, for each token's context. na, nb: names for the pair
    column. lex, counts: for the flags, as in compare(). Returns a list of dicts with the keys pair ("{na}-{nb}"), a_id,
    b_id, key, a_form, b_form (the marked forms), a_pattern, b_pattern, lexicon_ambiguous and attested_ambiguous (flags
    of the a-side token as 0 or 1), and a_context and b_context from context()."""
    ia, ib = {r["id"]: i for i, r in enumerate(a)}, {r["id"]: i for i, r in enumerate(b)}
    return [{"pair": f"{na}-{nb}", "a_id": x["id"], "b_id": y["id"], "key": x["key"], "a_form": x["gold"], "b_form": y["gold"],
             "a_pattern": x["gold_pattern"], "b_pattern": y["gold_pattern"],
             **{k: int(v) for k, v in flags(x, lex, counts).items() if k != "minority_reading"},
             "a_context": context(a, ia[x["id"]]), "b_context": context(b, ib[y["id"]])}
            for x, y in pairs if x["gold_pattern"] != y["gold_pattern"]]


def sample(rows, n=100, seed=1):
    """A seeded random sample of n rows (all of them, shuffled, when fewer), which also depends on the row order."""
    return random.Random(seed).sample(rows, min(n, len(rows)))


def hand_check(path, rows):
    """Label counts of the labelled sample, or a stale note when it no longer matches the current disagreements.

    path: a Path to the labelled TSV, which needs at least the columns a_id, b_id, label and labelled_by. rows: the
    current Harrison and Sharp vs Wyatt disagreement rows in the order main() writes them. The file is stale when its
    set of (a_id, b_id) pairs differs from sample(rows), the seed-1 sample of 100. Only the id pairs are compared, so
    changed patterns under the same ids go unnoticed. Returns {stale: True, note} or {n (rows in the file), seed,
    labels (label -> count), labelled_by (the sorted distinct values)}."""
    with open(path, encoding="utf8") as f:
        done = list(csv.DictReader(f, delimiter="\t"))
    if {(r["a_id"], r["b_id"]) for r in done} != {(r["a_id"], r["b_id"]) for r in sample(rows)}:
        return {"stale": True, "note": f"{path.name} is not the seed-1 sample of the current disagreements; relabel it"}
    return {"n": len(done), "seed": 1, "labels": dict(collections.Counter(r["label"] for r in done)),
            "labelled_by": sorted({r["labelled_by"] for r in done})}


def main():
    """Score the three editor pairs, write results/agreement/metrics.json and disagreements.tsv, and print a summary.

    Loads the training set and the Harrison and Sharp and Wyatt editions with experiment.build_data on data/ without
    Zupitza (decision D3), the lexicon with data/lex_wikipron.csv, the Tauber gold from data/brunetti-length.txt and,
    when present, data/ambiguity_classes.csv and results/agreement/disagreements_checked.tsv. Creates
    results/agreement if needed and overwrites both outputs. metrics.json holds editions, lexicon, classes, pairs
    (compare() per pair), gate (rule, note and gate() per pair) and, with the checked file, hand_check_hs_wyatt.
    disagreements.tsv lists the disagreements of all three pairs. Prints load_tauber's validation line, the alignment
    and per-population agreement and kappa of each pair, the gate note and fields, and the manual-check result."""
    train, _, hs, wy = build_data(ROOT / "data", EXCLUDE_D3)
    lex, counts = lexicon(train, str(ROOT / "data/lex_wikipron.csv"))
    eds = {"hs": hs, "wyatt": wy, "tauber": load_tauber(ROOT / "data/brunetti-length.txt")}
    cls = classes(ROOT / "data/ambiguity_classes.csv")
    metrics = {"editions": {"hs": "Harrison and Sharp 1883", "wyatt": "Wyatt 1894", "tauber": "Tauber gold (Klaeber 1922 via Perseus)"},
               "lexicon": "training forms without zupitza_grammar (decision D3) + data/lex_wikipron.csv, as results/wikipron",
               "classes": "data/ambiguity_classes.csv" if cls else "skipped: data/ambiguity_classes.csv not found",
               "pairs": {}, "gate": {"rule": GATE_RULE, "note": GATE_NOTE}}
    rows = []
    for na, nb in PAIRS:
        pairs, res = compare(eds[na], eds[nb], na, nb, lex, counts, cls)
        metrics["pairs"][f"{na}-{nb}"], metrics["gate"][f"{na}-{nb}"] = res, gate(res, na, nb)
        rows += disagreements(pairs, eds[na], eds[nb], na, nb, lex, counts)
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "disagreements.tsv", "w", encoding="utf8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter="\t")
        w.writeheader()
        w.writerows(rows)
    checked = OUT / "disagreements_checked.tsv"
    if checked.exists():
        metrics["hand_check_hs_wyatt"] = hand_check(checked, [r for r in rows if r["pair"] == "hs-wyatt"])
    (OUT / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf8")
    show = lambda v, f, m=1: "n/a" if v is None else format(m * v, f)  # noqa: E731
    for p, res in metrics["pairs"].items():
        na, nb = p.split("-")
        print(f"{p}: aligned {res['aligned']} ({100 * res[f'alignment_rate_{na}']:.2f}% of {na}, "
              f"{100 * res[f'alignment_rate_{nb}']:.2f}% of {nb})")
        for pop in [k for k in res if isinstance(res[k], dict)]:
            s = res[pop]
            print(f"  {pop:20} n {s['n']:6}  agreement {show(s['agreement'], '.2f', 100)}%  kappa {show(s['cohen_kappa'], '.3f')}")
    print("gate:", GATE_NOTE)
    for p, g in metrics["gate"].items():
        if isinstance(g, dict):
            for k, s in g.items():
                print(f"  {p:12} {k:34} n {s['n']:5}  ceiling {100 * s['ceiling']:.2f}  floor {100 * s['floor']:.2f}  "
                      f"gap {s['gap_points']:5.2f}  {'clears' if s['clears_bar'] else 'below'} the bar")
    if "hand_check_hs_wyatt" in metrics:
        print("manual check:", metrics["hand_check_hs_wyatt"])


if __name__ == "__main__":
    main()
