"""Editor agreement on Beowulf and the go/no-go gate.

Three length-marked editions of the poem: Harrison and Sharp (1883), Wyatt (1894) and the Tauber gold (marks
from the Perseus text of Klaeber 1922, the set the models are scored against). Each pair is aligned token by token
on bare forms, as in experiment.agreement, and aligned tokens are compared on their vowel-length patterns.
Ambiguity and the floor (the prior) use the headline training set and lexicon, as results/wikipron: training forms
without Zupitza (decision D3) + WikiPron.

  python -m oevl.agreement   -> results/agreement/metrics.json and disagreements.tsv

If data/ambiguity_classes.csv exists (columns skeleton and a final A/B/C label), agreement is also given per class.
If results/agreement/disagreements_checked.tsv exists (the seed-1 sample of 100 Harrison and Sharp vs Wyatt
disagreements), its label counts are added to metrics.json. One run of an LLM, Claude Opus 5.5 (model id
claude-opus-5-5), labelled the sample. A blind review by further runs of the same model on 2026-10-19 changed four
labels (see the labelled_by column). The labels were then manually checked, and none was changed.
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
    """Token pairs from the equal runs of the two key sequences."""
    sm = difflib.SequenceMatcher(None, [r["key"] for r in a], [r["key"] for r in b], autojunk=False)
    return [p for op, i, j, k, m in sm.get_opcodes() if op == "equal" for p in zip(a[i:j], b[k:m])]


def rate(x, y):
    return sum(p == q for p, q in zip(x, y)) / len(x) if x else None


def kappa(x, y):
    from sklearn.metrics import cohen_kappa_score
    return float(cohen_kappa_score(x, y)) if x and len(set(x + y)) > 1 else None


def classes(path):
    """skeleton -> A, B or C from the class file (LLM labels, manually checked); empty when the file is absent."""
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
    """Aligned pairs and, per population, n, raw agreement, kappa and the prior's accuracy against each side."""
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
    out = {}
    for pop in ("lexicon_ambiguous", "attested_ambiguous"):
        s = res[pop]
        for ref in (na, nb):
            gap = round(100 * (s["agreement"] - s[f"prior_vs_{ref}"]), 9)  # rounded so that exactly 10 points clears
            out[f"{pop}_floor_vs_{ref}"] = {"n": s["n"], "ceiling": s["agreement"], "floor": s[f"prior_vs_{ref}"],
                                             "gap_points": round(gap, 2), "clears_bar": gap >= GATE_POINTS and s["n"] >= GATE_N}
    return out


def context(rows, i, w=4):
    return " ".join(f"[{r['gold']}]" if j == i else r["gold"] for j, r in enumerate(rows[max(0, i - w):i + w + 1], max(0, i - w)))


def disagreements(pairs, a, b, na, nb, lex, counts):
    ia, ib = {r["id"]: i for i, r in enumerate(a)}, {r["id"]: i for i, r in enumerate(b)}
    return [{"pair": f"{na}-{nb}", "a_id": x["id"], "b_id": y["id"], "key": x["key"], "a_form": x["gold"], "b_form": y["gold"],
             "a_pattern": x["gold_pattern"], "b_pattern": y["gold_pattern"],
             **{k: int(v) for k, v in flags(x, lex, counts).items() if k != "minority_reading"},
             "a_context": context(a, ia[x["id"]]), "b_context": context(b, ib[y["id"]])}
            for x, y in pairs if x["gold_pattern"] != y["gold_pattern"]]


def sample(rows, n=100, seed=1):
    return random.Random(seed).sample(rows, min(n, len(rows)))


def hand_check(path, rows):
    """Label counts of the labelled sample, or a stale note when it no longer matches the current disagreements."""
    with open(path, encoding="utf8") as f:
        done = list(csv.DictReader(f, delimiter="\t"))
    if {(r["a_id"], r["b_id"]) for r in done} != {(r["a_id"], r["b_id"]) for r in sample(rows)}:
        return {"stale": True, "note": f"{path.name} is not the seed-1 sample of the current disagreements; relabel it"}
    return {"n": len(done), "seed": 1, "labels": dict(collections.Counter(r["label"] for r in done)),
            "labelled_by": sorted({r["labelled_by"] for r in done})}


def main():
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
