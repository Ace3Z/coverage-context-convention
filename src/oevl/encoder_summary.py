"""Pool encoder runs over seeds: accuracy per population, collapse flags and paired statistics.

Reads RUNS/<config>/<arm>_s<seed>/{metrics.json, <set>_predictions.jsonl} (one arm and one seed per folder, as written
by oevl.encoder_experiment), where <config> names the byte cap and anything else that differs (cap128, cap256, capfull,
cap256_nozup). Writes to --out:
  summary.json: per config, arm, held-out set and population the per-seed accuracies, mean, sample SD and n; each
    run's collapse flag, training loss and diagnostics; the frequency prior; and on the two test sets, for
    lexicon-ambiguous and class B tokens, a sentence bootstrap of seed-mean correctness and McNemar per seed.
  <set>_predictions.tsv.gz: one row per token: id, sentence, gold, then per config the prior, the population flags
    and class, then one column per run (<config>/<arm>_s<seed>) with its pattern. Read the patterns as strings.
Every accuracy and paired statistic in summary.json is computed from these tables alone.
Writes to --report: tab_encoder.tex and numbers_encoder.tex (numbers.tex is left alone).
Usage: python -m oevl.encoder_summary --runs DIR --out results/encoder
"""
import argparse
import gzip
import json
import re
import statistics
from pathlib import Path

from oevl.experiment import TESTS, mcnemar
from oevl.stats import bootstrap_diff

ROOT = Path(__file__).resolve().parents[2]
POPS = ("accuracy", "lexicon_ambiguous", "attested_ambiguous", "minority_reading",
        "class_A", "class_B", "class_C", "class_unlisted")
FLAGS = ("lexicon_ambiguous", "attested_ambiguous", "minority_reading")
SETTINGS = ("model", "epochs", "batch", "lr", "max_bytes", "train_limit", "train_examples", "exclude_train")
PAIRS = (("sentence", "word"), ("left", "word"), ("sentence", "left"), ("local2", "word"), ("local5", "word"))
PAIRED_SETS, PAIRED_POPS = ("soliloquies", "tauber_beowulf"), ("lexicon_ambiguous", "class_B")


def arm_order(arm):
    return ("word", "left", "sentence").index(arm) if arm in ("word", "left", "sentence") else 3 + int(arm[5:])


def in_pop(t, cfg, pop):
    if pop == "accuracy":
        return True
    return t[f"{cfg}:class"] == pop[6:] if pop.startswith("class_") else t[f"{cfg}:{pop}"] == "1"


def acc(table, ix, col):
    return sum(table[i][col] == table[i]["gold"] for i in ix) / len(ix) if ix else None


def collect(runs):
    """Read every run; return the per-set token tables and {config: {(arm, seed): metrics}}."""
    tables, metrics = {}, {}
    for cfg in sorted(p for p in Path(runs).iterdir() if p.is_dir()):
        for folder in sorted(cfg.iterdir(), key=lambda p: p.name):
            m = re.fullmatch(r"(.+)_s(\d+)", folder.name)
            if not m:
                continue
            arm, seed = m[1], int(m[2])
            if not (folder / "metrics.json").exists():
                raise SystemExit(f"{folder}: no metrics.json (unfinished run? move it out of {runs})")
            d = json.loads((folder / "metrics.json").read_text(encoding="utf8"))
            if d["seed"] != seed or arm not in d["arms"]:
                raise SystemExit(f"{folder}: metrics.json is not arm {arm}, seed {seed}")
            metrics.setdefault(cfg.name, {})[arm, seed] = d
            for s in TESTS:
                if s not in d["arms"][arm]:
                    continue
                with open(folder / f"{s}_predictions.jsonl", encoding="utf8") as f:
                    rows = [json.loads(line) for line in f]
                table = tables.setdefault(s, [{"id": r["id"], "sentence_id": r["sentence_id"], "gold": r["gold_pattern"]}
                                              for r in rows])
                if [t["id"] for t in table] != [r["id"] for r in rows]:
                    raise SystemExit(f"{folder}: the {s} tokens differ from the other runs")
                for t, r in zip(table, rows):
                    pop = {f"{cfg.name}:frequency": r["frequency"], **{f"{cfg.name}:{k}": str(int(r[k])) for k in FLAGS},
                           f"{cfg.name}:class": r["class"] or ""}
                    for k, v in pop.items():
                        if t.setdefault(k, v) != v:
                            raise SystemExit(f"{folder}: {s} {k} differs from the other runs of {cfg.name}")
                    t[f"{cfg.name}/{arm}_s{seed}"] = r[f"encoder_{arm}"]
    for cfg, runs_ in metrics.items():
        if len({json.dumps([d.get(k) for k in SETTINGS]) for d in runs_.values()}) > 1:
            raise SystemExit(f"{cfg}: runs differ in {SETTINGS}; put them in separate config folders")
    return tables, metrics


def summarise(tables, metrics):
    out = {}
    for cfg, runs in metrics.items():
        arms = sorted({a for a, _ in runs}, key=arm_order)
        seeds = {a: sorted(s for b, s in runs if b == a) for a in arms}
        C = out[cfg] = {"settings": {k: next(iter(runs.values())).get(k) for k in SETTINGS}, "frequency": {},
                        "arms": {a: {"seeds": seeds[a], "collapsed": [runs[a, s]["arms"][a]["collapsed"] for s in seeds[a]],
                                     "final_train_loss": [runs[a, s]["arms"][a]["final_train_loss"] for s in seeds[a]],
                                     "train_loss_per_epoch": [runs[a, s]["arms"][a]["train_loss_per_epoch"]
                                                              for s in seeds[a]], "sets": {}} for a in arms},
                        "paired": {}}
        for a in arms:
            C["arms"][a]["n_collapsed"] = sum(bool(x) for x in C["arms"][a]["collapsed"])
        for s, table in tables.items():
            if f"{cfg}:frequency" not in table[0]:
                continue
            idx = {p: [i for i, t in enumerate(table) if in_pop(t, cfg, p)] for p in POPS}
            C["frequency"][s] = {p: {"n": len(ix), "accuracy": acc(table, ix, f"{cfg}:frequency")} for p, ix in idx.items()}
            for a in arms:
                res = C["arms"][a]["sets"][s] = {k: [runs[a, sd]["arms"][a][s][k] for sd in seeds[a]]
                                                 for k in ("most_long_share", "prior_agreement")}
                for p, ix in idx.items():
                    vals = [acc(table, ix, f"{cfg}/{a}_s{sd}") for sd in seeds[a]]
                    for v, sd in zip(vals, seeds[a]):  # the tables must reproduce each run's own numbers
                        own = runs[a, sd]["arms"][a][s][p]["accuracy"]
                        if (v is None) != (own is None) or (v is not None and abs(v - own) > 1e-6):
                            raise SystemExit(f"{cfg}/{a}_s{sd} {s} {p}: {v} from the predictions, {own} in metrics.json")
                    ok = None not in vals
                    res[p] = {"n": len(ix), "values": vals, "mean": statistics.mean(vals) if ok else None,
                              "sd": statistics.stdev(vals) if ok and len(vals) > 1 else None}
            if s not in PAIRED_SETS:
                continue
            for p in PAIRED_POPS:
                ix = idx[p]
                gold = [table[i]["gold"] for i in ix]

                def col(system, sd):
                    return [table[i][f"{cfg}:frequency" if system == "frequency" else f"{cfg}/{system}_s{sd}"] for i in ix]

                def correct(system):  # per token, averaged over the system's seeds
                    cols = [col(system, sd) for sd in seeds.get(system, [None])]
                    return [sum(c[j] == g for c in cols) / len(cols) for j, g in enumerate(gold)]

                res = C["paired"].setdefault(s, {})[p] = {}
                for x, y in PAIRS + tuple((a, "frequency") for a in arms):
                    if x not in arms or (y != "frequency" and y not in arms):
                        continue
                    both = [sd for sd in seeds[x] if y == "frequency" or sd in seeds[y]]
                    mc = [{"seed": sd, **mcnemar(col(x, sd), col(y, sd), gold)} for sd in both]
                    res[f"{x}-{y}"] = {"n": len(ix), **(bootstrap_diff([table[i]["sentence_id"] for i in ix], correct(x),
                                                                       correct(y)) or {"diff": None, "lo": None, "hi": None}),
                                       "mcnemar": mc, "n_seeds_p_below_0.05": int(sum(r["p_exact"] < 0.05 for r in mc))}
    return out


def pct(x, signed=False):
    return "" if x is None else f"{100 * x:+.2f}" if signed else f"{100 * x:.2f}"


DIGITS = dict(zip("0123456789", ("Zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine")))
SET_TAG = {"john_ws": "Dev", "soliloquies": "Sol", "tauber_beowulf": "Beo"}
POP_TAG = {"accuracy": "All", "lexicon_ambiguous": "Lex", "attested_ambiguous": "Att", "minority_reading": "Min",
           "class_A": "ClsA", "class_B": "ClsB", "class_C": "ClsC", "class_unlisted": "ClsU"}
NAMES = {"frequency": "Frequency", "word": "Word", "left": "Left", "sentence": "Bilateral"}


def tag(name):
    """LaTeX macro names are letters only: cap128 -> CapOneTwoEight, sentence -> Bil (as in numbers.tex)."""
    return {"sentence": "Bil", "frequency": "Freq"}.get(name) or "".join(DIGITS.get(c, c) for c in name.title()
                                                                        if c.isalnum())


def numbers(summary):
    num = {}

    def put(k, v):
        if k in num:
            raise SystemExit(f"macro {k} defined twice")
        num[k] = v

    for cfg, C in summary.items():
        c = "Enc" + tag(cfg)
        for a, A in C["arms"].items():
            put(f"{c}{tag(a)}Seeds", str(len(A["seeds"])))
            put(f"{c}{tag(a)}Collapsed", str(A["n_collapsed"]))
        for s, F in C["frequency"].items():
            for p, f in F.items():
                k = f"{c}{SET_TAG[s]}{POP_TAG[p]}"
                put(k + "N", f"{f['n']:,}")
                put(k + "Freq", pct(f["accuracy"]))
                for a, A in C["arms"].items():
                    put(k + tag(a), pct(A["sets"][s][p]["mean"]))
                    put(k + tag(a) + "Sd", pct(A["sets"][s][p]["sd"]))
        for s, P in C["paired"].items():
            for p, comparisons in P.items():
                for name, r in comparisons.items():
                    x, y = name.split("-")
                    k = f"{c}{SET_TAG[s]}{POP_TAG[p]}{tag(x)}Vs{tag(y)}"
                    put(k, pct(r["diff"], True))
                    put(k + "Lo", pct(r["lo"], True))
                    put(k + "Hi", pct(r["hi"], True))
                    put(k + "Sig", str(r["n_seeds_p_below_0.05"]))
    return "% generated by src/oevl/encoder_summary.py from the encoder runs (see results/encoder/summary.json); " \
           "do not edit\n" + "".join(f"\\newcommand{{\\N{k}}}{{{v}}}\n" for k, v in sorted(num.items()))


def table_tex(summary):
    sets = (("soliloquies", "\\textit{Soliloquies} (prose)"), ("tauber_beowulf", "\\textit{Beowulf} (verse)"))
    pops = (("accuracy", "All"), ("lexicon_ambiguous", "Lex.-amb."), ("class_B", "Class B"))
    lines = ["\\begin{tabular}{@{}llrrrrrrrr@{}}", "\\toprule",
             " & & & & " + " & ".join(f"\\multicolumn{{3}}{{c}}{{{t}}}" for _, t in sets) + " \\\\",
             "\\cmidrule(lr){5-7}\\cmidrule(l){8-10}",
             "Cap & System & Seeds & Collapsed & " + " & ".join(lab for _ in sets for _, lab in pops) + " \\\\",
             "\\midrule"]
    for k, (cfg, C) in enumerate(summary.items()):
        if k:
            lines.append("\\addlinespace")
        cells = [pct(C["frequency"].get(s, {}).get(p, {}).get("accuracy")) for s, _ in sets for p, _ in pops]
        label = cfg.removeprefix("cap").replace("_", "\\_")
        lines.append(f"{label} & Frequency & & & " + " & ".join(cells) + " \\\\")
        for a, A in C["arms"].items():
            cells = []
            for s, _ in sets:
                for p, _ in pops:
                    r = A["sets"].get(s, {}).get(p, {})
                    cells.append(pct(r.get("mean")) + ("" if r.get("sd") is None else f"$\\pm${pct(r['sd'])}"))
            name = NAMES.get(a) or f"Local {a[5:]}"
            lines.append(f" & {name} & {len(A['seeds'])} & {A['n_collapsed']} & " + " & ".join(cells) + " \\\\")
    return "\n".join(lines + ["\\bottomrule", "\\end{tabular}"]) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True, help="folder of <config>/<arm>_s<seed> runs")
    ap.add_argument("--out", default=str(ROOT / "results" / "encoder"))
    ap.add_argument("--report", default=str(ROOT / "report" / "generated"), help="where the .tex files go")
    a = ap.parse_args()
    tables, metrics = collect(a.runs)
    if not metrics:
        raise SystemExit(f"no <config>/<arm>_s<seed> runs under {a.runs}")
    out, rep = Path(a.out), Path(a.report)
    out.mkdir(parents=True, exist_ok=True)
    rep.mkdir(parents=True, exist_ok=True)
    for s, table in tables.items():
        cols = list(table[0])
        text = "\t".join(cols) + "\n" + "".join("\t".join(t[c] for c in cols) + "\n" for t in table)
        (out / f"{s}_predictions.tsv.gz").write_bytes(gzip.compress(text.encode("utf8"), mtime=0))  # byte-stable
    summary = summarise(tables, metrics)
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf8")
    (rep / "tab_encoder.tex").write_text(table_tex(summary), encoding="utf8")
    (rep / "numbers_encoder.tex").write_text(numbers(summary), encoding="utf8")
    print(f"wrote {out / 'summary.json'}, {len(tables)} prediction tables, {rep / 'tab_encoder.tex'} and "
          f"{rep / 'numbers_encoder.tex'}")
    for cfg, C in summary.items():
        for arm, A in C["arms"].items():
            if A["n_collapsed"]:
                print(f"WARNING {cfg} {arm}: {A['n_collapsed']} of {len(A['seeds'])} runs collapsed "
                      f"(most_long_share >= 0.9 on soliloquies)")


if __name__ == "__main__":
    main()
