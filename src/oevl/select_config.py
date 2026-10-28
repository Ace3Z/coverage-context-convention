"""Apply the encoder selection rule (the selection search and decision D3) to the 24 dev runs and write
results/encoder/selection.json.

The runs are the ByT5 sentence arm, seeds 1 and 2, with and without Zupitza, lr 1e-4 and 3e-4, 1, 3 and 5 epochs
(scripts/encoder_grid.sh). By default their dev numbers come from the "runs" list of the committed selection.json, so
no GPU is needed. --runs DIR reads them from the run folders (DIR/<z|noz>_lr<LR>_ep<N>/sentence_s<seed>) instead.
The logistic john_ws comparison comes from results/wikipron (D3) and results/wikipron_with_zupitza (run_all.sh).
Usage: python -m oevl.select_config [--runs DIR]
"""
import argparse
import json
import statistics
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from oevl.experiment import load

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "encoder" / "selection.json"
RULE = ("Highest mean dev lexicon-ambiguous accuracy over seeds 1-2. Configurations within 0.1 points of the best "
        "count as tied; among them prefer fewer epochs, then keeping Zupitza, then the lower learning rate.")


def pct(x: float | None) -> float | None:
    """A fraction as a percent rounded to two decimals, None stays None."""
    return round(100 * x, 2) if x is not None else None


def read_runs(runs: str | Path, shared: set[str]) -> list[dict[str, Any]]:
    """One row per selection run folder, in the form of the committed "runs" list (scores in percent, two decimals).

    runs: the folder holding <z|noz>_lr<LR>_ep<N>/sentence_s<seed>/, each with the metrics.json and
    john_ws_predictions.jsonl that oevl.encoder_experiment writes. shared: the set of ids of the lexicon-ambiguous dev
    tokens without Zupitza (see main). Asserts that every run used a 2,560-byte cap, batch 32 and only the sentence
    arm, and that every run without Zupitza flags exactly the shared tokens as lexicon-ambiguous. A run counts as with
    Zupitza when its exclude_train list is empty. dev_lexamb scores a run on its own lexicon-ambiguous tokens,
    dev_lexamb_shared on the shared ones. The test scores (sol_*, beo_*) play no part in the rule, but build_report
    quotes those of the collapsed run. "cfg" is the configuration folder name, "minutes" the training time (one
    decimal), and "collapsed" the flag encoder_experiment sets from the soliloquies scores. Every run must have scored
    soliloquies and tauber_beowulf (KeyError otherwise), so "collapsed" is True or False here. Losses are copied as
    they are. Rows are sorted with Zupitza first, then by lr, epochs and seed.
    """
    rows = []
    for m in sorted(Path(runs).glob("*/sentence_s*/metrics.json")):
        d = json.loads(m.read_text(encoding="utf8"))
        a = d["arms"]["sentence"]
        j = a["john_ws"]
        assert d["max_bytes"] == 2560 and d["batch"] == 32 and d["arms"].keys() == {"sentence"}, m
        ps = load(m.parent / "john_ws_predictions.jsonl")
        if d["exclude_train"]:  # every run without Zupitza must flag the shared population
            assert {p["id"] for p in ps if p["lexicon_ambiguous"]} == shared, m
        ok = [p["encoder_sentence"] == p["gold_pattern"] for p in ps if p["id"] in shared]
        rows.append({"cfg": m.parent.parent.name, "zupitza": not d["exclude_train"], "lr": d["lr"],
                     "epochs": d["epochs"], "seed": d["seed"], "train_examples": d["train_examples"],
                     "dev_lexamb": pct(j["lexicon_ambiguous"]["accuracy"]), "dev_lexamb_n": j["lexicon_ambiguous"]["n"],
                     "dev_all": pct(j["accuracy"]["accuracy"]), "dev_classB": pct(j["class_B"]["accuracy"]),
                     "dev_most_long": pct(j["most_long_share"]), "dev_prior_agree": pct(j["prior_agreement"]),
                     "final_loss": a["final_train_loss"], "loss_per_epoch": a["train_loss_per_epoch"],
                     "minutes": round(a["train_seconds"] / 60, 1), "collapsed": a["collapsed"],
                     "sol_all": pct(a["soliloquies"]["accuracy"]["accuracy"]),
                     "sol_lexamb": pct(a["soliloquies"]["lexicon_ambiguous"]["accuracy"]),
                     "beo_all": pct(a["tauber_beowulf"]["accuracy"]["accuracy"]),
                     "beo_lexamb": pct(a["tauber_beowulf"]["lexicon_ambiguous"]["accuracy"]),
                     "dev_lexamb_shared": pct(sum(ok) / len(ok))})
    rows.sort(key=lambda r: (not r["zupitza"], r["lr"], r["epochs"], r["seed"]))
    return rows


def select(rows: Iterable[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Apply RULE to the run rows: every configuration scored best first, those tied with the best, and the choice.

    rows: run dicts as read_runs returns them (only zupitza, lr, epochs, seed, dev_lexamb, dev_lexamb_n,
    dev_lexamb_shared and collapsed are read). Runs are grouped by (zupitza, lr, epochs), and "mean" is the mean of
    the runs' dev_lexamb percents (already rounded to two decimals), rounded to three decimals. The seeds in a group
    are averaged as found, not checked, and dev_lexamb_n is the group's first run's. Configurations at most 0.1 points
    below the best mean count as tied (with a 1e-9 float tolerance), and the choice is the tied one with the fewest
    epochs, then Zupitza kept, then the lower lr. mean_on_shared_7639 (the same mean over dev_lexamb_shared) and
    collapses (the number of runs flagged collapsed) are reported but do not enter the rule.
    Returns (scored, tied, choice), where choice is one of the dicts in tied and scored.
    """
    configs: dict[tuple[bool, float, int], list[Mapping[str, Any]]] = {}
    for r in rows:
        configs.setdefault((r["zupitza"], r["lr"], r["epochs"]), []).append(r)
    scored = [{"zupitza": z, "lr": lr, "epochs": ep, "seeds": [x["seed"] for x in rs],
               "dev_lexamb": [x["dev_lexamb"] for x in rs], "dev_lexamb_n": rs[0]["dev_lexamb_n"],
               "mean": round(statistics.mean(x["dev_lexamb"] for x in rs), 3),
               "mean_on_shared_7639": round(statistics.mean(x["dev_lexamb_shared"] for x in rs), 3),
               "collapses": sum(bool(x["collapsed"]) for x in rs)} for (z, lr, ep), rs in configs.items()]
    scored.sort(key=lambda c: -c["mean"])
    tied = [c for c in scored if scored[0]["mean"] - c["mean"] <= 0.1 + 1e-9]
    choice = min(tied, key=lambda c: (c["epochs"], not c["zupitza"], c["lr"]))
    return scored, tied, choice


def logistic(folder: str, shared: set[str] | None = None) -> dict[str, dict[str, Any]]:
    """Accuracy of the logistic baselines on john_ws from results/<folder>/metrics.json, in percent.

    Returns a dict keyed by population: "lexicon_ambiguous", "accuracy" (all john_ws tokens) and "class_B". Each holds
    n (the population's token count, from the frequency entry), the frequency prior and oracle accuracy, and for the
    word, left and sentence (bilateral) arms the three seed accuracies (two decimals) with their mean and sample sd
    (computed before rounding). With shared (a set of token ids), it also reads
    results/<folder>/john_ws_predictions.jsonl, asserts that the number of rows with a shared id equals len(shared),
    and adds "lexicon_ambiguous_shared_7639": the same models scored on those tokens only (n, frequency, oracle, and
    per arm the mean and sd over seeds 1 to 3, without per-seed values). The key names 7639 whatever n is.
    """
    m = json.loads((ROOT / "results" / folder / "metrics.json").read_text(encoding="utf8"))["john_ws"]
    res: dict[str, dict[str, Any]] = {}
    for pop in ("lexicon_ambiguous", "accuracy", "class_B"):
        def get(k: str) -> dict[str, Any]:
            """The metrics entry k for this population ("accuracy" is the entry's top level, over all tokens)."""
            return m[k] if pop == "accuracy" else m[k][pop]
        res[pop] = {"n": get("frequency")["n"], "frequency": pct(get("frequency")["accuracy"]),
                    "oracle": pct(get("oracle")["accuracy"])}
        for arm in ("word", "left", "sentence"):
            exact = [100 * get(f"{arm}_seed{s}")["accuracy"] for s in (1, 2, 3)]
            res[pop][arm] = {"seeds": [round(x, 2) for x in exact], "mean": round(statistics.mean(exact), 2),
                             "sd": round(statistics.stdev(exact), 2)}
    if shared is not None:
        ps = [d for d in load(ROOT / "results" / folder / "john_ws_predictions.jsonl") if d["id"] in shared]
        assert len(ps) == len(shared)

        def a(k: str) -> float:
            """Percent of the shared tokens whose prediction column k equals the gold pattern."""
            return 100 * sum(d[k] == d["gold_pattern"] for d in ps) / len(ps)
        res["lexicon_ambiguous_shared_7639"] = {
            "n": len(ps), "frequency": round(a("frequency_prediction"), 2), "oracle": round(a("oracle_prediction"), 2),
            **{arm: {"mean": round(statistics.mean(a(f"{arm}_seed{s}_prediction") for s in (1, 2, 3)), 2),
                     "sd": round(statistics.stdev(a(f"{arm}_seed{s}_prediction") for s in (1, 2, 3)), 2)}
               for arm in ("word", "left", "sentence")}}
    return res


def main() -> None:
    """Write results/encoder/selection.json from the selection runs and the logistic D3 comparison, print the choice.

    Reads the lexicon-ambiguous john_ws ids of results/wikipron and results/wikipron_with_zupitza and asserts that the
    first (without Zupitza) is a strict subset of the second. The 24 run rows come from --runs (read_runs) or else
    from the "runs" list of the existing selection.json, which is then overwritten. Exits if there are not 24 rows, or
    if the configuration with the highest mean_on_shared_7639 (the first one on equal values) is not the rule's
    choice, since population_note says both scorings give the same choice. That check does not rerun the tie rule on
    the shared means. Nothing is written when a check fails. The descriptive fields (what, code_commit, command,
    grid, rule, population_note and their D3 counterparts) are fixed text, so the commit bad3695 and the token
    counts 7,782 and 7,639 in population_note are not checked here. The numbers come from the runs and from
    results/wikipron and results/wikipron_with_zupitza (metrics.json and john_ws_predictions.jsonl).
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", help="folder of <z|noz>_lr<LR>_ep<N>/sentence_s<seed> runs (default: the committed list)")
    a = ap.parse_args()
    # the shared population: the lexicon-ambiguous dev tokens without Zupitza, a strict subset of those with it
    amb = {f: {d["id"] for d in load(ROOT / "results" / f / "john_ws_predictions.jsonl") if d["lexicon_ambiguous"]}
           for f in ("wikipron", "wikipron_with_zupitza")}
    shared = amb["wikipron"]
    assert shared < amb["wikipron_with_zupitza"]
    rows = read_runs(a.runs, shared) if a.runs else json.loads(OUT.read_text(encoding="utf8"))["runs"]
    if len(rows) != 24:
        raise SystemExit(f"expected the 24 selection runs, found {len(rows)}")
    scored, tied, choice = select(rows)
    rec = {
        "what": "Model selection for the ByT5 encoder (the selection search and decision D3): sentence arm only, seeds 1 and 2, scored "
                "on the john_ws dev set. The test sets played no part in the choice.",
        "code_commit": "bad3695",
        "command": "python -m oevl.encoder_experiment --arms sentence --seed SEED --max-bytes 2560 --batch 32 --micro 8 "
                   "--lr LR --epochs EPOCHS [--exclude-train zupitza_grammar] --out "
                   "select/<z|noz>_lr<LR>_ep<EPOCHS>/sentence_s<SEED>",
        "grid": {"zupitza": [True, False], "lr": [0.0001, 0.0003], "epochs": [1, 3, 5], "seeds": [1, 2]},
        "rule": RULE,
        "population_note": "With Zupitza the dev lexicon-ambiguous set has 7,782 tokens, without it 7,639, a strict "
                           "subset. 'mean' scores each configuration on its own set as the rule says; "
                           "'mean_on_shared_7639' scores every run on the 7,639 shared tokens. Both give the same "
                           "choice.",
        "choice": {k: choice[k] for k in ("zupitza", "lr", "epochs", "mean", "mean_on_shared_7639")},
        "tied_with_choice": [{k: c[k] for k in ("zupitza", "lr", "epochs", "mean")} for c in tied if c is not choice],
        "configs": scored,
        "runs": rows,
        "d3_logistic_john_ws": {
            "what": "Logistic models (word, left, bilateral = 'sentence') on john_ws with the WikiPron lexicon, seeds "
                    "1-3, trained with and without Zupitza. Percent.",
            "command": "python -m oevl.experiment --lexicon data/lex_wikipron.csv --tests john_ws "
                       "[--exclude-train zupitza_grammar]",
            "code_commit": "bad3695",
            "with_zupitza": {"train_tokens": json.loads((ROOT / "results/wikipron_with_zupitza/metrics.json").read_text(
                encoding="utf8"))["dataset"]["train"], **logistic("wikipron_with_zupitza", shared)},
            "without_zupitza": {"train_tokens": json.loads((ROOT / "results/wikipron/metrics.json").read_text(
                encoding="utf8"))["dataset"]["train"], **logistic("wikipron")},
        },
    }
    if max(scored, key=lambda c: c["mean_on_shared_7639"]) is not choice:
        raise SystemExit("the shared-population scoring picks another configuration: fix population_note")
    OUT.write_text(json.dumps(rec, ensure_ascii=False, indent=1) + "\n", encoding="utf8")
    print(f"choice: {rec['choice']}, tied: {rec['tied_with_choice']}; wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
