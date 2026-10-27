"""Pool encoder runs over seeds: accuracy per population, collapse flags and paired statistics.

Reads RUNS/<config>/<arm>_s<seed>/{metrics.json, <set>_predictions.jsonl} (files of oevl.encoder_experiment, one arm
and one seed per folder as scripts/encoder_grid.sh lays them out), where <config> names the byte cap and anything else
that differs (cap128, cap256, capfull). Writes to --out:
  summary.json: per config its settings, the frequency prior's accuracy per held-out set and population, and per arm
    each run's collapse flag, training losses and diagnostics and, per held-out set and population, the per-seed
    accuracies, mean, sample SD and n. On the two test sets it adds, for lexicon-ambiguous and class B tokens, a
    sentence bootstrap of seed-mean correctness and McNemar per seed.
  <set>_predictions.tsv.gz: one row per token: id, sentence_id, gold, then for each config its prior
    (<config>:frequency), population flags and class, followed by one column per run of that config
    (<config>/<arm>_s<seed>) with its pattern. Read the patterns as strings.
Every accuracy and paired statistic in summary.json is computed from these tables alone, except that minority_reading
is recomputed from the data with experiment.flags (see minority()). The tables keep the flag the runs recorded,
which for the committed runs is the same.
local2 and local5 never reach a cap of the grid (see REPLICATES), so their runs at the caps of one config group (the
configs that differ only in max_bytes) are replicates: summary.json labels them (identical_inputs_at), pools them
under local_replicates and lists their spread per seed under replicate_spread. The table and the numbers give them,
like every other arm, at the full cap (the largest cap of their group) over its seeds, and the numbers also give them
pooled (EncRep...). identical_runs counts the distinct prediction sets among all runs and lists the runs whose
predictions agree on every token.
Writes to --report: tab_encoder.tex and numbers_encoder.tex (numbers.tex is left alone).
Usage: python -m oevl.encoder_summary --runs DIR --out results/encoder
       python -m oevl.encoder_summary --from-tsv   (no GPU runs: recomputes summary.json and the .tex files from the
         committed tables, taking from summary.json the arms and seeds of each config and what the tables cannot
         hold: settings, collapse flags, losses, most_long_share and prior_agreement. Its per-seed accuracies are
         read only to check the tables against them, and summary.json is overwritten in place. run_all.sh does this.)
"""
import argparse
import gzip
import json
import re
import statistics
from collections.abc import Collection
from pathlib import Path
from typing import Any

from oevl.experiment import TESTS, build_data, flags, heldout, lexicon, mcnemar
from oevl.stats import bootstrap_diff

ROOT = Path(__file__).resolve().parents[2]
POPS = ("accuracy", "lexicon_ambiguous", "attested_ambiguous", "minority_reading",
        "class_A", "class_B", "class_C", "class_unlisted")
FLAGS = ("lexicon_ambiguous", "attested_ambiguous", "minority_reading")
SETTINGS = ("model", "epochs", "batch", "lr", "max_bytes", "train_limit", "train_examples", "exclude_train")
PAIRS = (("sentence", "word"), ("left", "word"), ("sentence", "left"), ("local2", "word"), ("local5", "word"))
PAIRED_SETS, PAIRED_POPS = ("soliloquies", "tauber_beowulf"), ("lexicon_ambiguous", "class_B")
# make_input gives local2 at most 57 and local5 at most 103 bytes on these data (tests/test_encoder.py checks it), so
# configs that differ only in max_bytes, at caps of 103 bytes or more as in the grid, give these arms identical inputs,
# data order and seed. replicates() does not check the caps itself.
REPLICATES = ("local2", "local5")
REPLICATE_NOTE = ("local2 and local5 inputs never exceed 103 bytes (local2 57), below every cap, so the runs of one "
                  "arm and seed at caps 128, 256 and full share inputs, data order and seed: they are replicates, not "
                  "different caps. Their per-config entries are kept as run. Here each arm is pooled over all its runs "
                  "(mean and sample SD over runs), and the bootstrap uses each token's correctness averaged over them.")
BLOCKS = ("local_replicates", "replicate_spread", "identical_runs")  # summary.json keys that are not configs


def configs(summary: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """The per-config entries of a summary, leaving out the BLOCKS keys."""
    return {k: v for k, v in summary.items() if k not in BLOCKS}


def arm_order(arm: str) -> int:
    """Sort key for an arm name: word, left, sentence, then localK in order of K."""
    return ("word", "left", "sentence").index(arm) if arm in ("word", "left", "sentence") else 3 + int(arm[5:])


def in_pop(t: dict[str, str], cfg: str, pop: str) -> bool:
    """Whether table row t belongs to population pop under config cfg.

    accuracy takes every token, class_X the tokens whose <cfg>:class cell is X (empty for tokens that are not
    lexicon-ambiguous), and a flag population the tokens whose <cfg>:<pop> cell is "1"."""
    if pop == "accuracy":
        return True
    return t[f"{cfg}:class"] == pop[6:] if pop.startswith("class_") else t[f"{cfg}:{pop}"] == "1"


def acc(table: list[dict[str, str]], ix: list[int], col: str) -> float | None:
    """Share of the rows ix of table whose column col equals the gold pattern, or None when ix is empty."""
    return sum(table[i][col] == table[i]["gold"] for i in ix) / len(ix) if ix else None


def collect(runs: str | Path) -> tuple[dict[str, list[dict[str, str]]],
                                       dict[str, dict[tuple[str, int], dict[str, Any]]]]:
    """Read every run under the folder runs into per-token tables and run records.

    runs holds one folder per config with <arm>_s<seed> run folders inside (other names are skipped). Returns
    (tables, metrics). tables maps each held-out set to one dict per token, in the runs' token order, with id,
    sentence_id, gold and, per config, <cfg>:frequency (the prior's pattern), the three FLAGS as "0" or "1" and
    <cfg>:class ("" for tokens that are not lexicon-ambiguous), then one <cfg>/<arm>_s<seed> pattern per run. A set
    the run did not score is skipped for that run, and sentence_id and gold come from the first run that scored the
    set. metrics maps each config to {(arm, seed): parsed metrics.json}. Exits with an error if a run has no
    metrics.json (an unfinished run) or one for another arm or seed, if its tokens or a config's prior, flags or
    classes differ from the other runs, or if the runs of a config differ in SETTINGS."""
    tables: dict[str, list[dict[str, str]]]
    metrics: dict[str, dict[tuple[str, int], dict[str, Any]]]
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


def read_tables(out: str | Path) -> dict[str, list[dict[str, str]]]:
    """The <set>_predictions.tsv.gz tables main() wrote to folder out, back as collect() built them.

    Returns {set: rows} with every cell a string, leaving out a set with no table file."""
    tables: dict[str, list[dict[str, str]]] = {}
    for s in TESTS:
        p = Path(out) / f"{s}_predictions.tsv.gz"
        if p.exists():
            head, *lines = gzip.decompress(p.read_bytes()).decode("utf8").split("\n")[:-1]
            cols = head.split("\t")
            tables[s] = [dict(zip(cols, line.split("\t"), strict=True)) for line in lines]
    return tables


def runs_from_summary(summary: dict[str, dict[str, Any]]) -> dict[str, dict[tuple[str, int], dict[str, Any]]]:
    """Rebuild each run's record from summary.json in the shape collect() reads from metrics.json.

    Returns {config: {(arm, seed): record}}. Only the fields minority() and summarise() read are rebuilt: the
    settings (minority() needs exclude_train), the seed, the collapse flag, the losses and, per set,
    most_long_share, prior_agreement and each population's accuracy.
    The per-seed accuracies come along, so summarise() still checks the tables against them."""
    metrics: dict[str, dict[tuple[str, int], dict[str, Any]]] = {}
    for cfg, C in configs(summary).items():
        for a, A in C["arms"].items():
            for i, sd in enumerate(A["seeds"]):
                arm = {k: A[k][i] for k in ("collapsed", "final_train_loss", "train_loss_per_epoch")}
                for s, S in A["sets"].items():
                    arm[s] = {k: v[i] if isinstance(v, list) else {"accuracy": v["values"][i]} for k, v in S.items()}
                metrics.setdefault(cfg, {})[a, sd] = {**C["settings"], "seed": sd, "arms": {a: arm}}
    return metrics


def minority(tables: dict[str, list[dict[str, str]]],
             metrics: dict[str, dict[tuple[str, int], dict[str, Any]]]) -> set[str]:
    """Recompute each config's minority_reading column with experiment.flags on the data, in the tables in memory.

    The training counts come from build_data(ROOT / "data") without the config's exclude_train works, and each
    token's flag is looked up by id among the held-out sets in data/, so a table id that is not there raises
    KeyError. main() calls this after writing the tables, so the files keep the recorded flag. Returns the set of
    configs whose recorded column differed, which is empty for the committed runs."""
    by_ex, stale = {}, set()
    for cfg, runs in metrics.items():
        ex = tuple(next(iter(runs.values())).get("exclude_train") or ())
        if ex not in by_ex:
            counts = lexicon(build_data(ROOT / "data", ex)[0])[1]  # the flag needs only the training counts
            by_ex[ex] = {r["id"]: flags(r, {}, counts)["minority_reading"] for s in tables for r in heldout(ROOT / "data", s)}
        for table in tables.values():
            if f"{cfg}:frequency" in table[0]:
                for t in table:
                    flag = str(int(by_ex[ex][t["id"]]))
                    if t[f"{cfg}:minority_reading"] != flag:
                        stale.add(cfg)
                    t[f"{cfg}:minority_reading"] = flag
    return stale


def summarise(tables: dict[str, list[dict[str, str]]], metrics: dict[str, dict[tuple[str, int], dict[str, Any]]],
              stale: Collection[str] = ()) -> dict[str, dict[str, Any]]:
    """Pool every config's runs over seeds and pair the systems, from the token tables.

    tables: {set: rows} as collect() builds them. metrics: {config: {(arm, seed): record}}. stale: configs whose
    minority_reading column minority() replaced, so their minority accuracies are not checked against the runs.
    Returns per config the settings, the prior's n and accuracy per set and population ("frequency"), per arm
    ("arms") the seeds, collapse flags, n_collapsed, losses and, per set, most_long_share, prior_agreement and each
    population's n, per-seed values, mean and sample SD (None for an empty population, SD also None with one seed),
    and "paired". paired covers PAIRED_SETS and PAIRED_POPS: every pair in PAIRS whose arms both ran, and every arm
    against the prior, gets a sentence bootstrap of each token's correctness averaged over each system's seeds
    (diff, lo, hi), McNemar on each seed both systems ran, and how many of those seeds give p < 0.05. Sets without
    the config's columns are skipped. Exits with an error if an accuracy from the tables differs from the run's own
    by more than 1e-6."""
    out: dict[str, dict[str, Any]] = {}
    res: Any  # reused below for two JSON blocks of different shape
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
                        if p == "minority_reading" and cfg in stale:  # the runs scored the flag they recorded
                            continue
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

                def col(system: str, sd: int | None) -> list[str]:
                    """The patterns of the population's tokens in run (system, sd), or in the prior for frequency."""
                    return [table[i][f"{cfg}:frequency" if system == "frequency" else f"{cfg}/{system}_s{sd}"] for i in ix]

                def correct(system: str) -> list[float]:
                    """Each population token's correctness, averaged over the system's seeds (the prior is one run)."""
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


def replicates(tables: dict[str, list[dict[str, str]]], summary: dict[str, dict[str, Any]]) -> None:
    """Label, pool and spread the REPLICATES arms over configs that differ only in max_bytes, adding to summary.

    summary is summarise()'s output. Configs are grouped by their settings without max_bytes, and an arm of
    REPLICATES that ran in more than one config of a group has replicates. Each of its per-config entries gets
    identical_inputs_at (those configs). summary["local_replicates"] holds REPLICATE_NOTE and, per arm, its configs,
    runs, n_collapsed and per set and population the n, mean and sample SD over all its runs, plus in "paired" a
    sentence bootstrap against the word arm (all word runs of the group, when there are any) and the prior on
    PAIRED_SETS and PAIRED_POPS, with each token's correctness averaged over the runs (no McNemar).
    summary["replicate_spread"] gives per arm and seed the min, max and per-config value on those sets and
    populations. Nothing is added when no arm has replicates. Exits with an error if local2 and local5 have
    replicates in different groups, or if the group's configs differ in the prior or the populations. An arm with
    replicates in more than one group is pooled over the last of them only, and the caps are not checked against
    the arm's longest input (see REPLICATES)."""
    groups: dict[str, list[str]] = {}
    for cfg, C in summary.items():
        groups.setdefault(json.dumps({k: v for k, v in C["settings"].items() if k != "max_bytes"}), []).append(cfg)
    found = {a: g for g in groups.values() for a in REPLICATES if sum(a in summary[c]["arms"] for c in g) > 1}
    if not found:
        return
    group = next(iter(found.values()))
    if any(g != group for g in found.values()):  # ponytail: the grid has one group. Key the blocks by group if more
        raise SystemExit("replicate arms in more than one group of configs")
    pooled: dict[str, Any]
    spread: dict[str, Any]
    pooled, spread = {"note": REPLICATE_NOTE, "arms": {}, "paired": {}}, {}
    cols: dict[str, list[str]] = {}
    for a in sorted(found, key=arm_order):
        cfgs = [c for c in group if a in summary[c]["arms"]]
        for c in cfgs:
            summary[c]["arms"][a]["identical_inputs_at"] = cfgs
        cols[a] = [f"{c}/{a}_s{sd}" for c in cfgs for sd in summary[c]["arms"][a]["seeds"]]
        P = pooled["arms"][a] = {"configs": cfgs, "runs": cols[a],
                                 "n_collapsed": sum(summary[c]["arms"][a]["n_collapsed"] for c in cfgs), "sets": {}}
        for s in summary[cfgs[0]]["arms"][a]["sets"]:
            for p in POPS:
                v = [x for c in cfgs for x in summary[c]["arms"][a]["sets"][s][p]["values"]]
                ok = None not in v
                P["sets"].setdefault(s, {})[p] = {"n": summary[cfgs[0]]["arms"][a]["sets"][s][p]["n"],
                                                  "mean": statistics.mean(v) if ok else None,
                                                  "sd": statistics.stdev(v) if ok else None}
        for sd in sorted({sd for c in cfgs for sd in summary[c]["arms"][a]["seeds"]}):
            have = [c for c in cfgs if sd in summary[c]["arms"][a]["seeds"]]
            for s in (s for s in PAIRED_SETS if s in P["sets"]):
                for p in PAIRED_POPS:
                    by = {c: summary[c]["arms"][a]["sets"][s][p]["values"][summary[c]["arms"][a]["seeds"].index(sd)]
                          for c in have}
                    if None not in by.values():
                        spread.setdefault(a, {}).setdefault(f"s{sd}", {}).setdefault(s, {})[p] = {
                            "min": min(by.values()), "max": max(by.values()), "by_config": by}
    word = [f"{c}/word_s{sd}" for c in group if "word" in summary[c]["arms"] for sd in summary[c]["arms"]["word"]["seeds"]]
    c0 = group[0]
    for s, table in tables.items():
        if f"{c0}:frequency" not in table[0]:
            continue
        for c in group:  # one prior and one set of populations for the whole group
            if any(t[f"{c}:{k}"] != t[f"{c0}:{k}"] for t in table for k in ("frequency", "class") + FLAGS):
                raise SystemExit(f"{s}: {c} and {c0} differ in the prior or the populations")
        if s not in PAIRED_SETS:
            continue
        for p in PAIRED_POPS:
            ix = [i for i, t in enumerate(table) if in_pop(t, c0, p)]

            def correct(cs: list[str]) -> list[float]:
                """Each population token's correctness, averaged over the table columns cs (runs or the prior)."""
                return [sum(table[i][c] == table[i]["gold"] for c in cs) / len(cs) for i in ix]

            res = pooled["paired"].setdefault(s, {})[p] = {}
            for a in cols:
                for y, ys in (("word", word), ("frequency", [f"{c0}:frequency"])):
                    if ys:
                        res[f"{a}-{y}"] = {"n": len(ix), **(bootstrap_diff(
                            [table[i]["sentence_id"] for i in ix], correct(cols[a]), correct(ys))
                            or {"diff": None, "lo": None, "hi": None})}
    summary["local_replicates"], summary["replicate_spread"] = pooled, spread


def identical_runs(tables: dict[str, list[dict[str, str]]], summary: dict[str, dict[str, Any]]) -> None:
    """Group the runs whose predictions agree on every token of every set, adding summary["identical_runs"].

    The runs are the <config>/<arm>_s<seed> columns of the first table. The entry gives their number ("runs"), the
    number of distinct prediction sets ("distinct") and the groups of two or more identical runs ("groups")."""
    runs = [k for k in next(iter(tables.values()))[0] if "/" in k]
    groups: dict[tuple[str | None, ...], list[str]] = {}
    for k in runs:
        groups.setdefault(tuple(t.get(k) for table in tables.values() for t in table), []).append(k)
    summary["identical_runs"] = {"runs": len(runs), "distinct": len(groups),
                                 "groups": [g for g in groups.values() if len(g) > 1]}


def pct(x: float | None, signed: bool = False) -> str:
    """A fraction as a percentage with two decimals, with a sign when signed, or "" for None."""
    return "" if x is None else f"{100 * x:+.2f}" if signed else f"{100 * x:.2f}"


DIGITS = dict(zip("0123456789", ("Zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine")))
SET_TAG = {"john_ws": "Dev", "soliloquies": "Sol", "tauber_beowulf": "Beo"}
POP_TAG = {"accuracy": "All", "lexicon_ambiguous": "Lex", "attested_ambiguous": "Att", "minority_reading": "Min",
           "class_A": "ClsA", "class_B": "ClsB", "class_C": "ClsC", "class_unlisted": "ClsU"}
NAMES = {"frequency": "Frequency", "word": "Word", "left": "Left", "sentence": "Bilateral"}


def tag(name: str) -> str:
    """A config or system name as letters for a LaTeX macro name: cap128 -> CapOneTwoEight, local2 -> LocalTwo.

    sentence becomes Bil and frequency Freq, as in numbers.tex. Any other name is title-cased, its digits are spelt
    out and every other character that is not a letter is dropped."""
    return {"sentence": "Bil", "frequency": "Freq"}.get(name) or "".join(DIGITS.get(c, c) for c in name.title()
                                                                        if c.isalnum())


def full_caps(summary: dict[str, dict[str, Any]]) -> dict[str, str]:
    """Map each replicate arm to its full cap, the config of identical_inputs_at with the largest max_bytes.

    That is the only config whose table rows and per-config numbers show the arm."""
    return {a: max(A["identical_inputs_at"], key=lambda c: summary[c]["settings"]["max_bytes"])
            for C in configs(summary).values() for a, A in C["arms"].items() if "identical_inputs_at" in A}


def numbers(summary: dict[str, dict[str, Any]]) -> str:
    """The text of numbers_encoder.tex: a do-not-edit comment, then one LaTeX macro N<key> per reported number, sorted.

    Per config (Enc<Config>...): the seeds and collapsed runs of each arm, then per set and population the n, the
    prior's accuracy (Freq), each arm's mean and SD, and for each paired comparison the difference, its interval
    (Lo, Hi) and how many seeds give McNemar p < 0.05 (Sig). A replicate arm and its comparisons appear per config
    only at its full cap. EncRep... gives each replicate arm pooled over all its runs (runs, collapsed runs, mean and
    SD, and the differences to the word arm and the prior with Lo and Hi but no Sig) and its widest min-max spread
    over seeds, EncRunsSolClsB...Min and Max the range of prose class B accuracy over every run of an arm at every
    cap, and EncRunsDistinct the number of distinct prediction sets. Values are percentages with two decimals
    (signed for differences), empty for None, n has thousands separators and counts are plain integers. Exits with
    an error if a macro would be defined twice."""
    num: dict[str, str] = {}

    def put(k: str, v: str) -> None:
        """Record macro k with value v, exiting with an error if k is already set."""
        if k in num:
            raise SystemExit(f"macro {k} defined twice")
        num[k] = v

    full = full_caps(summary)  # a replicate arm per config only at its full cap, and all its runs pooled below
    for cfg, C in configs(summary).items():
        c = "Enc" + tag(cfg)
        arms = {a: A for a, A in C["arms"].items() if full.get(a, cfg) == cfg}
        for a, A in arms.items():
            put(f"{c}{tag(a)}Seeds", str(len(A["seeds"])))
            put(f"{c}{tag(a)}Collapsed", str(A["n_collapsed"]))
        for s, F in C["frequency"].items():
            for p, f in F.items():
                k = f"{c}{SET_TAG[s]}{POP_TAG[p]}"
                put(k + "N", f"{f['n']:,}")
                put(k + "Freq", pct(f["accuracy"]))
                for a, A in arms.items():
                    put(k + tag(a), pct(A["sets"][s][p]["mean"]))
                    put(k + tag(a) + "Sd", pct(A["sets"][s][p]["sd"]))
        for s, P in C["paired"].items():
            for p, comparisons in P.items():
                for name, r in comparisons.items():
                    x, y = name.split("-")
                    if x not in arms or (y != "frequency" and y not in arms):
                        continue
                    k = f"{c}{SET_TAG[s]}{POP_TAG[p]}{tag(x)}Vs{tag(y)}"
                    put(k, pct(r["diff"], True))
                    put(k + "Lo", pct(r["lo"], True))
                    put(k + "Hi", pct(r["hi"], True))
                    put(k + "Sig", str(r["n_seeds_p_below_0.05"]))
    R = summary.get("local_replicates", {"arms": {}, "paired": {}})
    for a, A in R["arms"].items():
        put(f"EncRep{tag(a)}Runs", str(len(A["runs"])))
        put(f"EncRep{tag(a)}Collapsed", str(A["n_collapsed"]))
        for s, S in A["sets"].items():
            for p, r in S.items():
                put(f"EncRep{SET_TAG[s]}{POP_TAG[p]}{tag(a)}", pct(r["mean"]))
                put(f"EncRep{SET_TAG[s]}{POP_TAG[p]}{tag(a)}Sd", pct(r["sd"]))
    for s, P in R["paired"].items():
        for p, comparisons in P.items():
            for name, r in comparisons.items():
                x, y = name.split("-")
                k = f"EncRep{SET_TAG[s]}{POP_TAG[p]}{tag(x)}Vs{tag(y)}"
                put(k, pct(r["diff"], True))
                put(k + "Lo", pct(r["lo"], True))
                put(k + "Hi", pct(r["hi"], True))
    for a in sorted({a for C in configs(summary).values() for a in C["arms"]}, key=arm_order):  # every run of an arm
        vals = [v for C in configs(summary).values() if a in C["arms"]
                for v in C["arms"][a]["sets"].get("soliloquies", {}).get("class_B", {}).get("values", [])]
        if vals:
            put(f"EncRunsSolClsB{tag(a)}Min", pct(min(vals)))
            put(f"EncRunsSolClsB{tag(a)}Max", pct(max(vals)))
    if "identical_runs" in summary:
        put("EncRunsDistinct", str(summary["identical_runs"]["distinct"]))
    for a, seeds in summary.get("replicate_spread", {}).items():  # the widest min-max range over the seeds
        for s in PAIRED_SETS:
            for p in PAIRED_POPS:
                ranges = [S[s][p]["max"] - S[s][p]["min"] for S in seeds.values() if p in S.get(s, {})]
                if ranges:
                    put(f"EncRep{SET_TAG[s]}{POP_TAG[p]}{tag(a)}Spread", pct(max(ranges)))
    return "% generated by src/oevl/encoder_summary.py from the encoder runs (see results/encoder/summary.json); " \
           "do not edit\n" + "".join(f"\\newcommand{{\\N{k}}}{{{v}}}\n" for k, v in sorted(num.items()))


def table_tex(summary: dict[str, dict[str, Any]]) -> str:
    """The LaTeX tabular of tab_encoder.tex: per config the prior and each arm on the two test sets.

    Columns: cap (the config name without "cap"), system, runs, collapsed runs, then for the Soliloquies and Beowulf
    the accuracy in percent on all, lexicon-ambiguous and class B tokens. The prior's row carries the cap, and each
    arm's row gives the mean over seeds plus or minus the sample SD. A replicate arm appears only at its full cap.
    Missing values are left empty."""
    sets = (("soliloquies", "\\textit{Soliloquies} (prose)"), ("tauber_beowulf", "\\textit{Beowulf} (verse)"))
    pops = (("accuracy", "All"), ("lexicon_ambiguous", "Lex.-amb."), ("class_B", "Class B"))
    lines = ["\\begin{tabular}{@{}llrrrrrrrr@{}}", "\\toprule",
             " & & & & " + " & ".join(f"\\multicolumn{{3}}{{c}}{{{t}}}" for _, t in sets) + " \\\\",
             "\\cmidrule(lr){5-7}\\cmidrule(l){8-10}",
             "Cap & System & Runs & Collapsed & " + " & ".join(lab for _ in sets for _, lab in pops) + " \\\\",
             "\\midrule"]

    def row(a: str, A: dict[str, Any]) -> str:
        """The row of arm a (its summary entry A), with an empty cap cell."""
        cells = []
        for s, _ in sets:
            for p, _ in pops:
                r = A["sets"].get(s, {}).get(p, {})
                cells.append(pct(r.get("mean")) + ("" if r.get("sd") is None else f"$\\pm${pct(r['sd'])}"))
        name = NAMES.get(a) or f"Local {a[5:]}"
        return f" & {name} & {len(A['seeds'])} & {A['n_collapsed']} & " + " & ".join(cells) + " \\\\"

    full = full_caps(summary)  # a replicate arm shows only at its full cap, like the numbers
    for k, (cfg, C) in enumerate(configs(summary).items()):
        if k:
            lines.append("\\addlinespace")
        cells = [pct(C["frequency"].get(s, {}).get(p, {}).get("accuracy")) for s, _ in sets for p, _ in pops]
        label = cfg.removeprefix("cap").replace("_", "\\_")
        lines.append(f"{label} & Frequency & & & " + " & ".join(cells) + " \\\\")
        lines += [row(a, A) for a, A in C["arms"].items() if full.get(a, cfg) == cfg]
    return "\n".join(lines + ["\\bottomrule", "\\end{tabular}"]) + "\n"


def main() -> None:
    """Command line entry point: summarise the runs or the committed tables and write the outputs.

    Exactly one of --runs and --from-tsv must be given. With --runs it reads the runs with collect() and writes the
    <set>_predictions.tsv.gz tables to --out (gzip mtime 0, so the bytes are stable). With --from-tsv it reads those
    tables and summary.json from --out and leaves the tables as they are. Either way it recomputes minority_reading
    (minority()) in memory, writes summary.json to --out and tab_encoder.tex and numbers_encoder.tex to --report
    (creating both folders), and prints a warning for every config and arm with collapsed runs. Exits with an error
    if no runs are found."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", help="folder of <config>/<arm>_s<seed> runs")
    ap.add_argument("--from-tsv", action="store_true", help="recompute from the tables and summary.json in --out")
    ap.add_argument("--out", default=str(ROOT / "results" / "encoder"))
    ap.add_argument("--report", default=str(ROOT / "report" / "generated"), help="where the .tex files go")
    a = ap.parse_args()
    if a.from_tsv == bool(a.runs):
        ap.error("give either --runs DIR or --from-tsv")
    out, rep = Path(a.out), Path(a.report)
    if a.from_tsv:
        tables = read_tables(out)
        metrics = runs_from_summary(json.loads((out / "summary.json").read_text(encoding="utf8")))
    else:
        tables, metrics = collect(a.runs)
    if not metrics:
        raise SystemExit(f"no <config>/<arm>_s<seed> runs under {a.runs or out}")
    out.mkdir(parents=True, exist_ok=True)
    rep.mkdir(parents=True, exist_ok=True)
    for s, table in tables.items() if a.runs else ():
        cols = list(table[0])
        text = "\t".join(cols) + "\n" + "".join("\t".join(t[c] for c in cols) + "\n" for t in table)
        (out / f"{s}_predictions.tsv.gz").write_bytes(gzip.compress(text.encode("utf8"), mtime=0))  # byte-stable
    summary = summarise(tables, metrics, minority(tables, metrics))
    replicates(tables, summary)
    identical_runs(tables, summary)
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf8")
    (rep / "tab_encoder.tex").write_text(table_tex(summary), encoding="utf8")
    (rep / "numbers_encoder.tex").write_text(numbers(summary), encoding="utf8")
    print(f"wrote {out / 'summary.json'}, {len(tables) if a.runs else 0} prediction tables, {rep / 'tab_encoder.tex'} and "
          f"{rep / 'numbers_encoder.tex'}")
    for cfg, C in configs(summary).items():
        for arm, A in C["arms"].items():
            if A["n_collapsed"]:
                print(f"WARNING {cfg} {arm}: {A['n_collapsed']} of {len(A['seeds'])} runs collapsed "
                      f"(most_long_share >= 0.9 on soliloquies)")


if __name__ == "__main__":
    main()
