"""Error typing for the ByT5 sentence arm and CharsiuG2P: agreement, kappa and type counts.

results/error_typing/labels.tsv holds 200 sampled errors per system. PROVENANCE, which summary.json also carries, says
who labelled them and how. The written guidelines are in results/label_review/materials/error_types.

  python -m oevl.error_typing   -> results/error_typing/summary.json
  python -m oevl.error_typing items --out items.tsv [--systems charsiug2p]   -> the item sheet the labellers typed
      (results/label_review/materials/error_types/items.tsv), drawn by the SAMPLING rule

oevl.build_report and oevl.paper_numbers read summary.json. The paper's error table (tab:errors in report/main.tex,
split into prose and verse) takes its counts from oevl.paper_numbers, which reads labels.tsv.
"""
import argparse
import collections
import csv
import gzip
import json
import random
import sys
from collections.abc import Collection, Iterable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar

from oevl.agreement import align, classes, kappa, rate
from oevl.experiment import EXCLUDE_D3, build_data, heldout, lexicon, load_tauber, pattern
from oevl.try_text import apply
from oevl.wikipron_to_marked import mark

if TYPE_CHECKING:
    T = TypeVar("T")

ROOT = Path(__file__).resolve().parents[2]
DIR = ROOT / "results" / "error_typing"
SYSTEMS = ("encoder_sentence", "charsiug2p")
SETS = ("soliloquies", "tauber_beowulf")  # pooled in this order, each in experiment.heldout order
# guideline order: the first type that applies wins
TYPES = ("ceiling", "segment", "truncation", "gold_in_doubt", "homograph", "grammatical", "editorial", "other")
PROVENANCE = ("label_a and label_b come from two separate runs of one LLM, Claude Opus 5.5 (model id claude-opus-5-5), "
              "with the same written guidelines, and final and note from a third run that settled every disagreement. "
              "Blind reviews by further runs of the same model (2026-10-19 for the 81 items kept from the first "
              "sample, 2026-10-27 for the 319 redrawn ones) then changed three final labels, noted in the note column. "
              "The labels were then manually checked, and none was changed (reported 2026-10-20 and 2026-10-27).")
SAMPLING = ("An error is a lexicon-ambiguous token (two or more length patterns for its skeleton in the system's own "
            "lexicon) of the soliloquies and tauber_beowulf test sets, pooled, whose predicted length pattern differs "
            "from gold. john_ws is not used. For each system, 200 errors were drawn with random.Random(1).sample. "
            "encoder_sentence: ByT5 sentence arm, seed 1, --max-bytes 2560, lr 3e-4, 5 epochs, zupitza_grammar left "
            "out of training (the selected configuration), lexicon = the other four training works plus WikiPron, "
            "pool 1,001 errors (532 prose, 469 verse). charsiug2p: CharsiuG2P IPA mapped to a length pattern, "
            "lexicon = all five training works plus WikiPron as charsiu_g2p.analyse used before D3, pool 2,624 errors "
            "(1,604 prose, 1,020 verse). charsiu_g2p.analyse now uses the D3 training set, which gives a pool of "
            "2,588 (1,580 prose, 1,008 verse), and 3 of the 200 sampled tokens are not lexicon-ambiguous there. Five "
            "tokens appear in both samples and were typed separately.")


def main() -> None:
    """Check the labels in labels.tsv, then write summary.json.

    Reads results/error_typing/labels.tsv (tab-separated, one row per sampled error) and uses its item, system, set,
    gold_in_candidates, label_a, label_b, final and note columns. Asserts, naming the item on failure, that label_a,
    label_b and final are each one of TYPES, that a row whose three labels are not all equal has a note, that final is
    ceiling exactly when gold_in_candidates is no, that a final of gold_in_doubt occurs only on tauber_beowulf and a
    final of segment or truncation only on charsiug2p, and that each system has 200 rows.
    The groups are the two SYSTEMS and overall (every row). results/error_typing/summary.json holds provenance,
    sampling_rule and types (the type order), and per group agreement (n, the raw agreement of label_a with label_b
    as a share, cohen_kappa and the number of disagreements), counts (rows per type in final) and sample_by_set (rows
    per test set). summary.json is overwritten, one line per group is printed, then its path. When a group's label_a
    and label_b hold one type only, kappa is None, summary.json gets null and printing the line raises TypeError."""
    with open(DIR / "labels.tsv", encoding="utf8") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    names = set(TYPES)
    for r in rows:  # the guideline's hard rules hold
        assert {r["label_a"], r["label_b"], r["final"]} <= names, r["item"]
        assert r["final"] == r["label_a"] == r["label_b"] or r["note"], r["item"]  # a final that departs from the annotators needs a note
        assert (r["final"] == "ceiling") == (r["gold_in_candidates"] == "no"), r["item"]
        assert r["final"] != "gold_in_doubt" or r["set"] == "tauber_beowulf", r["item"]
        assert r["final"] not in ("segment", "truncation") or r["system"] == "charsiug2p", r["item"]
    groups = {s: [r for r in rows if r["system"] == s] for s in SYSTEMS} | {"overall": rows}
    assert [len(groups[s]) for s in SYSTEMS] == [200, 200], "expected 200 items per system"
    summary: dict[str, Any]  # JSON: strings, a list and nested dicts
    summary = {"provenance": PROVENANCE, "sampling_rule": SAMPLING, "types": list(TYPES),
               "agreement": {}, "counts": {}, "sample_by_set": {}}
    for g, sub in groups.items():
        a, b = [r["label_a"] for r in sub], [r["label_b"] for r in sub]
        summary["agreement"][g] = {"n": len(sub), "agreement": rate(a, b), "cohen_kappa": kappa(a, b),
                                   "disagreements": sum(x != y for x, y in zip(a, b))}
        c = collections.Counter(r["final"] for r in sub)
        summary["counts"][g] = {t: c[t] for t in TYPES}
        summary["sample_by_set"][g] = dict(sorted(collections.Counter(r["set"] for r in sub).items()))
    (DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf8")

    cols = [*SYSTEMS, "overall"]
    ag = summary["agreement"]
    for g in cols:
        print(f"{g:17} n {ag[g]['n']}  agreement {100 * ag[g]['agreement']:.1f}%  kappa {ag[g]['cohen_kappa']:.3f}  "
              + ", ".join(f"{t} {k}" for t, k in summary["counts"][g].items() if k))
    print(f"wrote {DIR / 'summary.json'}")


def errors(system: str, tests: Mapping[str, Sequence[Mapping[str, Any]]],
           lex: Mapping[str, Mapping[str, set[str]]]) -> dict[str, list[tuple[Mapping[str, Any], str, str]]]:
    """The errors of one system, as {set: [(token row, system_form, charsiug2p_ipa), ...]} in test set order.

    tests: SETS -> token rows from experiment.heldout. lex: the system's lexicon from experiment.lexicon. An error is a
    lexicon-ambiguous token (two or more patterns for its key in lex) whose predicted pattern differs from gold.
    encoder_sentence reads the capfull/sentence_s1 column of results/encoder/<set>_predictions.tsv.gz, asserts that
    the file holds exactly the set's token ids, and puts the pattern on the lower-cased bare word. charsiug2p reads
    results/charsiu/ipa.tsv (a missing word is empty IPA) and maps the IPA with wikipron_to_marked.mark, skipping
    tokens without a vowel and tokens whose IPA does not align. charsiug2p_ipa is empty for the encoder."""
    if system == "charsiug2p":
        with open(ROOT / "results" / "charsiu" / "ipa.tsv", encoding="utf8") as f:
            ipa = {r["word"]: r["ipa"] for r in csv.DictReader(f, delimiter="\t")}
    out: dict[str, list[tuple[Mapping[str, Any], str, str]]] = {}
    form: str | None  # mark gives None when the IPA does not align
    for s in SETS:
        if system == "encoder_sentence":
            with gzip.open(ROOT / "results" / "encoder" / f"{s}_predictions.tsv.gz", "rt", encoding="utf8") as f:
                pred = {r["id"]: r["capfull/sentence_s1"] for r in csv.DictReader(f, delimiter="\t")}
            assert pred.keys() == {r["id"] for r in tests[s]}, f"results/encoder/{s}_predictions.tsv.gz is stale"
        out[s] = []
        for r in tests[s]:
            w = r["bare"].lower()
            if system == "encoder_sentence":
                p, form, i = pred[r["id"]], apply(w, pred[r["id"]]), ""
            else:
                i = ipa.get(w, "")
                form = mark(w, i)
                if not r["gold_pattern"] or form is None:
                    continue
                p = pattern(form)
            if len(lex.get(r["key"], ())) >= 2 and p != r["gold_pattern"]:
                out[s].append((r, form, i))
    return out


def draw(pool: "Mapping[str, Iterable[T]]", n: int = 200) -> "list[T]":
    """random.Random(1).sample of n errors from pool (SETS -> errors), the sets pooled in SETS order."""
    return random.Random(1).sample([e for s in SETS for e in pool[s]], n)


def items(systems: Collection[str]) -> list[dict[str, int | str]]:
    """The item sheet as row dicts: 200 sampled errors per system named in systems, in SYSTEMS order.

    Items are numbered 1 to 200 for encoder_sentence and 201 to 400 for charsiug2p, whichever are built. The lexicon is
    data/lex_wikipron.csv plus the training forms, without Zupitza (D3) for encoder_sentence and of all five works for
    charsiug2p. sentence is the sentence's bare tokens joined by spaces, the target in [[ ]]. candidates are the key's
    lexicon patterns in sorted order, each put on the lower-cased bare word, joined by ' | '. class is the final label
    from data/ambiguity_classes.csv (agreement.classes), empty when not listed. hs_form and wyatt_form are the Harrison
    and Sharp and Wyatt tokens that agreement.align pairs with the Tauber token, not_aligned when it pairs none, and
    empty for prose."""
    data = ROOT / "data"
    tests = {s: heldout(data, s) for s in SETS}
    tauber = load_tauber(data / "brunetti-length.txt")
    cls = classes(data / "ambiguity_classes.csv")
    rows = []
    for k, system in enumerate(SYSTEMS):
        if system not in systems:
            continue
        train, _, hs, wy = build_data(data, EXCLUDE_D3 if system == "encoder_sentence" else ())
        lex = lexicon(train, str(data / "lex_wikipron.csv"))[0]
        eds = [{t["id"]: e["gold"] for e, t in align(ed, tauber)} for ed in (hs, wy)]
        for j, (r, form, ipa) in enumerate(draw(errors(system, tests, lex))):
            verse = r["source"] == "tauber_beowulf"
            rows.append({"item": 200 * k + j + 1, "system": system, "set": r["source"], "id": r["id"],
                         "sentence": " ".join(f"[[{w}]]" if i == r["index"] else w
                                              for i, w in enumerate(r["sentence"])),
                         "gold": r["gold"], "system_form": form, "charsiug2p_ipa": ipa,
                         "candidates": " | ".join(apply(r["bare"].lower(), p) for p in sorted(lex[r["key"]])),
                         "class": cls.get(r["key"], ""),
                         "gold_in_candidates": "yes" if r["gold_pattern"] in lex[r["key"]] else "no",
                         "hs_form": eds[0].get(r["id"], "not_aligned") if verse else "",
                         "wyatt_form": eds[1].get(r["id"], "not_aligned") if verse else ""})
    return rows


if __name__ == "__main__":
    if sys.argv[1:2] == ["items"]:
        ap = argparse.ArgumentParser(prog="python -m oevl.error_typing items")
        ap.add_argument("--out", required=True, help="the item sheet to write (tab-separated)")
        ap.add_argument("--systems", nargs="+", choices=SYSTEMS, default=list(SYSTEMS))
        args = ap.parse_args(sys.argv[2:])
        sheet = items(args.systems)
        with open(args.out, "w", encoding="utf8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(sheet[0]), delimiter="\t", lineterminator="\n")
            w.writeheader()
            w.writerows(sheet)
    else:
        main()
