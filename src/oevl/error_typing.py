"""Error typing for the ByT5 sentence arm and CharsiuG2P: agreement, kappa and type counts.

results/error_typing/labels.tsv holds 200 sampled errors per system. label_a and label_b come from two separate runs
of one LLM, Claude Opus 5.5 (model id claude-opus-5-5), with the same written guidelines
(results/label_review/materials/error_types), and final and note from a third run that settled their disagreements.
A blind review by further runs of the same model (2026-10-19) changed five finals and explained them in the note
column. The labels were then manually checked, and none was changed.

  python -m oevl.error_typing   -> results/error_typing/summary.json and report/generated/tab_errors.tex

oevl.build_report and oevl.paper_numbers read summary.json. report/main.tex does not include tab_errors.tex. The
paper's error table (tab:errors, split into prose and verse) is built by oevl.paper_numbers from labels.tsv.
"""
import collections
import csv
import json
from pathlib import Path

from oevl.agreement import kappa, rate

ROOT = Path(__file__).resolve().parents[2]
DIR = ROOT / "results" / "error_typing"
SYSTEMS = (("encoder_sentence", "ByT5 sentence"), ("charsiug2p", "CharsiuG2P"))
# guideline order: the first type that applies wins
TYPES = (("ceiling", "Ceiling"), ("segment", "Segment"), ("truncation", "Truncation"),
         ("gold_in_doubt", "Gold in doubt"), ("homograph", "Homograph (A)"), ("grammatical", "Grammatical (B)"),
         ("editorial", "Editorial (C)"), ("other", "Other"))
PROVENANCE = ("label_a and label_b come from two separate runs of one LLM, Claude Opus 5.5 (model id claude-opus-5-5), "
              "with the same written guidelines, and final and note from a third run that settled every disagreement. "
              "A blind review by further runs of the same model (2026-10-19) then changed five final labels, noted in "
              "the note column. The labels were then manually checked, and none was changed (reported "
              "2026-10-20).")
SAMPLING = ("An error is a lexicon-ambiguous token (two or more length patterns for its skeleton in the system's own "
            "lexicon) of the soliloquies and tauber_beowulf test sets, pooled, whose predicted length pattern differs "
            "from gold. john_ws is not used. For each system, 200 errors were drawn with random.Random(1).sample. "
            "encoder_sentence: ByT5 sentence arm, seed 1, --max-bytes 2560, lr 3e-4, 5 epochs, zupitza_grammar left "
            "out of training (the selected configuration), lexicon = the other four training works plus WikiPron, "
            "pool 917 errors (468 prose, 449 verse). charsiug2p: CharsiuG2P IPA mapped to a length pattern, lexicon "
            "= all five training works plus WikiPron as charsiu_g2p.analyse used before D3, pool 2,625 errors (1,607 "
            "prose, 1,018 verse). charsiu_g2p.analyse now uses the D3 training set, which gives a pool of 2,589 "
            "(1,583 prose, 1,006 verse), and 2 of the 200 sampled tokens are not lexicon-ambiguous there. Ten tokens "
            "appear in both samples and were typed separately.")


def main():
    """Check the labels in labels.tsv, then write summary.json and tab_errors.tex.

    Reads results/error_typing/labels.tsv (tab-separated, one row per sampled error) and uses its item, system, set,
    gold_in_candidates, label_a, label_b, final and note columns. Asserts, naming the item on failure, that label_a,
    label_b and final are each one of TYPES, that a row whose three labels are not all equal has a note, that final is
    ceiling exactly when gold_in_candidates is no, that a final of gold_in_doubt occurs only on tauber_beowulf and a
    final of segment or truncation only on charsiug2p, and that each system has 200 rows.
    The groups are the two SYSTEMS and overall (every row). results/error_typing/summary.json holds provenance,
    sampling_rule and types (the type order), and per group agreement (n, the raw agreement of label_a with label_b
    as a share, cohen_kappa and the number of disagreements), counts (rows per type in final) and sample_by_set (rows
    per test set). report/generated/tab_errors.tex is a tabular with the final counts per type (n/a for segment and
    truncation under ByT5), then items, raw agreement in % and kappa, one column per group. Both files are
    overwritten, one line per group is printed, then the two paths. When a group's label_a and label_b hold one type
    only, kappa is None, summary.json gets null and formatting the table then raises TypeError."""
    with open(DIR / "labels.tsv", encoding="utf8") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    names = {t for t, _ in TYPES}
    for r in rows:  # the guideline's hard rules hold
        assert {r["label_a"], r["label_b"], r["final"]} <= names, r["item"]
        assert r["final"] == r["label_a"] == r["label_b"] or r["note"], r["item"]  # a final that departs from the annotators needs a note
        assert (r["final"] == "ceiling") == (r["gold_in_candidates"] == "no"), r["item"]
        assert r["final"] != "gold_in_doubt" or r["set"] == "tauber_beowulf", r["item"]
        assert r["final"] not in ("segment", "truncation") or r["system"] == "charsiug2p", r["item"]
    groups = {s: [r for r in rows if r["system"] == s] for s, _ in SYSTEMS} | {"overall": rows}
    assert [len(groups[s]) for s, _ in SYSTEMS] == [200, 200], "expected 200 items per system"
    summary = {"provenance": PROVENANCE, "sampling_rule": SAMPLING, "types": [t for t, _ in TYPES],
               "agreement": {}, "counts": {}, "sample_by_set": {}}
    for g, sub in groups.items():
        a, b = [r["label_a"] for r in sub], [r["label_b"] for r in sub]
        summary["agreement"][g] = {"n": len(sub), "agreement": rate(a, b), "cohen_kappa": kappa(a, b),
                                   "disagreements": sum(x != y for x, y in zip(a, b))}
        c = collections.Counter(r["final"] for r in sub)
        summary["counts"][g] = {t: c[t] for t, _ in TYPES}
        summary["sample_by_set"][g] = dict(sorted(collections.Counter(r["set"] for r in sub).items()))
    (DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf8")

    cols = [s for s, _ in SYSTEMS] + ["overall"]
    lines = ["% generated by src/oevl/error_typing.py from results/error_typing/labels.tsv "
             "(LLM labels, manually checked); do not edit",
             "\\begin{tabular}{@{}lrrr@{}}", "\\toprule",
             "Type & " + " & ".join(n for _, n in SYSTEMS) + " & Both \\\\", "\\midrule"]
    for t, name in TYPES:
        cells = ["n/a" if t in ("segment", "truncation") and g == "encoder_sentence" else str(summary["counts"][g][t])
                 for g in cols]
        lines.append(f"{name} & " + " & ".join(cells) + " \\\\")
    ag = summary["agreement"]
    lines += ["\\midrule", "Items & " + " & ".join(str(ag[g]["n"]) for g in cols) + " \\\\",
              "Raw agreement (\\%) & " + " & ".join(f"{100 * ag[g]['agreement']:.1f}" for g in cols) + " \\\\",
              "Cohen's $\\kappa$ & " + " & ".join(f"{ag[g]['cohen_kappa']:.3f}" for g in cols) + " \\\\",
              "\\bottomrule", "\\end{tabular}"]
    out = ROOT / "report" / "generated" / "tab_errors.tex"
    out.write_text("\n".join(lines) + "\n", encoding="utf8")
    for g in cols:
        print(f"{g:17} n {ag[g]['n']}  agreement {100 * ag[g]['agreement']:.1f}%  kappa {ag[g]['cohen_kappa']:.3f}  "
              + ", ".join(f"{t} {k}" for t, k in summary["counts"][g].items() if k))
    print(f"wrote {DIR / 'summary.json'} and {out}")


if __name__ == "__main__":
    main()
