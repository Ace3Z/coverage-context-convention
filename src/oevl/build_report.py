"""Generate report/generated/numbers.tex and tab_data.tex from data/ and results/, then compile the report.

Reads results/<setting>/metrics.json of the five logistic settings (training_only, wikipron, proposal, all,
wikipron_with_zupitza), results/training_only/soliloquies_per_word.csv, the WikiPron run's predictions
(results/wikipron/{soliloquies, tauber_beowulf}_predictions.jsonl), results/{agreement, charsiu, encoder,
error_typing}/ and data/ (manifest, ambiguity classes, leakage audit, WikiPron TSV, lex_wikipron.csv, Tauber's
Beowulf, by_source/*.jsonl).
Writes report/generated/numbers.tex (main.tex uses only some of its macros) and tab_data.tex. main.tex's other
generated macros come from numbers_encoder.tex (oevl.encoder_summary) and numbers_paper.tex (oevl.paper_numbers).
All work runs at module level, so importing the module runs the whole build. The run stops with sys.exit when a check
fails: the CharsiuG2P reproduction against its published WER and PER, the WikiPron conversion against
data/lex_wikipron.csv, and a few consistency checks between result files. A failed LaTeX pass also stops it.
When pdflatex is installed it runs pdflatex, bibtex and pdflatex twice in report/ and copies main.pdf to REPORT.pdf
(otherwise upload report/ to Overleaf).
Usage: python -m oevl.build_report
"""
import collections
import csv
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from oevl import experiment as E
from oevl.check_expected import acc

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "report" / "generated"
TESTS = ("soliloquies", "tauber_beowulf")


def load(folder: str) -> dict[str, Any]:
    """Return results/<folder>/metrics.json, one logistic setting's metrics from oevl.experiment, as a dict."""
    return json.loads((ROOT / "results" / folder / "metrics.json").read_text(encoding="utf8"))


W, T = load("wikipron"), load("training_only")
P, A, Z = load("proposal"), load("all"), load("wikipron_with_zupitza")  # proposal dictionaries, all four, D3 ablation


def pct(x: float) -> str:
    """Format a fraction as a percentage with two decimals and no % sign, e.g. 0.87836 gives "87.84"."""
    return f"{100 * x:.2f}"


def val(m: dict[str, Any], test: str, system: str, pop: str, sd: bool = False) -> str:
    """Accuracy of one system on one population of a held-out set, as a percentage string (see pct).

    m: a metrics.json dict, test: a held-out set key such as "soliloquies", pop: "accuracy" (all tokens) or a population
    such as "lexicon_ambiguous" or "class_B". "frequency" and "oracle" are deterministic and read directly, "word",
    "left" and "sentence" give the mean over seeds from seed_summary. With sd=True a logistic system also gets
    "$\\pm$" and its sample SD over seeds in points."""
    t = m[test]
    if system in ("frequency", "oracle"):
        return pct(acc(t[system], pop))
    s = t["seed_summary"][system][pop]
    return f"{pct(s['mean'])}$\\pm${100 * s['sd']:.2f}" if sd else pct(s["mean"])


def n(m: dict[str, Any], test: str, pop: str) -> int:
    """Number of scored tokens of a population of one held-out set, read from the frequency entry ("accuracy" = all)."""
    return m[test]["frequency"]["n"] if pop == "accuracy" else m[test]["frequency"][pop]["n"]


def p_fmt(p: float) -> str:
    """Format a p-value as plain text.

    Four decimals from 0.001 up, four decimals with trailing zeros stripped from 0.0001 up (0.000999 gives "0.001"),
    and below that one decimal in exponent form, e.g. "4.8e-13" (not LaTeX math)."""
    return f"{p:.4f}" if p >= 0.001 else f"{p:.4f}".rstrip("0") if p >= 0.0001 else f"{p:.1e}"


# ---------------------------------------------------------------- numbers for numbers.tex (main.tex uses only some)
num = {}
manifest = json.loads((ROOT / "data" / "manifest.json").read_text(encoding="utf8"))["sources"]
num["CorpusTokens"] = f"{sum(s['tokens'] for s in manifest.values()):,}"
num["TrainTokens"] = f"{W['dataset']['train']:,}"
num["ProseTokens"] = f"{W['dataset']['test_prose']:,}"
num["BeoTokens"] = f"{n(W, 'tauber_beowulf', 'accuracy'):,}"
for tag, m in (("W", W), ("T", T), ("P", P), ("A", A), ("Z", Z)):
    for test, short in (("soliloquies", "Sol"), ("tauber_beowulf", "Beo")):
        for pop, p in (("accuracy", "All"), ("lexicon_ambiguous", "Lex"), ("minority_reading", "Min"),
                       ("attested_ambiguous", "Att")):
            num[f"{tag}{short}{p}N"] = f"{n(m, test, pop):,}"
            for system, s in (("frequency", "Freq"), ("word", "Word"), ("left", "Left"),
                              ("sentence", "Bil"), ("oracle", "Orc")):
                num[f"{tag}{short}{p}{s}"] = val(m, test, system, pop)
        mc = m[test]["mcnemar_seed1"]
        for key, s in (("frequency_vs_word_seed1", "FW"), ("frequency_vs_left_seed1", "FL"),
                       ("frequency_vs_sentence_seed1", "FB"), ("word_seed1_vs_left_seed1", "WL"),
                       ("word_seed1_vs_sentence_seed1", "WB"), ("left_seed1_vs_sentence_seed1", "LB")):
            num[f"{tag}{short}P{s}"] = p_fmt(mc[key]["p_exact"])
            num[f"{tag}{short}A{s}"] = str(mc[key]["a_wrong_b_right"])
            num[f"{tag}{short}B{s}"] = str(mc[key]["a_right_b_wrong"])
        num[f"{tag}{short}MinP"] = p_fmt(min(r["p_exact"] for k, r in mc.items() if k.startswith("frequency_vs")))
    e = m["editor_agreement"]
    num[f"{tag}AgrAligned"] = f"{e['aligned']:,}"
    num[f"{tag}AgrAll"] = pct(e["all"]["agreement"])
    num[f"{tag}AgrAllK"] = f"{e['all']['cohen_kappa']:.3f}"
    num[f"{tag}AgrAmbN"] = f"{e['lexicon_ambiguous']['n']:,}"
    num[f"{tag}AgrAmb"] = pct(e["lexicon_ambiguous"]["agreement"])
    num[f"{tag}AgrAmbK"] = f"{e['lexicon_ambiguous']['cohen_kappa']:.3f}"
num["BeoGain"] = f"{100 * (W['tauber_beowulf']['frequency']['accuracy'] - T['tauber_beowulf']['frequency']['accuracy']):.2f}"
num["SolGain"] = f"{100 * (W['soliloquies']['frequency']['accuracy'] - T['soliloquies']['frequency']['accuracy']):.2f}"
sds = [W[t]["seed_summary"][a][p]["sd"] for t in TESTS for a in ("word", "left", "sentence")
       for p in ("accuracy", "lexicon_ambiguous")]
num["SdMaxMain"] = f"{100 * max(sds):.2f}"
num["SdMaxMin"] = f"{100 * max(W[t]['seed_summary'][a]['minority_reading']['sd'] for t in TESTS for a in ('word', 'left', 'sentence')):.2f}"
num["EditorGap"] = f"{100 * (W['editor_agreement']['lexicon_ambiguous']['agreement'] - W['tauber_beowulf']['seed_summary']['word']['lexicon_ambiguous']['mean']):.1f}"
num["AgrGap"] = f"{100 * (1 - W['editor_agreement']['lexicon_ambiguous']['agreement']):.1f}"

# ambiguity classes: annotator agreement and final counts
CLS = json.loads((ROOT / "data" / "ambiguity_classes.json").read_text(encoding="utf8"))
num["ClassKappa"] = f"{CLS['cohen_kappa']:.3f}"
num["ClassAgree"] = pct(CLS["agreement"])
for c in "ABC":
    num[f"Class{c}"] = str(CLS["counts"]["final"][c])

# D3: what training on Zupitza does to the prior on the prose test (points)
num["DThreeSolAllCost"] = f"{100 * (Z['soliloquies']['frequency']['accuracy'] - W['soliloquies']['frequency']['accuracy']):.2f}"


def rjson(path: str) -> dict[str, Any]:
    """Read a JSON file given by its path relative to the repository root."""
    return json.loads((ROOT / path).read_text(encoding="utf8"))


# editor agreement of the three Beowulf pairs, the gate and the checked HS-Wyatt sample (results/agreement)
AG = rjson("results/agreement/metrics.json")
for pair, short in (("hs-wyatt", "HW"), ("hs-tauber", "HT"), ("wyatt-tauber", "WT")):
    num[f"Pair{short}Aligned"] = f"{AG['pairs'][pair]['aligned']:,}"
    for pop, s in (("all", "All"), ("lexicon_ambiguous", "Lex"), ("class_A", "ClsA"), ("class_B", "ClsB"),
                   ("class_C", "ClsC")):
        r = AG["pairs"][pair][pop]
        num[f"Pair{short}{s}"], num[f"Pair{short}{s}K"] = pct(r["agreement"]), f"{r['cohen_kappa']:.3f}"
        num[f"Pair{short}{s}N"] = f"{r['n']:,}"
        for k, v in r.items():  # the prior scored against each edition of the pair
            if k.startswith("prior_vs_"):
                num[f"Pair{short}{s}Prior{k[len('prior_vs_'):].title()}"] = pct(v)
    for k, g in AG["gate"][pair].items():
        if k.startswith("lexicon_ambiguous_floor_vs_"):
            ed = k.rsplit("_", 1)[1].title()
            num[f"Gate{short}{ed}"], num[f"Gate{short}{ed}Floor"] = f"{g['gap_points']:.2f}", pct(g["floor"])
wt_b = AG["pairs"]["wyatt-tauber"]["class_B"]
num["PairWTClsBPriorGap"] = f"{100 * (wt_b['prior_vs_wyatt'] - wt_b['prior_vs_tauber']):.2f}"
# class B disagreements between Harrison & Sharp and Wyatt, and how many are ne (the PG revision prints nē, Wyatt ne)
CLASSES = E.load_classes(ROOT / "data" / "ambiguity_classes.csv")
cls_b = {s for s, c in CLASSES.items() if c == "B"}
hw_b = [r for r in csv.DictReader(open(ROOT / "results/agreement/disagreements.tsv", encoding="utf8"), delimiter="\t")
        if r["pair"] == "hs-wyatt" and r["key"] in cls_b]
r = AG["pairs"]["hs-wyatt"]["class_B"]
if len(hw_b) != r["n"] - round(r["agreement"] * r["n"]):
    sys.exit("disagreements.tsv does not match the HS-Wyatt class B agreement")
num["PairHWClsBDis"], num["PairHWClsBDisNe"] = str(len(hw_b)), str(sum(r["key"] == "ne" for r in hw_b))
HC = AG["hand_check_hs_wyatt"]["labels"]
num["CheckN"] = str(AG["hand_check_hs_wyatt"]["n"])
for label, s in (("genuine editorial difference", "Editorial"), ("normalisation fault", "Norm"),
                 ("alignment fault", "Align"), ("other", "Other")):
    num[f"Check{s}"] = str(HC.get(label, 0))

# CharsiuG2P: its own benchmark (part 1) and length on our tokens (part 2)
CB, CM = rjson("results/charsiu/benchmark.json"), rjson("results/charsiu/metrics.json")
b20, b50 = CB["max_length_20"], CB["max_length_50"]
if (b20["wer"], b20["per"]) != (CB["published"]["wer"], CB["published"]["per"]):
    sys.exit("CharsiuG2P: the reproduction no longer matches the published WER and PER")
num["CharsiuWer"], num["CharsiuPer"], num["CharsiuTrunc"] = f"{b20['wer']:.3f}", f"{b20['per']:.4f}", str(b20["truncation"])
num["CharsiuWerFixed"], num["CharsiuLenShare"] = f"{b50['wer']:.3f}", f"{100 * b50['length_only_share']:.1f}"
num["CharsiuLenOnly"], num["CharsiuWordErrors"] = str(b50["length_only"]), str(b50["word_errors"])
num["CharsiuN"] = str(b20["n"])
for test, short in (("soliloquies", "Sol"), ("tauber_beowulf", "Beo")):
    e = CM[test]["experiment_populations"]
    num[f"Charsiu{short}All"], num[f"Charsiu{short}Lex"] = pct(e["accuracy"]), pct(e["lexicon_ambiguous"]["accuracy"])
cs = CM["soliloquies"]  # its length errors on ambiguous prose tokens, and how many the seed-1 logistic models get right
num["CharsiuSolLexWrong"] = f"{cs['wrong_length_on_ambiguous']:,}"
num["CharsiuSolLexWordFix"], num["CharsiuSolLexBilFix"] = (f"{cs[k]:,}" for k in (
    "of_those_word_model_right", "of_those_bilateral_model_right"))
num["CharsiuSolLexWordFixShare"] = pct(cs["of_those_word_model_right"] / cs["wrong_length_on_ambiguous"])

# ByT5 model selection on dev and the D3 comparison (results/encoder/selection.json), grid size (summary.json)
SEL = rjson("results/encoder/selection.json")
num["SelRuns"], num["SelEpochs"] = str(len(SEL["runs"])), str(SEL["choice"]["epochs"])
num["SelDevNoZup"] = f"{SEL['choice']['mean']:.2f}"
num["SelDevZup"] = f"{max(c['mean'] for c in SEL['configs'] if c['zupitza']):.2f}"
d3 = SEL["d3_logistic_john_ws"]
for side, s in (("with_zupitza", "Zup"), ("without_zupitza", "NoZup")):
    num[f"DevFreq{s}"] = f"{d3[side]['lexicon_ambiguous']['frequency']:.2f}"
    num[f"DevBil{s}"] = f"{d3[side]['lexicon_ambiguous']['sentence']['mean']:.2f}"
# the same comparison on the dev tokens both settings share (the set without Zupitza is a strict subset)
sh = d3["with_zupitza"]["lexicon_ambiguous_shared_7639"]
if sh["n"] != d3["without_zupitza"]["lexicon_ambiguous"]["n"]:
    sys.exit("selection.json: the shared dev set is not the set without Zupitza")
num["DevLexShared"] = f"{sh['n']:,}"
num["DevFreqZupShared"], num["DevBilZupShared"] = f"{sh['frequency']:.2f}", f"{sh['sentence']['mean']:.2f}"
num["SelDevZupShared"] = f"{max(c['mean_on_shared_7639'] for c in SEL['configs'] if c['zupitza']):.2f}"
collapsed = [r for r in SEL["runs"] if r["collapsed"]]
if len(collapsed) != 1:
    sys.exit(f"selection.json: expected one collapsed run, found {len(collapsed)}")
num["CollapseSolAll"], num["CollapseSolLex"] = f"{collapsed[0]['sol_all']:.2f}", f"{collapsed[0]['sol_lexamb']:.2f}"
num["CollapseEpochs"] = str(collapsed[0]["epochs"])
num["CollapseDevLex"], num["CollapseDevMostLong"] = f"{collapsed[0]['dev_lexamb']:.2f}", f"{collapsed[0]['dev_most_long']:.2f}"
from oevl.encoder_summary import configs, read_tables  # noqa: E402
ENC = configs(rjson("results/encoder/summary.json"))
num["GridRuns"] = str(sum(len(A["seeds"]) for C in ENC.values() for A in C["arms"].values()))
num["GridCollapsed"] = str(sum(A["n_collapsed"] for C in ENC.values() for A in C["arms"].values()))
num["EncTrainExamples"] = f"{ENC['capfull']['settings']['train_examples']:,}"
num["EncFullCap"] = f"{ENC['capfull']['settings']['max_bytes']:,}"
for test, short in (("soliloquies", "Sol"), ("tauber_beowulf", "Beo")):  # best context arm minus prior, all tokens
    best = max(A["sets"][test]["accuracy"]["mean"] for C in ENC.values() for a, A in C["arms"].items() if a != "word")
    num[f"EncBestCtxGain{short}"] = f"{100 * (best - ENC['capfull']['frequency'][test]['accuracy']['accuracy']):.2f}"
    word = ENC["capfull"]["arms"]["word"]["sets"][test]["accuracy"]["mean"]  # the word arm over the same prior
    num[f"EncWordGain{short}"] = f"{100 * (word - ENC['capfull']['frequency'][test]['accuracy']['accuracy']):.2f}"
    # the prior with all four candidate sources, over training forms only and over training forms + WikiPron
    num[f"{short}GainDict"] = f"{100 * (A[test]['frequency']['accuracy'] - T[test]['frequency']['accuracy']):.2f}"
    num[f"{short}GainDictWiki"] = f"{100 * (A[test]['frequency']['accuracy'] - W[test]['frequency']['accuracy']):.2f}"

# leakage audit (data/leakage_excluded.json) and the dev set size
LK = rjson("data/leakage_excluded.json")["counts"]
num["LeakSolSent"], num["LeakSolTok"] = str(LK["soliloquies"]["sentences"]), str(LK["soliloquies"]["tokens"])
num["LeakDevSent"], num["LeakDevTok"] = str(LK["john_ws"]["sentences"]), str(LK["john_ws"]["tokens"])
num["DevTokens"] = f"{W['john_ws']['frequency']['n']:,}"

# error typing (results/error_typing/summary.json): LLM labels, manually checked
ET = rjson("results/error_typing/summary.json")
for system, s in (("encoder_sentence", "Enc"), ("charsiug2p", "Charsiu")):
    for t, k in ET["counts"][system].items():
        num[f"Err{s}{''.join(w.title() for w in t.split('_'))}"] = str(k)
num["ErrKappa"] = f"{ET['agreement']['overall']['cohen_kappa']:.3f}"
num["ErrItems"] = str(ET["agreement"]["encoder_sentence"]["n"])

# WikiPron conversion statistics, recomputed with the same converter used by run_all.sh
import re  # noqa: E402
from oevl.wikipron_to_marked import mark  # noqa: E402
wp_rows = wp_alpha = wp_ok = 0
wp_forms = set()
for line in open(ROOT / "data" / "ang_latn_broad.tsv", encoding="utf8"):
    bare_w, ipa = line.rstrip("\n").split("\t")[:2]
    wp_rows += 1
    wp_alpha += bool(re.fullmatch(r"[a-zþðæ]+", bare_w.lower()))
    if re.fullmatch(r"[a-zþðæ]+", bare_w.lower()) and (m := mark(bare_w, ipa)) is not None:
        wp_ok += 1
        wp_forms.add(m)
lex_rows = sum(1 for _ in open(ROOT / "data" / "lex_wikipron.csv", encoding="utf8")) - 1
if lex_rows != len(wp_forms):
    sys.exit(f"lex_wikipron.csv has {lex_rows} forms but the converter gives {len(wp_forms)}")
num["WikiRows"] = f"{wp_rows:,}"
num["WikiForms"] = f"{lex_rows:,}"
num["WikiAligned"] = f"{100 * wp_ok / wp_rows:.1f}"
num["WikiAlignedN"], num["WikiAlignedAlpha"] = f"{wp_ok:,}", f"{100 * wp_ok / wp_alpha:.1f}"  # of the alphabetic rows

# ---------------------------------------------------------------- per-word numbers (training-only run)
pw = {r["word"]: r for r in csv.DictReader(open(ROOT / "results" / "training_only" / "soliloquies_per_word.csv",
                                                encoding="utf8"))}
WORDS = ["þæt", "ge"]
missing = [w for w in WORDS if w not in pw]
if missing:
    sys.exit(f"per-word numbers: words missing from CSV: {missing}")


def counts(js: str) -> str:
    """Turn a per-word CSV cell, JSON {pattern: count} of a one-vowel word, into "short/long" from patterns "0", "1"."""
    c = json.loads(js)
    return f"{c.get('0', 0)}/{c.get('1', 0)}"


# Zupitza is not trained on (decision D3), so the per-word CSVs hold no counts for it: GeZup counts the corpus file
ZUP = collections.Counter((r["key"], r["gold_pattern"]) for r in E.tokens(E.load(ROOT / "data/by_source/zupitza_grammar.jsonl")))


def zup(w: str) -> str:
    """Counts of skeleton w with pattern "0" and "1" in the whole Zupitza file, as "short/long" (one-vowel words)."""
    return f"{ZUP[w, '0']}/{ZUP[w, '1']}"


the = json.loads(pw["þæt"]["test_pattern_counts"])
num["ThaetShort"] = str(the.get("0", 0))
num["ThaetN"] = pw["þæt"]["n"]
ge = pw["ge"]
num["GeZup"] = zup("ge")
num["GeMat"] = counts(ge["matthew_ws_pattern_counts"])
num["GeSol"] = counts(ge["test_pattern_counts"])

# ---------------------------------------------------------------- the WikiPron run's Soliloquies predictions
preds = E.load(ROOT / "results" / "wikipron" / "soliloquies_predictions.jsonl")
bysent: dict[str, list[dict[str, Any]]] = {}
for pr in preds:
    bysent.setdefault(pr["sentence_id"], []).append(pr)

# ---------------------------------------------------------------- coverage from data/, þe facts from predictions
# Beowulf coverage by the training forms alone (D3 training, no dictionary), and the rows of Tauber's file
train_toks, _, _, _ = E.build_data(str(ROOT / "data"), E.EXCLUDE_D3)
lex_t, _ = E.lexicon(train_toks)
beo = E.heldout(ROOT / "data", "tauber_beowulf")
no_cand = sum(r["gold_pattern"] not in lex_t.get(r["key"], set()) for r in beo)
unknown = sum(not lex_t.get(r["key"]) for r in beo)
num["BeoNoCandTrue"] = f"{100 * no_cand / len(beo):.1f}"
num["BeoUnknown"] = f"{100 * unknown / len(beo):.1f}"
tauber_rows = [line.rstrip("\n").split("|") for line in open(ROOT / "data" / "brunetti-length.txt", encoding="utf8")]
num["TauberRows"] = f"{len(tauber_rows):,}"
num["TauberTwo"] = str(sum(len(r[9].split()) == 2 for r in tauber_rows))
num["TauberThree"] = str(sum(len(r[9].split()) == 3 for r in tauber_rows))
the_all = [p for p in preds if p["key"] == "þe"]
num["TheTokens"] = str(len(the_all))
num["TheLongLeft"] = str(sum(p["left_seed1_prediction"] == "1" for p in the_all))
after_to = []
for toks in bysent.values():
    for i, t in enumerate(toks):
        if t["key"] == "þe" and i > 0 and toks[i - 1]["key"] == "to" and t["gold_pattern"] == "1":
            after_to.append(t)
num["ToTheN"] = str(len(after_to))
num["ToTheMissed"] = str(sum(t["left_seed1_prediction"] == "0" and t["sentence_seed1_prediction"] == "0" for t in after_to))
# the same tokens in the ByT5 runs (results/encoder): fewest and most long þē after tō per arm, over its runs at
# every cap (128, 256 and full)
ENC_SOL = {r["id"]: r for r in read_tables(ROOT / "results/encoder")["soliloquies"]}
enc_runs = [c for c in next(iter(ENC_SOL.values())) if "/" in c]
for arm, short in (("word", "Word"), ("local2", "LocalTwo"), ("sentence", "Bil")):
    hits = [sum(ENC_SOL[t["id"]][c] == "1" for t in after_to) for c in enc_runs if c.split("/")[1].startswith(arm + "_")]
    num[f"ToTheEnc{short}Min"], num[f"ToTheEnc{short}Max"] = str(min(hits)), str(max(hits))
    num[f"ToTheEnc{short}Runs"] = str(len(hits))
for a, short in (("word", "Word"), ("left", "Left"), ("sentence", "Bil")):
    num[f"SolGain{short}"] = f"{100 * (W['soliloquies']['seed_summary'][a]['accuracy']['mean'] - T['soliloquies']['seed_summary'][a]['accuracy']['mean']):.2f}"
    num[f"BeoGain{short}"] = f"{100 * (W['tauber_beowulf']['seed_summary'][a]['accuracy']['mean'] - T['tauber_beowulf']['seed_summary'][a]['accuracy']['mean']):.2f}"

# ---------------------------------------------------------------- Table 1 (tab_data.tex)
ORDER = ["zupitza_grammar", "bright_reader", "matthew_ws", "sweet_primer", "physiologus", "john_ws", "soliloquies",
         "hs_beowulf", "wyatt_beowulf", "tauber_beowulf"]
TITLES = {"zupitza_grammar": "Ælfric's \\textit{Grammar}", "bright_reader": "\\textit{Anglo-Saxon Reader}",
          "matthew_ws": "Gospel of Matthew", "sweet_primer": "\\textit{Anglo-Saxon Primer}",
          "physiologus": "\\textit{Physiologus}", "john_ws": "Gospel of John",
          "soliloquies": "Alfred's \\textit{Soliloquies}",
          "hs_beowulf": "\\textit{Beowulf}", "wyatt_beowulf": "\\textit{Beowulf}", "tauber_beowulf": "\\textit{Beowulf}"}
# shown as printed, not as the manifest has it: Matthew was transcribed from the 1910 printing, the Harrison and Sharp
# text is the 1894 printing of the 4th edition, Cook edited the Physiologus text (Pitman wrote the verse translation),
# Bright edited John (Harris wrote the glossary), and every text taken from Sweet's Primer is prose
EDITOR = {"matthew_ws": "Bright, 1910 printing", "hs_beowulf": "Harrison \\& Sharp 1894$^a$",
          "physiologus": "Cook 1921", "john_ws": "Bright 1904",
          "wyatt_beowulf": "Wyatt 1894$^b$", "tauber_beowulf": "Klaeber 1922"}
GENRE = {"sweet_primer": "prose"}
ROLE = {"zupitza_grammar": "ablation", "john_ws": "dev", "soliloquies": "prose test", "tauber_beowulf": "verse test",
        "hs_beowulf": "agreement", "wyatt_beowulf": "agreement"}
tauber_meta = {"editor": "Klaeber", "year": 1922, "genre": "verse", "origin": "Tauber"}
fin = [sum(len(r["tokens"]) for r in E.load(ROOT / "data/by_source" / f"{k}.jsonl") if "finnsburg" in r["section"].lower())
       for k in ("wyatt_beowulf", "hs_beowulf")]
rows, sentences = [], len({t["sentence_id"] for t in beo})
for k in ORDER:
    s = manifest.get(k, tauber_meta)
    toks = beo if k == "tauber_beowulf" else list(E.tokens(E.load(ROOT / "data/by_source" / f"{k}.jsonl")))
    sentences += 0 if k == "tauber_beowulf" else s["units"]
    long = 100 * sum(t["gold_pattern"].count("1") for t in toks) / sum(len(t["gold_pattern"]) for t in toks)
    role = ROLE.get(k, "train")
    if k in ("john_ws", "soliloquies", "tauber_beowulf"):  # the share of scored tokens unseen in training
        held = beo if k == "tauber_beowulf" else E.heldout(ROOT / "data", k)
        role += f" ({100 * sum(not lex_t.get(t['key']) for t in held) / len(held):.2f})"
    src = "PG" if "gutenberg" in s["origin"] else "WS" if "wikisource" in s["origin"] else "Tauber"
    editor = EDITOR.get(k) or s["editor"].replace("&", r"\&") + f" {s['year']}"
    rows.append(f"{TITLES[k]} & {editor} & {src} & {GENRE.get(k, s['genre'])} & {len(toks):,} & {long:.2f} & {role} \\\\")
release = sum(s["tokens"] for s in manifest.values()) + len(beo)
notes = (f"$^a$Project Gutenberg revision of the 4th edition (first edition 1883). $^b$Its \\textit{{Finnsburh Fragment}} "
         f"lacks lines 1 to 18 in our file ({fin[0]} of {fin[1]} tokens). "
         f"Scored tokens: dev {W['john_ws']['frequency']['n']:,}, prose "
         f"{n(W, 'soliloquies', 'accuracy'):,}, verse {n(W, 'tauber_beowulf', 'accuracy'):,}. Training without Zupitza: "
         f"{W['dataset']['train']:,} tokens. Hugging Face release: {sentences:,} sentences, {release:,} tokens.")
(OUT / "tab_data.tex").write_text(
    "\\begin{tabular}{@{}llllrrl@{}}\n\\toprule\nEdition & Editor, printing & Source & Genre & Tokens & \\% long & "
    "Role (\\% unseen) \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n"
    f"\\multicolumn{{7}}{{@{{}}p{{\\linewidth}}@{{}}}}{{\\footnotesize {notes}}} \\\\\n\\end{{tabular}}\n", encoding="utf8")

# Paired sentence bootstrap (oevl.stats.bootstrap_diff) of the prior and the logistic models of the + WikiPron run
# (results/wikipron): each arm's per-token correctness is its mean over seeds 1-3. Differences in points with a 95%
# interval. Class B means lexicon-ambiguous tokens whose skeleton is class B, as in metrics.json
from oevl.stats import bootstrap_diff  # noqa: E402
PAIRS = (("word", "frequency"), ("left", "frequency"), ("sentence", "frequency"), ("left", "word"),
         ("sentence", "word"), ("sentence", "left"))
BPOPS = ("lexicon_ambiguous", "class_B")
boot = {}
for t in TESTS:
    trows = preds if t == "soliloquies" else E.load(ROOT / "results" / "wikipron" / f"{t}_predictions.jsonl")
    for pop in BPOPS:
        sub = [r for r in trows if r["lexicon_ambiguous"] and (pop == "lexicon_ambiguous" or
                                                              CLASSES.get(r["key"], "unlisted") == "B")]
        ok: dict[str, list[float]] = {"frequency": [r["frequency_prediction"] == r["gold_pattern"] for r in sub]}
        for a in ("word", "left", "sentence"):
            ok[a] = [sum(r[f"{a}_seed{k}_prediction"] == r["gold_pattern"] for k in (1, 2, 3)) / 3 for r in sub]
        for x, y in PAIRS:
            boot[t, pop, x, y] = bootstrap_diff([r["sentence_id"] for r in sub], ok[x], ok[y])
TAG = {"soliloquies": "Sol", "tauber_beowulf": "Beo", "lexicon_ambiguous": "Lex", "class_B": "ClsB", "frequency": "Freq",
       "word": "Word", "left": "Left", "sentence": "Bil"}
for (t, p, x, y), r in boot.items():
    k = f"Boot{TAG[t]}{TAG[p]}{TAG[x]}Vs{TAG[y]}"
    num[k], num[k + "Lo"], num[k + "Hi"] = (f"{100 * r[q]:+.2f}" for q in ("diff", "lo", "hi"))

# ---------------------------------------------------------------- write numbers.tex
OUT.mkdir(parents=True, exist_ok=True)
with open(OUT / "numbers.tex", "w", encoding="utf8") as f:
    f.write("% generated by src/oevl/build_report.py from data/ and results/ -- do not edit\n")
    for k, v in sorted(num.items()):
        name = "N" + "".join(ch for ch in k if ch.isalpha())
        if name != "N" + k:
            sys.exit(f"macro name must be letters only: {k}")
        f.write(f"\\newcommand{{\\{name}}}{{{v}}}\n")


print(f"wrote {len(num)} numbers and tab_data.tex to {OUT.relative_to(ROOT)}")

# ---------------------------------------------------------------- compile
if shutil.which("pdflatex"):
    for cmd in (["pdflatex"], ["bibtex"], ["pdflatex"], ["pdflatex"]):  # bibtex reads references.bib after pass one
        args = cmd + (["-interaction=nonstopmode", "-halt-on-error", "main.tex"] if cmd[0] == "pdflatex" else ["main"])
        r = subprocess.run(args, cwd=ROOT / "report", capture_output=True, text=True,
                           errors="replace",  # pdflatex echoes T1 bytes (þ, ý) that are not UTF-8
                           env={**os.environ, "SOURCE_DATE_EPOCH": "1792454400"})  # fixed PDF dates and ID: same bytes
        if r.returncode:
            print(r.stdout[-3000:])
            sys.exit(f"{cmd[0]} failed; see report/main.log or main.blg")
    shutil.copy(ROOT / "report" / "main.pdf", ROOT / "REPORT.pdf")
    print("compiled REPORT.pdf")
else:
    print("pdflatex not found: upload the report/ folder to Overleaf and compile main.tex there")
