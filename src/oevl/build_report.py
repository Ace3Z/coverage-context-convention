"""Generate every number and table used in report/main.tex, and an accuracy figure, from data/ and results/.

Writes report/generated/{numbers.tex, tab_data.tex, tab_main.tex, tab_ablation.tex, tab_perword.tex,
tab_classes.tex, tab_bootstrap.tex, fig_accuracy.pdf}. It also checks that the worked examples quoted in the report match the
predictions, and compiles report/main.tex to REPORT.pdf when pdflatex is installed (otherwise upload report/ to
Overleaf).
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

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "report" / "generated"
SYSTEMS = (("frequency", "Frequency"), ("word", "Word"), ("left", "Left"), ("sentence", "Bilateral"))
TESTS = (("soliloquies", "Soliloquies"), ("tauber_beowulf", "Beowulf"))


def load(folder):
    return json.loads((ROOT / "results" / folder / "metrics.json").read_text(encoding="utf8"))


W, T = load("wikipron"), load("training_only")
P, A, Z = load("proposal"), load("all"), load("wikipron_with_zupitza")  # proposal dictionaries, all four, D3 ablation


def pct(x):
    return f"{100 * x:.2f}"


def acc(d, pop):
    return d["accuracy"] if pop == "accuracy" else d[pop]["accuracy"]


def val(m, test, system, pop, sd=False):
    t = m[test]
    if system in ("frequency", "oracle"):
        return pct(acc(t[system], pop))
    s = t["seed_summary"][system][pop]
    return f"{pct(s['mean'])}$\\pm${100 * s['sd']:.2f}" if sd else pct(s["mean"])


def n(m, test, pop):
    return m[test]["frequency"]["n"] if pop == "accuracy" else m[test]["frequency"][pop]["n"]


def p_fmt(p):
    return f"{p:.4f}" if p >= 0.001 else f"{p:.4f}".rstrip("0") if p >= 0.0001 else f"{p:.1e}"


# ---------------------------------------------------------------- numbers used in the prose
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
sds = [W[t]["seed_summary"][a][p]["sd"] for t, _ in TESTS for a in ("word", "left", "sentence")
       for p in ("accuracy", "lexicon_ambiguous")]
num["SdMaxMain"] = f"{100 * max(sds):.2f}"
num["SdMaxMin"] = f"{100 * max(W[t]['seed_summary'][a]['minority_reading']['sd'] for t, _ in TESTS for a in ('word', 'left', 'sentence')):.2f}"
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


def rjson(path):
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
# class B disagreements between Harrison & Sharp and Wyatt, and how many are ne/nē (the PG revision's ne = nē)
cls_b = {r["skeleton"] for r in csv.DictReader(open(ROOT / "data" / "ambiguity_classes.csv", encoding="utf8"))
         if r["final"] == "B"}
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
cs = CM["soliloquies"]  # its length errors on ambiguous prose tokens, and how many the logistic models get right
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
from oevl.encoder_summary import configs  # noqa: E402
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

# ---------------------------------------------------------------- per-word table (training-only run)
pw = {r["word"]: r for r in csv.DictReader(open(ROOT / "results" / "training_only" / "soliloquies_per_word.csv",
                                                encoding="utf8"))}
WORDS = ["þæt", "þe", "þu", "ne", "to", "se", "ge"]
missing = [w for w in WORDS if w not in pw]
if missing:
    sys.exit(f"per-word table: words missing from CSV: {missing}")


def counts(js):
    c = json.loads(js)
    return f"{c.get('0', 0)}/{c.get('1', 0)}"


# Zupitza is not trained on (decision D3), so the per-word CSVs hold no counts for it: its column counts the corpus file
from oevl import experiment as E  # noqa: E402
ZUP = collections.Counter((r["key"], r["gold_pattern"]) for r in E.tokens(E.load(ROOT / "data/by_source/zupitza_grammar.jsonl")))


def zup(w):
    return f"{ZUP[w, '0']}/{ZUP[w, '1']}"


the = json.loads(pw["þæt"]["test_pattern_counts"])
num["ThaetShort"] = str(the.get("0", 0))
num["ThaetN"] = pw["þæt"]["n"]
ge = pw["ge"]
num["GeZup"] = zup("ge")
num["GeMat"] = counts(ge["matthew_ws_pattern_counts"])
num["GeSol"] = counts(ge["test_pattern_counts"])

# ---------------------------------------------------------------- worked examples: verify the claims
preds = [json.loads(line) for line in open(ROOT / "results" / "wikipron" / "soliloquies_predictions.jsonl",
                                           encoding="utf8")]
bysent = {}
for p in preds:
    bysent.setdefault(p["sentence_id"], []).append(p)
EXAMPLES = {
    "Ic clypie tō þē Drihten þū þe æall geworhtest": "ex1",
    "Tō þē ic clypige þe nēfre nān yfel ne worhtest": "ex2",
}
found = {}
for sid, toks in bysent.items():
    text = " ".join(t["gold"] for t in toks)
    for start, tag in EXAMPLES.items():
        if text.startswith(start):
            found[tag] = toks
if set(found) != set(EXAMPLES.values()):
    sys.exit(f"worked examples not found in predictions: {set(EXAMPLES.values()) - set(found)}")
arms = ["frequency_prediction"] + [f"{a}_seed{s}_prediction" for a in ("word", "left", "sentence") for s in (1, 2, 3)]
for tag, toks in found.items():
    first_the = next(t for i, t in enumerate(toks) if t["key"] == "þe" and i > 0 and toks[i - 1]["key"] == "to")
    if first_the["gold_pattern"] != "1" or any(first_the[a] != "0" for a in arms):
        sys.exit(f"{tag}: 'tō þē' claim no longer holds: {first_the}")
    rel = next(t for i, t in enumerate(toks) if t["key"] == "þe" and t["gold_pattern"] == "0")
    if any(rel[a] != "0" for a in arms):
        sys.exit(f"{tag}: relative 'þe' claim no longer holds: {rel}")

# ---------------------------------------------------------------- coverage and þe facts recomputed from the data
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
# the same tokens in the ByT5 runs (results/encoder): fewest and most long þē after tō per arm, over its runs
import gzip  # noqa: E402
head, *enc_rows = gzip.decompress((ROOT / "results/encoder/soliloquies_predictions.tsv.gz").read_bytes()).decode(
    "utf8").split("\n")[:-1]
ENC_SOL = {r[0]: dict(zip(head.split("\t"), r)) for r in (line.split("\t") for line in enc_rows)}
enc_runs = [c for c in head.split("\t") if "/" in c]
for arm, short in (("word", "Word"), ("local2", "LocalTwo"), ("sentence", "Bil")):
    hits = [sum(ENC_SOL[t["id"]][c] == "1" for t in after_to) for c in enc_runs if c.split("/")[1].startswith(arm + "_")]
    num[f"ToTheEnc{short}Min"], num[f"ToTheEnc{short}Max"] = str(min(hits)), str(max(hits))
    num[f"ToTheEnc{short}Runs"] = str(len(hits))
# ex1, "tō þē ... þū þe": every local2 run gets both right, every word run marks both short
ex1 = found["ex1"]
the_long = next(t for i, t in enumerate(ex1) if t["key"] == "þe" and i > 0 and ex1[i - 1]["key"] == "to")
the_rel = next(t for t in ex1 if t["key"] == "þe" and t["gold_pattern"] == "0")
for c in enc_runs:
    arm = c.split("/")[1].rsplit("_", 1)[0]
    if arm == "local2" and (ENC_SOL[the_long["id"]][c], ENC_SOL[the_rel["id"]][c]) != ("1", "0"):
        sys.exit(f"ex1: the local2 claim no longer holds for {c}")
    if arm == "word" and (ENC_SOL[the_long["id"]][c], ENC_SOL[the_rel["id"]][c]) != ("0", "0"):
        sys.exit(f"ex1: the word-arm claim no longer holds for {c}")
# Beowulf 2796 "þē ic hēr on starie": gold and Harrison and Sharp long, Wyatt short, every full-cap ByT5 run short
head, *rows = gzip.decompress((ROOT / "results/encoder/tauber_beowulf_predictions.tsv.gz").read_bytes()).decode(
    "utf8").split("\n")[:-1]
b2796 = next(dict(zip(head.split("\t"), r.split("\t"))) for r in rows if r.startswith("tauber:685:44\t"))
lab = next(r for r in csv.DictReader(open(ROOT / "results/error_typing/labels.tsv", encoding="utf8"), delimiter="\t")
           if r["id"] == "tauber:685:44")
if (b2796["gold"], lab["hs_form"], lab["wyatt_form"], lab["final"]) != ("1", "þē", "þe", "gold_in_doubt") or any(
        v != "0" for k, v in b2796.items() if k.startswith("capfull/")):
    sys.exit("Beowulf 2796 example no longer holds")
for a, short in (("word", "Word"), ("left", "Left"), ("sentence", "Bil")):
    num[f"SolGain{short}"] = f"{100 * (W['soliloquies']['seed_summary'][a]['accuracy']['mean'] - T['soliloquies']['seed_summary'][a]['accuracy']['mean']):.2f}"
    num[f"BeoGain{short}"] = f"{100 * (W['tauber_beowulf']['seed_summary'][a]['accuracy']['mean'] - T['tauber_beowulf']['seed_summary'][a]['accuracy']['mean']):.2f}"

# ---------------------------------------------------------------- tables
ORDER = ["zupitza_grammar", "bright_reader", "matthew_ws", "sweet_primer", "physiologus", "john_ws", "soliloquies",
         "hs_beowulf", "wyatt_beowulf", "tauber_beowulf"]
TITLES = {"zupitza_grammar": "Ælfric's \\textit{Grammar}", "bright_reader": "\\textit{Anglo-Saxon Reader}",
          "matthew_ws": "Gospel of Matthew", "sweet_primer": "\\textit{Anglo-Saxon Primer}",
          "physiologus": "\\textit{Physiologus}", "john_ws": "Gospel of John",
          "soliloquies": "Alfred's \\textit{Soliloquies}",
          "hs_beowulf": "\\textit{Beowulf}", "wyatt_beowulf": "\\textit{Beowulf}", "tauber_beowulf": "\\textit{Beowulf}"}
# shown as printed, not as the manifest has it: Matthew was transcribed from the 1910 printing, and every text taken
# from Sweet's Primer is prose
EDITOR = {"matthew_ws": "Bright, 1910 printing", "hs_beowulf": "Harrison \\& Sharp 1883$^a$",
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
notes = (f"$^a$Project Gutenberg revision of the 4th edition (copyright 1883). $^b$Its \\textit{{Finnsburh Fragment}} is "
         f"incomplete ({fin[0]} of {fin[1]} tokens). Scored tokens: dev {W['john_ws']['frequency']['n']:,}, prose "
         f"{n(W, 'soliloquies', 'accuracy'):,}, verse {n(W, 'tauber_beowulf', 'accuracy'):,}. Training without Zupitza: "
         f"{W['dataset']['train']:,} tokens. Hugging Face release: {sentences:,} sentences, {release:,} tokens.")
(OUT / "tab_data.tex").write_text(
    "\\begin{tabular}{@{}llllrrl@{}}\n\\toprule\nEdition & Editor, printing & Source & Genre & Tokens & \\% long & "
    "Role (\\% unseen) \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n"
    f"\\multicolumn{{7}}{{@{{}}p{{\\linewidth}}@{{}}}}{{\\footnotesize {notes}}} \\\\\n\\end{{tabular}}\n", encoding="utf8")

hdr = ("\\begin{tabular}{@{}lrrrrrr@{}}\n\\toprule\n & \\multicolumn{3}{c}{\\textit{Soliloquies} (prose)} & "
       "\\multicolumn{3}{c}{\\textit{Beowulf} (verse)} \\\\\n\\cmidrule(lr){2-4}\\cmidrule(l){5-7}\n")
pops = ("accuracy", "lexicon_ambiguous", "minority_reading")
lab = {"accuracy": "All", "lexicon_ambiguous": "Lex.-amb.", "minority_reading": "Minority"}
head = "System & " + " & ".join(f"{lab[p]}" for p in pops) + " & " + " & ".join(f"{lab[p]}" for p in pops) + " \\\\\n"
nrow = "\\textit{n} & " + " & ".join(f"{n(W, t, p):,}" for t, _ in TESTS for p in pops) + " \\\\\n\\midrule\n"
body = []
for s, name in SYSTEMS + (("oracle", "Oracle"),):
    body.append(name + " & " + " & ".join(val(W, t, s, p) for t, _ in TESTS for p in pops) + " \\\\")
(OUT / "tab_main.tex").write_text(hdr + head + nrow + "\n".join(body) + "\n\\bottomrule\n\\end{tabular}\n",
                                  encoding="utf8")

abl = ["\\begin{tabular}{@{}llrrrr@{}}\n\\toprule\n & & \\multicolumn{2}{c}{Training-only} & "
       "\\multicolumn{2}{c}{+ WikiPron} \\\\\n\\cmidrule(lr){3-4}\\cmidrule(l){5-6}\n"
       "Test & System & All & Lex.-amb. & All & Lex.-amb. \\\\\n\\midrule"]
for t, tname in TESTS:
    for s, name in (("frequency", "Frequency"), ("sentence", "Bilateral"), ("oracle", "Oracle")):
        abl.append(f"{tname} & {name} & {val(T, t, s, 'accuracy')} & {val(T, t, s, 'lexicon_ambiguous')} & "
                   f"{val(W, t, s, 'accuracy')} & {val(W, t, s, 'lexicon_ambiguous')} \\\\")
    abl.append("\\addlinespace" if t == "soliloquies" else "")
(OUT / "tab_ablation.tex").write_text("\n".join(abl) + "\n\\bottomrule\n\\end{tabular}\n", encoding="utf8")

pwr = ["\\begin{tabular}{@{}lrrrrrrr@{}}\n\\toprule\nWord & \\textit{n} & Gold s/l & Freq. & Bil. & "
       "Zupitza s/l & Matthew s/l & Bright s/l \\\\\n\\midrule"]
for w in WORDS:
    r = pw[w]
    pwr.append(f"\\textit{{{w}}} & {r['n']} & {counts(r['test_pattern_counts'])} & "
               f"{100 * float(r['frequency_accuracy']):.1f} & {100 * float(r['sentence_seed1_accuracy']):.1f} & "
               f"{zup(w)} & {counts(r['matthew_ws_pattern_counts'])} & "
               f"{counts(r['bright_reader_pattern_counts'])} \\\\")
(OUT / "tab_perword.tex").write_text("\n".join(pwr) + "\n\\bottomrule\n\\end{tabular}\n", encoding="utf8")

cls = ["\\begin{tabular}{@{}llrrrrrr@{}}\n\\toprule\n"
       "Test & Class & \\textit{n} & Freq. & Word & Left & Bil. & Oracle \\\\\n\\midrule"]
for t, tname in TESTS:
    for c, cname in (("A", "A lexical"), ("B", "B grammatical"), ("C", "C editorial"), ("unlisted", "Outside top 60")):
        pop = "class_" + c
        accs = " & ".join(val(W, t, s, pop) for s in ("frequency", "word", "left", "sentence", "oracle"))
        cls.append(f"{tname if c == 'A' else ''} & {cname} & {n(W, t, pop):,} & {accs} \\\\")
    cls.append("\\addlinespace" if t == "soliloquies" else "")
(OUT / "tab_classes.tex").write_text("\n".join(cls) + "\n\\bottomrule\n\\end{tabular}\n", encoding="utf8")

# Paired sentence bootstrap (oevl.stats) of the logistic models, + WikiPron run: each arm's per-token correctness is
# its mean over seeds 1-3; differences in points with a 95% interval
from oevl.stats import bootstrap_diff  # noqa: E402
CLASSES = E.load_classes(ROOT / "data" / "ambiguity_classes.csv")
PAIRS = (("word", "frequency"), ("left", "frequency"), ("sentence", "frequency"), ("left", "word"),
         ("sentence", "word"), ("sentence", "left"))
SHORT = {"frequency": "Freq.", "word": "Word", "left": "Left", "sentence": "Bil."}
BPOPS = (("lexicon_ambiguous", "Lex.-amb."), ("class_B", "Class B"))
boot, bn = {}, {}
for t, _ in TESTS:
    rows = preds if t == "soliloquies" else [json.loads(line) for line in open(
        ROOT / "results" / "wikipron" / f"{t}_predictions.jsonl", encoding="utf8")]
    for pop, _ in BPOPS:
        sub = [r for r in rows if r["lexicon_ambiguous"] and (pop == "lexicon_ambiguous" or
                                                              CLASSES.get(r["key"], "unlisted") == "B")]
        bn[t, pop] = len(sub)
        ok = {"frequency": [r["frequency_prediction"] == r["gold_pattern"] for r in sub]}
        for a in ("word", "left", "sentence"):
            ok[a] = [sum(r[f"{a}_seed{k}_prediction"] == r["gold_pattern"] for k in (1, 2, 3)) / 3 for r in sub]
        for x, y in PAIRS:
            boot[t, pop, x, y] = bootstrap_diff([r["sentence_id"] for r in sub], ok[x], ok[y])
bt = ["\\begin{tabular}{@{}lllll@{}}\n\\toprule\n & \\multicolumn{2}{c}{\\textit{Soliloquies} (prose)} & "
      "\\multicolumn{2}{c}{\\textit{Beowulf} (verse)} \\\\\n\\cmidrule(lr){2-3}\\cmidrule(l){4-5}\n"
      "Difference & " + " & ".join(lab for _ in TESTS for _, lab in BPOPS) + " \\\\\n"
      "\\textit{n} & " + " & ".join(f"{bn[t, p]:,}" for t, _ in TESTS for p, _ in BPOPS) + " \\\\\n\\midrule"]
for x, y in PAIRS:
    bt.append(f"{SHORT[x]} $-$ {SHORT[y]} & " + " & ".join(
        "${:+.2f}\\ [{:+.2f}, {:+.2f}]$".format(*(100 * boot[t, p, x, y][k] for k in ("diff", "lo", "hi")))
        for t, _ in TESTS for p, _ in BPOPS) + " \\\\")
(OUT / "tab_bootstrap.tex").write_text("\n".join(bt) + "\n\\bottomrule\n\\end{tabular}\n", encoding="utf8")

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


# ---------------------------------------------------------------- figure (drawn at its printed size: 15 cm wide)
# Dot plot (not bars): the left panel needs a zoomed axis, and bars must start at zero.
plt.rcParams.update({"font.family": "serif", "font.serif": ["Liberation Serif", "Nimbus Roman", "Times New Roman",
                     "DejaVu Serif"], "font.size": 11, "axes.titlesize": 11, "axes.labelsize": 11,
                     "xtick.labelsize": 11, "ytick.labelsize": 11, "legend.fontsize": 11, "pdf.fonttype": 42})
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # validated categorical slots 1-4 (dataviz palette)
MARKERS = ["o", "s", "^", "D"]  # second channel, so identity never relies on colour alone
fig, axes = plt.subplots(1, 2, figsize=(15 / 2.54, 6.2 / 2.54))
for ax, pop, title in ((axes[0], "lexicon_ambiguous", "Lexicon-ambiguous tokens"),
                       (axes[1], "minority_reading", "Minority-reading tokens")):
    for i, (s, name) in enumerate(SYSTEMS):
        xs, ys, es = [], [], []
        for j, (t, _) in enumerate(TESTS):
            d = W[t]
            if s == "frequency":
                y, e = 100 * acc(d["frequency"], pop), 0.0
            else:
                y, e = 100 * d["seed_summary"][s][pop]["mean"], 100 * d["seed_summary"][s][pop]["sd"]
            xs.append(j + (i - 1.5) * 0.16); ys.append(y); es.append(e)
        ax.errorbar(xs, ys, yerr=es, fmt=MARKERS[i], color=COLORS[i], markersize=6.5, elinewidth=1.2,
                    capsize=0, label=name, markeredgecolor="white", markeredgewidth=0.8)
    ax.set_xticks([0, 1], [name for _, name in TESTS])
    ax.set_xlim(-0.5, 1.5)
    if pop == "minority_reading":  # the other panel keeps matplotlib's zoomed range around its data
        ax.set_ylim(bottom=0)
    ax.set_title(title)
    ax.set_ylabel("Accuracy (%)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.grid(True, color="#e6e6e6", linewidth=0.6)
    ax.set_axisbelow(True)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.02),
           handletextpad=0.3, columnspacing=1.2)
fig.tight_layout(rect=(0, 0.1, 1, 1), w_pad=1.5)
fig.savefig(OUT / "fig_accuracy.pdf", metadata={"CreationDate": None})  # no timestamp: reruns are byte-identical
plt.close(fig)

print(f"wrote {len(num)} numbers, 6 tables and the figure to {OUT.relative_to(ROOT)}")
print("worked examples verified against results/wikipron/soliloquies_predictions.jsonl")

# ---------------------------------------------------------------- compile
if shutil.which("pdflatex"):
    for cmd in (["pdflatex"], ["bibtex"], ["pdflatex"], ["pdflatex"]):  # bibtex reads references.bib after the first pass
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
