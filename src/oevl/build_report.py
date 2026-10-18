"""Generate every number, table and the figure used in report/main.tex from the results files.

Writes report/generated/{numbers.tex, tab_data.tex, tab_main.tex, tab_ablation.tex, tab_perword.tex,
tab_classes.tex, fig_accuracy.pdf}. It also checks that the worked examples quoted in the report match the
predictions, and compiles report/main.tex to REPORT.pdf when pdflatex is installed (otherwise upload report/ to
Overleaf).
Usage: python -m oevl.build_report
"""
import csv
import json
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
for tag, m in (("W", W), ("T", T)):
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

# S6 ambiguity classes: annotator agreement and final counts
CLS = json.loads((ROOT / "data" / "ambiguity_classes.json").read_text(encoding="utf8"))
num["ClassKappa"] = f"{CLS['cohen_kappa']:.3f}"
num["ClassAgree"] = pct(CLS["agreement"])
for c in "ABC":
    num[f"Class{c}"] = str(CLS["counts"]["final"][c])

# WikiPron conversion statistics, recomputed with the same converter used by run_all.sh
import re  # noqa: E402
from oevl.wikipron_to_marked import mark  # noqa: E402
wp_rows = wp_ok = 0
wp_forms = set()
for line in open(ROOT / "data" / "ang_latn_broad.tsv", encoding="utf8"):
    bare_w, ipa = line.rstrip("\n").split("\t")[:2]
    wp_rows += 1
    if re.fullmatch(r"[a-zþðæ]+", bare_w.lower()) and (m := mark(bare_w, ipa)) is not None:
        wp_ok += 1
        wp_forms.add(m)
lex_rows = sum(1 for _ in open(ROOT / "data" / "lex_wikipron.csv", encoding="utf8")) - 1
if lex_rows != len(wp_forms):
    sys.exit(f"lex_wikipron.csv has {lex_rows} forms but the converter gives {len(wp_forms)}")
num["WikiRows"] = f"{wp_rows:,}"
num["WikiForms"] = f"{lex_rows:,}"
num["WikiAligned"] = f"{100 * wp_ok / wp_rows:.1f}"

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


the = json.loads(pw["þæt"]["test_pattern_counts"])
num["ThaetShort"] = str(the.get("0", 0))
num["ThaetN"] = pw["þæt"]["n"]
ge = pw["ge"]
num["GeZup"] = counts(ge["zupitza_grammar_pattern_counts"])
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
from oevl import experiment as E  # noqa: E402
train_toks, _, _, _ = E.build_data(str(ROOT / "data"))
lex_t, _ = E.lexicon(train_toks)
beo = E.load_tauber(ROOT / "data" / "brunetti-length.txt")
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
for a, short in (("word", "Word"), ("left", "Left"), ("sentence", "Bil")):
    num[f"SolGain{short}"] = f"{100 * (W['soliloquies']['seed_summary'][a]['accuracy']['mean'] - T['soliloquies']['seed_summary'][a]['accuracy']['mean']):.2f}"
    num[f"BeoGain{short}"] = f"{100 * (W['tauber_beowulf']['seed_summary'][a]['accuracy']['mean'] - T['tauber_beowulf']['seed_summary'][a]['accuracy']['mean']):.2f}"

# ---------------------------------------------------------------- write numbers.tex
OUT.mkdir(parents=True, exist_ok=True)
with open(OUT / "numbers.tex", "w", encoding="utf8") as f:
    f.write("% generated by src/oevl/build_report.py from results/*/metrics.json -- do not edit\n")
    for k, v in sorted(num.items()):
        name = "N" + "".join(ch for ch in k if ch.isalpha())
        if name != "N" + k:
            sys.exit(f"macro name must be letters only: {k}")
        f.write(f"\\newcommand{{\\{name}}}{{{v}}}\n")

# ---------------------------------------------------------------- tables
ORDER = ["zupitza_grammar", "bright_reader", "matthew_ws", "sweet_primer", "physiologus", "soliloquies",
         "hs_beowulf", "wyatt_beowulf"]
TITLES = {"zupitza_grammar": "Ælfric's \\textit{Grammar}", "bright_reader": "\\textit{Anglo-Saxon Reader}",
          "matthew_ws": "Gospel of Matthew (W.~Saxon)", "sweet_primer": "\\textit{Anglo-Saxon Primer}",
          "physiologus": "\\textit{Physiologus}", "soliloquies": "Alfred's \\textit{Soliloquies}",
          "hs_beowulf": "\\textit{Beowulf}", "wyatt_beowulf": "\\textit{Beowulf}"}
ROLE = {"soliloquies": "prose test", "hs_beowulf": "agreement", "wyatt_beowulf": "agreement"}
GENRE = {"prose": "prose", "verse": "verse", "mixed": "mixed"}
rows = []
for k in ORDER:
    s = manifest[k]
    editor = s["editor"].replace("&", r"\&")  # no backslash inside f-string braces (Python 3.11)
    rows.append(f"{TITLES[k]} & {editor}, {s['year']} & {GENRE[s['genre']]} & "
                f"{s['tokens']:,} & {ROLE.get(k, 'train')} \\\\")
(OUT / "tab_data.tex").write_text(
    "\\begin{tabular}{@{}lllrl@{}}\n\\toprule\nEdition & Editor, year & Genre & Tokens & Role \\\\\n\\midrule\n"
    + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n", encoding="utf8")

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
               f"{counts(r['zupitza_grammar_pattern_counts'])} & {counts(r['matthew_ws_pattern_counts'])} & "
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

# ---------------------------------------------------------------- figure (drawn at its printed size: 15 cm wide)
# Dot plot (not bars): the left panel needs a zoomed axis, and bars must start at zero.
plt.rcParams.update({"font.family": "serif", "font.serif": ["Liberation Serif", "Nimbus Roman", "Times New Roman",
                     "DejaVu Serif"], "font.size": 11, "axes.titlesize": 11, "axes.labelsize": 11,
                     "xtick.labelsize": 11, "ytick.labelsize": 11, "legend.fontsize": 11, "pdf.fonttype": 42})
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # validated categorical slots 1-4 (dataviz palette)
MARKERS = ["o", "s", "^", "D"]  # second channel, so identity never relies on colour alone
fig, axes = plt.subplots(1, 2, figsize=(15 / 2.54, 6.2 / 2.54))
for ax, pop, title, ylim in ((axes[0], "lexicon_ambiguous", "Lexicon-ambiguous tokens", (87.5, 89.5)),
                             (axes[1], "minority_reading", "Minority-reading tokens", (0, 10))):
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
    ax.set_ylim(*ylim)
    ax.set_title(title)
    ax.set_ylabel("Accuracy (%)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.grid(True, color="#e6e6e6", linewidth=0.6)
    ax.set_axisbelow(True)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.02),
           handletextpad=0.3, columnspacing=1.2)
fig.tight_layout(rect=(0, 0.1, 1, 1), w_pad=1.5)
fig.savefig(OUT / "fig_accuracy.pdf")
plt.close(fig)

print(f"wrote {len(num)} numbers, 5 tables and the figure to {OUT.relative_to(ROOT)}")
print("worked examples verified against results/wikipron/soliloquies_predictions.jsonl")

# ---------------------------------------------------------------- compile
if shutil.which("pdflatex"):
    for _ in range(2):
        r = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "main.tex"],
                           cwd=ROOT / "report", capture_output=True, text=True)
        if r.returncode:
            print(r.stdout[-3000:])
            sys.exit("pdflatex failed; see report/main.log")
    shutil.copy(ROOT / "report" / "main.pdf", ROOT / "REPORT.pdf")
    print("compiled REPORT.pdf")
else:
    print("pdflatex not found: upload the report/ folder to Overleaf and compile main.tex there")
