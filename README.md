<div align="center">

<div align="center">
<table>
  <tr>
    <td align="center" valign="middle">
      <a href="https://www.uzh.ch/"><picture>
        <source media="(prefers-color-scheme: dark)" srcset="media/figures/logos/uzh-logo-white.png"/>
        <img src="media/figures/logos/uzh-logo-black.png" height="70" alt="University of Zurich"/>
      </picture></a>
    </td>
  </tr>
</table>
</div>

<p align="center">
  <a href="https://huggingface.co/datasets/Ace3Z/old-english-vowel-length">
    <img src="https://img.shields.io/badge/Hugging%20Face-Dataset-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black" alt="Dataset"/>
  </a>
  <!-- TODO: link the report PDF once it is uploaded -->
  <img src="https://img.shields.io/badge/Report-PDF%20TODO-b31b1b?style=for-the-badge" alt="Report: TODO"/>
  <a href="LICENSE">
    <img src="https://img.shields.io/badge/Licence-MIT-2ea44f?style=for-the-badge" alt="Licence"/>
  </a>
  <a href="REPRODUCE.md">
    <img src="https://img.shields.io/badge/Python-3.11%20to%203.14-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"/>
  </a>
</p>

# Coverage, Context, Convention: Restoring Vowel Length in Old English

**Old English editions mark long vowels with macrons that the manuscripts never wrote. We ask what limits putting
them back automatically, and measure four limits on the same tokens: dictionary coverage, sentence context,
editorial convention and editor disagreement.**

[Overview](#-overview) •
[Headline results](#-headline-results) •
[Pipeline](#-pipeline) •
[Quickstart](#%EF%B8%8F-quickstart) •
[Dataset](#-dataset) •
[Documentation](#-documentation)

</div>

---

<div align="center">
<table>
  <tr>
    <td align="center"><b>Input</b> (bare)</td>
    <td align="center"><i>Ic þe lære þeah þæt þu hyt ongynne</i></td>
  </tr>
  <tr>
    <td align="center"><b>Frequency prior</b>, <b>logistic word model</b></td>
    <td align="center"><i>Ic þe lǣre þēah þæt þū hyt ongynne</i></td>
  </tr>
  <tr>
    <td align="center"><b>Logistic model</b> with two words each side</td>
    <td align="center"><i>Ic <b>þē</b> lǣre þēah þæt þū hyt ongynne</i></td>
  </tr>
</table>

*A held-out sentence from Alfred's* Soliloquies *(`soliloquies:00530`), restored with `oevl.try_text`. Only models
that see the neighbouring words pick the pronoun* þē *'thee' over the relative particle* þe, *as the edition does.*

</div>

---

## 📖 Overview

Old English manuscripts do not show vowel length, so *god* 'God' and *gōd* 'good' look the same, and so do the
relative particle *þe* and the pronoun *þē* 'thee'. Modern editions add macrons by hand. A system for this task sees
the bare word, takes the length patterns a dictionary lists for it, and picks one.

We built a length-marked corpus from nine editions and hold out one of them, Alfred's *Soliloquies*, as the prose
test. The verse test is a tenth source, Tauber's token-level *Beowulf*. On exactly the same tokens we compare a
frequency prior, logistic classifiers, a fine-tuned ByT5 model and CharsiuG2P, and three editions of *Beowulf* show
how far editors agree. `bash run_all.sh` reproduces every number and table in the report.

| Limit | Question | Where to look |
|---|---|---|
| Coverage | Is the right spelling among the candidates? | dictionary ablation, `results/<setting>/` |
| Context | Do the neighbouring words pick the right candidate? | `results/encoder/`, `results/wikipron/` |
| Convention | How much of the remaining error is editorial habit? | ambiguity classes, `results/error_typing/` |
| Disagreement | How often do two editors mark a word the same way? | `results/agreement/` |

---

## 📊 Headline results

The ByT5 rows are means over three seeds at the full byte cap, and the intervals are 95% sentence-level bootstrap
intervals on each token's correctness averaged over seeds. Lexicon-ambiguous tokens have two or more candidate patterns, and class B holds grammatical homographs,
words spelled alike that differ in function.

| Finding | Tokens | Result |
|---|---|---|
| Coverage matters most on verse | *Beowulf*, all tokens, frequency prior | 82.40% with training forms, 92.38% with all four dictionaries |
| Context helps most on grammatical homographs | *Soliloquies*, class B (*þe/þē*, *ne/nē*), ByT5 | 62.66% from the word alone, 76.47% with two words each side (+13.82, 11.39 to 16.18) |
| Context helps prose overall | *Soliloquies*, lexicon-ambiguous tokens, ByT5 | 88.91% from the word alone, 91.31% with two words each side (+2.40, 1.92 to 2.89) |
| On verse the word alone is enough | *Beowulf*, lexicon-ambiguous tokens, ByT5 | 90.04% from the word alone, 89.83% with two words each side (−0.20, −0.62 to +0.21), and one unseen name, *Gēata* (39 tokens), decides the sign |
| Editors differ most on grammatical words | Wyatt against Klaeber (Tauber), lexicon-ambiguous / class B | 96.08% / 78.07% agreement |
| CharsiuG2P's errors are mostly about length | its own 500-word Old English test set | word error rate 0.208 as published and 0.188 with 50 decoding steps, where 75.5% of the word errors differ only in vowel length |

An LLM, Claude Opus 5.5 (model id `claude-opus-5-5`), made the 560 labels. Two separate runs of it labelled the
ambiguity classes and the error types with the same written guidelines, and a third run settled their disagreements.
A single run labelled a sample of 100 editor disagreements. A blind review by further runs of the same model agreed
with 538 of the 560 labels, and of the 22 it disputed another run kept 13 and corrected nine. Mahbod Tajdini then
checked all 560 labels by hand and changed none. The guidelines, the item sheets and the labels of the two runs
(`labels_a.tsv`, `labels_b.tsv`) are in `results/label_review/materials/`, and the blind review is in
`results/label_review/expert_review.json`. The settled labels are in `data/ambiguity_classes.csv` and
`results/error_typing/labels.tsv`, and the single run's labels of the disagreements in
`results/agreement/disagreements_checked.tsv`.
The report has the full tables, the caveats and the run-to-run variance (TODO: link the PDF).

---

## 🧩 Pipeline

The code is the Python package `oevl` in `src/oevl/`. `bash run_all.sh` runs every CPU stage from the committed
inputs and rebuilds all tables, figures and the numbers the report prints. Rebuilding the corpus or the dictionaries
from their sources, the CharsiuG2P benchmark and the GPU runs have their own commands in [REPRODUCE.md](REPRODUCE.md).

| # | Stage | Module | Writes |
|---|---|---|---|
| 1 | Corpus build and clean-up | `build_corpus.py`, `clean_corpus.py`, `fetch_tauber.py` | `data/by_source/`, `data/manifest.json`, `data/brunetti-length.txt` |
| 2 | Leakage audit | `audit_splits.py` | `data/leakage_excluded.json` |
| 3 | Dictionaries | `wikipron_to_marked.py`, `build_lexicons.py` | `data/lex_*.csv` |
| 4 | Frequency prior and logistic models | `experiment.py` | `results/<setting>/` |
| 5 | Editor agreement | `agreement.py` | `results/agreement/` |
| 6 | ByT5 grid and selection (GPU) | `encoder_experiment.py`, `select_config.py`, `encoder_summary.py`, `stats.py`, `scripts/encoder_grid.sh` | `results/encoder/`, `report/generated/tab_encoder.tex` and `numbers_encoder.tex` |
| 7 | CharsiuG2P error analysis | `charsiu_g2p.py` | `results/charsiu/` |
| 8 | Error typing | `error_typing.py` | `results/error_typing/`, `report/generated/tab_errors.tex` |
| 9 | Frozen check, paper numbers and report | `check_expected.py`, `paper_numbers.py`, `build_report.py` | `report/generated/`, `REPORT.pdf` |

`try_text.py` adds macrons to your own sentences, `export_hf.py` writes the Hugging Face copy of the data, and
`pilot.py` runs the GPU pilot from the Kaggle and Colab notebooks in `notebooks/`.

---

## ▶️ Quickstart

All three steps run on a laptop CPU.

```bash
# Step 1. Clone and install. Details in REPRODUCE.md.
git clone https://github.com/Ace3Z/coverage-context-convention.git
cd coverage-context-convention
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Step 2. Reproduce every number and table, and REPORT.pdf if pdflatex is installed
#         (about two minutes). Includes the frozen check of 660 values.
bash run_all.sh

# Step 3. Restore macrons in your own sentences: write them one per line to my_tests/sentences.txt.
PYTHONPATH=src python -m oevl.try_text my_tests/sentences.txt
```

`requirements.txt` includes torch and transformers, so the unit tests and the `--tiny` smoke runs work without a GPU.
Training ByT5 needs one. The exact commands are in `scripts/encoder_grid.sh`, and [REPRODUCE.md](REPRODUCE.md)
explains them.

---

## 💾 Dataset

The Hugging Face dataset [`Ace3Z/old-english-vowel-length`](https://huggingface.co/datasets/Ace3Z/old-english-vowel-length)
holds the corpus, the dictionaries and the ambiguity class files. Loading it needs the `datasets` package
(`pip install datasets`), which `requirements.txt` does not include.

| Split | Content | Role |
|---|---|---|
| `train` | Bright's *Reader*, Matthew, Sweet's *Primer*, *Physiologus*, Zupitza's *Grammar* | training (the final systems leave Zupitza out, decision D3) |
| `dev` | Gospel of John (West Saxon) | model selection only |
| `test_prose` | Alfred's *Soliloquies* | prose test |
| `test_verse` | Tauber's token-level *Beowulf* | verse test |
| `agreement` | Harrison & Sharp and Wyatt *Beowulf* | evaluation only: editor agreement and the prior's score against each edition |

```python
from datasets import load_dataset
corpus = load_dataset("Ace3Z/old-english-vowel-length")
```

The split is by work, and no *Beowulf* text is in `train`. [DATA.md](DATA.md) covers the sources,
the leakage audit, decision D3 and the licences.

---

## 📚 Documentation

| File | What it covers |
|---|---|
| [REPRODUCE.md](REPRODUCE.md) | installation, the systems and their evaluation, and every command |
| [DATA.md](DATA.md) | corpus, splits, dev set, leakage audit, decision D3, data layout and licences |
| [notebooks/README.md](notebooks/README.md) | Kaggle and Colab launchers for the ByT5 pilot |
| Report (PDF) | TODO: link the PDF once it is uploaded. The LaTeX source is [report/main.tex](report/main.tex), and `bash run_all.sh` compiles it to `REPORT.pdf` when `pdflatex` is installed. |
| [LICENSE](LICENSE), [data/LICENSE.md](data/LICENSE.md) | code licence (MIT) and the licence of every data file |

---

## 🗂️ Repository layout

```
coverage-context-convention/
├── README.md                overview and quickstart
├── LICENSE                  MIT licence of the code
├── pyproject.toml           package metadata (pip install -e .)
├── REPRODUCE.md             installation, systems, evaluation, every command
├── DATA.md                  corpus, splits, leakage audit, D3, licences
├── run_all.sh               the whole CPU pipeline, from inputs to REPORT.pdf
├── src/oevl/                the package: one or more modules per pipeline stage
├── scripts/encoder_grid.sh  exact GPU commands behind results/encoder/
├── tests/                   unit tests, run by run_all.sh
├── data/                    corpus (by_source/), Tauber and WikiPron inputs, dictionaries, class files
├── results/                 one folder per experiment: metrics, predictions, logs, labels
├── report/                  main.tex, the UZH logo and generated/ tables, numbers and figures
├── notebooks/               Kaggle and Colab launchers for the ByT5 pilot
├── release/                 dataset card and licence file for the Hugging Face copy
├── media/figures/logos/     logos
└── requirements.txt         every direct dependency, pinned (torch has a lower bound)
```

---

## 🖥️ Compute

- `bash run_all.sh` takes about two minutes on an Apple M-series laptop, including 89 unit tests (one is skipped unless
  the Wikisource download is cached), 660 frozen checks and the report.
- `requirements.txt` installs and passes those checks on macOS with Python 3.11 and 3.14, and on Linux with Python 3.14
  and a CUDA GPU.
- A ByT5-small dev selection run took 12 to 70 minutes on a shared lab GPU (`results/encoder/selection.json` records
  each run's time).
- CharsiuG2P's own 500-word benchmark takes a few minutes on a recent laptop CPU. Its predictions on our test sets were
  made on a GPU.

---

## 📜 Licence

The code is released under the MIT licence in [LICENSE](LICENSE). The data is CC BY-SA 4.0 unless
[data/LICENSE.md](data/LICENSE.md) says otherwise, and each source keeps its own terms. The files in `results/` quote
corpus text, so they are CC BY-SA 4.0 as well.

---

## 🙏 Acknowledgements

- [James Tauber](https://github.com/old-english-learner-texts/old-english-texts) for the token-level *Beowulf* with
  length marks, built on Brunetti's analysis and the Perseus text of Klaeber's edition
- [WikiPron](https://github.com/CUNY-CL/wikipron) and the Wiktionary contributors, also through
  [kaikki.org](https://kaikki.org/dictionary/Old%20English/)
- [CharsiuG2P](https://github.com/lingjzhu/CharsiuG2P) and [ByT5](https://huggingface.co/google/byt5-small)
- Bosworth-Toller on [LINDAT](http://hdl.handle.net/11234/1-3532) (Tichý et al.) and the
  [Clark Hall TEI conversion](https://github.com/Serg078/Conversion-of-The-Clark-Hall-Dictionary-of-Old-English-to-TEI)
- The Project Gutenberg and Wikisource volunteers who transcribed the editions in the corpus
- The *Essentials in Text and Speech Processing* course staff at the University of Zurich

---

<div align="center">

**[Back to top](#coverage-context-convention-restoring-vowel-length-in-old-english)**

</div>
