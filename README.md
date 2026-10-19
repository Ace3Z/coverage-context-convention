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
[Headline Results](#-headline-results) •
[Pipeline](#-pipeline) •
[Quickstart](#-quickstart) •
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
    <td align="center"><b>Frequency prior</b>, <b>word model</b></td>
    <td align="center"><i>Ic þe lǣre þēah þæt þū hyt ongynne</i></td>
  </tr>
  <tr>
    <td align="center"><b>Context model</b> (two words each side)</td>
    <td align="center"><i>Ic <b>þē</b> lǣre þēah þæt þū hyt ongynne</i></td>
  </tr>
</table>

*A held-out sentence from Alfred's* Soliloquies *(`soliloquies:00530`), restored with `oevl.try_text`. Only the
model that reads the neighbouring words picks the pronoun* þē *'thee' over the relative particle* þe*, as the edition
does.*

</div>

---

## 📖 Overview

This is our project for *Essentials in Text and Speech Processing* at the University of Zurich (HS 2026). Old English
spelling hides vowel length: *god* 'God' and *gōd* 'good', *man* 'man' and *mān* 'crime', the relative particle *þe*
and the pronoun *þē* 'thee'. Editors add the marks by hand. Given a bare token, a system chooses among the length
patterns a dictionary proposes for it.

We build a length-marked corpus of nine editions, hold out prose (Alfred's *Soliloquies*) and verse (Tauber's
token-level *Beowulf*) by work, and compare a frequency prior, logistic classifiers, a fine-tuned ByT5 candidate scorer
and CharsiuG2P on exactly the same tokens. Three editions of *Beowulf* give the editor-agreement reference. One
command, `bash run_all.sh`, reproduces every number, table and the figure in the report.

| Limit | Question | Where it is measured |
|---|---|---|
| **Coverage** | Is the right spelling among the candidates at all? | dictionary ablation, `results/<setting>/` |
| **Context** | Do neighbouring words pick the right candidate? | ByT5 arms and logistic models, `results/encoder/`, `results/wikipron/` |
| **Convention** | How much of the remaining error is editorial habit? | ambiguity classes A/B/C, error typing, `results/error_typing/` |
| **Disagreement** | How often do two editors mark the same word alike? | three-way *Beowulf* agreement and the gate, `results/agreement/` |

---

## 📊 Headline Results

All numbers come from the committed results and are the ones `bash run_all.sh` rebuilds into the report. Intervals are
sentence-level bootstrap 95% intervals over seed-averaged predictions.

| Finding | Population | Result |
|---|---|---|
| Coverage moves the most on verse | *Beowulf*, all tokens, frequency prior | **82.40%** with training forms, **92.38%** with all four dictionaries |
| Context resolves grammatical homographs | *Soliloquies*, class B (*þe/þē*, *ne/nē*) | **62.66%** word only, **76.39%** with two words each side (+13.73, 11.29 to 16.03) |
| Context helps prose overall | *Soliloquies*, lexicon-ambiguous | **88.91%** word only, **91.30%** with two words each side (+2.39, 1.91 to 2.87) |
| On verse the word alone is enough | *Beowulf*, lexicon-ambiguous | **90.04%** word only, **89.43%** full sentence (−0.61, −1.06 to −0.15) |
| Editors differ most on grammar words | Wyatt against Klaeber (Tauber), ambiguous / class B | **96.08%** / **78.07%** agreement |
| CharsiuG2P's errors are mostly length | its own 500-word Old English test | WER **0.208** reproduced, **0.188** with 50 decoding steps, where **75.5%** of word errors are length only |

The ambiguity classes and the error types were labelled by two independent LLM annotators and are still pending
review by the team. Full tables, caveats and the run-to-run variance are in the report (**TODO:** link the PDF).

---

## 🧩 Pipeline

The code is the package `oevl` in `src/oevl/`. `bash run_all.sh` runs the CPU stages from the committed inputs
(corpus files, dictionaries, CharsiuG2P and ByT5 predictions) and rebuilds every table from them. Rebuilding the corpus
or the dictionaries from their sources and the GPU runs have their own commands in [REPRODUCE.md](REPRODUCE.md).

| # | Stage | Module | Writes |
|---|---|---|---|
| 1 | Corpus build and clean-up | `build_corpus.py`, `clean_corpus.py`, `fetch_tauber.py` | `data/by_source/`, `data/manifest.json` |
| 2 | Leakage audit | `audit_splits.py` | `data/leakage_excluded.json` |
| 3 | Dictionaries | `wikipron_to_marked.py`, `build_lexicons.py` | `data/lex_*.csv` |
| 4 | Frequency prior and logistic models | `experiment.py` | `results/<setting>/` |
| 5 | Editor agreement and the gate | `agreement.py` | `results/agreement/` |
| 6 | ByT5 grid and selection (GPU) | `encoder_experiment.py`, `select_config.py`, `encoder_summary.py`, `stats.py`, `scripts/encoder_grid.sh` | `results/encoder/` |
| 7 | CharsiuG2P error analysis | `charsiu_g2p.py` | `results/charsiu/` |
| 8 | Error typing | `error_typing.py` | `results/error_typing/` |
| 9 | Frozen check and report | `check_expected.py`, `build_report.py` | `report/generated/`, `REPORT.pdf` |

Two further modules: `try_text.py` adds macrons to your own sentences, and `export_hf.py` writes the Hugging Face
copy of the data. `pilot.py` is the GPU pilot runner behind the Kaggle and Colab notebooks in `notebooks/`.

---

## ▶️ Quickstart

All three steps run on a laptop CPU.

```bash
# Step 1. Clone and install. Details in REPRODUCE.md.
git clone https://github.com/Ace3Z/cl-uzh-26-essentials-code.git
cd cl-uzh-26-essentials-code
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Step 2. Reproduce every number, table and REPORT.pdf (a few minutes).
#         Ends with the frozen check of 660 headline values.
bash run_all.sh

# Step 3. Restore macrons in your own sentences, one per line.
PYTHONPATH=src python -m oevl.try_text my_sentences.txt
```

`requirements.txt` also installs torch and transformers, so the unit tests and the `--tiny` smoke runs work on a CPU.
The ByT5 training runs need a GPU. Their exact commands are in
`scripts/encoder_grid.sh`, and [REPRODUCE.md](REPRODUCE.md) explains each one.

---

## 💾 Dataset

The corpus, the dictionaries and the ambiguity class file are on the Hugging Face Hub as
[`Ace3Z/old-english-vowel-length`](https://huggingface.co/datasets/Ace3Z/old-english-vowel-length), private until it
is released together with this repository.

| Split | Content | Role |
|---|---|---|
| `train` | Bright's *Reader*, Matthew, Sweet's *Primer*, *Physiologus*, Zupitza's *Grammar* | training (the final systems leave Zupitza out, decision D3) |
| `dev` | Gospel of John (West Saxon) | model selection only |
| `test_prose` | Alfred's *Soliloquies* | prose test |
| `test_verse` | Tauber's token-level *Beowulf* | verse test |
| `agreement` | Harrison & Sharp and Wyatt *Beowulf* | editor agreement only |

```python
from datasets import load_dataset
corpus = load_dataset("Ace3Z/old-english-vowel-length")  # needs a Hugging Face token while the repo is private
```

The split is by work, never by sentence, and no *Beowulf* text is in `train`. [DATA.md](DATA.md) gives the sources,
the leakage audit, decision D3 and every licence.

---

## 📚 Documentation

The README is the map. The details live in focused files:

| File | What it covers |
|---|---|
| [REPRODUCE.md](REPRODUCE.md) | Installation, the systems and the evaluation protocol, and every command: the full run, single experiments, corpus and dictionary rebuilds, the ByT5 GPU grid, encoder tables without a GPU, CharsiuG2P, smoke tests, `try_text`, the dataset export and upload, and the layout of `results/`. |
| [DATA.md](DATA.md) | The corpus and its splits, the dev set, the leakage audit, decision D3, the layout of `data/`, and the licence of every source. |
| [notebooks/README.md](notebooks/README.md) | The Kaggle and Colab launchers for the ByT5 pilot. |
| Report (PDF) | **TODO:** link the PDF once it is uploaded. Its LaTeX source is [report/main.tex](report/main.tex), and `bash run_all.sh` compiles it to `REPORT.pdf`. |
| [LICENSE](LICENSE), [data/LICENSE.md](data/LICENSE.md) | Code licence (MIT) and the licence of every data file. |

---

## 🗂️ Repository Layout

```
cl-uzh-26-essentials-code/
├── README.md                this file (the map)
├── REPRODUCE.md             installation, systems, evaluation, every command
├── DATA.md                  corpus, splits, leakage audit, D3, licences
├── run_all.sh               the whole CPU pipeline, from inputs to REPORT.pdf
├── src/oevl/                the package: one module per pipeline stage
├── scripts/encoder_grid.sh  exact GPU commands behind results/encoder/
├── tests/                   unit tests, run by run_all.sh
├── data/                    corpus (by_source/), Tauber and WikiPron inputs, dictionaries, class file
├── results/                 one folder per experiment: metrics, predictions, logs
├── report/                  main.tex and generated/ tables, numbers and figure
├── notebooks/               Kaggle and Colab launchers for the ByT5 pilot
├── release/hf_card.md       template of the Hugging Face dataset card
├── media/figures/logos/     logos
└── requirements.txt         every dependency, pinned (torch has a lower bound)
```

---

## 🖥️ Compute

- **CPU pipeline:** `bash run_all.sh` takes about a minute and a half on a laptop (Apple M-series), including the 75
  unit tests, the 660 frozen checks and the report.
- **GPU runs:** ByT5-small, one run per arm and seed. On an NVIDIA RTX PRO 6000 shared with other jobs, a dev
  selection run took 12 to 70 minutes and a grid run 13 to 152 minutes (`scripts/encoder_grid.sh`).
- **CharsiuG2P:** its own 500-word benchmark runs in about 15 minutes on a laptop CPU, and prediction on our test sets
  was made on a GPU.

---

## 📜 Licence

The code is released under the MIT licence in [LICENSE](LICENSE). The data is CC BY-SA 4.0 unless
[data/LICENSE.md](data/LICENSE.md) says otherwise, and each source keeps its own terms.

---

## 🙏 Acknowledgments

- [James Tauber](https://github.com/old-english-learner-texts/old-english-texts): the token-level *Beowulf* with
  length marks, building on Brunetti's analysis and the Perseus text of Klaeber's edition
- [WikiPron](https://github.com/CUNY-CL/wikipron) and the Wiktionary contributors, also through
  [kaikki.org](https://kaikki.org/dictionary/Old%20English/)
- [CharsiuG2P](https://github.com/lingjzhu/CharsiuG2P) and [ByT5](https://huggingface.co/google/byt5-small)
- Bosworth–Toller on [LINDAT](http://hdl.handle.net/11234/1-3532) (Tichý et al.) and the
  [Clark Hall TEI conversion](https://github.com/Serg078/Conversion-of-The-Clark-Hall-Dictionary-of-Old-English-to-TEI)
- The transcribers of Project Gutenberg and Wikisource, whose proofread editions make up the corpus
- The *Essentials in Text and Speech Processing* course staff at the University of Zurich

---

<div align="center">

**[Back to top](#coverage-context-convention-restoring-vowel-length-in-old-english)**

</div>
