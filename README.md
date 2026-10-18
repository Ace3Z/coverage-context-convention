# What Limits Vowel-Length Restoration in Old English? — code

Course project, *Essentials in Text and Speech Processing*, UZH, HS 2026.
Mahbod Tajdini, Hanna Miranovich, Arjun Singh Bhadoria.

One command reproduces every number, table and the figure in the report (`REPORT.pdf`, source in `report/`).

## Setup

Python 3.11 to 3.14 (check with `python3 --version`); the pinned packages have no 3.15 builds yet.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
bash run_all.sh
```

The code is the package `oevl` in `src/oevl/`. `run_all.sh` puts `src` on `PYTHONPATH`, so nothing has to be
installed. To run modules by hand, either `pip install -e .` once (editable only: the code finds `data/` and `results/`
relative to the repository) or prefix commands with `PYTHONPATH=src`.

`run_all.sh` takes a few minutes and does six steps:

1. **Inputs.** Uses `data/brunetti-length.txt` (Tauber's *Beowulf* gold) and `data/ang_latn_broad.tsv` (WikiPron Old
   English) if present, otherwise downloads them. Prints their SHA-256. `oevl.audit_splits` then prints the word
   6-gram overlap of each held-out set with each training work and writes `data/leakage_excluded.json`.
2. **Unit tests** (`tests/`).
3. **Dictionaries:** WikiPron IPA length marks to spellings (`data/lex_wikipron.csv`), plus the three dictionaries named
   in the proposal: Clark Hall, Wiktionary inflection tables and Bosworth–Toller (`data/lex_clarkhall.csv`,
   `lex_wiktionary.csv`, `lex_bt.csv`; rebuilt from the original sources with `python -m oevl.build_lexicons`).
4. **Experiments:** four dictionary settings (training only, + WikiPron, + the three proposal dictionaries, + all four)
   × both test sets, and the John dev set once `data/by_source/john_ws.jsonl` exists, one set per run, written to
   `results/<setting>/`. Then `oevl.agreement` compares the three *Beowulf* editions (Harrison & Sharp, Wyatt, the
   Tauber gold) pairwise and applies the gate of plan rule D2, written to `results/agreement/`.
5. **`oevl.check_expected`:** fails if any of 336 headline values differs from the frozen results.
6. **`oevl.build_report`:** writes `report/generated/` (numbers, tables, figure), checks the worked examples against
   the predictions, and compiles `REPORT.pdf` if `pdflatex` is installed. Without LaTeX, upload `report/` to Overleaf
   and compile `main.tex`.

A single experiment can also be run on its own. Results for the same `--out` folder are merged:

```bash
PYTHONPATH=src python -m oevl.experiment --data data --out results/wikipron --lexicon data/lex_wikipron.csv --tests soliloquies
PYTHONPATH=src python -m oevl.experiment --data data --out results/wikipron --lexicon data/lex_wikipron.csv --tests tauber_beowulf
```

## Neural experiments (GPU)

They need `pip install -r requirements-neural.txt` (pins `transformers==4.57.6`) and a GPU.

- **Encoder, full data:** ByT5-small fine-tuned on the same examples with the word only, the left context or the whole
  sentence. `PYTHONPATH=src python -m oevl.encoder_experiment` writes `results/encoder/`. No full-data encoder output
  is committed yet. Other context arms: `--arms local2,local5` (two or five words either side); a fixed training
  subset: `--train-limit N`. Add the dev set with `--tests john_ws,soliloquies,tauber_beowulf` (it needs
  `results/wikipron/john_ws_predictions.jsonl` from `run_all.sh`).
- **Encoder pilot:** the five context arms on 10,000 training examples, run from `notebooks/` on Kaggle or Colab.
  See `notebooks/README.md`. The logic is `oevl.pilot`.
- **CharsiuG2P error analysis:** `PYTHONPATH=src python -m oevl.charsiu_g2p benchmark` runs the published checkpoint
  on the repository's own 500 Old English test words (`data/charsiu_ang_test.tsv`) and writes
  `results/charsiu/benchmark.json` and a per-word `benchmark.tsv` (committed, about 15 minutes on a laptop CPU).
  `PYTHONPATH=src python -m oevl.charsiu_g2p predict` writes `results/charsiu/ipa.tsv` for our test sets (committed,
  made on a GPU); `run_all.sh` runs the `analyse` step on it.

GPU training is not bit-for-bit repeatable, so a new run can differ slightly. CPU smoke tests without downloads:
`PYTHONPATH=src python -m oevl.encoder_experiment --tiny --out /tmp/t`,
`PYTHONPATH=src python -m oevl.pilot --tiny --allow-cpu --skip-model-check --max-bytes 128 --out /tmp/pilot` for the whole
pilot loop, and `PYTHONPATH=src python -m oevl.charsiu_g2p predict --tiny`, which writes `results/charsiu/ipa_tiny.tsv`.

## Try it on your own sentences

After `bash run_all.sh` has run once:

```bash
PYTHONPATH=src python -m oevl.try_text my_tests/sentences.txt
PYTHONPATH=src python -m oevl.try_text my_tests/sentences.txt --gold my_tests/sentences_marked.txt --tsv my_tests/words.tsv
```

- `sentences.txt`: one sentence per line. Any macrons in it are removed first.
- `--gold` (optional): the same lines with correct macrons. Then accuracy is printed.
- `--tsv` (optional): one row per word: output, known word or not, changed or not, correct or not.
- `--model frequency|word|left|bilateral` (default `bilateral`), `--lexicon none` for training texts only.

It trains the same system as the report (seed 1) in under a minute and changes no existing file. Unknown words
are left unchanged. On text that is not Old English there is no correct answer, so count the words it wrongly
changed instead of reporting accuracy.

## Protocol

- **Train:** Bright's *Reader*, Matthew (West Saxon), Sweet's *Primer*, *Physiologus*, Zupitza's Ælfric
  *Grammar* (93,038 tokens). **Dev:** Gospel of John (`john_ws`), scored like the test sets.
  **Prose test:** Alfred's *Soliloquies* (15,431 of 15,480 tokens). **Verse test:** Tauber's
  *Beowulf* (17,361 tokens from 17,244 rows). **Editor agreement only:** Harrison & Sharp and Wyatt *Beowulf*.
  Splits are by work. No *Beowulf* text is used for training. Dev and test sentences that share a word 6-gram
  (bare, lower-cased, ð folded to þ) with a training work are left out of evaluation. For the *Soliloquies* that is
  one sentence of 49 tokens, for *Beowulf* none (`python -m oevl.audit_splits`).
- **Systems:** frequency prior (most frequent pattern in training, and for a word not seen in training the pattern
  attested by the most dictionaries, then the one with the fewest long vowels); candidate-constrained logistic
  classifiers (word / +2 left words / +2 left and 2 right words), seeds 1–3; oracle with the same all-short fallback
  for unknown words.
- **Populations:** lexicon-ambiguous, attested-ambiguous, minority-reading. The lexicon-ambiguous tokens are also
  split by ambiguity class (`class_A`, `class_B`, `class_C` in `metrics.json`) for the 60 skeletons in
  `data/ambiguity_classes.csv`, and `class_unlisted` for the rest. The class labels come from two automated
  annotators and an automated adjudicator and still need review by the team (`data/ambiguity_classes.json`).
- **Statistics:** mean ± sample SD over seeds; exact McNemar on seed 1 (token level).
- **Dictionaries:** the report's tables use training forms only (`results/training_only`) and training forms +
  WikiPron (`results/wikipron`). `run_all.sh` also runs the three proposal dictionaries (`results/proposal`) and all
  four together (`results/all`), and `check_expected` checks all four settings.

`check_expected` passes all 336 checks on Python 3.14 with the pinned versions in `requirements.txt`.

## Data and licences

- Corpus: `data/by_source/`, cleaned by `python -m oevl.clean_corpus` (explicit character mappings, counted in
  `data/manifest.json`). Built from Project Gutenberg (public domain) and Wikisource (CC BY-SA 4.0); released as
  CC BY-SA 4.0. A Hugging Face copy is planned at `oe-vowel-length/old-english-vowel-length` but is not published yet.
  `PYTHONPATH=src python -m oevl.build_corpus --source <name> --out data/by_source` fetches an edition into
  `data/raw/corpus/` and rebuilds its file and manifest entry (URL, retrieval date and SHA-256 of each input, and the
  work's `role`: train, dev, test or agreement). The original extraction scripts were not kept, so the rules were
  inferred from the files: `matthew_ws`, `bright_reader`, `physiologus` and `hs_beowulf` rebuild byte for byte once
  `clean_corpus` has run on the output. The other four rebuilds contain 65,988 of their 65,992 committed tokens in
  order but differ in some sentences, mostly where the original language filter dropped short lines (the module
  docstring says more); they were not overwritten. `john_ws.jsonl` (Gospel of John, Bright 1904, the dev set) is built
  by the same procedure as Matthew. The builder still writes the old sentence-level `split` field, which
  `clean_corpus` removes; the split is by work, as the manifest `role` says.
- Tauber gold: https://github.com/old-english-learner-texts/old-english-texts (CC BY-SA 4.0).
- WikiPron: https://github.com/CUNY-CL/wikipron (Apache-2.0).
- Clark Hall, *A Concise Anglo-Saxon Dictionary*, TEI conversion by Serg078 (Unlicense):
  https://github.com/Serg078/Conversion-of-The-Clark-Hall-Dictionary-of-Old-English-to-TEI
- Wiktionary Old English entries via kaikki.org (CC BY-SA 4.0): https://kaikki.org/dictionary/Old%20English/
- Bosworth–Toller data dump, Tichý et al., LINDAT (CC BY 4.0): http://hdl.handle.net/11234/1-3532
- CharsiuG2P (MIT): https://github.com/lingjzhu/CharsiuG2P; `data/charsiu_ang_dict_words.txt` lists the words of its
  Old English training dictionary, used only to split the error analysis into seen and unseen words.
  `data/charsiu_ang_test.tsv` is its `data/test/ang.tsv` at commit `0c929390759fb94f8ecdfc05cc0bc5f2ff2dc0f4`.

## Layout

```
src/oevl/            the package
  experiment.py        data, dictionaries, frequency/logistic models, metrics (the traditional system)
  build_corpus.py, wikipron_to_marked.py, build_lexicons.py, fetch_tauber.py    corpus, inputs and dictionaries
  clean_corpus.py, audit_splits.py                             corpus clean-up and the 6-gram leakage audit
  agreement.py                                                 editor agreement on Beowulf and the gate
  check_expected.py, build_report.py                           frozen-value check, report tables and figure
  encoder_experiment.py, charsiu_g2p.py                        neural experiments (GPU)
  pilot.py                                                     the GPU pilot runner used by the notebooks
  try_text.py                                                  macrons for your own sentences
notebooks/           thin Kaggle and Colab launchers for oevl.pilot
tests/               unit tests
data/                corpus (by_source/), manifest, Tauber and WikiPron inputs, dictionaries (lex_*.csv),
                     ambiguity classes (ambiguity_classes.csv and .json)
results/             one folder per experiment: metrics.json, per-word CSVs, token-level predictions, run logs
report/              main.tex (+ generated/ tables, numbers, figure)
run_all.sh           the whole traditional pipeline, from inputs to REPORT.pdf
```
