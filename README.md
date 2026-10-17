# What Limits Vowel-Length Restoration in Old English? — code

Course project, *Essentials in Text and Speech Processing*, UZH, HS 2026.
Mahbod Tajdini, Hanna Miranovich, Arjun Singh Bhadoria.

One command reproduces every number, table and the figure in the report (`REPORT.pdf`, source in `report/`).

## Setup

Python 3.11, 3.12, 3.13 or 3.14 (check with `python3 --version`).

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
bash run_all.sh
```

`run_all.sh` takes a few minutes and does six steps:

1. **Inputs.** Uses `data/brunetti-length.txt` (Tauber's *Beowulf* gold) and `data/ang_latn_broad.tsv` (WikiPron Old
   English) if present, otherwise downloads them. Prints their SHA-256.
2. **Unit tests** (12 tests, none skipped when the inputs are present).
3. **WikiPron conversion:** IPA length marks to spellings, giving `data/lex_wikipron.csv`.
4. **Experiments:** both dictionaries × both test sets, one test set per run.
5. **`src/check_expected.py`:** fails if any of 156 headline values differs from the frozen results.
6. **`src/build_report.py`:** writes `report/generated/` (numbers, tables, figure), checks the worked examples
   against the predictions, and compiles `REPORT.pdf` if `pdflatex` is installed. Without LaTeX, upload `report/`
   to Overleaf and compile `main.tex`.

A single experiment can also be run on its own. Results for the same `--out` folder are merged:

```bash
python src/experiment.py --data data --out results_wikipron --lexicon data/lex_wikipron.csv --tests soliloquies
python src/experiment.py --data data --out results_wikipron --lexicon data/lex_wikipron.csv --tests tauber_beowulf
```

## Try it on your own sentences

After `bash run_all.sh` has run once:

```bash
python src/try_text.py my_tests/sentences.txt
python src/try_text.py my_tests/sentences.txt --gold my_tests/sentences_marked.txt --tsv my_tests/words.tsv
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
  *Grammar* (93,038 tokens). **Prose test:** Alfred's *Soliloquies* (15,480). **Verse test:** Tauber's
  *Beowulf* (17,361 tokens from 17,244 rows). **Editor agreement only:** Harrison & Sharp and Wyatt *Beowulf*.
  Splits are by work. No *Beowulf* text is used for training.
- **Systems:** frequency prior; candidate-constrained logistic classifiers (word / +2 left words / +2 left and
  2 right words), seeds 1–3; oracle with the same all-short fallback for unknown words.
- **Populations:** lexicon-ambiguous, attested-ambiguous, minority-reading.
- **Statistics:** mean ± sample SD over seeds; exact McNemar on seed 1 (token level).
- **Dictionaries:** training forms only, or training forms + WikiPron. Clark Hall and Bosworth–Toller are not used.

Verified: all 156 checked values are identical on Python 3.11 and 3.14 with the pinned versions, and with the newest
versions tested (scikit-learn 1.9.1, scipy 1.18.1).

## Data and licences

- Corpus: `data/by_source/`, also at https://huggingface.co/datasets/oe-vowel-length/old-english-vowel-length.
  Built from Project Gutenberg (public domain) and Wikisource (CC BY-SA 4.0); released as CC BY-SA 4.0.
- Tauber gold: https://github.com/old-english-learner-texts/old-english-texts (CC BY-SA 4.0).
- WikiPron: https://github.com/CUNY-CL/wikipron (Apache-2.0).

## Layout

```
data/            corpus (by_source/), manifest, Tauber and WikiPron inputs
src/             experiment.py, wikipron_to_marked.py, fetch_tauber.py, check_expected.py, build_report.py,
                 try_text.py
tests/           unit tests
report/          main.tex (+ generated/ tables, numbers, figure)
results_*/       metrics.json, per-word CSVs, token-level predictions, run logs
```
