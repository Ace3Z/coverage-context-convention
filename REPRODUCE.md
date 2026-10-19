# Reproducing the results

Every command, what each run does, and where its output goes. The data, the splits and the licences are in
[DATA.md](DATA.md). Back to the [README](README.md).

## Installation

### Environment setup

Python 3.11 to 3.14 (check with `python3 --version`). The pinned packages have no 3.15 builds yet.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`run_all.sh` puts `src` on `PYTHONPATH`, so nothing has to be installed. To run modules by hand, either
`pip install -e .` once (editable only: the code finds `data/` and `results/` relative to the repository) or prefix
commands with `PYTHONPATH=src`.

### Dependencies

Key dependencies include:
- scikit-learn 1.8.0, scipy 1.16.3, numpy 2.3.5, matplotlib 3.10.8 for the CPU pipeline
- torch 2.2 or later and transformers 4.57.6 for ByT5 and CharsiuG2P (training needs a GPU). The committed ByT5
  runs used torch 2.11.0 with CUDA 12.8

All of them are in the one `requirements.txt`.
- `pdflatex` to compile `REPORT.pdf` (optional)
- `huggingface_hub` for the dataset upload (its `hf` command line)

## Systems

- **Frequency prior:** the most frequent pattern in training, and for a word not seen in training the pattern
  attested by the most dictionaries, then the one with the fewest long vowels.
- **Logistic models:** candidate-constrained logistic classifiers (word / +2 left words / +2 left and 2 right words),
  seeds 1–3.
- **Oracle:** with the same all-short fallback for unknown words.
- **ByT5 encoder:** ByT5-small fine-tuned on the same examples with the word only, the left context or the whole
  sentence (and two or five words either side in the grid). The dictionary proposes the candidate patterns and the
  model scores each one.
- **CharsiuG2P:** the published multilingual checkpoint, used for an error analysis.

## Evaluation

- **Populations:** lexicon-ambiguous, attested-ambiguous, minority-reading. The lexicon-ambiguous tokens are also
  split by ambiguity class (`class_A`, `class_B`, `class_C` in `metrics.json`) for the 60 skeletons in
  `data/ambiguity_classes.csv`, and `class_unlisted` for the rest. The class labels come from two automated
  annotators and an automated adjudicator and still need review by the team (`data/ambiguity_classes.json`). The 60
  were chosen by frequency over the corpus with Zupitza and are kept under D3. The `d3` block of that file records
  that `hi` is no longer lexicon-ambiguous with the D3 training forms + WikiPron, and that 6,759 of the 36,203
  lexicon-ambiguous tokens counted there fall outside the list.
- **Statistics:** mean ± sample SD over seeds, and exact McNemar on seed 1 (token level).
- **Dictionaries:** the report's tables use training forms only (`results/training_only`) and training forms +
  WikiPron (`results/wikipron`). `run_all.sh` also runs the three proposal dictionaries (`results/proposal`) and all
  four together (`results/all`), and `check_expected` checks all four settings and the Zupitza ablation.

`check_expected` passes all 660 checks on Python 3.14 with the pinned versions in `requirements.txt`.

## 1. Reproduce everything

```bash
bash run_all.sh
```

It takes a few minutes and does six steps:

1. **Inputs.** Uses `data/brunetti-length.txt` (Tauber's *Beowulf* gold) and `data/ang_latn_broad.tsv` (WikiPron Old
   English) if present, otherwise downloads them. Prints their SHA-256. `oevl.audit_splits` then prints the word
   6-gram overlap of each held-out set with each training work and writes `data/leakage_excluded.json`.
2. **Unit tests** (`tests/`).
3. **Dictionaries:** `oevl.wikipron_to_marked` writes `data/lex_wikipron.csv`. The three proposal dictionaries ship in
   `data/`, so `oevl.build_lexicons --unions-only` only rebuilds the unions. Without them it downloads and converts
   the original sources.
4. **Experiments:** four dictionary settings (training only, + WikiPron, + the three proposal dictionaries, + all four)
   × the John dev set and both test sets, one set per run, written to `results/<setting>/`. All four train without
   Zupitza (`--exclude-train zupitza_grammar`, see D3 in [DATA.md](DATA.md)). One ablation keeps all five training
   works: `results/wikipron_with_zupitza/` is the WikiPron setting with Zupitza, on the same three sets. Then
   `oevl.charsiu_g2p analyse` runs on the committed CharsiuG2P output and `oevl.agreement` writes
   `results/agreement/`.
5. **`oevl.check_expected`:** fails if any of 660 headline values differs from the frozen results (four settings and
   the Zupitza ablation × three sets, and the editor agreement of each *Beowulf* pair).
6. **`oevl.build_report`:** writes `report/generated/` (numbers, tables, figure), checks the worked examples against
   the predictions, and compiles `REPORT.pdf` if `pdflatex` is installed. Without LaTeX, upload `report/` to Overleaf
   and compile `main.tex`. Before it, `oevl.error_typing`, `oevl.encoder_summary --from-tsv` and `oevl.select_config`
   rebuild the error table, the encoder table and numbers, and the encoder selection record from committed files (no
   GPU, see section 5 below).

## 2. Run a single experiment

Results for the same `--out` folder are merged:

```bash
PYTHONPATH=src python -m oevl.experiment --data data --out results/wikipron --lexicon data/lex_wikipron.csv --tests soliloquies
PYTHONPATH=src python -m oevl.experiment --data data --out results/wikipron --lexicon data/lex_wikipron.csv --tests tauber_beowulf
```

## 3. Rebuild the corpus and the dictionaries

```bash
PYTHONPATH=src python -m oevl.build_corpus --source <name> --out data/by_source
PYTHONPATH=src python -m oevl.clean_corpus
PYTHONPATH=src python -m oevl.build_lexicons
```

`oevl.build_corpus` fetches an edition into `data/raw/corpus/` and rebuilds its file and manifest entry (URL,
retrieval date and SHA-256 of each input, and the work's `role`: train, dev, test or agreement). The original
extraction scripts were not kept, so the rules were inferred from the files: `matthew_ws`, `bright_reader`,
`physiologus` and `hs_beowulf` rebuild byte for byte once `clean_corpus` has run on the output. The other four
rebuilds contain 65,988 of their 65,992 committed tokens in order but differ in some sentences, mostly where the
original language filter dropped short lines (the module docstring says more). They were not overwritten.
`john_ws.jsonl` (Gospel of John, Bright 1904, the dev set) is built by the same procedure as Matthew. The builder
still writes the old sentence-level `split` field, which `clean_corpus` removes. The split is by work, as the
manifest `role` says. `oevl.build_lexicons` rebuilds the three proposal dictionaries from the original sources.

## 4. ByT5 encoder (GPU)

Training needs a GPU. `requirements.txt` already holds torch and the pinned `transformers==4.57.6`.

```bash
PYTHONPATH=src python -m oevl.encoder_experiment
bash scripts/encoder_grid.sh [select|grid|all] [RUNS_DIR]
```

- `oevl.encoder_experiment` fine-tunes ByT5-small on the same examples with the word only, the left context or the
  whole sentence. Like `run_all.sh` it trains without Zupitza by default (D3), and `--exclude-train ""` trains on all
  five works. Other context arms: `--arms local2,local5` (two or five words either side). A fixed training subset:
  `--train-limit N`. It scores the John dev set and both test sets (`--tests` picks fewer).
- `scripts/encoder_grid.sh` holds the exact commands behind `results/encoder/`: 24 dev selection runs, whose rule
  `oevl.select_config` applies (`selection.json`), then 39 runs of the chosen setting, which
  `oevl.encoder_summary --runs` pools into `summary.json`, per-token prediction tables (`*_predictions.tsv.gz`),
  `report/generated/tab_encoder.tex` and `numbers_encoder.tex`. The run folders are not committed.
- **Encoder pilot:** the five context arms on 10,000 training examples, run from `notebooks/` on Kaggle or Colab.
  See `notebooks/README.md`. The logic is `oevl.pilot`. The pilots ran before D3, so `oevl.pilot` still trains on all
  five works.

GPU training is not bit-for-bit repeatable, so a new run can differ slightly.

## 5. Encoder tables from the committed predictions (no GPU)

```bash
PYTHONPATH=src python -m oevl.encoder_summary --from-tsv
PYTHONPATH=src python -m oevl.select_config
```

`run_all.sh` runs both. They rebuild the encoder table and numbers and `results/encoder/selection.json` from the
committed tables and run records.

## 6. CharsiuG2P error analysis

```bash
PYTHONPATH=src python -m oevl.charsiu_g2p benchmark
PYTHONPATH=src python -m oevl.charsiu_g2p predict
PYTHONPATH=src python -m oevl.charsiu_g2p analyse
```

`benchmark` runs the published checkpoint on the repository's own 500 Old English test words
(`data/charsiu_ang_test.tsv`) and writes `results/charsiu/benchmark.json` and a per-word `benchmark.tsv` (committed,
about 15 minutes on a laptop CPU). `predict` writes `results/charsiu/ipa.tsv` for our test sets (committed, made on a
GPU). `run_all.sh` runs the `analyse` step on it.

## 7. CPU smoke tests without downloads

```bash
PYTHONPATH=src python -m oevl.encoder_experiment --tiny --out /tmp/t
PYTHONPATH=src python -m oevl.pilot --tiny --allow-cpu --skip-model-check --max-bytes 128 --out /tmp/pilot
PYTHONPATH=src python -m oevl.charsiu_g2p predict --tiny
```

The second runs the whole pilot loop. The third writes `results/charsiu/ipa_tiny.tsv`.

## 8. Try it on your own sentences

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

## 9. Export and upload the Hugging Face dataset

```bash
PYTHONPATH=src python -m oevl.export_hf --out release/hf
hf upload Ace3Z/old-english-vowel-length release/hf . --repo-type=dataset --private
```

The export writes `release/hf/` (gitignored, regenerated on each run): one JSONL file per split (`train`, `dev`,
`test_prose` = *Soliloquies*, `test_verse` = Tauber's *Beowulf*, `agreement` = the Harrison & Sharp and Wyatt
*Beowulf*), the dictionaries (`lexicons/`), the ambiguity class files, `data/LICENSE.md` as `LICENSE.md`, the logos
(`media/figures/logos/`), and the dataset card `README.md`, made from `release/hf_card.md` with the token counts read
from the written files. Sentences that the leakage audit leaves out of evaluation are kept, with `leakage_excluded`
set to true. `train` holds all five training works, Zupitza included.

The upload needs a Hugging Face account with write access to the `Ace3Z` namespace. `--private` keeps a
newly created repository private, and the repository stays private until it is released together with this one.
`hf` is the command line of `huggingface_hub`. Version 1.0 removed the older `huggingface-cli`. With a 0.x version
the same upload is
`huggingface-cli upload Ace3Z/old-english-vowel-length release/hf . --repo-type=dataset --private`.

## Output data

```
results/
├── training_only/              one folder per dictionary setting
│   ├── metrics.json
│   ├── <set>_per_word.csv      per-word results, <set> = john_ws, soliloquies, tauber_beowulf
│   ├── <set>_predictions.jsonl token-level predictions
│   └── results_training_only_<set>.log
├── wikipron/
├── proposal/
├── all/
├── wikipron_with_zupitza/      the Zupitza ablation
├── agreement/                  metrics.json, disagreements.tsv, disagreements_checked.tsv, agreement.log
├── encoder/                    selection.json, summary.json, <set>_predictions.tsv.gz
├── charsiu/                    benchmark.json, benchmark.tsv, ipa.tsv, metrics.json
├── error_typing/               labels.tsv, summary.json
└── run_all.log                 log of an earlier full run_all.sh
report/
├── main.tex
└── generated/                  tables, numbers and figure written by oevl.build_report
release/
└── hf_card.md                  template of the Hugging Face dataset card
```
