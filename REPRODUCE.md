# Reproducing the results

This file lists every command, what it does and where its output goes. The data, the splits and the licences are in
[DATA.md](DATA.md). Back to the [README](README.md).

## Installation

You need Python 3.11 to 3.14 (check with `python3 --version`). The pinned packages have no builds for 3.15 yet.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` holds every direct dependency. The CPU pipeline uses scikit-learn 1.8.0, scipy 1.16.3, numpy 2.3.5 and
matplotlib 3.10.8. ByT5 and CharsiuG2P use torch 2.2 or later and transformers 4.57.6. Training needs a GPU, and the
committed ByT5 runs used torch 2.11.0 with CUDA 12.8. `pdflatex` is optional and compiles `REPORT.pdf`. The `hf`
command for uploading the dataset comes with `huggingface_hub`, which transformers already installs.

`run_all.sh` puts `src` on `PYTHONPATH`, so nothing has to be installed. To run modules by hand, prefix commands
with `PYTHONPATH=src`, or install the package once with `pip install -e .`. The install must be editable, because
the code finds `data/` and `results/` relative to the repository.

## Systems

- Frequency prior: the most frequent pattern in training. For a word not seen in training it takes the pattern that
  most dictionaries list, then the one with the fewest long vowels.
- Logistic models: logistic classifiers that choose among the dictionary's candidates. They see the word, the word
  plus two words on the left, or the word plus two words on each side, and run with seeds 1, 2 and 3.
- Oracle: picks the gold pattern whenever it is a candidate. Like every system, it writes a word with no candidates
  as all short.
- ByT5: ByT5-small fine-tuned on the same examples. One arm sees the word only, one the left context, one the whole
  sentence, and two arms see two or five words on each side. The dictionary proposes the candidate patterns and the
  model scores each one.
- CharsiuG2P: the published multilingual checkpoint, used for an error analysis.

## Evaluation

Accuracy is reported on all tokens and on three harder populations: lexicon-ambiguous tokens (two or more candidate
patterns), attested-ambiguous tokens (at least two patterns, each seen at least twice in training), and minority
readings (the gold pattern is seen in training but is not the most frequent one there). The lexicon-ambiguous tokens are also split by ambiguity
class (`class_A`, `class_B`, `class_C` in `metrics.json`) for the 60 skeletons (spellings without macrons) in `data/ambiguity_classes.csv`, with
`class_unlisted` for the rest. Two LLM annotators and an automated adjudicator assigned the classes, a second, blind LLM review agreed with 58 of them and an adjudicator kept the other two (`results/label_review/`),
and the team has not reviewed them yet. The 60 skeletons were picked by their frequency in the training works (Zupitza included) and both test
sets, and they are kept under decision D3. With training forms plus WikiPron, the `d3` block of `data/ambiguity_classes.json` records that `hi` is no longer lexicon-ambiguous without
Zupitza, and that 6,759 of the 36,203 lexicon-ambiguous tokens counted there fall outside the list.

Results are the mean and sample standard deviation over seeds. A difference between two systems gets a 95%
sentence-level bootstrap interval over seed-averaged predictions (`oevl.stats`). Exact McNemar tests are
stored as well (per seed for ByT5, seed 1 for the logistic models).

`run_all.sh` runs four candidate settings: training forms only, training forms plus WikiPron (the main setting), the
three proposal dictionaries, and all four sources together. `check_expected` freezes 660 values from these settings,
the Zupitza ablation and the editor agreement. It passes on macOS with Python 3.11 and 3.14 and on Linux with
Python 3.14. On Linux a few McNemar p-values differ from the committed ones in the 15th significant digit, which the
check tolerates.

## 1. Reproduce everything

```bash
bash run_all.sh
```

It takes about two minutes and has six steps:

1. Inputs. It uses `data/brunetti-length.txt` (Tauber's *Beowulf* gold) and `data/ang_latn_broad.tsv` (WikiPron Old
   English) if they are present and downloads them otherwise, then prints their SHA-256. `oevl.audit_splits` prints
   the word 6-gram overlap of each held-out set with each training work and writes `data/leakage_excluded.json`.
2. Unit tests in `tests/`.
3. Dictionaries. `oevl.wikipron_to_marked` writes `data/lex_wikipron.csv`. The three proposal dictionaries ship in
   `data/`, so `oevl.build_lexicons --unions-only` only rebuilds the unions. Without them it downloads and converts
   the original sources.
4. Experiments. Each of the four candidate settings runs on the John dev set and both test sets, one set per run,
   and writes `results/<setting>/`. All four train without Zupitza (`--exclude-train zupitza_grammar`, see D3 in
   [DATA.md](DATA.md)). One ablation keeps all five training works: `results/wikipron_with_zupitza/` is the WikiPron
   setting with Zupitza, on the same three sets. Then `oevl.charsiu_g2p analyse` runs on the committed CharsiuG2P
   output and `oevl.agreement` writes `results/agreement/`.
5. `oevl.check_expected` fails if any of the 660 frozen values changes.
6. `oevl.error_typing`, `oevl.encoder_summary --from-tsv` and `oevl.select_config` rebuild the error table, the
   ByT5 table and numbers, and `results/encoder/selection.json` from committed files without a GPU (section 5). Then
   `oevl.build_report` writes `report/generated/` (numbers, tables, figure), checks the worked examples against the
   predictions, and compiles `REPORT.pdf` if `pdflatex` is installed. Without LaTeX, upload `report/` to Overleaf and
   compile `main.tex`.

## 2. Run a single experiment

Results for the same `--out` folder are merged. Keep `--exclude-train zupitza_grammar` (decision D3), because
`experiment.py` refuses, before writing anything, to add to a folder made with a different training set:

```bash
PYTHONPATH=src python -m oevl.experiment --data data --out results/wikipron --lexicon data/lex_wikipron.csv --exclude-train zupitza_grammar --tests soliloquies
PYTHONPATH=src python -m oevl.experiment --data data --out results/wikipron --lexicon data/lex_wikipron.csv --exclude-train zupitza_grammar --tests tauber_beowulf
```

## 3. Rebuild the corpus and the dictionaries

```bash
PYTHONPATH=src python -m oevl.build_corpus --source <name> --out data/by_source
PYTHONPATH=src python -m oevl.clean_corpus
PYTHONPATH=src python -m oevl.build_lexicons
```

`oevl.build_corpus` fetches an edition into `data/raw/corpus/` and rebuilds its file and manifest entry: the URL,
retrieval date and SHA-256 of each input, and the work's `role` (train, dev, test or agreement). The first extraction
scripts were lost, so the rules were worked out from the files. After `clean_corpus`, `matthew_ws`, `bright_reader`,
`physiologus`, `hs_beowulf` and `john_ws` rebuild byte for byte. The other four rebuilds contain all but four of their 65,991
committed tokens in order, but some sentences differ, mostly where the original language filter dropped short lines (the
module docstring has the details). Those four committed files were kept as they are. `john_ws.jsonl` (Gospel of
John, the dev set) is built the same way as Matthew. The builder still writes an old sentence-level `split` field,
which `clean_corpus` removes, because the split is by work as the manifest `role` says. `oevl.build_lexicons`
rebuilds the three proposal dictionaries from the original sources.

## 4. ByT5 (GPU)

Training needs a GPU. `requirements.txt` already includes torch and the pinned `transformers==4.57.6`.

```bash
PYTHONPATH=src python -m oevl.encoder_experiment --out my_runs/encoder
bash scripts/encoder_grid.sh [select|grid|all] [RUNS_DIR]
```

`oevl.encoder_experiment` fine-tunes ByT5-small with the word only, the left context or the whole sentence. Like
`run_all.sh` it trains without Zupitza by default (D3), and `--exclude-train ""` trains on all five works.
`--arms` picks the arms (default `word,left,sentence`), so `--arms word,left,sentence,local2,local5` also runs the
arms with two or five words on each side. `--train-limit N` trains on a fixed subset,
and `--tests` scores fewer than the three held-out sets. Without `--out` it writes to `results/encoder/`, next to
the committed files, so pass `--out` for new runs.

`scripts/encoder_grid.sh` has the exact commands behind `results/encoder/`: 24 dev selection runs, whose rule
`oevl.select_config` applies (`selection.json`), then 39 runs of the chosen setting. `oevl.encoder_summary --runs`
pools those into `summary.json`, the per-token prediction tables (`*_predictions.tsv.gz`),
`report/generated/tab_encoder.tex` and `numbers_encoder.tex`. The run folders themselves are not committed. GPU
training is not bit-for-bit repeatable, so a new run can differ slightly.

The ByT5 pilot trains all five arms on 10,000 training examples, using the notebooks in `notebooks/` on Kaggle or
Colab (see `notebooks/README.md`). Its logic is in `oevl.pilot`. The pilots ran before D3, so `oevl.pilot` still trains on all
five works.

## 5. ByT5 tables from the committed predictions (no GPU)

```bash
PYTHONPATH=src python -m oevl.encoder_summary --from-tsv
PYTHONPATH=src python -m oevl.select_config
```

`run_all.sh` runs both. They rebuild the ByT5 table and numbers and `results/encoder/selection.json` from the
committed prediction tables and run records.

## 6. CharsiuG2P error analysis

```bash
PYTHONPATH=src python -m oevl.charsiu_g2p benchmark
PYTHONPATH=src python -m oevl.charsiu_g2p predict
PYTHONPATH=src python -m oevl.charsiu_g2p analyse
```

`benchmark` runs the published checkpoint on CharsiuG2P's own 500 Old English test words
(`data/charsiu_ang_test.tsv`) and writes `results/charsiu/benchmark.json` and a per-word `benchmark.tsv`. It takes
a few minutes on a recent laptop CPU. `predict` writes `results/charsiu/ipa.tsv` for our held-out sets, and the committed
copy was made on a GPU. `run_all.sh` runs the `analyse` step on it.

## 7. CPU smoke tests without downloads

```bash
PYTHONPATH=src python -m oevl.encoder_experiment --tiny --out /tmp/t
PYTHONPATH=src python -m oevl.pilot --tiny --allow-cpu --skip-model-check --max-bytes 128 --out /tmp/pilot
PYTHONPATH=src python -m oevl.charsiu_g2p predict --tiny
```

The second runs the whole pilot loop. The third writes `results/charsiu/ipa_tiny.tsv`.

## 8. Try it on your own sentences

From the repository root:

```bash
PYTHONPATH=src python -m oevl.try_text my_tests/sentences.txt
PYTHONPATH=src python -m oevl.try_text my_tests/sentences.txt --gold my_tests/sentences_marked.txt --tsv my_tests/words.tsv
```

`sentences.txt` has one sentence per line, and any macrons in it are removed first. With `--gold` (the same lines
with correct macrons) it prints the accuracy. `--tsv` writes one row per word: the output, and whether the word is
known, changed and correct. `--model` picks `frequency`, `word`, `left` or `bilateral` (the default), and
`--lexicon none` uses training texts only.

It trains the report's logistic model (bilateral by default, seed 1) in under a minute and changes no existing file. Unknown words stay
unchanged. Text that is not Old English has no correct answer, so count the words it wrongly changed instead.

## 9. Export and upload the Hugging Face dataset

```bash
PYTHONPATH=src python -m oevl.export_hf --out release/hf
hf upload Ace3Z/old-english-vowel-length release/hf . --repo-type=dataset
```

The export writes `release/hf/`, which git ignores and every run regenerates. It holds one JSONL file per split:
`train`, `dev`, `test_prose` (the *Soliloquies*), `test_verse` (Tauber's *Beowulf*) and `agreement` (the
Harrison & Sharp and Wyatt *Beowulf*). It also holds the dictionaries in `lexicons/`, the ambiguity class files,
`release/hf_license.md` as `LICENSE.md`, and the dataset card `README.md`, made from `release/hf_card.md` with the
token counts read from the written files. Sentences that the leakage audit leaves out of evaluation are kept and
marked with `leakage_excluded`. `train` holds all five training works, Zupitza included.

The upload needs a Hugging Face account with write access to the `Ace3Z` namespace. Older versions of
`huggingface_hub` only have `huggingface-cli`. With those, the same upload is
`huggingface-cli upload Ace3Z/old-english-vowel-length release/hf . --repo-type=dataset`.

## Output data

```
results/
├── training_only/              one folder per candidate setting
│   ├── metrics.json
│   ├── <set>_per_word.csv      per-word results, <set> = john_ws, soliloquies, tauber_beowulf
│   ├── <set>_predictions.jsonl token-level predictions
│   └── results_training_only_<set>.log
├── wikipron/
├── proposal/
├── all/
├── wikipron_with_zupitza/      the Zupitza ablation
├── agreement/                  metrics.json, disagreements.tsv, agreement.log, and disagreements_checked.tsv
│                               (100 disagreements labelled by an LLM and checked by a blind LLM review,
│                               not yet reviewed by the team)
├── encoder/                    selection.json, summary.json, <set>_predictions.tsv.gz
├── charsiu/                    benchmark.json, benchmark.tsv, ipa.tsv, metrics.json
├── error_typing/               labels.tsv, summary.json
└── label_review/               expert_review.json: the blind LLM review of all 560 LLM labels
report/
├── main.tex
└── generated/                  tables, numbers and figure (build_report, encoder_summary, error_typing)
release/
├── hf_card.md                  template of the Hugging Face dataset card
└── hf_license.md               licence file of the Hugging Face copy
```
