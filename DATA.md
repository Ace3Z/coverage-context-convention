# Data, splits and licences

This file describes the corpus, the held-out sets, the decisions behind them and the layout of `data/`. The commands are
in [REPRODUCE.md](REPRODUCE.md). Back to the [README](README.md).

The corpus, the dictionaries and the class labels are also on Hugging Face as
[`Ace3Z/old-english-vowel-length`](https://huggingface.co/datasets/Ace3Z/old-english-vowel-length).

## Splits

The names in brackets in the Role column are the splits of the Hugging Face copy.

| Role | Works | Tokens |
|---|---|---|
| Train (`train`) | Bright's *Reader*, Matthew (West Saxon), Sweet's *Primer*, *Physiologus* | 69,087 |
| Train, ablation only (in `train`) | Zupitza's edition of Ælfric's *Grammar* (left out under D3, see below) | 23,940 |
| Dev (`dev`) | Gospel of John (`john_ws`) | 16,649 of 17,310 scored |
| Prose test (`test_prose`) | Alfred's *Soliloquies* | 15,426 of 15,475 scored |
| Verse test (`test_verse`) | Tauber's *Beowulf* | 17,361 (some of the 17,244 rows in Tauber's file hold two or three words) |
| Evaluation only (`agreement`): editor agreement and the prior's score against each edition | Harrison & Sharp and Wyatt *Beowulf*, each with the *Finnsburh Fragment* (Wyatt's file lacks the first 18 lines of the fragment) | 35,250 (17,682 and 17,568) |

The split is by work, and no *Beowulf* text is used for training. A dev or test sentence that shares a word 6-gram
(bare, lower-cased, ð folded to þ, hyphens and apostrophes dropped) with a training work is left out of scoring. That
removes 34 John sentences (661 tokens, shared with Matthew and Sweet), one *Soliloquies* sentence (49 tokens) and
nothing from *Beowulf*.
`PYTHONPATH=src python -m oevl.audit_splits` repeats the check.

A few short Latin and markup fragments survived the extraction, for example a stray `nop` token in `matthew_ws:00907`.
In `sweet_primer:00236` the extraction split a þā into `þ` and a short `a`, which gives one wrong training label. They
are part of the training data the results used.

## Decision D3: Zupitza

Zupitza's *Grammar* is a Latin grammar written in Old English. It was kept out of dev and test, and whether to train
on it was left open. We decided on the John dev set, on the 7,639 lexicon-ambiguous tokens (training forms plus
WikiPron) that both settings share. Dropping Zupitza raised the prior from 95.12% to 95.60% and the bilateral
logistic model from 94.70% to 95.31% (seeds 1 to 3). The best ByT5 sentence-arm setting without Zupitza also beats the
best one with it (95.70% against 95.21%, seeds 1 and 2), but at the five matched settings without a collapsed run the
change ranges from -7.40 to +0.49 points, so ByT5 gains slightly from the decision in all but its least trained
setting (1 epoch at learning rate 1e-4).

So the four candidate settings (`results/training_only`, `wikipron`, `proposal` and `all`) train without Zupitza. So
do `oevl.try_text`, the ByT5 default, the coverage numbers in the report, the CharsiuG2P populations, and the prior
that `oevl.agreement` compares the editors with.
`results/wikipron_with_zupitza` repeats the WikiPron setting with Zupitza in training, for comparison. The leakage
audit still checks all five training works, as the ablation needs. Zupitza shares no 6-gram with the dev or test sets,
so the exclusions are the same either way.

## Input data

```
data/
├── by_source/                  the corpus, one file per work
│   ├── bright_reader.jsonl     train
│   ├── matthew_ws.jsonl        train
│   ├── sweet_primer.jsonl      train
│   ├── physiologus.jsonl       train
│   ├── zupitza_grammar.jsonl   train (left out under D3)
│   ├── john_ws.jsonl           dev
│   ├── soliloquies.jsonl       prose test
│   ├── hs_beowulf.jsonl        editor agreement
│   └── wyatt_beowulf.jsonl     editor agreement
├── manifest.json               origin, role and cleaning counts of each work
├── brunetti-length.txt         Tauber's Beowulf gold, the verse test
├── ang_latn_broad.tsv          WikiPron Old English
├── wikipron_provenance.json
├── lex_wikipron.csv            dictionaries (marked_form, and sources in the two unions)
├── lex_clarkhall.csv
├── lex_wiktionary.csv
├── lex_bt.csv
├── lex_proposal.csv
├── lex_all.csv
├── lexicons_provenance.json
├── ambiguity_classes.csv       60 skeletons with their A/B/C class (LLM labels, manually checked)
├── ambiguity_classes.json      label provenance, agreement, counting rule, D3 notes
├── leakage_excluded.json       dev and test sentences left out of scoring
├── charsiu_ang_test.tsv        CharsiuG2P's Old English test words
├── charsiu_ang_dict_words.txt  words of CharsiuG2P's Old English pronunciation dictionary (with its dev and test
│                               words), used only to split the error analysis
├── LICENSE.md                  licence of every file here
└── raw/                        downloads for rebuilding (gitignored)
```

## Licences

The code is MIT ([LICENSE](LICENSE)). [data/LICENSE.md](data/LICENSE.md) gives the source, licence and attribution of
every file in `data/`, and the page each statement was checked against.
