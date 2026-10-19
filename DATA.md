# Data, splits and licences

This file describes the corpus, the held-out sets, the decisions behind them, the layout of `data/` and the licence of
every source. The commands are in [REPRODUCE.md](REPRODUCE.md). Back to the [README](README.md).

The same data is on Hugging Face as
[`Ace3Z/old-english-vowel-length`](https://huggingface.co/datasets/Ace3Z/old-english-vowel-length).

## Splits

| Role | Works | Tokens |
|---|---|---|
| Train | Bright's *Reader*, Matthew (West Saxon), Sweet's *Primer*, *Physiologus* | 69,098 |
| Train, ablation only | Zupitza's edition of Ælfric's *Grammar* (left out under D3, see below) | 23,940 |
| Dev | Gospel of John (`john_ws`) | 16,649 of 17,310 scored |
| Prose test | Alfred's *Soliloquies* | 15,431 of 15,480 scored |
| Verse test | Tauber's *Beowulf* | 17,361 (some of the 17,244 rows in Tauber's file hold two or three words) |
| Editor agreement only | Harrison & Sharp and Wyatt *Beowulf* | 35,250 (17,682 and 17,568) |

The split is by work, and no *Beowulf* text is used for training. A dev or test sentence that shares a word 6-gram
(bare, lower-cased, ð folded to þ) with a training work is left out of scoring. That removes 34 John sentences
(661 tokens, shared with Matthew and Sweet), one *Soliloquies* sentence (49 tokens) and nothing from *Beowulf*.
`PYTHONPATH=src python -m oevl.audit_splits` repeats the check.

A few short English, Latin and markup fragments survived the extraction, for example English notes in
`bright_reader:00937` to `00943` and a stray `nop` token in `matthew_ws:00907`. They are part of the training data
the results used.

## Decision D3: Zupitza

Zupitza's *Grammar* is a Latin grammar written in Old English. The plan kept it out of dev and test and left open
whether to train on it. We decided on the John dev set, on the 7,639 lexicon-ambiguous tokens (training forms plus
WikiPron) that both settings share. Dropping Zupitza raised the ByT5 sentence arm from 95.28% (its best setting with
Zupitza, seeds 1 and 2) to 96.00%, the prior from 95.12% to 95.60%, and the bilateral logistic model from 94.72% to
95.23% (seeds 1 to 3).

So every system in `results/<setting>/` trains without Zupitza. So do `oevl.try_text`, the ByT5 default, the coverage
numbers in the report, the CharsiuG2P populations, and the prior that `oevl.agreement` compares the editors with.
`results/wikipron_with_zupitza` keeps the earlier run with Zupitza for comparison. The leakage audit still checks all
five training works, as the ablation needs. Zupitza shares no 6-gram with the dev or test sets, so the exclusions are
the same either way.

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
├── ambiguity_classes.csv       60 skeletons with their A/B/C class
├── ambiguity_classes.json      label provenance, agreement, counting rule, D3 notes
├── leakage_excluded.json       dev and test sentences left out of scoring
├── charsiu_ang_test.tsv        CharsiuG2P's Old English test words
├── charsiu_ang_dict_words.txt  words of CharsiuG2P's Old English training dictionary
├── LICENSE.md                  licence of every file here
└── raw/                        downloads for rebuilding (gitignored)
```

## Licences

The code is MIT (`LICENSE`). The data is CC BY-SA 4.0 unless `data/LICENSE.md` says otherwise. That file gives the
licence of every file in `data/` and the licence text or source page that each statement was checked against.

- Corpus: `data/by_source/`, cleaned by `PYTHONPATH=src python -m oevl.clean_corpus` with explicit character mappings that are
  counted in `data/manifest.json`. It comes from Project Gutenberg editions (public domain in the USA) and Wikisource
  transcriptions (CC BY-SA 4.0), and is released as CC BY-SA 4.0.
- Tauber gold: https://github.com/old-english-learner-texts/old-english-texts (CC BY-SA 4.0, attribution to James
  Tauber).
- WikiPron: https://github.com/CUNY-CL/wikipron. Its code is Apache-2.0, but the scraped data is Wiktionary content
  under Wiktionary's own terms, so `data/ang_latn_broad.tsv` and `data/lex_wikipron.csv` are CC BY-SA 4.0.
- Clark Hall, *A Concise Anglo-Saxon Dictionary* (Project Gutenberg 31543, public domain in the USA), in the TEI
  conversion by Serg078 (Unlicense): https://github.com/Serg078/Conversion-of-The-Clark-Hall-Dictionary-of-Old-English-to-TEI
- Wiktionary's Old English entries via kaikki.org (Wiktionary content, CC BY-SA 4.0):
  https://kaikki.org/dictionary/Old%20English/
- The Bosworth-Toller data dump by Tichý et al. on LINDAT (CC BY 4.0): http://hdl.handle.net/11234/1-3532
- The dictionary unions `data/lex_proposal.csv` and `data/lex_all.csv` take the strictest licence of their sources,
  CC BY-SA 4.0.
- CharsiuG2P (MIT): https://github.com/lingjzhu/CharsiuG2P. `data/charsiu_ang_dict_words.txt` lists the words of its
  Old English training dictionary and is used only to split the error analysis into seen and unseen words.
  `data/charsiu_ang_test.tsv` is its `data/test/ang.tsv` at commit `0c929390759fb94f8ecdfc05cc0bc5f2ff2dc0f4`.
  CharsiuG2P took its Old English data from WikiPron, so both files are CC BY-SA 4.0.
