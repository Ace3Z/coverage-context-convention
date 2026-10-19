# Data, splits and licences

The corpus, the held-out sets, the decisions that shape them, the layout of `data/`, and the licence of every
source. The commands are in [REPRODUCE.md](REPRODUCE.md). Back to the [README](README.md).

The same data is on Hugging Face as
[`Ace3Z/old-english-vowel-length`](https://huggingface.co/datasets/Ace3Z/old-english-vowel-length), private until it
is released together with this repository.

## Data and protocol

- **Train:** Bright's *Reader*, Matthew (West Saxon), Sweet's *Primer*, *Physiologus* (69,098 tokens). Zupitza's
  Ælfric *Grammar* (23,940 tokens) stays in the corpus and the dataset release but is left out of training (D3
  below), except in the ablation `results/wikipron_with_zupitza` (93,038 training tokens).
  **Dev:** Gospel of John (`john_ws`, 16,649 of 17,310 tokens), scored like the test sets.
  **Prose test:** Alfred's *Soliloquies* (15,431 of 15,480 tokens). **Verse test:** Tauber's
  *Beowulf* (17,361 tokens from 17,244 rows). **Editor agreement only:** Harrison & Sharp and Wyatt *Beowulf*.
- Splits are by work. No *Beowulf* text is used for training. Dev and test sentences that share a word 6-gram
  (bare, lower-cased, ð folded to þ) with a training work are left out of evaluation. For John that is 34 sentences
  of 661 tokens (shared with Matthew and Sweet), for the *Soliloquies* one sentence of 49 tokens, for *Beowulf* none
  (`python -m oevl.audit_splits`).
- **D3, Zupitza:** plan decision D3 kept Zupitza's *Grammar*, a Latin grammar written in Old English, out of dev and
  test and left open whether to train on it. It was decided on the John dev set, on lexicon-ambiguous tokens with
  training forms + WikiPron. The ByT5 sentence arm trained without Zupitza scored 96.00% (mean dev accuracy), against
  95.06% for the best setting trained with it. The prior scored 95.60% without it and 94.87% with it, and the
  bilateral logistic model 95.23% and 94.44% (means over seeds 1 to 3). So every system in `results/<setting>/`,
  `oevl.agreement` (its floor is the prior), `oevl.try_text`, the encoder's default, the coverage numbers of
  `oevl.build_report` and the CharsiuG2P populations use the training set without Zupitza.
  `results/wikipron_with_zupitza` documents the comparison: it is the earlier headline run, unchanged. The leakage
  audit still checks all five works, as the ablation needs. Zupitza shares no 6-gram with the dev or test sets, so
  the exclusions are the same either way. The report's per-word table still shows how Zupitza marks each word,
  counted from its corpus file.

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
├── leakage_excluded.json       dev and test sentences left out of evaluation
├── charsiu_ang_test.tsv        CharsiuG2P's Old English test words
├── charsiu_ang_dict_words.txt  words of CharsiuG2P's Old English training dictionary
├── LICENSE.md                  licence of every file here
└── raw/                        downloads for rebuilding (gitignored)
```

## Data and licences

The code is MIT (`LICENSE`). The data is CC BY-SA 4.0 unless `data/LICENSE.md` says otherwise. That file gives the
licence of every file in `data/` and the licence text or source page each statement was checked against.

- Corpus: `data/by_source/`, cleaned by `python -m oevl.clean_corpus` (explicit character mappings, counted in
  `data/manifest.json`). Built from Project Gutenberg editions (public domain in the USA) and Wikisource
  transcriptions (CC BY-SA 4.0), released as CC BY-SA 4.0. The Hugging Face copy is
  `Ace3Z/old-english-vowel-length`, private until the release.
- Tauber gold: https://github.com/old-english-learner-texts/old-english-texts (CC BY-SA 4.0, attribution to James
  Tauber).
- WikiPron: https://github.com/CUNY-CL/wikipron. The code is Apache-2.0, but the scraped data is Wiktionary content
  with Wiktionary's own terms, so `data/ang_latn_broad.tsv` and `data/lex_wikipron.csv` are CC BY-SA 4.0.
- Clark Hall, *A Concise Anglo-Saxon Dictionary* (Project Gutenberg 31543, public domain in the USA), TEI conversion
  by Serg078 (Unlicense): https://github.com/Serg078/Conversion-of-The-Clark-Hall-Dictionary-of-Old-English-to-TEI
- Wiktionary Old English entries via kaikki.org (Wiktionary content, CC BY-SA 4.0):
  https://kaikki.org/dictionary/Old%20English/
- Bosworth–Toller data dump, Tichý et al., LINDAT (CC BY 4.0): http://hdl.handle.net/11234/1-3532
- The dictionary unions `data/lex_proposal.csv` and `data/lex_all.csv` take the strictest licence of their sources,
  CC BY-SA 4.0.
- CharsiuG2P (MIT): https://github.com/lingjzhu/CharsiuG2P. `data/charsiu_ang_dict_words.txt` lists the words of its
  Old English training dictionary, used only to split the error analysis into seen and unseen words.
  `data/charsiu_ang_test.tsv` is its `data/test/ang.tsv` at commit `0c929390759fb94f8ecdfc05cc0bc5f2ff2dc0f4`.
  CharsiuG2P took its Old English data from WikiPron, so both files are CC BY-SA 4.0.
