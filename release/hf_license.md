# Licences

The dataset is released under the Creative Commons Attribution-ShareAlike 4.0 International licence (CC BY-SA 4.0,
https://creativecommons.org/licenses/by-sa/4.0/), because it contains CC BY-SA material. A file that combines several
sources takes the strictest licence among them. The statements below were checked on 2026-10-18 against the licence
file or source page of each source. The code repository, https://github.com/Ace3Z/cl-uzh-26-essentials-code, keeps
a fuller version of this table for its own `data/` folder in `data/LICENSE.md`.

## Corpus splits

| File | Built from | Licence |
|---|---|---|
| `train.jsonl` | Bright's *Anglo-Saxon Reader* and the Gospel of Matthew (Wikisource transcriptions, CC BY-SA 4.0), Sweet's *Anglo-Saxon Primer*, the *Old English Physiologus* and Zupitza's edition of Ælfric's *Grammar* (Project Gutenberg, public domain in the USA) | CC BY-SA 4.0 |
| `dev.jsonl` | Gospel of John (Wikisource transcription, CC BY-SA 4.0) | CC BY-SA 4.0 |
| `test_prose.jsonl` | Alfred's *Soliloquies*, Hargrove (Wikisource transcription, CC BY-SA 4.0) | CC BY-SA 4.0 |
| `test_verse.jsonl` | James Tauber's `beowulf/raw-data/brunetti-length.txt`, https://github.com/old-english-learner-texts/old-english-texts (CC BY-SA 4.0, attribution to James Tauber), turned into sentence records | CC BY-SA 4.0 |
| `agreement.jsonl` | Wyatt's *Beowulf* (Wikisource transcription, CC BY-SA 4.0) and Harrison & Sharp's *Beowulf* (Project Gutenberg, public domain in the USA) | CC BY-SA 4.0 |

The source page of every work is in the Source data table of `README.md`. Attribution for the Wikisource works: the
Wikisource contributors of each page. The Project Gutenberg header and licence are not part of the extracted text.

## Dictionaries

| File | Built from | Licence |
|---|---|---|
| `lexicons/lex_wikipron.csv` | WikiPron Old English (`ang_latn_broad.tsv`), https://github.com/CUNY-CL/wikipron. The WikiPron code is Apache-2.0, but the data is Wiktionary content | CC BY-SA 4.0 |
| `lexicons/lex_clarkhall.csv` | Clark Hall, *A Concise Anglo-Saxon Dictionary*, TEI conversion by Serg078, https://github.com/Serg078/Conversion-of-The-Clark-Hall-Dictionary-of-Old-English-to-TEI (Project Gutenberg ebook 31543, public domain in the USA) | Unlicense, as its source |
| `lexicons/lex_wiktionary.csv` | Old English entries of the English Wiktionary via kaikki.org, https://kaikki.org/dictionary/Old%20English/ (cite Ylonen 2022, *Wiktextract*) | CC BY-SA 4.0 |
| `lexicons/lex_bt.csv` | Tichý et al. 2021, *Bosworth-Toller's Anglo-Saxon Dictionary online*, LINDAT/CLARIAH-CZ, http://hdl.handle.net/11234/1-3532 | CC BY 4.0, attribution to Tichý et al. |
| `lexicons/lex_proposal.csv` | union of Clark Hall, Wiktionary and Bosworth-Toller | CC BY-SA 4.0, with the attributions above |
| `lexicons/lex_all.csv` | union of the three and WikiPron | CC BY-SA 4.0, with the attributions above |
| `lexicons/lexicons_provenance.json`, `lexicons/wikipron_provenance.json` | URLs, licences and checksums of the dictionary sources | CC BY-SA 4.0 |

## Other files

| File | Licence |
|---|---|
| `ambiguity_classes.csv`, `ambiguity_classes.json` | CC BY-SA 4.0. They list spellings and candidate patterns taken from the dictionaries above |
| `README.md`, `LICENSE.md` | CC BY-SA 4.0 |

The provenance files and `ambiguity_classes.json` are copied unchanged from the code repository, so the paths they
mention (`data/...`, `results/...`) refer to that repository.
