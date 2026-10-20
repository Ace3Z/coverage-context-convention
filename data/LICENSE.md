# Data licences

The code in this repository is MIT (see `../LICENSE`). The files in `data/` are not code and carry the licences
below. Everything here that we made or derived from CC BY-SA material is released under the Creative Commons
Attribution-ShareAlike 4.0 International licence (CC BY-SA 4.0, https://creativecommons.org/licenses/by-sa/4.0/).
A file that combines several sources takes the strictest licence among them. `data/raw/` holds downloads for
rebuilding and is not part of the repository.

Each statement was checked on 2026-10-20 against the licence file or the source page named in the last column.

## Corpus: `by_source/*.jsonl` and `manifest.json`

The corpus bundle is released under CC BY-SA 4.0, because five of the nine works are Wikisource
transcriptions. The printed editions themselves are in the public domain in the United States. `manifest.json` records
the source URL of each work, and, for the five works whose committed file `oevl.build_corpus` produced, the retrieval
date and SHA-256 of each input.

| File | Work, editor, year | Source | Licence of the source | Checked |
|---|---|---|---|---|
| `bright_reader.jsonl` | *Anglo-Saxon Reader* (selections), Bright, 1913 | https://en.wikisource.org/wiki/Bright%27s_Anglo-Saxon_Reader | Wikisource transcription, CC BY-SA 4.0. The edition is public domain in the United States | page footer and the page's public-domain notice |
| `matthew_ws.jsonl` | Gospel of Matthew (West Saxon), Bright, 1904 (corrected impression of 1910) | https://en.wikisource.org/wiki/Evangelium_Secundum_Mattheum:_the_Gospel_of_Saint_Matthew_in_West-Saxon | Wikisource transcription, CC BY-SA 4.0. The scan is tagged PD-scan on Wikimedia Commons | page footer, Commons file page |
| `john_ws.jsonl` | Gospel of John (West Saxon), Bright & Harris, 1904 | https://en.wikisource.org/wiki/Index:Gospel_of_Saint_John_in_West-Saxon.djvu | Wikisource transcription, CC BY-SA 4.0. The scan is tagged PD-scan on Wikimedia Commons | page footer, Commons file page |
| `soliloquies.jsonl` | Alfred's *Soliloquies*, Hargrove, 1902 | https://en.wikisource.org/wiki/King_Alfred%27s_Old_English_Version_of_St._Augustine%27s_Soliloquies | Wikisource transcription, CC BY-SA 4.0. The edition is public domain in the United States | page footer and the page's public-domain notice |
| `wyatt_beowulf.jsonl` | *Beowulf*, Wyatt, 1894 | https://en.wikisource.org/wiki/Beowulf_(Wyatt) | Wikisource transcription, CC BY-SA 4.0. The edition is public domain in the United States | page footer and the page's public-domain notice |
| `hs_beowulf.jsonl` | *Beowulf*, Harrison & Sharp, 4th revised edition (first 1883), in the Project Gutenberg revision whose producer modernised some quantities | https://www.gutenberg.org/ebooks/9701 | Project Gutenberg: "Public domain in the USA." | ebook page |
| `zupitza_grammar.jsonl` | Ælfric's *Grammar*, Zupitza, 1880 | https://www.gutenberg.org/ebooks/45861 | Project Gutenberg: "Public domain in the USA." | ebook page |
| `sweet_primer.jsonl` | *Anglo-Saxon Primer* (texts), Sweet, 1905 | https://www.gutenberg.org/ebooks/34316 | Project Gutenberg: "Public domain in the USA." | ebook page |
| `physiologus.jsonl` | *Old English Physiologus*, Cook & Pitman, 1921 | https://www.gutenberg.org/ebooks/14529 | Project Gutenberg: "Public domain in the USA." | ebook page |

The Project Gutenberg header and licence are not part of the extracted text. Attribution for the Wikisource works:
the Wikisource contributors of each page, at the URLs above.

## Inputs copied from other projects

| File | Source | Licence | Checked |
|---|---|---|---|
| `brunetti-length.txt` | James Tauber, *old-english-texts*, `beowulf/raw-data/brunetti-length.txt`, https://github.com/old-english-learner-texts/old-english-texts | CC BY-SA 4.0, attribution to James Tauber. Tauber built it from Brunetti's analysis and the Perseus text of *Beowulf*, both of which his README calls CC-licensed | repository `LICENSE` (CC BY-SA 4.0), `README.md` ("All other text and data is made available under a Creative Commons Attribution-ShareAlike 4.0 International License with attribution to James Tauber"), `beowulf/README.md` |
| `ang_latn_broad.tsv` | WikiPron, `data/scrape/tsv/ang_latn_broad.tsv`, https://github.com/CUNY-CL/wikipron | CC BY-SA 4.0 (Wiktionary content). The WikiPron code is Apache-2.0, but its README says the Wiktionary data in `data/` "has its own licensing terms" and links Wiktionary:Copyrights, which dual-licenses entry text under CC BY-SA 4.0 and the GFDL. We use it under CC BY-SA 4.0 | WikiPron `README.md` (License section) and `LICENSE.txt`, https://en.wiktionary.org/wiki/Wiktionary:Copyrights |
| `charsiu_ang_test.tsv` | CharsiuG2P `data/test/ang.tsv` at commit `0c929390759fb94f8ecdfc05cc0bc5f2ff2dc0f4`, https://github.com/lingjzhu/CharsiuG2P | CC BY-SA 4.0 (Wiktionary content). The CharsiuG2P repository is MIT, but its `sources/info/wikipron/ang_latn_broad_info.txt` and `ang_latn_narrow_info.txt` name WikiPron's Old English files as the source of its Old English data. Those files label the data Apache 2.0, which is WikiPron's code licence. We follow WikiPron's README for the data | CharsiuG2P `LICENSE` (MIT), `README.md`, `sources/info/wikipron/` |
| `charsiu_ang_dict_words.txt` | the distinct words of CharsiuG2P's Old English pronunciation dictionary `dicts/ang.tsv` at the same commit, without the pronunciations | CC BY-SA 4.0, as above | as above |

## Dictionaries derived for this project

| File | Built from | Licence | Checked |
|---|---|---|---|
| `lex_wikipron.csv` | `ang_latn_broad.tsv` (`oevl.wikipron_to_marked`) | CC BY-SA 4.0 | as for `ang_latn_broad.tsv` |
| `lex_clarkhall.csv` | Clark Hall, *A Concise Anglo-Saxon Dictionary*, TEI conversion by Serg078, https://github.com/Serg078/Conversion-of-The-Clark-Hall-Dictionary-of-Old-English-to-TEI (commit `b53fc7d`). The conversion's source is Project Gutenberg ebook 31543 | Unlicense (public-domain dedication), as its source. Project Gutenberg lists ebook 31543 as "Public domain in the USA." | repository `LICENSE` (The Unlicense), the repository's `clark hall dict/` folder, https://www.gutenberg.org/ebooks/31543 |
| `lex_wiktionary.csv` | Old English entries of the English Wiktionary as extracted by kaikki.org (wiktextract), https://kaikki.org/dictionary/Old%20English/ | CC BY-SA 4.0 (Wiktionary content, which Wiktionary also offers under the GFDL). kaikki.org states no licence of its own for the data and asks users to cite Ylonen (2022), *Wiktextract: Wiktionary as Machine-Readable Structured Data*, LREC 2022. Wiktextract's code is MIT | kaikki.org Old English page and home page, wiktextract `README.md`, Wiktionary:Copyrights and the Wiktionary page footer (CC BY-SA 4.0) |
| `lex_bt.csv` | Tichý, Ondřej et al. 2021, *Bosworth-Toller's Anglo-Saxon Dictionary online*, LINDAT/CLARIAH-CZ, http://hdl.handle.net/11234/1-3532 | CC BY 4.0, attribution to Tichý et al. | LINDAT item page: "licensed under: Creative Commons - Attribution 4.0 International (CC BY 4.0)" |
| `lex_proposal.csv` | union of `lex_clarkhall.csv`, `lex_wiktionary.csv`, `lex_bt.csv` | CC BY-SA 4.0 (the strictest of its sources), with the attributions above | |
| `lex_all.csv` | union of `lex_proposal.csv` and `lex_wikipron.csv` | CC BY-SA 4.0, with the attributions above | |

## Files made for this project

`ambiguity_classes.csv`, `ambiguity_classes.json`, `leakage_excluded.json`, `lexicons_provenance.json` and
`wikipron_provenance.json` are released under CC BY-SA 4.0. The class files list spellings and candidate
patterns taken from the training works and `lex_wikipron.csv`.
