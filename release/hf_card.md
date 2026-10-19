---
license: cc-by-sa-4.0
language:
- ang
pretty_name: Old English Vowel Length
task_categories:
- token-classification
tags:
- old-english
- vowel-length
- macron-restoration
- diacritic-restoration
size_categories:
- 1K<n<10K
configs:
- config_name: corpus
  default: true
  data_files:
  - split: train
    path: train.jsonl
  - split: dev
    path: dev.jsonl
  - split: test_prose
    path: test_prose.jsonl
  - split: test_verse
    path: test_verse.jsonl
  - split: agreement
    path: agreement.jsonl
---

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
  <a href="https://github.com/Ace3Z/cl-uzh-26-essentials-code">
    <img src="https://img.shields.io/badge/GitHub-Code-181717?style=for-the-badge&logo=github&logoColor=white" alt="Code"/>
  </a>
  <a href="https://github.com/Ace3Z/cl-uzh-26-essentials-code/blob/main/report/main.tex">
    <img src="https://img.shields.io/badge/Report-LaTeX%20source-b31b1b?style=for-the-badge" alt="Report"/>
  </a>
  <a href="https://creativecommons.org/licenses/by-sa/4.0/">
    <img src="https://img.shields.io/badge/Licence-CC%20BY--SA%204.0-ef9421?style=for-the-badge" alt="Licence"/>
  </a>
</p>

# Coverage, Context, Convention: Restoring Vowel Length in Old English

**Old English sentences from nineteenth- and early twentieth-century editions, each with the editor's long-vowel marks
(macrons) and without them, plus the dictionaries and the ambiguity class file of the project.**

[Overview](#-overview) •
[Corpus](#-corpus) •
[Splits](#-splits) •
[Dictionaries](#-dictionaries) •
[Ambiguity Classes](#-ambiguity-classes) •
[Usage](#-usage) •
[Limitations](#-limitations) •
[Citation](#-citation)

</div>

---

## 📖 Overview

This dataset holds Old English sentences from nineteenth- and early twentieth-century editions, each with the editor's
vowel-length marks (macrons) and without them: $sentences sentences and $tokens tokens. It also holds the four
dictionaries and the ambiguity class file of the project. The task it was made for is putting the macrons back into
bare text. The dataset is private until it is released together with the code.

Course project of Mahbod Tajdini, Hanna Miranovich and Arjun Singh Bhadoria (Essentials in Text and Speech
Processing, University of Zurich, autumn 2026). The code that builds and evaluates the data and reproduces the report
is the `oevl` package at https://github.com/Ace3Z/cl-uzh-26-essentials-code, and `python -m oevl.export_hf` writes
this folder. There is no code and there are no experiment results here.

---

## 📦 Corpus


One JSON object per line, one sentence per object. Every split file has the same fields.

| Field | Type | Content |
|---|---|---|
| `id` | string | `<source>:<number>`, or `tauber:<number>` in `test_verse` |
| `source` | string | short name of the edition (see the table of sources) |
| `work`, `editor` | string | the work and the editor of the edition |
| `year` | integer | year of the edition |
| `genre` | string | `prose`, `verse` or `mixed` |
| `section` | string | chapter or Wikisource page the sentence comes from, empty in `test_verse` |
| `text` | string | the sentence with the edition's length marks |
| `text_bare` | string | the same sentence with the length marks removed |
| `tokens` | list of strings | the words of the sentence, with length marks, without punctuation |
| `tokens_bare` | list of strings | the same words without length marks |
| `length_patterns` | list of strings | one per token, one digit per vowel letter (a e i o u y æ): `1` if the edition marks it long, `0` if not |
| `leakage_excluded` | boolean | `true` for a dev or test sentence that shares a word 6-gram with a training work |

Length is written only with ā ē ī ō ū ȳ ǣ (and their capitals). For example the token `gōdne` has the bare form
`godne` and the pattern `10`. There is no `split` field in the records: the file is the split.

---

## 🔀 Splits


$table

The split is by work, never by sentence, and no *Beowulf* text is in `train`. `test_prose` is the prose test set and
`test_verse` the verse test set. `agreement` holds two further editions of *Beowulf* (each with the *Finnsburh
Fragment*, 21 sentences and 484 tokens in all), used only to measure how often two editors mark the same word the same
way.

`train` holds all five training works. The project's final systems leave Zupitza's Ælfric *Grammar*
(`zupitza_grammar`) out of training (decision D3, taken on the dev set) and train on it only in an ablation. To train
as the report does, drop the rows whose `source` is `zupitza_grammar`.

Nothing is left out. A dev or test sentence that shares a word 6-gram (bare, lower-cased, ð folded to þ) with any
training work is kept with `leakage_excluded: true`. The project leaves these sentences out when it scores a system,
so drop them to compare with its numbers.

### Sources and provenance


| `source` | Split | Work | Editor, year | URL | Licence of the source |
|---|---|---|---|---|---|
| `bright_reader` | train | *Anglo-Saxon Reader* (selections) | Bright, 1913 | https://en.wikisource.org/wiki/Bright%27s_Anglo-Saxon_Reader | Wikisource, CC BY-SA 4.0 |
| `matthew_ws` | train | Gospel of Matthew (West Saxon) | Bright, 1904 | https://en.wikisource.org/wiki/Evangelium_Secundum_Mattheum | Wikisource, CC BY-SA 4.0 |
| `sweet_primer` | train | *Anglo-Saxon Primer* (texts) | Sweet, 1905 | https://www.gutenberg.org/ebooks/34316 | Project Gutenberg, public domain in the USA |
| `physiologus` | train | *Old English Physiologus* | Cook & Pitman, 1921 | https://www.gutenberg.org/ebooks/14529 | Project Gutenberg, public domain in the USA |
| `zupitza_grammar` | train | Ælfric's *Grammar* | Zupitza, 1880 | https://www.gutenberg.org/ebooks/45861 | Project Gutenberg, public domain in the USA |
| `john_ws` | dev | Gospel of John (West Saxon) | Bright & Harris, 1904 | https://en.wikisource.org/wiki/Index:Gospel_of_Saint_John_in_West-Saxon.djvu | Wikisource, CC BY-SA 4.0 |
| `soliloquies` | test_prose | Alfred's *Soliloquies* | Hargrove, 1902 | https://en.wikisource.org/wiki/King_Alfred%27s_Old_English_Version_of_St._Augustine%27s_Soliloquies | Wikisource, CC BY-SA 4.0 |
| `tauber_beowulf` | test_verse | *Beowulf* | Tauber, with the length marks of the Perseus text of Klaeber's 1922 edition | https://github.com/old-english-learner-texts/old-english-texts (`beowulf/raw-data/brunetti-length.txt`) | CC BY-SA 4.0, attribution to James Tauber |
| `hs_beowulf` | agreement | *Beowulf*, *Finnsburh Fragment* | Harrison & Sharp, 1883 | https://www.gutenberg.org/ebooks/9701 | Project Gutenberg, public domain in the USA |
| `wyatt_beowulf` | agreement | *Beowulf*, *Finnsburh Fragment* | Wyatt, 1894 | https://en.wikisource.org/wiki/Beowulf_(Wyatt) | Wikisource, CC BY-SA 4.0 |

---

## 📚 Dictionaries


The folder `lexicons/` holds the candidate spellings that the project's systems choose from. Each file is a CSV with
one spelling with macrons per row (`marked_form`). The two unions add a `sources` column, the space-separated names of
the dictionaries that list the form. The files are copies of `data/lex_*.csv` in the code repository, which reads
them with `--lexicon`. `lexicons_provenance.json` gives the URL, licence and SHA-256 of the three downloaded sources
and the number of forms taken from each.

| File | Forms | Built from | Licence |
|---|---:|---|---|
| `lex_wikipron.csv` | 54,054 | WikiPron Old English (`ang_latn_broad.tsv`), IPA length marks turned into spellings by `oevl.wikipron_to_marked` | CC BY-SA 4.0 (Wiktionary content) |
| `lex_clarkhall.csv` | 34,287 | headwords of Clark Hall's *Concise Anglo-Saxon Dictionary*, TEI conversion by Serg078 | Unlicense, as its public-domain source |
| `lex_wiktionary.csv` | 127,899 | headwords and inflection-table forms of the Old English entries of the English Wiktionary, via kaikki.org | CC BY-SA 4.0 (Wiktionary content) |
| `lex_bt.csv` | 45,187 | headwords of Bosworth–Toller (LINDAT dump of Tichý et al.), acute accents read as macrons | CC BY 4.0 |
| `lex_proposal.csv` | 169,388 | union of Clark Hall, Wiktionary and Bosworth–Toller | CC BY-SA 4.0 |
| `lex_all.csv` | 169,525 | union of the three and WikiPron | CC BY-SA 4.0 |

The report's tables use training forms only and training forms with WikiPron. The three other dictionaries enter
only its coverage comparison.

---

## 🏷️ Ambiguity Classes


`ambiguity_classes.csv` lists the 60 spellings without macrons (`skeleton`) with the most lexicon-ambiguous tokens,
their candidate spellings (`candidates`, separated by `|`) and a class:

- **A, lexical:** two different words (god/gōd, man/mān, is/īs)
- **B, grammatical:** same spelling, different word class or function, both with dictionary backing (þe relative
  particle against þē "thee")
- **C, editorial:** the same word marked long or short by editorial habit, mostly unstressed function words (se/sē)

`label_a` and `label_b` come from two independent LLM annotators, and `final` and `adjudication_note` from an
automated adjudicator. The labels are pending review by the team. The two annotators agree on 58 of the 60 (Cohen's
kappa 0.927), and the final labels give 13 A, 5 B and 42 C. `ambiguity_classes.json` records the class definitions,
the agreement, the adjudication rule and the counting rule. The 60 were ranked over the corpus with Zupitza in
training, and its `d3` block records what changes when Zupitza is left out, as in the final systems.

---

## ▶️ Usage


Loading needs the `datasets` library. While the repository is private, it also needs a Hugging Face token with
access to it.

### 1. Load the corpus

```python
from datasets import load_dataset

corpus = load_dataset("Ace3Z/old-english-vowel-length")  # config "corpus": the five splits
```

### 2. Select the data the way the project scores it

```python
train = corpus["train"].filter(lambda r: r["source"] != "zupitza_grammar")  # decision D3
dev = corpus["dev"].filter(lambda r: not r["leakage_excluded"])
```

### 3. Load a dictionary or the class file

```python
base = "hf://datasets/Ace3Z/old-english-vowel-length"
lex = load_dataset("csv", data_files=f"{base}/lexicons/lex_wikipron.csv", keep_default_na=False)["train"]
classes = load_dataset("csv", data_files=f"{base}/ambiguity_classes.csv", keep_default_na=False)["train"]
```

Keep `keep_default_na=False`: without it the skeleton `nan` (nān, "none") is read as a missing value. The Hub reads
one file format per dataset repository, so the CSV files are plain files here and not configs next to the corpus.

### 4. Rebuild this folder from the code repository

```bash
PYTHONPATH=src python -m oevl.export_hf --out release/hf
```

---

## 🧩 How the Data Was Built


1. **Corpus build** (`src/oevl/build_corpus.py`)
   - Reads Wikisource pages through the MediaWiki API and Project Gutenberg books from their HTML editions
   - Keeps only the Old English text. Notes, glossaries, translations, footnotes and page and line numbers are
     removed, and leftover English and Latin lines and sentences are filtered out. The manifest counts what each
     filter dropped
   - Removes digits, brackets, daggers, asterisks, underscores and middle dots. Sentences end after `.`, `?` or `!`
     inside one paragraph
   - Text is Unicode NFC. Length is shown only by the macron vowels: Zupitza's circumflexes became macrons, and
     palatal dots (ċ ġ), hooks (ę ǫ) and stray accents were dropped
   - Tokens are runs of letters, joined over inner hyphens and apostrophes, which are removed
   - The code repository's `data/manifest.json` records each work's origin and cleaning counts, and for the works
     rebuilt with `oevl.build_corpus` the URL, retrieval date and SHA-256 of every input page

2. **Clean-up** (`src/oevl/clean_corpus.py`)
   - Maps Đ/đ typed for eth to Ð/ð and Cyrillic look-alike letters to Latin ones, and repairs three mistyped words
   - Each change is counted in the manifest

3. **Verse test set** (`src/oevl/experiment.py`, `load_tauber`)
   - `test_verse` comes from Tauber's token file. Palatal dots and hooks are dropped and circumflexes become macrons
   - Rows that hold two or three words are split into tokens, and a sentence ends at a row whose punctuation
     contains `.`, `!` or `?`
   - The file has no running text, so `text` and `text_bare` are the tokens joined by spaces, without punctuation.
     38 of its tokens keep a hyphen and 7 an apostrophe, which tokens in the other files never contain

4. **Leakage audit** (`src/oevl/audit_splits.py`)
   - Compares the word 6-grams (bare, lower-cased, ð folded to þ) of each dev and test set with each training work
   - A dev or test sentence that shares one is kept, with `leakage_excluded: true`

5. **Dictionaries** (`src/oevl/wikipron_to_marked.py`, `src/oevl/build_lexicons.py`)
   - WikiPron's Old English IPA turned into spellings with macrons, and Clark Hall, Wiktionary and Bosworth–Toller
     (see "Dictionaries")

6. **Ambiguity classes** (`ambiguity_classes.csv`, `ambiguity_classes.json`)
   - 60 frequent ambiguous spellings, each labelled lexical, grammatical or editorial (see "Ambiguity classes")

7. **Export** (`src/oevl/export_hf.py`)
   - One JSONL file per split, then the dictionaries, the class files, the data licence file and this card

---

## 🗂️ Data Format


```
old-english-vowel-length/
├── README.md                   this card
├── LICENSE.md                  the licence of every data file
├── media/figures/logos/      UZH logo, white and black
├── train.jsonl                 config corpus: the five training works
├── dev.jsonl                   Gospel of John
├── test_prose.jsonl            Alfred's Soliloquies
├── test_verse.jsonl            Tauber's Beowulf
├── agreement.jsonl             Harrison & Sharp and Wyatt Beowulf
├── ambiguity_classes.csv       60 skeletons, candidates, two LLM labels and the adjudicated class
├── ambiguity_classes.json      class definitions, agreement, counting rule, D3 notes
└── lexicons/
    ├── lex_wikipron.csv
    ├── lex_clarkhall.csv
    ├── lex_wiktionary.csv
    ├── lex_bt.csv
    ├── lex_proposal.csv        Clark Hall + Wiktionary + Bosworth–Toller
    ├── lex_all.csv             the three + WikiPron
    └── lexicons_provenance.json
```

`LICENSE.md` is the code repository's `data/LICENSE.md`, so its paths are those of that folder. Its corpus table
names the files `by_source/*.jsonl`, from which the split files here are built.

---

## 🎯 Intended Use


- Training and evaluating vowel-length (macron) restoration for Old English: from `tokens_bare`, predict `tokens` or
  `length_patterns`. Train on `train`, tune on `dev`, report `test_prose` and `test_verse` separately.
- Measuring editor agreement on *Beowulf* by aligning the two `agreement` editions with each other and with
  `test_verse`.
- Studying how editions of this period mark length.

---

## ⚠️ Limitations


- The marks are nineteenth- and early twentieth-century editorial macronisation, not pronunciation. Manuscripts do not
  mark length. Editors added macrons from their own reconstruction, and they disagree, most of all on unstressed
  function words such as se/sē and þe/þē. A system that matches these marks has learned an editorial convention. The
  data is not a phonetic transcription.
- Every split is one editor's (or one team's) opinion, `test_verse` included. It is not a gold standard of Old English
  length.
- The corpus is small, and West Saxon prose dominates. In `train` the only verse work is the short *Physiologus*.
  Bright's *Reader* and Sweet's *Primer* mix prose and verse.
- The texts are transcriptions and may contain errors the cleaning did not catch. Wikisource lists the scan of the
  Gospel of John as not proofread. The language filter may have dropped some Old English lines or kept a few Latin or
  English words.
- The scripts of the first extraction were not kept. `oevl.build_corpus` rebuilds four works byte for byte and the
  other four with almost all tokens but some sentences different, as the repository README explains.
- The dictionaries descend from the same nineteenth-century philology as the editions, and WikiPron's spellings come
  from a heuristic alignment of the IPA with the spelling.
- The ambiguity classes are LLM labels pending review, and the 60 skeletons were chosen on a corpus that still
  included Zupitza.

---

## 📜 Licence and Attribution


The dataset is released under the Creative Commons Attribution-ShareAlike 4.0 International licence (CC BY-SA 4.0,
https://creativecommons.org/licenses/by-sa/4.0/), because it contains CC BY-SA material. A file that combines several
sources takes the strictest licence among them. `LICENSE.md` gives the licence of every file and what each statement
was checked against. When you use the data, attribute:

- the Wikisource contributors of the transcriptions of Bright's *Reader*, Matthew, John, the *Soliloquies* and
  Wyatt's *Beowulf* (pages linked above).
- James Tauber for `test_verse`. His file builds on Brunetti's analysis of *Beowulf* and the Perseus text, which
  his README describes as CC-licensed.
- for the dictionaries, the Wiktionary contributors (through WikiPron, https://github.com/CUNY-CL/wikipron, and
  kaikki.org, https://kaikki.org/dictionary/Old%20English/, which asks users to cite Ylonen 2022, *Wiktextract*),
  Tichý et al. for Bosworth–Toller (http://hdl.handle.net/11234/1-3532) and Serg078 for the Clark Hall conversion
  (https://github.com/Serg078/Conversion-of-The-Clark-Hall-Dictionary-of-Old-English-to-TEI).

Project Gutenberg lists the editions of Sweet, Cook & Pitman, Zupitza and Harrison & Sharp, and Clark Hall's
dictionary, as public domain in the USA. Their Project Gutenberg header and licence are not part of this dataset.
Check the status of the editions where you are.

---

## 📝 Citation


```bibtex
@misc{tajdini2026oevl,
  author       = {Tajdini, Mahbod and Miranovich, Hanna and Bhadoria, Arjun Singh},
  title        = {Coverage, Context, Convention: Restoring Vowel Length in {Old English}},
  year         = {2026},
  howpublished = {\url{https://huggingface.co/datasets/Ace3Z/old-english-vowel-length}},
  note         = {Course project, Essentials in Text and Speech Processing, University of Zurich}
}
```

If you use `test_verse`, please also cite Tauber's data:

```bibtex
@misc{tauber-old-english-texts,
  author       = {Tauber, James},
  title        = {old-english-texts},
  howpublished = {\url{https://github.com/old-english-learner-texts/old-english-texts}},
  note         = {beowulf/raw-data/brunetti-length.txt, CC BY-SA 4.0}
}
```

---

## 👥 Team

- [Mahbod Tajdini](https://github.com/Ace3Z) ([LinkedIn](https://www.linkedin.com/in/mahbodtajdini/))
- Hanna Miranovich
- [Arjun Singh Bhadoria](https://github.com/ArjunSingh48)

---

<div align="center">

**[Back to top](#coverage-context-convention-restoring-vowel-length-in-old-english)**

</div>
