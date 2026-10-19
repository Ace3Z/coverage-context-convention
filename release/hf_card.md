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

# Old English Vowel Length

This dataset holds $sentences Old English sentences ($tokens tokens) from nineteenth- and early twentieth-century
editions, each with the editor's long-vowel marks (macrons) and without them. The task is to put the macrons back into
bare text. It also includes the candidate dictionaries and two ambiguity class files from the project *Coverage,
Context, Convention: Restoring Vowel Length in Old English* (University of Zurich, 2026). The code is at
https://github.com/Ace3Z/cl-uzh-26-essentials-code.

Language: Old English (`ang`). Licence: CC BY-SA 4.0.

## Dataset structure

Each line is one JSON object for one sentence.

| Field | Type | Content |
|---|---|---|
| `id` | string | `<source>:<number>`, or `tauber:<number>` in `test_verse` |
| `source` | string | short name of the edition (see the table of sources) |
| `work`, `editor` | string | the work and the editor of the edition |
| `year` | integer | year of the edition (for Harrison & Sharp, the year of the first edition) |
| `genre` | string | `prose`, `verse` or `mixed`, given per work (Bright's *Reader* and Sweet's *Primer* are `mixed`) |
| `section` | string | chapter or Wikisource page the sentence comes from, empty in `test_verse` |
| `text` | string | the sentence with the edition's length marks |
| `text_bare` | string | the same sentence with the length marks removed |
| `tokens` | list of strings | the words of the sentence, with length marks, without punctuation (in `test_verse`, 38 tokens keep a compound hyphen and 7 an apostrophe, as in Tauber's file) |
| `tokens_bare` | list of strings | the same words without length marks |
| `length_patterns` | list of strings | one per token, one digit per vowel letter (a e i o u y æ): `1` if the edition marks it long, `0` if not |
| `leakage_excluded` | boolean | `true` for a dev or test sentence that shares a word 6-gram with a training work |

Length is written only with the macron vowels ā ē ī ō ū ȳ ǣ and their capitals. For example, `gōdne` has the bare
form `godne` and the pattern `10`.

### Splits

$table

Flagged sentences and tokens are those with `leakage_excluded: true`.

The split is by work, never by sentence, and no *Beowulf* text is in `train`. `agreement` holds two more *Beowulf*
editions and is only used to measure how often editors agree. The project's final systems leave `zupitza_grammar` out
of training and skip sentences with `leakage_excluded: true` when scoring.

### Other files

- `lexicons/` has candidate spellings from WikiPron, Clark Hall, Wiktionary and Bosworth-Toller, plus two unions. Each
  CSV has a `marked_form` column, and the unions add a `sources` column.
- `lexicons/lexicons_provenance.json` (Clark Hall, Wiktionary, Bosworth-Toller) and `lexicons/wikipron_provenance.json`
  give the URL, licence and checksum of each dictionary source.
- `ambiguity_classes.csv` lists the 60 most frequent ambiguous spellings, each labelled lexical (A), grammatical (B)
  or editorial (C). Two LLM annotators labelled them and an automated step settled their two disagreements. No
  person has reviewed the labels yet. `ambiguity_classes.json` records the class definitions, the annotator
  agreement and the counting rule.

## Source data

| `source` | Split | Work | Editor, year | URL | Licence of the source |
|---|---|---|---|---|---|
| `bright_reader` | train | *Anglo-Saxon Reader* (selections) | Bright, 1913 | https://en.wikisource.org/wiki/Bright%27s_Anglo-Saxon_Reader | Wikisource, CC BY-SA 4.0 |
| `matthew_ws` | train | Gospel of Matthew (West Saxon) | Bright, 1904 | https://en.wikisource.org/wiki/Evangelium_Secundum_Mattheum:_the_Gospel_of_Saint_Matthew_in_West-Saxon | Wikisource, CC BY-SA 4.0 |
| `sweet_primer` | train | *Anglo-Saxon Primer* (texts) | Sweet, 1905 | https://www.gutenberg.org/ebooks/34316 | Project Gutenberg, public domain in the USA |
| `physiologus` | train | *Old English Physiologus* | Cook & Pitman, 1921 | https://www.gutenberg.org/ebooks/14529 | Project Gutenberg, public domain in the USA |
| `zupitza_grammar` | train | Ælfric's *Grammar* | Zupitza, 1880 | https://www.gutenberg.org/ebooks/45861 | Project Gutenberg, public domain in the USA |
| `john_ws` | dev | Gospel of John (West Saxon) | Bright & Harris, 1904 | https://en.wikisource.org/wiki/Index:Gospel_of_Saint_John_in_West-Saxon.djvu | Wikisource, CC BY-SA 4.0 |
| `soliloquies` | test_prose | Alfred's *Soliloquies* | Hargrove, 1902 | https://en.wikisource.org/wiki/King_Alfred%27s_Old_English_Version_of_St._Augustine%27s_Soliloquies | Wikisource, CC BY-SA 4.0 |
| `tauber_beowulf` | test_verse | *Beowulf* | Tauber, with the length marks of the Perseus text of Klaeber's 1922 edition | https://github.com/old-english-learner-texts/old-english-texts (`beowulf/raw-data/brunetti-length.txt`) | CC BY-SA 4.0, attribution to James Tauber |
| `hs_beowulf` | agreement | *Beowulf*, *Finnsburh Fragment* | Harrison & Sharp, 4th revised edition (first 1883) | https://www.gutenberg.org/ebooks/9701 | Project Gutenberg, public domain in the USA |
| `wyatt_beowulf` | agreement | *Beowulf*, *Finnsburh Fragment* | Wyatt, 1894 | https://en.wikisource.org/wiki/Beowulf_(Wyatt) | Wikisource, CC BY-SA 4.0 |

Each edition's Old English text was extracted, normalised to Unicode NFC with length marked only by macrons
(Zupitza's circumflexes were converted), and split into sentences and tokens. Notes, translations and Latin and
English lines were removed, but a few short English, Latin and markup fragments remain, for example in
`bright_reader:00937` to `00943`.

## Uses

The dataset is for training and evaluating vowel-length restoration in Old English. Train on `train`, tune on `dev`,
and report `test_prose` and `test_verse` separately.

```python
from datasets import load_dataset
corpus = load_dataset("Ace3Z/old-english-vowel-length")
train = corpus["train"].filter(lambda r: r["source"] != "zupitza_grammar")
```

Load the CSV files with `load_dataset("csv", data_files=..., keep_default_na=False)`, so that the spelling `nan` is
not read as a missing value.

## Limitations

- The marks record how editors macronised the text, not how it was pronounced, and editors disagree, most of all on
  unstressed function words such as *ne/nē* and *þe/þē*.
- The corpus is small and mostly West Saxon prose. The verse in `train` is the *Physiologus* and four sections of
  Bright's *Reader*, about 6,300 tokens in all.
- The Harrison & Sharp file is the Project Gutenberg revision of their fourth edition, whose producer changed some
  marks to modern values (for example *ne* to *nē* as a conjunction), so it is not purely the editors' marking.
- The texts are transcriptions and may contain errors. Wikisource does not yet mark the John transcription as fully
  proofread.

## Licence

CC BY-SA 4.0. Please credit the Wikisource contributors (Bright's *Reader*, Matthew, John, the *Soliloquies* and
Wyatt's *Beowulf*), James Tauber (`test_verse`), and, for the dictionaries, WikiPron, Wiktionary via kaikki.org,
Tichý et al. (Bosworth-Toller) and Serg078 (Clark Hall). `LICENSE.md` gives the licence of every file.
