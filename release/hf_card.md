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
- config_name: default
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

Old English sentences from nineteenth- and early twentieth-century editions, each with the editor's vowel-length
marks (macrons) and without them: $sentences sentences and $tokens tokens. The task it was made for is putting the
macrons back into bare text. It accompanies the course project *What Limits Vowel-Length Restoration in Old
English?* by Mahbod Tajdini, Hanna Miranovich and Arjun Singh Bhadoria (Essentials in Text and Speech Processing,
University of Zurich, autumn 2026). The code that builds and evaluates it is the `oevl` package at
https://github.com/Ace3Z/cl-uzh-26-essentials-code, and `python -m oevl.export_hf` writes this folder.

## Description

One JSON object per line, one sentence per object. Every file has the same fields.

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

## Sources and provenance

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

`oevl.build_corpus` reads Wikisource pages through the MediaWiki API and Project Gutenberg books from their HTML
editions. The repository's `data/manifest.json` records each work's origin and cleaning counts, and for the works rebuilt with
`oevl.build_corpus` the URL, retrieval date and SHA-256 of every input page.

## Construction and cleaning

- Only the Old English text was kept. Notes, glossaries, translations, footnotes and page and line numbers were
  removed, and leftover English and Latin lines and sentences were filtered out. The manifest counts what each filter
  dropped.
- Digits, brackets, daggers, asterisks, underscores and middle dots were removed. Sentences end after `.`, `?` or `!`
  inside one paragraph.
- Text is Unicode NFC. Length is shown only by the macron vowels above: Zupitza's circumflexes became macrons, and
  palatal dots (ċ ġ), hooks (ę ǫ) and stray accents were dropped. `oevl.clean_corpus` also maps Đ/đ typed for eth to
  Ð/ð and Cyrillic look-alike letters to Latin ones, and repairs three mistyped words. Each change is counted in the
  manifest.
- Tokens are runs of letters, joined over inner hyphens and apostrophes, which are removed.
- `test_verse` comes from Tauber's token file. Palatal dots and hooks are dropped and circumflexes become macrons.
  Rows that hold two or three words are split into tokens, and a sentence ends at a row whose punctuation contains
  `.`, `!` or `?`. The file has no running text, so `text` and `text_bare` are the tokens joined by spaces, without
  punctuation. 38 of its tokens keep a hyphen and 7 an apostrophe, which tokens in the other files never contain.

## Splits

$table

The split is by work, never by sentence, and no *Beowulf* text is in `train`. `test_prose` is the prose test set and
`test_verse` the verse test set. `agreement` holds two further editions of *Beowulf* (each with the *Finnsburh Fragment*, 21 sentences and 484 tokens
in all), used only to measure how often two editors mark the same word the same way.

Nothing is left out. A dev or test sentence that shares a word 6-gram (bare, lower-cased, ð folded to þ) with any
training work is kept with `leakage_excluded: true`. The project leaves these sentences out when it scores a system,
so drop them to compare with its numbers.

## Intended use

- Training and evaluating vowel-length (macron) restoration for Old English: from `tokens_bare`, predict `tokens` or
  `length_patterns`. Train on `train`, tune on `dev`, report `test_prose` and `test_verse` separately.
- Measuring editor agreement on *Beowulf* by aligning the two `agreement` editions with each other and with
  `test_verse`.
- Studying how editions of this period mark length.

## Limitations

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

## Licence and attribution

The dataset is released under the Creative Commons Attribution-ShareAlike 4.0 International licence (CC BY-SA 4.0,
https://creativecommons.org/licenses/by-sa/4.0/), because it contains CC BY-SA material. When you use it, attribute:

- the Wikisource contributors of the transcriptions of Bright's *Reader*, Matthew, John, the *Soliloquies* and
  Wyatt's *Beowulf* (pages linked above).
- James Tauber for `test_verse`. His file builds on Brunetti's analysis of *Beowulf* and the Perseus text, which
  his README describes as CC-licensed.

Project Gutenberg lists the editions of Sweet, Cook & Pitman, Zupitza and Harrison & Sharp as public domain in the
USA. Their Project Gutenberg header and licence are not part of this dataset. Check the status of the editions where
you are.

## Citation

```bibtex
@misc{tajdini2026oevl,
  author       = {Tajdini, Mahbod and Miranovich, Hanna and Bhadoria, Arjun Singh},
  title        = {Old English Vowel Length: length-marked {Old English} sentences},
  year         = {2026},
  howpublished = {\url{https://huggingface.co/datasets/oe-vowel-length/old-english-vowel-length}},
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
