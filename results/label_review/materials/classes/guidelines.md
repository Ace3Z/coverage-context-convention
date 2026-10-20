# Ambiguity classes: annotation guidelines

> Note added in this repository: this is the text given to the two runs of one LLM, Claude Opus 5.5 (model id
> `claude-opus-5-5`), that labelled the classes, with two references to internal planning notes removed. "The other
> annotator" is the other run, and a third run of the same LLM settled their disagreements instead of a joint
> discussion. The item sheet was made from the corpus and dictionaries of commit f9deda8, before later fixes to both,
> and uses all five training works (Zupitza included), so some of its counts and sources differ from the current
> `data/`. The `clarkhall` match ignored case and a final hyphen, so prefix entries such as `ge-` are included, and the
> value for `to` and part of the value for `for` are lines from the Project Gutenberg front matter and licence.

Give each skeleton in `items.tsv` exactly one class, A, B or C. Work alone and do not
look at the other annotator's labels. The two label sets are compared with Cohen's
kappa and the differences are then resolved together.

## Class definitions

| Class | Definition | Examples |
|---|---|---|
| A, lexical | Two different words | `god` and `gōd`, `man` and `mān`, `is` and `īs` |
| B, grammatical | Same spelling, different word class or function, both with dictionary backing | `þe` relative particle and `þē` "thee" |
| C, editorial | Same word, marked long or short by editorial habit, mostly unstressed function words | `se` and `sē` |

Class C is the function-word category the proposal promises to report separately.

## Columns in items.tsv

- `rank`, `skeleton`: the bare form (lower case, `ð` written `þ`, no hyphens) and its
  frequency rank.
- `tokens_counted`: tokens with this skeleton in the training text plus both test sets
  (Soliloquies and Tauber's Beowulf).
- `n_readings`: number of candidate length patterns in the lexicon (training forms plus
  WikiPron).
- For each reading `rN`:
  - `form` and `pattern`: the length-marked form and its pattern, one digit per vowel,
    1 = long.
  - `train_tokens`: how often the training text writes the skeleton with this pattern.
  - `sources`: which lists attest this pattern: the training text, WikiPron, Clark Hall,
    Wiktionary, Bosworth-Toller.
  - `tauber_lemma_pos_gloss`: lemma, part of speech and gloss from Tauber's Beowulf for
    tokens he marks this way, with counts (up to three).
  - `clarkhall`: the start of any Clark Hall entry whose headword is exactly this form.
  - `examples`: up to three passages where an edition marks the word this way. The word
    is in «guillemets» and the source is in brackets. Fewer than three means the corpus
    has fewer.

Tauber's part-of-speech codes as they appear in the data: v verb, m f n noun (by
gender), a adjective, p pronoun, d demonstrative or article, r relative particle,
av adverb, pp preposition, c conjunction, np proper name, nu numeral, e interjection.
