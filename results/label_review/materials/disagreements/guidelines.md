# Labelling editor disagreements (Harrison & Sharp vs Wyatt, Beowulf)

> Note added in this repository: this is the text given to the single run of one LLM, Claude Opus 5.5 (model id
> `claude-opus-5-5`), that labelled these 100 disagreements. Its labels and reasons are the `label` and `note` columns
> of `results/agreement/disagreements_checked.tsv`, where a later blind review changed four labels.

Each row is one aligned token where the two editions mark vowel length differently: `a_form` and `a_pattern` from
Harrison & Sharp (Project Gutenberg revision of their 4th edition), `b_form` and `b_pattern` from Wyatt (1894,
Wikisource), with each edition's context. Patterns have one digit per vowel letter (a e i o u y æ): 1 long, 0 short.
Give each row exactly one label:

- `genuine editorial difference`: both editions print the same word in the same place, and they really mark its
  vowel length differently (a deliberate editorial choice, including a different reading of which word it is).
- `normalisation fault`: the difference was created by our processing, not by the editors, for example a macron
  moved to the other letter of a digraph (eā vs ēa, iō vs īo) or a character mapping, so the editions actually agree.
- `alignment fault`: the two aligned tokens are not the same word in the same place.
- `other`: anything else, for example a probable printing or transcription slip in one edition (say which and why).

Write a one-line reason with your evidence (grammar, dictionary entry, the line in context).
