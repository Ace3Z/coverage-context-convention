# Error typing guidelines

> Note added in this repository: this is the text given to the two LLM runs that typed the errors, with references to
> internal planning notes removed. The "two annotators" are these two runs, and a third run of the same LLM settled
> their disagreements instead of a joint discussion. A closing section on regenerating the items is left out, because
> its script is not in this repository.

Two annotators type the same 400 errors independently. Cohen's kappa is reported, then the differences are resolved together.

## What is in `items.tsv`

Each row is one token that a system got wrong. That means its length pattern differs from the gold pattern on a lexicon-ambiguous token, a token whose skeleton has two or more length patterns in the lexicon. Items come from the two test sets pooled (`soliloquies` prose, `tauber_beowulf` verse). The dev set `john_ws` is not used. For each system, 200 errors were drawn with `random.Random(1).sample`.

| System | Errors in the pool | Prose / verse in the pool | Prose / verse in the sample | Items |
|---|---:|---|---|---|
| `encoder_sentence`: ByT5 sentence arm, capfull, seed 1 (`exp_grid/capfull/sentence_s1`) | 917 | 468 / 449 | 96 / 104 | 1 to 200 |
| `charsiug2p`: CharsiuG2P, IPA mapped to a length pattern (`results/charsiu/ipa.tsv`, `wikipron_to_marked.mark`) | 2,625 | 1,607 / 1,018 | 111 / 89 | 201 to 400 |

Each system is scored with the lexicon behind its own published numbers. The encoder run left Zupitza out of training (`--exclude-train zupitza_grammar`), so its lexicon is the forms in the other four training works plus WikiPron. CharsiuG2P uses all five training works plus WikiPron, as `charsiu_g2p.analyse` does. CharsiuG2P tokens whose IPA cannot be aligned to the vowel letters are skipped. Among lexicon-ambiguous tokens in these two test sets there are none (`results/charsiu/metrics.json`, `not_alignable` 0). Ten tokens appear in both samples. Type each row on its own.

| Column | Meaning |
|---|---|
| `item` | Row number, used to match the two annotators' files |
| `system`, `set`, `id` | System, test set and token id (`soliloquies:<sentence>:<index>`, `tauber:<sentence>:<index>`) |
| `sentence` | The unmarked sentence the system read, target in `[[ ]]` |
| `gold` | Gold marked form |
| `system_form` | The system's marked form. For the encoder this is its chosen pattern put on the bare word. For CharsiuG2P it is the IPA mapped back to letters (a diphthong's mark sits on its first letter) |
| `charsiug2p_ipa` | CharsiuG2P's raw IPA output. Empty for the encoder |
| `candidates` | Every lexicon pattern for the skeleton, as marked forms |
| `class` | Skeleton label (A, B or C) from `data/ambiguity_classes.csv`, column `final`. Empty when the skeleton is not one of the 60 listed |
| `gold_in_candidates` | `yes` if the gold pattern is among the candidates. `no` means no lexicon-constrained system could get it right |
| `hs_form`, `wyatt_form` | Beowulf only: the Harrison and Sharp (1883) and Wyatt (1894) token at the same position, aligned on bare forms as in `oevl.agreement`. `not_aligned` when the alignment did not reach it. Empty for prose |

## How to label

1. Copy `items.tsv` to `items_<your initials>.tsv` and add two columns at the end, `type` and `note`.
2. Give every row exactly one `type` from the list below. Do not look at the other annotator's file until both are finished.
3. Write a short `note` whenever the call was hard, when you use `other`, or when your call differs from the `class` column.
4. Read the `sentence`. For Beowulf, Tauber's lemma, part of speech, morphology and gloss are in `data/brunetti-length.txt` and settle most homograph and grammatical calls without fluent Old English.

The types overlap, so go through them in this order and stop at the first that applies.

### 1. `ceiling`

`gold_in_candidates` is `no`. The gold reading is not in the lexicon, so the error belongs to the lexicon, not to the choice between candidates. This is mechanical and applies to both systems. For CharsiuG2P, which does not read the lexicon, it still marks a gold reading that no lexicon-constrained system could produce.

### 2. `segment` or `truncation` (CharsiuG2P only)

Read `charsiug2p_ipa` against the written word. Normal Old English pronunciation rules are not segment errors: `θ` for þ or ð, `j` for palatal g, `x` for h, `ɑ` for a, `æ͜ɑ` and `e͜o` for the diphthongs.

- `segment`: the IPA renders a different word from the one written, and the wrong length comes with that change. Examples are a vowel replaced by another vowel, or a consonant added, dropped or changed so that the output matches some other word.
- `truncation`: the IPA stops before the word ends, so final consonants are missing. All vowels must still be present, since tokens with missing vowels were skipped.

If the IPA is faithful apart from the length mark `ː`, use neither. Go on to step 3.

### 3. `gold_in_doubt` (Beowulf only)

`hs_form` or `wyatt_form` carries the same length marks as `system_form` on the target, while the gold does not. Compare vowel by vowel and ignore capitals, þ against ð and hyphens. One of the two editions agreeing with the system is enough. Before you use this label, check in the sentence that the aligned token is the same word in the same place. If the alignment is clearly wrong, ignore it and say so in `note`.

Prose items never get this label, because the data hold no second edition of the *Soliloquies*. If you think a prose gold is wrong, say so in `note` and type the item by steps 4 to 6.

### 4. `homograph` (class A)

The gold and the system reading are two different words with different meanings, listed under different dictionary headwords, and at least one is a content word. Examples from the class file: `god` "god" and `gōd` "good", `is` "is" and `īs` "ice", `ac` "but" and `āc` "oak", `man` and `mān` "crime", `for` and `fōr`.

### 5. `grammatical` (class B)

Same spelling, and the two readings differ in word class, grammatical function or inflection, each with dictionary backing. At least one reading is a function word, a pronoun or an inflected form, and the question the sentence answers is which function the token fills here. Examples from the class file: `þe` relative particle and `þē` "thee", `ne` "not" and `nē` "nor", `ge` and `gē`.

### 6. `editorial` (class C)

The gold and the system reading are the same word with the same function in this sentence. Only the marking differs, because editors or dictionaries mark that word long or short by habit. This is mostly unstressed function words (`se` and `sē`, `he` and `hē`, `þa` and `þā`, `to` and `tō`, `swa` and `swā`). It also covers a content word whose length the dictionaries in the lexicon disagree on.

Decide steps 4 to 6 from the two readings in this sentence. The `class` column labels the skeleton, not the token. When it is empty, or when the sentence shows that this token is a different case from the class label, decide by the definitions above and add a `note`.

### `other`

Use this only when none of the types fits, for example a gold form that is plainly a typo or a normalisation fault. A `note` is required. If `other` passes 5% of your items, raise it with the team before you finish.

## Agreement and resolution

After both files are complete, compute Cohen's kappa on `type` over all 400 items and for each system separately (for example `sklearn.metrics.cohen_kappa_score`). Report the raw agreement next to kappa. Then go through the disagreements together, agree on one type per item, and keep the resolved file as `items_resolved.tsv` with a `resolution_note` column. Report counts per type and system from the resolved file only.
