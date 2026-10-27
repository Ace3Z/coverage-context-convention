"""The proposal's key experiment: fine-tune the same ByT5 checkpoint on identical data once per arm,
seeing only the word, the word with its left context, the whole sentence, or K words either side (localK).

Like the logistic models, it is candidate-constrained: the dictionary (training forms + WikiPron) proposes
the length patterns, the fine-tuned model scores each one (sum of log-probabilities of the pattern string,
e.g. "01", and of its end-of-sequence token), and the best-scoring pattern wins. Only candidates with one digit
per vowel of the word count. A word with one such candidate takes it, and a word with none (an unknown word)
stays all short.

Run on a GPU, and write new runs outside results/encoder/ (scripts/encoder_grid.sh has the report's exact commands):
  python -m oevl.encoder_experiment --out my_runs/encoder   -> my_runs/encoder/metrics.json + predictions
Quick CPU smoke test with a tiny random model (no download, under a minute):
  python -m oevl.encoder_experiment --tiny --out /tmp/enc_test
"""
import argparse
import hashlib
import json
import math
import os
import random
import re
import time
from collections.abc import Collection, Iterable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar

from oevl.experiment import EXCLUDE_D3, TESTS, TRAIN, build_data, flags, heldout, lexicon, load_classes, predict, zeros

if TYPE_CHECKING:
    from transformers import BatchEncoding

    T = TypeVar("T")

ROOT = Path(__file__).resolve().parents[2]

ARMS = ("word", "left", "sentence")
MARK_L, MARK_R = "[[", "]]"


def make_input(r: Mapping[str, Any], arm: str, max_bytes: int) -> str:
    """Text the encoder sees for token r, lower-cased: the bare word alone, or in context with the target in [[ ]].

    r needs "sentence" (the bare tokens) and "index" (the target's position). arm is "word" (the target alone, no
    markers, max_bytes ignored), "left" (all preceding words), "sentence" (all words) or "localK" (at most K words on
    each side). Context is cut to fit max_bytes UTF-8 bytes in all, each side keeping the words nearest the target.
    The room is max_bytes less the marked target and two spaces. The left side may use half of it (rounded down), or
    more when the right side needs less, and all of it for "left". The right side gets what the left leaves. A side
    that was cut is trimmed back to whole words: a word split by the cut is dropped, and so is a whole word whose
    outer edge (its start on the left, its end on the right) falls exactly on the cut. The result exceeds max_bytes
    only when the marked target alone does.
    """
    words, i = [w.lower() for w in r["sentence"]], r["index"]
    target = words[i]
    if arm == "word":
        return target
    core = f"{MARK_L}{target}{MARK_R}"
    k = int(arm[5:]) if arm.startswith("local") else len(words)
    left = " ".join(words[max(0, i - k):i])
    right = " ".join(words[i + 1:i + 1 + k]) if arm != "left" else ""
    room = max(0, max_bytes - len(core.encode()) - 2)
    lb, rb = left.encode(), right.encode()
    keep_l = min(len(lb), room) if arm == "left" else min(len(lb), max(room // 2, room - len(rb)))
    keep_r = min(len(rb), room - keep_l)  # unused space on one side goes to the other
    l_txt, r_txt = lb[len(lb) - keep_l:].decode(errors="ignore"), rb[:keep_r].decode(errors="ignore")
    if keep_l < len(lb):
        l_txt = l_txt.partition(" ")[2]  # drop the outermost word, which the cut may have split
    if keep_r < len(rb):
        r_txt = r_txt.rpartition(" ")[0]
    return " ".join(x for x in (l_txt, core, r_txt) if x)


def limit(examples: "list[T]", n: int | None) -> "list[T]":
    """A fixed subset of n examples, kept in their order in examples. It depends only on n and the examples list,
    never on --seed, --arms or --max-bytes, so every arm of a pilot trains on the same examples. With n None or at
    least len(examples) the list itself is returned."""
    if n is None or n >= len(examples):
        return examples
    keep = sorted(random.Random(0).sample(range(len(examples)), n))
    return [examples[j] for j in keep]


def candidates(r: Mapping[str, Any], lex: Mapping[str, Collection[str]]) -> tuple[list[str], str]:
    """The sorted lexicon patterns of r's key with one digit per vowel of the word, and the all-short fallback.

    Returns (candidates, fallback). Only the vowel count of r["gold"] is read, never its macrons, and it equals the bare
    word's. A key keeps every vowel letter, so all its lexicon patterns already have that length and the filter is a
    safeguard that removes nothing."""
    fallback = zeros(r["gold"])
    return sorted(p for p in lex.get(r["key"], set()) if len(p) == len(fallback)), fallback


def score(rows: Iterable[Mapping[str, Any]], preds: Iterable[str], prior: Iterable[str],
          lex: Mapping[str, Collection[str]], counts: Mapping[str, Mapping[str, int]],
          cls: Mapping[str, str]) -> dict[str, dict[str, int | float | None] | float | None]:
    """Accuracy per population (the classes split the lexicon-ambiguous tokens, as in experiment.evaluate), plus two
    checks for a collapsed run on the lexicon-ambiguous tokens: the share whose pick has the most long marks among
    its candidates, and the share whose pick equals the frequency prior.

    rows, preds and prior run in parallel (one predicted and one prior pattern per token), lex and counts come from
    experiment.lexicon, and cls maps a key to its class. Returns {population: {"n", "accuracy"}} for accuracy (all
    tokens), lexicon_ambiguous, attested_ambiguous, minority_reading and class_A, class_B, class_C and class_unlisted,
    each accuracy a fraction rounded to six places or None for an empty population, plus most_long_share and
    prior_agreement (None without lexicon-ambiguous tokens). A pick tied for the most long marks counts as most long,
    and a token without candidates of the right length has its pick compared with itself, so it counts too."""
    pops: dict[str, list[bool]]
    pops = {k: [] for k in ("accuracy", "lexicon_ambiguous", "attested_ambiguous", "minority_reading",
                            "class_A", "class_B", "class_C", "class_unlisted")}
    most_long, same = [], []
    for r, p, f in zip(rows, preds, prior):
        ok = p == r["gold_pattern"]
        pops["accuracy"].append(ok)
        fl = flags(r, lex, counts)
        for name, on in fl.items():
            if on:
                pops[name].append(ok)
        if fl["lexicon_ambiguous"]:
            pops["class_" + cls.get(r["key"], "unlisted")].append(ok)
            opts = candidates(r, lex)[0] or [p]
            most_long.append(p.count("1") == max(o.count("1") for o in opts))
            same.append(p == f)
    out: dict[str, dict[str, int | float | None] | float | None]
    out = {k: {"n": len(v), "accuracy": round(sum(v) / len(v), 6) if v else None} for k, v in pops.items()}
    out["most_long_share"] = round(sum(most_long) / len(most_long), 6) if most_long else None
    out["prior_agreement"] = round(sum(same) / len(same), 6) if same else None
    return out


def main() -> None:
    """Fine-tune and score a fresh model for each arm, then write metrics.json and one predictions file per test set.

    Reads the training works under data/by_source minus --exclude-train (build_data also loads the soliloquies and
    the two Beowulf editions, unused here), the --tests sets without their leaked sentences (experiment.heldout),
    data/lex_wikipron.csv and data/ambiguity_classes.csv. The training examples are the training tokens with at least
    one vowel and at least two lexicon patterns (with --tiny, 256 of them after a Random(0) shuffle, and the first 300
    rows of each test set), cut by --train-limit and visited in one --seed order that is the same for every arm and
    every epoch. Each arm starts from the --model checkpoint (or a tiny random T5) under torch.manual_seed(--seed) and
    trains with AdamW, a linear schedule with 5% warm-up and gradient clipping at 1.0, one step per --batch examples
    fed --micro at a time. A test token with two or more candidates of the right length gets the best-scoring one (an
    exact tie goes to the larger pattern string), any other its one candidate or all short.
    --out/metrics.json (settings, losses, score() per test set and the collapse flag of each arm) is rewritten after
    every arm. --out/<test>_predictions.jsonl (gold, frequency prior, flags, class and an encoder_<arm> pattern per
    arm) is written only after the last arm.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/byt5-small")
    ap.add_argument("--out", default=str(ROOT / "results" / "encoder"))
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch", type=int, default=32, help="examples per optimizer step (same for every arm)")
    ap.add_argument("--micro", type=int, default=8, help="examples per forward pass; lower it if the GPU runs out of memory")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--max-bytes", type=int, default=256)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--arms", default=",".join(ARMS), help="comma list of word, left, sentence, localK (K words each side)")
    ap.add_argument("--train-limit", type=int, help="train on a fixed subset of this many examples (see limit())")
    ap.add_argument("--tests", default=",".join(TESTS), help="comma list of john_ws (dev), soliloquies, tauber_beowulf")
    ap.add_argument("--exclude-train", default=",".join(EXCLUDE_D3), help="comma list of training works to leave out: "
                    "they leave the training set, the lexicon's training forms and the prior. Default zupitza_grammar "
                    "(decision D3, as run_all.sh). Pass an empty string to train on all five works")
    ap.add_argument("--tiny", action="store_true", help="tiny random model + small data subset (CPU smoke test)")
    a = ap.parse_args()
    a.arms = ",".join(dict.fromkeys(a.arms.split(",")))
    bad = [x for x in a.arms.split(",") if x not in ARMS and not re.fullmatch(r"local[1-9][0-9]*", x)]
    if bad:
        ap.error(f"unknown arm(s) {bad}: use word, left, sentence or localK")
    if a.train_limit is not None and a.train_limit < 1:
        ap.error("--train-limit must be at least 1")
    if set(a.tests.split(",")) - set(TESTS):
        ap.error(f"unknown --tests {sorted(set(a.tests.split(',')) - set(TESTS))}: use {','.join(TESTS)}")
    exclude = [x for x in a.exclude_train.split(",") if x]
    if set(exclude) - set(TRAIN):
        ap.error(f"unknown --exclude-train {sorted(set(exclude) - set(TRAIN))}: use {','.join(TRAIN)}")

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    import torch
    from transformers import AutoTokenizer, T5Config, T5ForConditionalGeneration, get_linear_schedule_with_warmup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    train, _, _, _ = build_data(ROOT / "data", exclude)
    tests = {t: heldout(ROOT / "data", t) for t in a.tests.split(",")}
    lex, counts = lexicon(train, str(ROOT / "data/lex_wikipron.csv"))
    examples = [r for r in train if r["gold_pattern"] and len(lex.get(r["key"], set())) >= 2]
    if a.tiny:
        random.Random(0).shuffle(examples)
        examples = examples[:256]
        tests = {k: v[:300] for k, v in tests.items()}
    prior = {t: predict(train, rows, lex, counts, "frequency") for t, rows in tests.items()}  # follows --exclude-train
    cls = load_classes(ROOT / "data" / "ambiguity_classes.csv")
    examples = limit(examples, a.train_limit)
    order = list(range(len(examples)))
    random.Random(a.seed).shuffle(order)  # the same order for every arm: identical data, identical schedule

    if a.tiny:
        from transformers import ByT5Tokenizer
        tok = ByT5Tokenizer()
        cfg = T5Config(vocab_size=384, d_model=64, d_ff=128, num_layers=2, num_decoder_layers=1, num_heads=2,
                       d_kv=32, decoder_start_token_id=0, pad_token_id=0, eos_token_id=1)
    else:
        tok = AutoTokenizer.from_pretrained(a.model)
    print(f"device={device} model={'tiny-random' if a.tiny else a.model} train_examples={len(examples)}", flush=True)

    def enc(texts: list[str], n: int) -> "BatchEncoding":
        """Tokenise texts as one padded batch on the device, each at most n tokens (ByT5: one per byte, plus EOS)."""
        return tok(texts, padding=True, truncation=True, max_length=n, return_tensors="pt").to(device)

    summary = {"model": "tiny-random" if a.tiny else a.model, "lexicon": "training + WikiPron", "seed": a.seed,
               "exclude_train": exclude,
               "epochs": a.epochs, "batch": a.batch, "lr": a.lr, "max_bytes": a.max_bytes,
               "train_examples": len(examples), "train_limit": a.train_limit,
               "train_subset_sha256": hashlib.sha256("\n".join(r["id"] for r in examples).encode()).hexdigest(),
               "arms": {}}
    all_preds: dict[str, dict[str, dict[str, str]]] = {t: {r["id"]: {} for r in rows} for t, rows in tests.items()}
    for arm in a.arms.split(","):
        torch.manual_seed(a.seed)
        model = (T5ForConditionalGeneration(cfg) if a.tiny else T5ForConditionalGeneration.from_pretrained(a.model)).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=a.lr)
        steps = a.epochs * math.ceil(len(order) / a.batch)
        sched = get_linear_schedule_with_warmup(opt, int(0.05 * steps), steps)
        model.train()
        t0, step, epoch_losses = time.time(), 0, []
        for e in range(a.epochs):
            epoch_loss = 0.0
            for b in range(0, len(order), a.batch):
                batch = [examples[j] for j in order[b:b + a.batch]]
                loss_sum = 0.0
                for m in range(0, len(batch), a.micro):  # gradient accumulation: one optimizer step per batch
                    part = batch[m:m + a.micro]
                    x = enc([make_input(r, arm, a.max_bytes) for r in part], a.max_bytes + 8)
                    y = enc([r["gold_pattern"] for r in part], 32).input_ids
                    y[y == tok.pad_token_id] = -100
                    loss = model(**x, labels=y).loss * len(part) / len(batch)
                    loss.backward()
                    loss_sum += loss.item()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()
                opt.zero_grad()
                step += 1
                epoch_loss += loss_sum
                if step % 100 == 0 or step == steps:
                    print(f"[{arm}] step {step}/{steps} loss {loss_sum:.4f} ({time.time() - t0:.0f}s)", flush=True)
            epoch_losses.append(round(epoch_loss / math.ceil(len(order) / a.batch), 6))
            print(f"[{arm}] epoch {e + 1}/{a.epochs} mean training loss {epoch_losses[-1]:.4f}", flush=True)
        train_seconds = time.time() - t0

        model.eval()
        # final_train_loss: mean step loss over the last epoch
        arm_summary = {"train_seconds": round(train_seconds), "train_loss_per_epoch": epoch_losses,
                       "final_train_loss": epoch_losses[-1]}
        for test, rows in tests.items():
            pairs = []  # (row index, candidate) for tokens the model must decide
            preds: list[str | None] = []
            for k, r in enumerate(rows):
                opts, fallback = candidates(r, lex)
                if len(opts) >= 2:
                    pairs += [(k, p) for p in opts]
                    preds.append(None)
                else:
                    preds.append(opts[0] if opts else fallback)
            scores: dict[int, list[tuple[float, str]]] = {}
            with torch.no_grad():
                for b in range(0, len(pairs), 4 * a.micro):
                    chunk = pairs[b:b + 4 * a.micro]
                    x = enc([make_input(rows[k], arm, a.max_bytes) for k, _ in chunk], a.max_bytes + 8)
                    y = enc([p for _, p in chunk], 32).input_ids
                    logits = model(**x, labels=y).logits
                    lp = torch.log_softmax(logits, -1).gather(-1, y.unsqueeze(-1)).squeeze(-1)
                    lp = (lp * (y != tok.pad_token_id)).sum(-1)
                    for (k, p), s in zip(chunk, lp.tolist()):
                        scores.setdefault(k, []).append((s, p))
            for k, sc in scores.items():
                preds[k] = max(sc)[1]
            for r, p in zip(rows, preds):
                all_preds[test][r["id"]][f"encoder_{arm}"] = p
            arm_summary[test] = score(rows, preds, prior[test], lex, counts, cls)
            print(f"[{arm}] {test}: " + ", ".join(f"{n}={d['accuracy'] if isinstance(d, dict) else d}"
                                                   for n, d in arm_summary[test].items()), flush=True)
        # A run that collapsed to "most long marks" (seen at lr 3e-4) is flagged on the prose test set:
        # most_long_share of at least 0.9 on soliloquies, None when --tests leaves soliloquies out
        sol = arm_summary.get("soliloquies", {}).get("most_long_share")
        arm_summary["collapsed"] = None if sol is None else sol >= 0.9
        summary["arms"][arm] = arm_summary
        (out / "metrics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf8")  # save after each arm
        del model, opt
        if device == "cuda":
            torch.cuda.empty_cache()

    for test, rows in tests.items():  # paired tests: oevl.encoder_summary, from these files
        with open(out / f"{test}_predictions.jsonl", "w", encoding="utf8") as f:
            for r, fp in zip(rows, prior[test]):
                fl = flags(r, lex, counts)
                f.write(json.dumps({"id": r["id"], "sentence_id": r["sentence_id"], "gold_pattern": r["gold_pattern"],
                                    "frequency": fp, **fl,
                                    "class": cls.get(r["key"], "unlisted") if fl["lexicon_ambiguous"] else None,
                                    **all_preds[test][r["id"]]}, ensure_ascii=False) + "\n")
    (out / "metrics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf8")
    print(f"wrote {out / 'metrics.json'}")


if __name__ == "__main__":
    main()
