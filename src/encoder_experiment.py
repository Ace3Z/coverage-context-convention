"""The proposal's key experiment: fine-tune the same ByT5 checkpoint on identical data three times,
seeing only the word, the word with its left context, or the whole sentence.

Like the logistic models, it is candidate-constrained: the dictionary (training forms + WikiPron) proposes
the length patterns, the fine-tuned model scores each one (sum of log-probabilities of the pattern string,
e.g. "01"), and the best-scoring pattern wins. Words with one candidate take it; unknown words stay short.

Run on a GPU (free Colab T4: about 1 hour for all three):
  python src/encoder_experiment.py                    -> results_encoder/metrics.json + predictions
Quick CPU smoke test with a tiny random model (no download, a few minutes):
  python src/encoder_experiment.py --tiny --out /tmp/enc_test
"""
import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from experiment import build_data, flags, lexicon, load_tauber, mcnemar, zeros  # noqa: E402

ARMS = ("word", "left", "sentence")
MARK_L, MARK_R = "[[", "]]"


def make_input(r, arm, max_bytes):
    """Text the encoder sees for token r. The target word is wrapped in [[ ]] when context is shown."""
    words, i = [w.lower() for w in r["sentence"]], r["index"]
    target = words[i]
    if arm == "word":
        return target
    core = f"{MARK_L}{target}{MARK_R}"
    left = " ".join(words[:i])
    right = " ".join(words[i + 1:]) if arm == "sentence" else ""
    room = max(0, max_bytes - len(core.encode()) - 2)
    lb, rb = left.encode(), right.encode()
    keep_l = min(len(lb), room) if arm == "left" else min(len(lb), max(room // 2, room - len(rb)))
    keep_r = min(len(rb), room - keep_l)  # unused space on one side goes to the other
    l_txt, r_txt = lb[len(lb) - keep_l:].decode(errors="ignore"), rb[:keep_r].decode(errors="ignore")
    if keep_l < len(lb):
        l_txt = l_txt.partition(" ")[2]  # drop the word that was cut in half
    if keep_r < len(rb):
        r_txt = r_txt.rpartition(" ")[0]
    return " ".join(x for x in (l_txt, core, r_txt) if x)


def candidates(r, lex):
    fallback = zeros(r["gold"])
    return sorted(p for p in lex.get(r["key"], set()) if len(p) == len(fallback)), fallback


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/byt5-small")
    ap.add_argument("--out", default=str(ROOT / "results_encoder"))
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--max-bytes", type=int, default=256)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--tiny", action="store_true", help="tiny random model + small data subset (CPU smoke test)")
    a = ap.parse_args()

    import torch
    from transformers import AutoTokenizer, T5Config, T5ForConditionalGeneration, get_linear_schedule_with_warmup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    train, prose, _, _ = build_data(ROOT / "data")
    tests = {"soliloquies": prose, "tauber_beowulf": load_tauber(ROOT / "data/brunetti-length.txt")}
    lex, counts = lexicon(train, str(ROOT / "data/lex_wikipron.csv"))
    examples = [r for r in train if r["gold_pattern"] and len(lex.get(r["key"], set())) >= 2]
    if a.tiny:
        random.Random(0).shuffle(examples)
        examples = examples[:256]
        tests = {k: v[:300] for k, v in tests.items()}
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

    freq = {}
    for test in tests:
        with open(ROOT / "results_wikipron" / f"{test}_predictions.jsonl", encoding="utf8") as f:
            for line in f:
                d = json.loads(line)
                freq[d["id"]] = d

    def enc(texts, n):
        return tok(texts, padding=True, truncation=True, max_length=n, return_tensors="pt").to(device)

    summary = {"model": "tiny-random" if a.tiny else a.model, "lexicon": "training + WikiPron", "seed": a.seed,
               "epochs": a.epochs, "batch": a.batch, "lr": a.lr, "max_bytes": a.max_bytes,
               "train_examples": len(examples), "arms": {}}
    all_preds = {t: {r["id"]: {} for r in rows} for t, rows in tests.items()}
    for arm in a.arms.split(","):
        torch.manual_seed(a.seed)
        model = (T5ForConditionalGeneration(cfg) if a.tiny else T5ForConditionalGeneration.from_pretrained(a.model)).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=a.lr)
        steps = a.epochs * math.ceil(len(order) / a.batch)
        sched = get_linear_schedule_with_warmup(opt, int(0.05 * steps), steps)
        model.train()
        t0, step = time.time(), 0
        for _ in range(a.epochs):
            for b in range(0, len(order), a.batch):
                batch = [examples[j] for j in order[b:b + a.batch]]
                x = enc([make_input(r, arm, a.max_bytes) for r in batch], a.max_bytes + 8)
                y = enc([r["gold_pattern"] for r in batch], 32).input_ids
                y[y == tok.pad_token_id] = -100
                loss = model(**x, labels=y).loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()
                opt.zero_grad()
                step += 1
                if step % 100 == 0 or step == steps:
                    print(f"[{arm}] step {step}/{steps} loss {loss.item():.4f} ({time.time() - t0:.0f}s)", flush=True)
        train_seconds = time.time() - t0

        model.eval()
        arm_summary = {"train_seconds": round(train_seconds)}
        for test, rows in tests.items():
            pairs = []  # (row index, candidate) for tokens the model must decide
            preds = []
            for k, r in enumerate(rows):
                opts, fallback = candidates(r, lex)
                if len(opts) >= 2:
                    pairs += [(k, p) for p in opts]
                    preds.append(None)
                else:
                    preds.append(opts[0] if opts else fallback)
            scores = {}
            with torch.no_grad():
                for b in range(0, len(pairs), 128):
                    chunk = pairs[b:b + 128]
                    x = enc([make_input(rows[k], arm, a.max_bytes) for k, _ in chunk], a.max_bytes + 8)
                    y = enc([p for _, p in chunk], 32).input_ids
                    logits = model(**x, labels=y).logits
                    lp = torch.log_softmax(logits, -1).gather(-1, y.unsqueeze(-1)).squeeze(-1)
                    lp = (lp * (y != tok.pad_token_id)).sum(-1)
                    for (k, p), s in zip(chunk, lp.tolist()):
                        scores.setdefault(k, []).append((s, p))
            for k, sc in scores.items():
                preds[k] = max(sc)[1]
            pops = {"accuracy": [], "lexicon_ambiguous": [], "attested_ambiguous": [], "minority_reading": []}
            for r, p in zip(rows, preds):
                ok = p == r["gold_pattern"]
                pops["accuracy"].append(ok)
                for name, on in flags(r, lex, counts).items():
                    if on:
                        pops[name].append(ok)
                all_preds[test][r["id"]][f"encoder_{arm}"] = p
            arm_summary[test] = {name: {"n": len(v), "accuracy": round(sum(v) / len(v), 6) if v else None}
                                 for name, v in pops.items()}
            print(f"[{arm}] {test}: " + ", ".join(f"{n}={d['accuracy']}" for n, d in arm_summary[test].items()), flush=True)
        summary["arms"][arm] = arm_summary
        del model, opt
        if device == "cuda":
            torch.cuda.empty_cache()

    summary["mcnemar"] = {}
    for test, rows in tests.items():
        gold = [r["gold_pattern"] for r in rows]
        cols = {"frequency": [freq[r["id"]]["frequency_prediction"] for r in rows],
                "logistic_word": [freq[r["id"]]["word_seed1_prediction"] for r in rows],
                "logistic_bilateral": [freq[r["id"]]["sentence_seed1_prediction"] for r in rows]}
        for arm in summary["arms"]:
            cols[f"encoder_{arm}"] = [all_preds[test][r["id"]][f"encoder_{arm}"] for r in rows]
        pairs = [("frequency", f"encoder_{x}") for x in summary["arms"]]
        pairs += [("encoder_word", f"encoder_{x}") for x in summary["arms"] if x != "word"]
        pairs += [("logistic_word", "encoder_word"), ("logistic_bilateral", "encoder_sentence")]
        summary["mcnemar"][test] = {f"{p}_vs_{q}": mcnemar(cols[p], cols[q], gold)
                                    for p, q in pairs if p in cols and q in cols}
        with open(out / f"{test}_predictions.jsonl", "w", encoding="utf8") as f:
            for r in rows:
                f.write(json.dumps({"id": r["id"], "gold_pattern": r["gold_pattern"], **all_preds[test][r["id"]]},
                                   ensure_ascii=False) + "\n")
    (out / "metrics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf8")
    print(f"wrote {out / 'metrics.json'}")


if __name__ == "__main__":
    main()
