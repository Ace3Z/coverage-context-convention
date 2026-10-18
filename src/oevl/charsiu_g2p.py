"""Error analysis of CharsiuG2P (Zhu et al. 2022): its own Old English benchmark, then our test sets.

Part 1 (needs torch + transformers; about 15 minutes on a laptop CPU):
  python -m oevl.charsiu_g2p benchmark    -> results/charsiu/benchmark.json and benchmark.tsv
It runs the published checkpoint on the repository's 500 Old English test words with the repository's evaluation
settings (prefix "<ang>:", one word per call, 5 beams, the default max_length of 20), then again with max_length=50,
and types every word error: length only (equal once "ː" is removed from both sides), else truncation (decoding hit
max_length before the end-of-sequence byte), else segment.

Part 2, step 1 (needs torch + transformers; a few minutes on a GPU, done once):
  python -m oevl.charsiu_g2p predict      -> results/charsiu/ipa.tsv (one IPA string per test word type)
  python -m oevl.charsiu_g2p predict --tiny   (smoke test, tiny random model, writes ipa_tiny.tsv)
  Both steps cover the experiment's dev set (john_ws) and two test sets, minus the leakage audit's exclusions.
Part 2, step 2 (no neural libraries needed; run_all.sh runs this):
  python -m oevl.charsiu_g2p analyse      -> results/charsiu/metrics.json

The IPA output is turned into a vowel-length pattern with the same aligner we use for WikiPron
(wikipron_to_marked.mark). Each token is then: correct length, wrong length, or not alignable
(a vowel was added, dropped or changed, so the length cannot be read off).
"""
import collections
import csv
import json
import sys
import urllib.request
from pathlib import Path

from oevl.experiment import EXCLUDE_D3, TESTS, build_data, evaluate, heldout, lexicon, pattern
from oevl.wikipron_to_marked import mark

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "charsiu"
MODEL, REVISION = "charsiu/g2p_multilingual_byT5_small_100", "ed5cf99707fb39f5f3ccc461530d56b6fb531fd3"
TOKENIZER, PREFIX = "google/byt5-small", "<ang>:"  # no space: data_utils.py builds '<'+language+'>:'+word
REPO = "https://raw.githubusercontent.com/lingjzhu/CharsiuG2P/0c929390759fb94f8ecdfc05cc0bc5f2ff2dc0f4/"
BENCH = ROOT / "data/charsiu_ang_test.tsv"  # REPO + data/test/ang.tsv
PUBLISHED = {"per": 0.04102693178957966, "wer": 0.208}  # REPO + multilingual_results/multilingual/byt5_small_100, ang


def tests():
    """The experiment's dev and test sets, without the sentences the leakage audit excludes."""
    return {t: heldout(ROOT / "data", t) for t in TESTS}


def load_model(tiny=False):
    import torch
    from transformers import AutoTokenizer, T5Config, T5ForConditionalGeneration
    if tiny:  # smoke test: random tiny model, no download
        from transformers import ByT5Tokenizer
        tok = ByT5Tokenizer()
        model = T5ForConditionalGeneration(T5Config(vocab_size=384, d_model=32, d_ff=64, num_layers=1, num_heads=2, d_kv=16,
                                                    decoder_start_token_id=0, pad_token_id=0, eos_token_id=1)).eval()
    else:
        tok = AutoTokenizer.from_pretrained(TOKENIZER)
        model = T5ForConditionalGeneration.from_pretrained(MODEL, revision=REVISION).eval()
    return tok, model.cuda() if torch.cuda.is_available() else model


def lev(a, b):
    row = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        prev, row[0] = row[0], i
        for j, y in enumerate(b, 1):
            prev, row[j] = row[j], min(row[j] + 1, row[j - 1] + 1, prev + (x != y))
    return row[-1]


def error_type(ref, hyp, truncated):
    if ref == hyp:
        return "correct"
    if ref.replace("ː", "") == hyp.replace("ː", ""):  # before truncation: a complete output can fill max_length exactly
        return "length_only"
    return "truncation" if truncated else "segment"


def score(refs, hyps, types):
    c = collections.Counter(types)
    errors = len(refs) - c["correct"]
    return {"n": len(refs),
            "per": sum(lev(r, h) for r, h in zip(refs, hyps)) / sum(map(len, refs)),  # jiwer cer, as in evaluate.py
            "wer": sum(lev(r.split(), h.split()) for r, h in zip(refs, hyps)) / sum(len(r.split()) for r in refs),
            "word_errors": errors, **{k: c[k] for k in ("length_only", "segment", "truncation")},
            "length_only_share": round(c["length_only"] / errors, 4) if errors else None}


def benchmark():
    import torch
    if not BENCH.exists():
        urllib.request.urlretrieve(REPO + "data/test/ang.tsv", BENCH)
    rows = [line.strip().split("\t") for line in BENCH.read_text(encoding="utf8").splitlines() if line.strip()]
    words, refs = [w for w, _ in rows], [p.split(",")[0] for _, p in rows]  # first variant, as data_utils.py
    tok, model = load_model()
    summary = {"model": MODEL, "revision": REVISION, "tokenizer": TOKENIZER, "repository": REPO, "prefix": PREFIX,
               "num_beams": 5, "published": PUBLISHED}
    cols = {}
    for max_length in (20, 50):  # 20 is the transformers default that evaluate.py runs with
        hyps, types = [], []
        for i, word in enumerate(words):  # one word per call, no padding, as evaluate.py
            enc = tok([PREFIX + word], add_special_tokens=False, return_tensors="pt").to(model.device)
            with torch.no_grad():
                ids = model.generate(enc.input_ids, num_beams=5, max_length=max_length)[0].tolist()
            hyps.append(tok.decode(ids, skip_special_tokens=True))
            types.append(error_type(refs[i], hyps[-1], tok.eos_token_id not in ids))
            print(f"max_length {max_length}: {i + 1}/{len(words)}", end="\r", flush=True)
        summary[f"max_length_{max_length}"] = score(refs, hyps, types)
        cols[f"prediction_max{max_length}"], cols[f"type_max{max_length}"] = hyps, types
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "benchmark.tsv", "w", encoding="utf8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["word", "reference", *cols])
        w.writerows(zip(words, refs, *cols.values()))
    (OUT / "benchmark.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf8")
    print("\n" + json.dumps(summary, ensure_ascii=False, indent=2))


def predict(batch_size=64, tiny=False):
    import torch
    torch.manual_seed(0)
    words = sorted({r["bare"].lower() for rows in tests().values() for r in rows if r["gold_pattern"]})
    tok, model = load_model(tiny)
    if tiny:
        words = words[:200]
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / ("ipa_tiny.tsv" if tiny else "ipa.tsv")  # the smoke test must not replace the GPU output
    with open(path, "w", encoding="utf8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["word", "ipa"])
        for i in range(0, len(words), batch_size):
            chunk = words[i:i + batch_size]
            enc = tok([PREFIX + x for x in chunk], padding=True, add_special_tokens=False, return_tensors="pt").to(model.device)
            with torch.no_grad():
                ids = model.generate(**enc, num_beams=1, max_length=50)  # greedy, as recommended by the authors
            for word, ipa in zip(chunk, tok.batch_decode(ids.tolist(), skip_special_tokens=True)):
                w.writerow([word, ipa])
            print(f"{min(i + batch_size, len(words))}/{len(words)}", end="\r", flush=True)
    print(f"\nwrote {len(words)} predictions to {path}")


def analyse():
    with open(OUT / "ipa.tsv", encoding="utf8") as f:
        ipa = {r["word"]: r["ipa"] for r in csv.DictReader(f, delimiter="\t")}
    charsiu_train = set()
    dict_path = ROOT / "data/charsiu_ang_dict_words.txt"   # words in CharsiuG2P's own Old English dictionary
    if dict_path.exists():
        charsiu_train = set(dict_path.read_text(encoding="utf8").split())
    train, _, _, _ = build_data(ROOT / "data", EXCLUDE_D3)  # populations as in results/wikipron
    lex, counts = lexicon(train, str(ROOT / "data/lex_wikipron.csv"))
    summary = {"model": MODEL, "prefix": PREFIX}
    for test, rows in tests().items():
        preds = {}
        p = ROOT / "results" / "wikipron" / f"{test}_predictions.jsonl"
        with open(p, encoding="utf8") as f:
            for line in f:
                d = json.loads(line)
                preds[d["id"]] = d
        cats, amb_cats, seen_cats = collections.Counter(), collections.Counter(), collections.Counter()
        wrong_len_amb = fixed_by_word = fixed_by_bilateral = 0
        direction = collections.Counter()
        pats = []  # predicted length pattern per token, None when not alignable
        for r in rows:
            m = mark(r["bare"].lower(), ipa.get(r["bare"].lower(), ""))
            pats.append(None if m is None else pattern(m))
            if not r["gold_pattern"]:
                continue
            if m is None:
                cat = "not_alignable"
            elif pattern(m) == r["gold_pattern"]:
                cat = "correct"
            else:
                cat = "wrong_length"
                pm, g = pattern(m), r["gold_pattern"]
                direction["too_long"] += sum(a == "1" and b == "0" for a, b in zip(pm, g))
                direction["too_short"] += sum(a == "0" and b == "1" for a, b in zip(pm, g))
            ambiguous = len(lex.get(r["key"], set())) >= 2
            cats[cat] += 1
            if ambiguous:
                amb_cats[cat] += 1
            if charsiu_train:
                seen_cats[("seen" if r["bare"].lower() in charsiu_train else "unseen", cat)] += 1
            if cat == "wrong_length" and ambiguous:
                wrong_len_amb += 1
                d = preds[r["id"]]
                fixed_by_word += d["word_seed1_prediction"] == r["gold_pattern"]
                fixed_by_bilateral += d["sentence_seed1_prediction"] == r["gold_pattern"]
        n, na = sum(cats.values()), sum(amb_cats.values())
        summary[test] = {
            "tokens_with_vowels": n,
            "all": {k: cats[k] for k in ("correct", "wrong_length", "not_alignable")},
            "all_share": {k: round(cats[k] / n, 4) for k in ("correct", "wrong_length", "not_alignable")},
            "lexicon_ambiguous_n": na,
            "lexicon_ambiguous": {k: amb_cats[k] for k in ("correct", "wrong_length", "not_alignable")},
            "wrong_length_slots": dict(direction),
            "wrong_length_on_ambiguous": wrong_len_amb,
            "of_those_word_model_right": fixed_by_word,
            "of_those_bilateral_model_right": fixed_by_bilateral,
            "by_charsiu_dictionary": {f"{a}/{b}": v for (a, b), v in sorted(seen_cats.items())},
            "experiment_populations": evaluate(rows, pats, lex, counts),  # same n and subsets as experiment.py
        }
    (OUT / "metrics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "analyse"
    if cmd == "predict":
        predict(tiny="--tiny" in sys.argv)
    elif cmd == "benchmark":
        benchmark()
    else:
        analyse()
