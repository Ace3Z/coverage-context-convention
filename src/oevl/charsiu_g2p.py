"""Error analysis of CharsiuG2P (Zhu et al. 2022) on our two test sets.

Step 1 (needs torch + transformers; a few minutes on a Colab GPU, done once):
  python -m oevl.charsiu_g2p predict      -> results/charsiu/ipa.tsv (one IPA string per test word type)
  python -m oevl.charsiu_g2p predict --tiny   (smoke test, tiny random model, writes ipa_tiny.tsv)
Step 2 (no neural libraries needed; run_all.sh runs this):
  python -m oevl.charsiu_g2p analyse      -> results/charsiu/metrics.json

The IPA output is turned into a vowel-length pattern with the same aligner we use for WikiPron
(wikipron_to_marked.mark). Each token is then: correct length, wrong length, or not alignable
(a vowel was added, dropped or changed, so the length cannot be read off).
"""
import collections
import csv
import json
import sys
from pathlib import Path

from oevl.experiment import build_data, lexicon, load_tauber, pattern
from oevl.wikipron_to_marked import mark

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "charsiu"
MODEL, TOKENIZER, PREFIX = "charsiu/g2p_multilingual_byT5_small_100", "google/byt5-small", "<ang>:"  # as data_utils.py


def tests():
    _, prose, _, _ = build_data(ROOT / "data")
    return {"soliloquies": prose, "tauber_beowulf": load_tauber(ROOT / "data/brunetti-length.txt")}


def predict(batch_size=64, tiny=False):
    import torch
    from transformers import AutoTokenizer, T5Config, T5ForConditionalGeneration
    torch.manual_seed(0)
    words = sorted({r["bare"].lower() for rows in tests().values() for r in rows if r["gold_pattern"]})
    if tiny:  # smoke test: random tiny model, 200 words, no download
        from transformers import ByT5Tokenizer
        words, tok = words[:200], ByT5Tokenizer()
        model = T5ForConditionalGeneration(T5Config(vocab_size=384, d_model=32, d_ff=64, num_layers=1, num_heads=2, d_kv=16,
                                                    decoder_start_token_id=0, pad_token_id=0, eos_token_id=1)).eval()
    else:
        tok = AutoTokenizer.from_pretrained(TOKENIZER)
        model = T5ForConditionalGeneration.from_pretrained(MODEL).eval()
    if torch.cuda.is_available():
        model = model.cuda()
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
    train, _, _, _ = build_data(ROOT / "data")
    lex, _ = lexicon(train, str(ROOT / "data/lex_wikipron.csv"))
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
        for r in rows:
            if not r["gold_pattern"]:
                continue
            m = mark(r["bare"].lower(), ipa.get(r["bare"].lower(), ""))
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
        }
    (OUT / "metrics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "analyse"
    if cmd == "predict":
        predict(tiny="--tiny" in sys.argv)
    else:
        analyse()
