"""Write the corpus as a Hugging Face dataset folder. Local only: nothing is uploaded (README, "Publishing the dataset").

One JSONL file per split, one sentence per line, all with the same fields:
  train       the five training works (experiment.TRAIN)   dev         Gospel of John
  test_prose  Alfred's Soliloquies                         test_verse  Tauber's Beowulf, in the corpus schema
  agreement   the Harrison & Sharp and Wyatt Beowulf, used only for editor agreement
Every sentence is written. The dev and test sentences that the 6-gram audit leaves out of evaluation
(data/leakage_excluded.json) are kept with leakage_excluded true. There is no split field: the file is the split.
Tauber's file has no running text, so a test_verse sentence's text is its tokens joined by spaces.
README.md is the card release/hf_card.md with the split table counted from the written files.
Usage: python -m oevl.export_hf [--data data] [--tauber data/brunetti-length.txt] [--out release/hf]
"""
import argparse
import itertools
import json
from pathlib import Path
from string import Template

from oevl.experiment import TRAIN, load, load_tauber, pattern

ROOT = Path(__file__).resolve().parents[2]
SPLITS = {"train": TRAIN, "dev": ["john_ws"], "test_prose": ["soliloquies"], "test_verse": ["tauber_beowulf"],
          "agreement": ["hs_beowulf", "wyatt_beowulf"]}
META = ("source", "work", "editor", "year", "genre", "section")
TAUBER = {"source": "tauber_beowulf", "work": "Beowulf", "editor": "Tauber (Klaeber 1922 via Perseus)", "year": 1922,
          "genre": "verse", "section": ""}


def tauber(path):
    """Tauber's Beowulf as sentence records, with the sentences and tokens of experiment.load_tauber."""
    out = []
    for sid, rows in itertools.groupby(load_tauber(path), key=lambda r: r["sentence_id"]):
        rows = list(rows)
        out.append({"id": sid, **TAUBER, "text": " ".join(r["gold"] for r in rows),
                    "text_bare": " ".join(r["bare"] for r in rows),
                    "tokens": [r["gold"] for r in rows], "tokens_bare": [r["bare"] for r in rows]})
    return out


def record(r, excluded):
    """The released fields only, so no stale field (such as an old split) can leak into the files."""
    return {"id": r["id"], **{k: r[k] for k in META}, "text": r["text"], "text_bare": r["text_bare"],
            "tokens": r["tokens"], "tokens_bare": r["tokens_bare"],
            "length_patterns": [pattern(t) for t in r["tokens"]], "leakage_excluded": r["id"] in excluded}


def export(data, tauber_path, out):
    """Write <out>/<split>.jsonl and <out>/README.md; return the counts read back from the files."""
    data, out = Path(data), Path(out)
    audit = json.loads((data / "leakage_excluded.json").read_text(encoding="utf8"))["excluded"]
    excluded = {s for ids in audit.values() for s in ids}
    out.mkdir(parents=True, exist_ok=True)
    counts = {}
    for split, sources in SPLITS.items():
        rows = [r for s in sources
                for r in (tauber(tauber_path) if s == "tauber_beowulf" else load(data / "by_source" / f"{s}.jsonl"))]
        path = out / f"{split}.jsonl"
        path.write_text("".join(json.dumps(record(r, excluded), ensure_ascii=False) + "\n" for r in rows),
                        encoding="utf8")
        back = load(path)
        flagged = [r for r in back if r["leakage_excluded"]]
        counts[split] = {"sources": sources, "sentences": len(back), "tokens": sum(len(r["tokens"]) for r in back),
                         "flagged_sentences": len(flagged), "flagged_tokens": sum(len(r["tokens"]) for r in flagged)}
    table = ["| Split | File | Works (`source`) | Sentences | Tokens | Flagged sentences | Flagged tokens |",
             "|---|---|---|---:|---:|---:|---:|"]
    table += [f"| `{s}` | `{s}.jsonl` | {', '.join(f'`{w}`' for w in c['sources'])} | {c['sentences']:,} | "
              f"{c['tokens']:,} | {c['flagged_sentences']:,} | {c['flagged_tokens']:,} |" for s, c in counts.items()]
    card = Template((ROOT / "release" / "hf_card.md").read_text(encoding="utf8")).substitute(
        table="\n".join(table), sentences=f"{sum(c['sentences'] for c in counts.values()):,}",
        tokens=f"{sum(c['tokens'] for c in counts.values()):,}")
    (out / "README.md").write_text(card, encoding="utf8")
    return counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--tauber", default="data/brunetti-length.txt")
    ap.add_argument("--out", default="release/hf")
    a = ap.parse_args()
    if not Path(a.tauber).exists():
        raise SystemExit("Tauber gold missing: put brunetti-length.txt in data/ or run python -m oevl.fetch_tauber")
    for split, c in export(a.data, a.tauber, a.out).items():
        print(f"{split:11}{c['sentences']:>7,} sentences{c['tokens']:>8,} tokens, flagged {c['flagged_sentences']} "
              f"sentences ({c['flagged_tokens']} tokens)")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
