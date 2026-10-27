"""The numbers the paper may print that the other modules do not, from data/ and results/ (CPU, deterministic).

Writes report/generated/numbers_paper.tex (one \\newcommand per number, all named \\NP...) and Figure 1,
report/generated/fig_coverage.pdf: the frequency prior and the achievable ceiling for the four candidate sources, with
ByT5 local2 at + WikiPron. It asserts the worked examples the paper quotes (soliloquies:00026, Gēata, the Beowulf
article se, Beowulf line 2796) and the other facts its numbers rest on before it writes anything, so a change in the
data stops the run here. run_all.sh runs it before build_report.

The word-conjoined logistic model (experiment.features, arm "conjoined") is a post hoc diagnostic: it was designed
after the ByT5 results, has no tuning, and runs only here. Every ByT5 mean and contrast from the test grid (the
encoder tables and summary.json) is at the full cap over three seeds, local2 and local5 included. The exceptions
range over the whole grid: EncCtxGainMin over all 12 context arm and cap combinations, PriorCopyRuns over the 36
context runs, counts of single runs (the best run, MostLongMax, the runs that mark a token long) over all 39 runs,
and the runs with an identical twin over the nine local2 and the nine local5 runs. The word arm runs only at the full
cap. The selection numbers (SelDev..., DThreeEnc..., EncSolAll..., EncDThreeSolCost) come from the selection runs in
results/encoder/selection.json instead: sentence arm, full cap, seeds 1 and 2.
Usage: python -m oevl.paper_numbers
"""
import collections
import csv
import json
import math
import re
import statistics
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib
from scipy.stats import chi2_contingency, fisher_exact

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from oevl import experiment as E  # noqa: E402
from oevl.agreement import align, classes, compare, kappa, rate  # noqa: E402
from oevl.audit_splits import grams  # noqa: E402
from oevl.encoder_summary import configs, read_tables  # noqa: E402
from oevl.stats import bootstrap_diff, bootstrap_diff_runs  # noqa: E402
from oevl.wikipron_to_marked import mark  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DATA, RES, GEN = ROOT / "data", ROOT / "results", ROOT / "report" / "generated"
SETTINGS = (("T", "training_only", None), ("W", "wikipron", "lex_wikipron.csv"), ("P", "proposal", "lex_proposal.csv"),
            ("A", "all", "lex_all.csv"))  # the four candidate sources of run_all.sh
SETS = (("Dev", "john_ws"), ("Sol", "soliloquies"), ("Beo", "tauber_beowulf"))
TESTS = SETS[1:]
EDITIONS = (("Zupitza", "zupitza_grammar"), ("Reader", "bright_reader"), ("Matthew", "matthew_ws"),
            ("Sweet", "sweet_primer"), ("Physiologus", "physiologus"), ("John", "john_ws"),
            ("Soliloquies", "soliloquies"), ("HarrisonSharp", "hs_beowulf"), ("Wyatt", "wyatt_beowulf"))
READER_VERSE = ("Caedmon_s_Genesis", "Battle_of_Brunanburh", "The_Wanderer", "The_Phoenix")  # the rest is prose
FINNSBURH = "finnsburg"  # in the section names of both agreement editions
SPEECH_FORMULA = ("þa", "cwæþ", "se", "hælend", "to", "him")  # þā cwæð sē Hǣlend tō him
AEGTHER = re.compile(r"[æe]gþ?[æe]r")  # keys of ǣgðer in ǣgðer ge ... ge 'both ... and'
POPS = (("All", "accuracy"), ("Lex", "lexicon_ambiguous"), ("ClsA", "class_A"), ("ClsB", "class_B"),
        ("ClsC", "class_C"))
CLASS_TYPE = {"A": "homograph", "B": "grammatical", "C": "editorial"}
TWO_LEVEL_DRAWS = 40000  # with 4,000 draws an endpoint moved by up to 0.5 points with the RNG seed


class Numbers(dict):
    """Macro name after \\NP -> value as printed. The sections call an instance as put(name, value)."""

    def __call__(self, name: str, value: object) -> None:
        """Store str(value) under name. Exits when name is not all ASCII letters or is already taken."""
        if not (name.isascii() and name.isalpha()) or name in self:
            raise SystemExit(f"paper_numbers: bad or repeated macro name {name!r}")
        self[name] = str(value)


def need(ok: object, what: str) -> None:
    """Stop the run with SystemExit ("paper_numbers: <what> no longer holds") unless ok is true."""
    if not ok:
        raise SystemExit(f"paper_numbers: {what} no longer holds")


def pct(x: float, digits: int = 2) -> str:
    """A share in [0, 1] as a percentage with digits decimals: 0.9 -> "90.00"."""
    return f"{100 * x:.{digits}f}"


def pts(x: float, digits: int = 2) -> str:
    """A difference of shares as signed percentage points with digits decimals: 0.0123 -> "+1.23"."""
    return f"{100 * x:+.{digits}f}"


def num(x: int) -> str:
    """An integer with thousands commas: 12345 -> "12,345"."""
    return f"{x:,}"


def share(flags: Iterable[bool]) -> float:
    """The share of true values in flags, any iterable of booleans (ZeroDivisionError when it is empty)."""
    flags = list(flags)
    return sum(flags) / len(flags)


def word_tag(k: str) -> str:
    """A key as letters for a macro name: þe -> The, forþam -> Fortham."""
    return k.replace("þ", "th").replace("ð", "th").replace("æ", "ae").capitalize()


def reachable(r: dict[str, Any], lex: Mapping[str, Mapping[str, set[str]]]) -> bool:
    """Whether any candidate chooser can be right on token r: its gold is a candidate, or the word has no candidate
    and the gold is all short (every system writes such a word all short). This is the oracle of experiment.py."""
    return r["gold_pattern"] in lex.get(r["key"], {}) or (not lex.get(r["key"]) and "1" not in r["gold_pattern"])


def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    """The 95% Wilson score interval (lo, hi) of k successes in n trials, as shares."""
    p, d = k / n, 1 + z * z / n
    c, h = (p + z * z / (2 * n)) / d, z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def loose(k: str) -> str:
    """A spelling-tolerant key: æa, æo, io, æ, ea and eo become e, ie and y become i, a doubled letter is written once.

    k: an experiment.key. The replacements run in the order æa, æo, ie, y, io, æ, ea, eo, and the letter pairs are
    halved left to right, so a run of three equal letters keeps two."""
    for a, b in (("æa", "ea"), ("æo", "eo"), ("ie", "i"), ("y", "i"), ("io", "eo"), ("æ", "e"), ("ea", "e"),
                 ("eo", "e")):
        k = k.replace(a, b)
    return re.sub(r"(.)\1", r"\1", k)


def longest_run(sentences: Sequence[Sequence[str]], stream: Sequence[str]) -> tuple[int, tuple[str, ...]]:
    """(n, words) of the longest run of n consecutive keys that a sentence shares with stream, a key sequence.

    sentences: one key list per sentence, a run never crosses their ends; stream: one flat key list, so a run in it
    may cross sentence ends. words is the first such run in sentence order. Returns (0, ()) when no key is shared."""
    best: tuple[int, tuple[str, ...]]
    best, n = (0, ()), 1
    while True:
        have = {tuple(stream[i:i + n]) for i in range(len(stream) - n + 1)}
        hit = next((g for s in sentences for g in (tuple(s[i:i + n]) for i in range(len(s) - n + 1)) if g in have),
                   None)
        if hit is None:
            return best
        best, n = (n, hit), n + 1


def counts_from_pcts(values: Sequence[float], n: int) -> list[int]:
    """The token counts behind percentages of n tokens rounded to two decimals (unique while n < 10,000).

    values: percentages from 0 to 100. Returns one integer count per value. Exits when n is 10,000 or more or when a
    value is not 100 * count / n rounded to two decimals."""
    out = [round(v * n / 100) for v in values]
    need(n < 10000 and all(round(100 * c / n, 2) == v for c, v in zip(out, values)), f"exact counts of {values}")
    return out


def fixed_rule(r: dict[str, str]) -> str | None:
    """The error type that follows from the item sheet alone: gold not a candidate, then another edition with the
    system's pattern (Beowulf only), then the class of the word. None when no rule applies.

    r: a row of results/error_typing/labels.tsv. Returns ceiling, gold_in_doubt, homograph, grammatical, editorial or
    None."""
    def same(form: str) -> bool:
        """Whether form, another edition's spelling of the token, is aligned and has the system's pattern."""
        return bool(form) and form != "not_aligned" and E.pattern(form) == E.pattern(r["system_form"])
    if r["gold_in_candidates"] == "no":
        return "ceiling"
    if r["set"] == "tauber_beowulf" and (same(r["hs_form"]) or same(r["wyatt_form"])):
        return "gold_in_doubt"
    return CLASS_TYPE.get(r["class"])


def tauber_columns(path: str | Path) -> list[tuple[str, str]]:
    """Tauber's line number and gloss of every token, in the order of experiment.load_tauber.

    path: Tauber's pipe-separated file (data/brunetti-length.txt). Returns one (line, gloss) pair of strings, columns 4
    and 16 counted from zero, per word of column 9, so the words of one row share them. The line number is zero-padded
    to four digits ("0084", "2587"). No sentence is left out, so the list lines up with
    experiment.heldout(..., "tauber_beowulf") only while the leakage audit excludes no Beowulf sentence (Context checks
    the lengths)."""
    out = []
    for line in Path(path).read_text(encoding="utf8").splitlines():
        if line.strip():
            c = line.split("|")
            out += [(c[4], c[16])] * len(E.clean_tauber(c[9]).split())
    return out


def jsonl(path: str | Path) -> list[Any]:
    """The JSON value on each line of a JSON Lines file, in order (the file must have no blank lines)."""
    with open(path, encoding="utf8") as f:
        return [json.loads(line) for line in f]


def runs_of(row: dict[str, str], arm: str | None = None, cap: str | None = None) -> list[str]:
    """The ByT5 run columns of an encoder table row, optionally of one arm and one config.

    Run columns are named <config>/<arm>_s<seed>, such as capfull/local2_s1. arm: word, left, sentence, local2 or
    local5, and cap: cap128, cap256 or capfull. With neither, all 39 runs (the word arm runs only at capfull). Every
    row has the same columns, so any row will do."""
    return [c for c in row if "/" in c and (arm is None or c.split("/")[1].rsplit("_", 1)[0] == arm)
            and (cap is None or c.startswith(cap + "/"))]


def correct(row: dict[str, str], cols: Sequence[str]) -> float:
    """Share of the columns that give the gold pattern on this encoder table row (cols must not be empty)."""
    return sum(row[c] == row["gold"] for c in cols) / len(cols)


# ---------------------------------------------------------------- worked examples (asserted, so the paper stays true)
def example_soliloquies_00026(preds: Iterable[dict[str, Any]], enc: dict[str, dict[str, str]]) -> None:
    """In 'Ic clypie tō þē, Drihten, þū þe æall geworhtest' the first þē is the pronoun and the second þe the relative
    particle. The prior, every word, left and bilateral logistic run and every ByT5 word run mark both short, and every
    local2 run gets both right (and every conjoined run, asserted in conjoined).

    preds: the WikiPron prediction rows of the Soliloquies; enc: its encoder table by token id. The local2 check covers
    all three caps, the word arm runs only at the full cap. Returns None. Exits when any of this no longer holds."""
    toks = [p for p in preds if p["sentence_id"] == "soliloquies:00026"]
    need(" ".join(t["gold"] for t in toks).startswith("Ic clypie tō þē Drihten þū þe æall geworhtest"),
         "the text of soliloquies:00026")
    pron, rel = toks[3], toks[6]
    need((pron["key"], pron["gold_pattern"], rel["key"], rel["gold_pattern"]) == ("þe", "1", "þe", "0"),
         "the two þe of soliloquies:00026")
    logit = ["frequency_prediction"] + [f"{a}_seed{s}_prediction" for a in ("word", "left", "sentence")
                                        for s in (1, 2, 3)]
    need(all(t[k] == "0" for t in (pron, rel) for k in logit), "soliloquies:00026: the prior and logistic runs")
    a, b = enc[pron["id"]], enc[rel["id"]]
    need(all((a[c], b[c]) == ("0", "0") for c in runs_of(a, "word")), "soliloquies:00026: the ByT5 word runs")
    need(all((a[c], b[c]) == ("1", "0") for c in runs_of(a, "local2")), "soliloquies:00026: the local2 runs")


def example_geata(preds: Iterable[dict[str, Any]], enc: dict[str, dict[str, str]],
                  lex: Mapping[str, Mapping[str, set[str]]],
                  counts: Mapping[str, collections.Counter[str]]) -> list[dict[str, Any]]:
    """Gēata: 39 Beowulf tokens, all long and unseen in training. WikiPron lists gēata and geata 'gates', so the prior
    falls back to fewer long vowels. Returns the tokens.

    It also asserts that every ByT5 word run (full cap) writes gēata on every token and no full-cap sentence run writes
    it on more than two. preds: the WikiPron prediction rows of Beowulf; enc: its encoder table by token id; lex,
    counts: the WikiPron lexicon and training counts from experiment.lexicon. The tokens come back as prediction rows."""
    g = [p for p in preds if p["key"] == "geata"]
    need(len(g) == 39 and {p["gold_pattern"] for p in g} == {"100"}, "Gēata: 39 tokens, all long in the gold")
    need(not counts.get("geata") and set(lex["geata"]) == {"000", "100"},
         "Gēata: unseen in training, two WikiPron forms")
    need(all(p["frequency_prediction"] == "000" for p in g), "Gēata: the prior marks it short")
    row = enc[g[0]["id"]]
    need(all(enc[p["id"]][c] == "100" for p in g for c in runs_of(row, "word", "capfull")), "Gēata: the word arm")
    need(all(sum(enc[p["id"]][c] == "100" for p in g) <= 2 for c in runs_of(row, "sentence", "capfull")),
         "Gēata: the sentence arm almost none")
    return g


def example_se(tb: Sequence[dict[str, Any]], cols: Sequence[tuple[str, str]], preds: Iterable[dict[str, Any]],
               enc: dict[str, dict[str, str]]) -> list[tuple[dict[str, Any], str]]:
    """In Beowulf line 2587, þæt se mǣra maga Ecgðēowes, Klaeber writes the article se short, and the prior and every
    ByT5 run mark it long. Returns every se token of Tauber's Beowulf with its gloss.

    tb: Tauber's token rows; cols: tauber_columns of the same file, aligned with tb; preds: the WikiPron prediction rows
    of Beowulf; enc: its encoder table by token id. "Every ByT5 run" covers all caps. The code checks one short se on
    line 2587, not its gloss. Returns (token row, gloss) pairs."""
    se = [(t, gloss) for t, (_, gloss) in zip(tb, cols) if t["key"] == "se"]
    line = [t for t, (ln, _) in zip(tb, cols) if t["key"] == "se" and ln == "2587"]
    need(len(line) == 1 and line[0]["gold_pattern"] == "0", "Beowulf 2587: one short se")
    p, row = next(x for x in preds if x["id"] == line[0]["id"]), enc[line[0]["id"]]
    need(p["frequency_prediction"] == "1" and all(row[c] == "1" for c in runs_of(row)), "Beowulf 2587: all mark se long")
    return se


def example_2796(enc: dict[str, dict[str, str]], ctx: "Context") -> int:
    """Beowulf 2796, ēcum dryhtne þē ic hēr on starie: Tauber and Harrison and Sharp print þē, Wyatt þe, and every
    full-cap ByT5 run writes þe. Returns the number of runs over all caps that write þē.

    The token is tauber:685:44 (line 2796 in Tauber's columns, which the code does not check). The Harrison and Sharp
    and Wyatt forms come from aligning each edition with the scored Tauber rows (agreement.align). enc: the Beowulf
    encoder table by token id; ctx: the Context (hs, wy and tb)."""
    b = enc["tauber:685:44"]
    hs, wy = ({t["id"]: e["gold"] for e, t in align(ed, ctx.tb)}.get("tauber:685:44") for ed in (ctx.hs, ctx.wy))
    need((b["gold"], hs, wy) == ("1", "þē", "þe") and all(b[c] == "0" for c in runs_of(b, cap="capfull")),
         "Beowulf 2796")
    return sum(b[c] == "1" for c in runs_of(b))


# ---------------------------------------------------------------- the sections of the paper
class Context:
    """Everything the sections share, loaded once."""

    def __init__(self) -> None:
        """Load the data, predictions and results the sections read, and check that they line up.

        train: the training token rows without Zupitza (decision D3); hs, wy: every Harrison and Sharp and Wyatt
        Beowulf token; lex: tag (T, W, P, A of SETTINGS) -> (lexicon, training counts) from experiment.lexicon;
        held: set name -> scored token rows (experiment.heldout); tb: the scored Tauber rows; tb_cols: tauber_columns,
        aligned with tb (only the lengths are checked); preds: (tag, set) -> rows of
        results/<folder>/<set>_predictions.jsonl, checked to cover the held tokens in order; metrics: tag ->
        results/<folder>/metrics.json; enc: set -> {token id: row of results/encoder/<set>_predictions.tsv.gz, every
        cell a string}; summary: results/encoder/summary.json; cls: skeleton -> class letter of
        data/ambiguity_classes.csv (experiment.load_classes); labels: the rows of results/error_typing/labels.tsv.
        Exits when a check fails. experiment.load_tauber prints one validation line."""
        self.train, _, self.hs, self.wy = E.build_data(DATA, E.EXCLUDE_D3)
        self.lex = {tag: E.lexicon(self.train, str(DATA / f) if f else None) for tag, _, f in SETTINGS}
        self.held = {s: E.heldout(DATA, s) for _, s in SETS}
        self.tb = self.held["tauber_beowulf"]
        self.tb_cols = tauber_columns(DATA / "brunetti-length.txt")
        need(len(self.tb_cols) == len(self.tb), "Tauber's columns line up with its tokens")
        self.preds = {(tag, s): jsonl(RES / folder / f"{s}_predictions.jsonl")
                      for tag, folder, _ in SETTINGS for _, s in SETS}
        for (tag, s), p in self.preds.items():
            need([r["id"] for r in p] == [r["id"] for r in self.held[s]], f"{tag} {s}: the predictions cover the tokens")
        self.metrics = {tag: json.loads((RES / folder / "metrics.json").read_text(encoding="utf8"))
                        for tag, folder, _ in SETTINGS}
        self.enc = {s: {t["id"]: t for t in table} for s, table in read_tables(RES / "encoder").items()}
        self.summary = json.loads((RES / "encoder" / "summary.json").read_text(encoding="utf8"))
        self.cls = E.load_classes(DATA / "ambiguity_classes.csv")
        self.labels = list(csv.DictReader(open(RES / "error_typing" / "labels.tsv", encoding="utf8"), delimiter="\t"))


def corpus(put: Numbers, ctx: Context) -> None:
    """The corpus and its splits: sizes and long-vowel shares per edition and genre (Corpus..., Ed..., Reader...,
    VerseLong..., ProseLong..., Finnsburh...), the training set without and with Zupitza (Train..., ZupTrain...), the
    tokens of the two agreement editions (AgreementTokens), the scored sentences and unseen tokens of each set
    (ScoredSentences..., Unseen...), the leakage figures (LeakFormula, LongestRun), the release (Release...), the
    corpus tokens per Tauber token (CorpusOverTauber), apparent slips in Bright's Reader (Slip...), the scored
    Soliloquies sentences that are only the token A (SpeakerA), and Harrison and Sharp against Wyatt without
    Finnsburh (FinnsburhAligned, NoFinnsburh...).

    The edition counts cover every sentence, the scored ones and the leakage-excluded alike. Tauber's (EdTauber...)
    come from ctx.tb, the scored rows, which Context's length check makes all of Tauber's. LongestRun compares loose
    keys of all five experiment.TRAIN works, Zupitza included. Reads data/by_source, data/manifest.json,
    data/leakage_excluded.json, data/lex_wikipron.csv and data/ambiguity_classes.csv. Exits when the manifest token
    counts, the four verse sections of the Reader, the training set size or the longest run (feor oþþe nēah) no
    longer hold."""
    rows = {w: E.load(DATA / "by_source" / f"{w}.jsonl") for _, w in EDITIONS}
    toks = {w: list(E.tokens(r)) for w, r in rows.items()}
    manifest = json.loads((DATA / "manifest.json").read_text(encoding="utf8"))["sources"]
    need(all(len(toks[w]) == manifest[w]["tokens"] for w in toks), "the manifest token counts")

    def long(ts: Sequence[dict[str, Any]]) -> float:
        """Share of the vowel letters of tokens ts that are marked long."""
        return sum(t["gold_pattern"].count("1") for t in ts) / sum(len(t["gold_pattern"]) for t in ts)

    corpus_toks = [t for ts in toks.values() for t in ts]
    sentences = sum(map(len, rows.values()))
    put("CorpusEditions", len(rows))
    put("CorpusSentences", num(sentences))
    put("CorpusVowels", num(sum(len(t["gold_pattern"]) for t in corpus_toks)))
    put("CorpusLongVowels", num(sum(t["gold_pattern"].count("1") for t in corpus_toks)))
    put("CorpusLong", pct(long(corpus_toks)))
    put("CorpusMacronTokens", pct(share("1" in t["gold_pattern"] for t in corpus_toks)))
    reader_verse = [r for r in rows["bright_reader"] if any(v in r["section"] for v in READER_VERSE)]
    sections = {r["section"] for r in rows["bright_reader"]}
    need(len({r["section"] for r in reader_verse}) == len(READER_VERSE), "the four verse sections of Bright's Reader")
    rv = list(E.tokens(reader_verse))
    rp = [t for t in toks["bright_reader"] if t["sentence_id"] not in {r["id"] for r in reader_verse}]
    tb_sentences = len({t["sentence_id"] for t in ctx.tb})
    for name, ts, ns in [(name, toks[w], len(rows[w])) for name, w in EDITIONS] + [("Tauber", ctx.tb, tb_sentences)]:
        put(f"Ed{name}Tokens", num(len(ts)))
        put(f"Ed{name}Sentences", num(ns))
        put(f"Ed{name}Long", pct(long(ts)))
    put("ReaderSections", len(sections))
    put("ReaderVerseSections", len(READER_VERSE))
    put("ReaderVerseTokens", num(len(rv)))
    put("ReaderVerseLong", pct(long(rv)))
    put("ReaderProseLong", pct(long(rp)))
    # ranges over parts, not editions: the Reader's verse and prose parts count separately, each Beowulf edition once
    verse = [long(rv), long(toks["physiologus"]), long(toks["hs_beowulf"]), long(toks["wyatt_beowulf"]), long(ctx.tb)]
    prose = [long(rp)] + [long(toks[w]) for w in ("zupitza_grammar", "matthew_ws", "sweet_primer", "john_ws",
                                                  "soliloquies")]
    put("VerseLongMin", pct(min(verse), 1))
    put("VerseLongMax", pct(max(verse), 1))
    put("ProseLongMin", pct(min(prose), 1))
    put("ProseLongMax", pct(max(prose), 1))
    fin = {w: [t for r in rows[w] if FINNSBURH in r["section"].lower() for t in E.tokens([r])]
           for w in ("hs_beowulf", "wyatt_beowulf")}
    put("FinnsburhHarrisonSharp", len(fin["hs_beowulf"]))
    put("FinnsburhWyatt", len(fin["wyatt_beowulf"]))

    train = [w for w in E.TRAIN if w not in E.EXCLUDE_D3]
    n_train = sum(len(toks[w]) for w in train)
    bright = len(toks["bright_reader"]) + len(toks["matthew_ws"])
    verse_train = len(rv) + len(toks["physiologus"])
    need(n_train == len(ctx.train), "the training set")
    put("TrainSentences", num(sum(len(rows[w]) for w in train)))
    put("TrainBrightTokens", num(bright))
    put("TrainBrightShare", pct(bright / n_train, 1))
    put("TrainVerseTokens", num(verse_train))
    put("TrainVerseShare", pct(verse_train / n_train, 1))
    put("TrainProseShare", pct(1 - verse_train / n_train, 1))
    zup = E.build_data(DATA)[0]
    zlex = E.lexicon(zup, str(DATA / "lex_wikipron.csv"))[0]
    put("ZupTrainTokens", num(len(zup)))
    put("ZupTrainExamples", num(sum(bool(r["gold_pattern"]) and len(zlex.get(r["key"], ())) >= 2 for r in zup)))
    put("AgreementTokens", num(len(ctx.hs) + len(ctx.wy)))
    for st, s in SETS:
        put(f"ScoredSentences{st}", num(len({r["sentence_id"] for r in ctx.held[s]})))

    seen = set().union(*(g for w in E.TRAIN for g in grams(toks[w]).values()))  # the audit's five works
    excluded = json.loads((DATA / "leakage_excluded.json").read_text(encoding="utf8"))["excluded"]["john_ws"]
    john = grams(toks["john_ws"])
    put("LeakFormula", sum(john[s] & seen == {SPEECH_FORMULA} for s in excluded))
    run = longest_run([[loose(E.key(b)) for b in r["tokens_bare"]] for w in E.TRAIN for r in rows[w]],
                      [loose(t["key"]) for t in ctx.tb])
    need(run[1] == ("fer", "oþe", "neh", "findan", "mehte"), "the longest run shared with Beowulf (feor oþþe nēah)")
    put("LongestRun", run[0])
    counts = ctx.lex["T"][1]
    for st, s in SETS:
        unseen = [not counts.get(r["key"]) for r in ctx.held[s]]
        put(f"Unseen{st}N", num(sum(unseen)))
        put(f"Unseen{st}", pct(share(unseen)))
    put("ReleaseSentences", num(sentences + tb_sentences))
    put("ReleaseTokens", num(len(corpus_toks) + len(ctx.tb)))

    reader = collections.Counter(E.norm(t["gold"]).replace("ð", "þ") for t in toks["bright_reader"])
    for name, form in (("Be", "bē"), ("Hit", "hīt"), ("Waes", "wǣs"), ("Cwaeth", "cwǣþ"), ("Aefter", "ǣfter")):
        put(f"Slip{name}", reader[form])
    scored = {r["sentence_id"] for r in ctx.held["soliloquies"]}
    put("SpeakerA", sum(r["tokens"] == ["A"] and r["id"] in scored for r in rows["soliloquies"]))
    put("CorpusOverTauber", f"{len(corpus_toks) / len(ctx.tb):.1f}")

    lex, cnt = ctx.lex["W"]
    cls = classes(DATA / "ambiguity_classes.csv")
    fin_ids = {t["id"] for ts in fin.values() for t in ts}
    put("FinnsburhAligned", sum(x["id"] in fin_ids for x, _ in align(ctx.hs, ctx.wy)))
    _, res = compare([t for t in ctx.hs if t["id"] not in fin_ids], [t for t in ctx.wy if t["id"] not in fin_ids],
                     "hs", "wyatt", lex, cnt, cls)
    for name, pop in (("All", "all"), ("Lex", "lexicon_ambiguous"), ("ClsB", "class_B")):
        put(f"NoFinnsburh{name}", pct(res[pop]["agreement"]))


def candidates(put: Numbers, ctx: Context) -> None:
    """The candidate lists and the labels: rows per list (LexForms...), WikiPron forms also in the Wiktionary list
    (WikiInWiktionary...), the lexicon-ambiguous share of each test set and its split by class (LexShare..., ClsB...,
    ClsShare...), the dev minority-reading count (MinDevN), the blind review (Review...), the two labelling runs of the
    classes and the error types (ClassLabels..., ErrLabels...), the distinct patterns among the logistic training
    examples (LogitClasses) and the optimiser steps of a full-cap ByT5 run, epochs times batches (EncSteps).

    Reads data/lex_*.csv, results/label_review/expert_review.json, data/ambiguity_classes.csv and
    results/error_typing/labels.tsv. Exits unless the logistic models and ByT5 train on the same number of examples."""
    rows = {n: [r["marked_form"] for r in csv.DictReader(open(DATA / f"lex_{n}.csv", encoding="utf8"))]
            for n in ("clarkhall", "wiktionary", "bt", "proposal", "all", "wikipron")}
    for name, n in (("ClarkHall", "clarkhall"), ("Wiktionary", "wiktionary"), ("BT", "bt"), ("Proposal", "proposal"),
                    ("All", "all")):
        put(f"LexForms{name}", num(len(rows[n])))
    wiktionary = set(rows["wiktionary"])
    in_wiktionary = sum(f in wiktionary for f in rows["wikipron"])
    put("WikiInWiktionary", num(in_wiktionary))
    put("WikiInWiktionaryShare", pct(in_wiktionary / len(rows["wikipron"])))
    for st, s in TESTS:
        p = ctx.preds["W", s]
        amb = [r for r in p if r["lexicon_ambiguous"]]
        put(f"LexShare{st}", pct(len(amb) / len(p), 1))
        b = collections.Counter(r["key"] for r in amb if ctx.cls.get(r["key"]) == "B")
        for k in sorted(b):
            put(f"ClsB{st}{word_tag(k)}", num(b[k]))
        c = collections.Counter(ctx.cls.get(r["key"], "unlisted") for r in amb)
        for x in "ABC":
            put(f"ClsShare{st}{x}", pct(c[x] / len(amb), 1))
        put(f"ClsShare{st}", pct(1 - c["unlisted"] / len(amb), 1))
    put("MinDevN", num(ctx.metrics["W"]["john_ws"]["frequency"]["minority_reading"]["n"]))
    review = json.loads((RES / "label_review" / "expert_review.json").read_text(encoding="utf8"))["areas"]
    put("ReviewItems", sum(a["summary"]["items"] for a in review.values()))
    put("ReviewAgreed", sum(a["summary"]["expert_agrees_with_existing"] for a in review.values()))
    put("ReviewChanged", sum(len(a["summary"]["changed"]) for a in review.values()))
    for name, path, sep in (("Class", DATA / "ambiguity_classes.csv", ","),
                            ("Err", RES / "error_typing" / "labels.tsv", "\t")):  # the two labelling runs
        with open(path, encoding="utf8") as f:
            labels = list(csv.DictReader(f, delimiter=sep))
        put(f"{name}Labels", num(len(labels)))
        put(f"{name}LabelsAgree", num(sum(r["label_a"] == r["label_b"] for r in labels)))
    lex = ctx.lex["W"][0]
    eligible = [r for r in ctx.train if r["gold_pattern"] and len(lex.get(r["key"], ())) >= 2]
    full = configs(ctx.summary)["capfull"]["settings"]
    need(len(eligible) == full["train_examples"], "the logistic and ByT5 models train on the same examples")
    put("LogitClasses", len({r["gold_pattern"] for r in eligible}))
    put("EncSteps", num(full["epochs"] * math.ceil(full["train_examples"] / full["batch"])))


def coverage(put: Numbers, ctx: Context) -> None:
    """Coverage: the all-short baseline (AllShort..., NoCandShort...), per candidate source the test tokens with no
    candidate and the share of the prior's errors that no chooser could fix (NoCand..., OutOfReach...Share), for
    WikiPron alone the dev ceiling and those errors in counts (CeilDev, PriorErr..., OutOfReach..., Headroom...), the
    prior's back-off on unseen words (Backoff...), the prior on seen and unseen words (SeenPrior..., UnseenPrior...,
    DictWikiExtra..., AvsP...) and the net Beowulf gain of all four sources over training forms and over WikiPron,
    with its part on words found in no other text (BeoGain..., BeoOnly...).

    Reads data/by_source for the words of the other texts. Exits when reachable() and the experiment's oracle disagree
    on a count, or when a dictionary changes the prior on a word seen in training."""
    for st, s in SETS:  # the all-short baseline
        p = ctx.preds["W", s]
        put(f"AllShort{st}", pct(share("1" not in r["gold_pattern"] for r in p)))
        if s != "john_ws":
            lex = ctx.lex["W"][0]
            put(f"AllShort{st}Lex", pct(share("1" not in r["gold_pattern"] for r in p if r["lexicon_ambiguous"])))
            put(f"NoCandShort{st}", pct(share(not lex.get(r["key"]) and "1" not in r["gold_pattern"] for r in p)))
    for tag, folder, _ in SETTINGS:
        lex = ctx.lex[tag][0]
        for st, s in SETS:
            p = ctx.preds[tag, s]
            reach = [reachable(r, lex) for r in p]
            need(sum(reach) == sum(r["oracle_prediction"] == r["gold_pattern"] for r in p), f"{folder} {s}: the oracle")
            if s == "john_ws":
                if tag == "W":
                    put("CeilDev", pct(share(reach)))
                continue
            put(f"NoCand{tag}{st}", pct(share(not lex.get(r["key"]) for r in p)))
            err = [(r, ok) for r, ok in zip(p, reach) if r["frequency_prediction"] != r["gold_pattern"]]
            out = [r for r, ok in err if not ok]
            put(f"OutOfReach{tag}{st}Share", pct(len(out) / len(err), 1))
            if tag == "W":
                put(f"PriorErr{st}", num(len(err)))
                put(f"OutOfReach{st}", num(len(out)))
                put(f"OutOfReach{st}Unknown", num(sum(not lex.get(r["key"]) for r in out)))
                put(f"OutOfReach{st}Other", num(sum(bool(lex.get(r["key"])) for r in out)))
                put(f"Headroom{st}", num(len(err) - len(out)))
                put(f"Headroom{st}Pts", f"{100 * (len(err) - len(out)) / len(p):.2f}")
    lex, counts = ctx.lex["W"]
    for st, s in TESTS:  # words unseen in training with two or more candidates, where the back-off decides
        back = [r for r in ctx.preds["W", s] if not counts.get(r["key"]) and len(lex.get(r["key"], ())) >= 2]
        put(f"BackoffTokens{st}", len(back))
        put(f"BackoffRight{st}", sum(r["frequency_prediction"] == r["gold_pattern"] for r in back))
    counts = ctx.lex["T"][1]
    for st, s in TESTS:  # dictionaries change the prior only on words unseen in training
        seen = [bool(counts.get(r["key"])) for r in ctx.held[s]]
        prior = {tag: [r["frequency_prediction"] for r in ctx.preds[tag, s]] for tag, _, _ in SETTINGS}
        need(all(prior[tag][i] == prior["T"][i] for tag in prior for i, k in enumerate(seen) if k),
             f"{s}: the prior never changes on a seen word")
        gold = [r["gold_pattern"] for r in ctx.held[s]]
        put(f"SeenPrior{st}", pct(share(prior["T"][i] == gold[i] for i, k in enumerate(seen) if k)))
        for tag in prior:
            put(f"UnseenPrior{tag}{st}", pct(share(prior[tag][i] == gold[i] for i, k in enumerate(seen) if not k)))
        diff = [i for i in range(len(gold)) if prior["A"][i] != prior["P"][i]]
        acc = {tag: share(x == g for x, g in zip(prior[tag], gold)) for tag in ("P", "A")}
        put(f"DictWikiExtra{st}", f"{100 * (acc['A'] - acc['P']):.2f}")
        if s == "tauber_beowulf":
            put("AvsPDiffer", len(diff))
            put("AvsPRightA", sum(prior["A"][i] == gold[i] for i in diff))
            put("AvsPRightP", sum(prior["P"][i] == gold[i] for i in diff))
    # the net verse gain of the dictionaries (tokens gained minus tokens lost), and its part on word types found in no
    # other text of the corpus
    other = {t["key"] for w in E.TRAIN + ["john_ws", "soliloquies"]
             for t in E.tokens(E.load(DATA / "by_source" / f"{w}.jsonl"))}
    ok = {tag: [r["frequency_prediction"] == r["gold_pattern"] for r in ctx.preds[tag, "tauber_beowulf"]]
          for tag in "TWA"}
    only = [r["key"] not in other for r in ctx.tb]
    for base in ("T", "W"):
        gain = [a - b for a, b in zip(ok["A"], ok[base])]
        put(f"BeoGainTokens{base}", num(sum(gain)))
        put(f"BeoOnlyGainTokens{base}", num(sum(g for g, o in zip(gain, only) if o)))
        if base == "T":
            put("BeoOnlyGainPts", f"{100 * sum(g for g, o in zip(gain, only) if o) / len(gain):.2f}")
            put("BeoOnlyGainShare", f"{100 * sum(g for g, o in zip(gain, only) if o) / sum(gain):.0f}")


def context(put: Numbers, ctx: Context) -> None:
    """Context: the logistic bilateral row (Bil...), ByT5 all-token gains over the WikiPron prior (EncGain...,
    EncCtxGainMin...), intervals that resample runs as well as sentences (TwoLevel...), token-equivalents of local2 over
    the word arm (Te...), prose class B per word (PwSol..., SolClsBNoTheGain), ge after ǣgðer (GeShort, GeAegther),
    verse class B without þe (BeoClsBNoTheGain...), Gēata (Geata..., BeoLexNoGeata...), runs with an identical twin
    (Identical...), the highest share of ambiguous prose tokens on which a run picks a candidate with the most long
    vowels (MostLongMax) and the context runs that copy the prior on every prose class B token (PriorCopyRuns).

    ByT5 numbers use the full-cap runs, except EncCtxGainMin, Identical..., MostLongMax and PriorCopyRuns, which range
    over all caps. Asserts the worked examples soliloquies:00026 and Gēata, the twelve context arm and cap
    combinations, and that the word arm equals the prior on þe, ne and ge in prose class B."""
    sol, beo = ctx.enc["soliloquies"], ctx.enc["tauber_beowulf"]
    full = configs(ctx.summary)["capfull"]
    s: str | int  # a set name, then a seed
    pop: str | None  # a population name, then an optional class letter
    for st, s in TESTS:  # the logistic bilateral row
        for name, pop in POPS:
            m = ctx.metrics["W"][s]["seed_summary"]["sentence"][pop]
            put(f"Bil{st}{name}", pct(m["mean"]))
            put(f"Bil{st}{name}Sd", pct(m["sd"]))
    for st, s in TESTS:  # all-token gains over the WikiPron prior
        prior = full["frequency"][s]["accuracy"]["accuracy"]
        for name, arm in (("Bil", "sentence"), ("LocalTwo", "local2")):
            put(f"EncGain{name}{st}", f"{100 * (full['arms'][arm]['sets'][s]['accuracy']['mean'] - prior):.2f}")
        ctx_gains = [C["arms"][a]["sets"][s]["accuracy"]["mean"] - prior for C in configs(ctx.summary).values()
                     for a in C["arms"] if a != "word"]
        need(len(ctx_gains) == 12, "twelve context arm and cap combinations")
        put(f"EncCtxGainMin{st}", f"{100 * min(ctx_gains):.2f}")
    put("TwoLevelDraws", num(TWO_LEVEL_DRAWS))
    for st, table in (("Sol", sol), ("Beo", beo)):  # intervals that resample runs as well as sentences
        for name, pop in (("Lex", None), ("ClsB", "B")):
            ix = [t for t in table.values()
                  if t["capfull:lexicon_ambiguous"] == "1" and pop in (None, t["capfull:class"])]
            for arm, a in (("LocalTwo", "local2"), ("Bil", "sentence")):
                b = bootstrap_diff_runs([t["sentence_id"] for t in ix],
                                        [[t[c] == t["gold"] for t in ix] for c in runs_of(ix[0], a, "capfull")],
                                        [[t[c] == t["gold"] for t in ix] for c in runs_of(ix[0], "word", "capfull")],
                                        n=TWO_LEVEL_DRAWS)
                for k in ("Lo", "Hi"):
                    put(f"TwoLevel{st}{name}{arm}VsWord{k}", pts(b[k.lower()]))
    l2, word = runs_of(next(iter(sol.values())), "local2", "capfull"), runs_of(next(iter(sol.values())), "word", "capfull")

    def te(rows: Iterable[dict[str, str]]) -> float:
        """Token-equivalents: local2 minus the word arm (full cap, share of runs right), summed over encoder rows."""
        return sum(correct(t, l2) - correct(t, word) for t in rows)

    amb = {st: [t for t in table.values() if t["capfull:lexicon_ambiguous"] == "1"]
           for st, table in (("Sol", sol), ("Beo", beo))}
    for c, name in (("A", "A"), ("B", "B"), ("C", "C"), ("unlisted", "U")):
        put(f"TeSolCls{name}", f"{te(t for t in amb['Sol'] if t['capfull:class'] == c):+.1f}")
    put("TeSolLex", f"{te(amb['Sol']):+.1f}")
    keys = {t["id"]: t["key"] for _, s in TESTS for t in ctx.held[s]}
    b_sol = [t for t in amb["Sol"] if t["capfull:class"] == "B"]
    for k in ("þe", "ne", "ge"):  # per word, prose class B
        w = [t for t in b_sol if keys[t["id"]] == k]
        need(all(correct(t, word) == (t["capfull:frequency"] == t["gold"]) for t in w), f"{k}: the word arm is the prior")
        put(f"PwSol{word_tag(k)}N", num(len(w)))
        put(f"PwSol{word_tag(k)}Prior", pct(share(t["capfull:frequency"] == t["gold"] for t in w), 1))
        put(f"PwSol{word_tag(k)}LocalTwo", pct(sum(correct(t, l2) for t in w) / len(w), 1))
        put(f"TeSol{word_tag(k)}", f"{te(w):+.1f}")
    no_the = [t for t in b_sol if keys[t["id"]] != "þe"]
    put("SolClsBNoTheGain", pts(te(no_the) / len(no_the)))
    ge = [r for r in ctx.held["soliloquies"] if r["key"] == "ge"]
    sent = collections.defaultdict(list)
    for r in ctx.held["soliloquies"]:
        sent[r["sentence_id"]].append(r["key"])
    put("GeShort", sum(r["gold_pattern"] == "0" for r in ge))
    after = [r for r in ge if any(AEGTHER.fullmatch(k) for k in sent[r["sentence_id"]][:r["index"]])]
    put("GeAegther", sum(r["gold_pattern"] == "0" for r in after))
    b_beo = [t for t in amb["Beo"] if t["capfull:class"] == "B"]
    put("TeBeoClsB", f"{te(b_beo):+.1f}")
    put("TeBeoThe", f"{te(t for t in b_beo if keys[t['id']] == 'þe'):+.1f}")
    no_the = [t for t in b_beo if keys[t["id"]] != "þe"]
    b = bootstrap_diff([t["sentence_id"] for t in no_the], [correct(t, l2) for t in no_the],
                       [correct(t, word) for t in no_the])
    for k, v in (("", "diff"), ("Lo", "lo"), ("Hi", "hi")):
        put(f"BeoClsBNoTheGain{k}", pts(b[v]))
    g = example_geata(ctx.preds["W", "tauber_beowulf"], beo, *ctx.lex["W"])
    put("GeataN", len(g))
    for s, name in ((1, "One"), (2, "Two"), (3, "Three")):
        put(f"GeataLocalTwoRun{name}", sum(beo[p["id"]][f"capfull/local2_s{s}"] == "100" for p in g))
    put("TeBeoUnlisted", f"{te(t for t in amb['Beo'] if t['capfull:class'] == 'unlisted'):+.1f}")
    put("TeBeoGeata", f"{te(beo[p['id']] for p in g):+.1f}")
    put("GeataPts", f"{100 * len(g) / len(beo):.2f}")  # every word run right, the prior wrong
    rest = [t for t in amb["Beo"] if keys[t["id"]] != "geata"]
    for name, a in (("LocalTwo", "local2"), ("Bil", "sentence")):
        cols = runs_of(rest[0], a, "capfull")
        b = bootstrap_diff([t["sentence_id"] for t in rest], [correct(t, cols) for t in rest],
                           [correct(t, word) for t in rest])
        for k, v in (("", "diff"), ("Lo", "lo"), ("Hi", "hi")):
            put(f"BeoLexNoGeata{name}VsWord{k}", pts(b[v]))
    twins = {c for g in ctx.summary["identical_runs"]["groups"] for c in g}  # runs with an identical twin, all caps
    for name, a in (("LocalTwo", "local2"), ("LocalFive", "local5")):
        cols = runs_of(next(iter(sol.values())), a)
        put(f"Identical{name}Runs", sum(c in twins for c in cols))
        put(f"Identical{name}Of", len(cols))
    example_soliloquies_00026(ctx.preds["W", "soliloquies"], sol)
    put("MostLongMax", pct(max(v for C in configs(ctx.summary).values() for A in C["arms"].values()
                               for v in A["sets"]["soliloquies"]["most_long_share"])))
    put("PriorCopyRuns", sum(all(t[c] == t["capfull:frequency"] for t in b_sol) for c in runs_of(b_sol[0])
                             if "/word_" not in c))  # context runs; the word arm is the prior on class B


def selection(put: Numbers, ctx: Context) -> None:
    """The ByT5 setting and decision D3: exact dev counts behind the chosen setting (SelDev...), the logistic word means
    with and without Zupitza (DevWord...), the ByT5 change from dropping Zupitza per setting (DThreeEnc...), what
    Zupitza's Grammar changes in the prior (ZupShort..., DevGain..., ZupWord..., ZupOnly..., DThreeSolNet), and the
    prose cost at the chosen setting (EncSolAll..., EncDThreeSolCost).

    The ByT5 numbers come from the selection runs in results/encoder/selection.json (sentence arm, full cap, seeds 1
    and 2), the prior's from results/wikipron, wikipron_with_zupitza and proposal. DThreeEnc leaves out every learning
    rate and epoch setting with a collapsed run, with or without Zupitza. Exits unless every percentage gives an exact
    count, the exact counts pick the chosen setting, Zupitza marks þes, gyt and lytel short, and forþam, swaswa and
    forþi are unknown without Zupitza."""
    sel = json.loads((RES / "encoder" / "selection.json").read_text(encoding="utf8"))
    n = sel["d3_logistic_john_ws"]["without_zupitza"]["lexicon_ambiguous"]["n"]
    by = collections.defaultdict(list)
    for r in sel["runs"]:
        by[r["zupitza"], r["lr"], r["epochs"]].append(r)

    def exact(rs: Sequence[dict[str, Any]], key: str) -> float:
        """The share of token-runs right over runs rs on the n dev tokens both settings share, from exact counts.

        key: the run field holding a two-decimal percentage on those tokens (dev_lexamb or dev_lexamb_shared)."""
        return sum(counts_from_pcts([r[key] for r in rs], n)) / (n * len(rs))

    nozup = {k: exact(rs, "dev_lexamb") for k, rs in by.items() if not k[0]}
    best = sorted(nozup.items(), key=lambda kv: -kv[1])
    choice = (False, sel["choice"]["lr"], sel["choice"]["epochs"])
    need(best[0][0] == choice, "the exact counts pick the same setting")
    prior = sel["d3_logistic_john_ws"]["without_zupitza"]["lexicon_ambiguous"]["frequency"]
    prior = counts_from_pcts([prior], n)[0] / n
    put("SelDevCorrect", num(sum(counts_from_pcts([r["dev_lexamb"] for r in by[choice]], n))))
    put("SelDevTokenRuns", num(n * len(by[choice])))
    put("SelDevExact", pct(best[0][1]))
    put("SelDevRunnerUp", pct(best[1][1]))
    put("SelDevOverRunnerUp", f"{100 * (best[0][1] - best[1][1]):.2f}")  # unrounded: may differ by 0.01 from the
    # difference of the two printed values
    put("SelDevOverPrior", f"{100 * (best[0][1] - prior):.2f}")
    d3 = sel["d3_logistic_john_ws"]
    put("DevWordZupShared", f"{d3['with_zupitza']['lexicon_ambiguous_shared_7639']['word']['mean']:.2f}")
    put("DevWordNoZup", f"{d3['without_zupitza']['lexicon_ambiguous']['word']['mean']:.2f}")
    collapsed = {(c["lr"], c["epochs"]) for c in sel["configs"] if c["collapses"]}
    matched = {}
    for (z, lr, ep), rs in by.items():
        if not z and (lr, ep) not in collapsed:
            matched[lr, ep] = nozup[False, lr, ep] - exact(by[True, lr, ep], "dev_lexamb_shared")
    for (lr, ep), d in sorted(matched.items()):
        put(f"DThreeEnc{'Low' if lr < 2e-4 else 'High'}Lr{['One', 'Three', 'Five'][[1, 3, 5].index(ep)]}Ep", pts(d))
    put("DThreeEncSettings", len(matched))
    put("DThreeEncMin", pts(min(matched.values())))
    put("DThreeEncMax", pts(max(matched.values())))
    # what Zupitza's Grammar changes (decision D3)
    zup = list(E.tokens(E.load(DATA / "by_source" / "zupitza_grammar.jsonl")))
    W, Z = ctx.preds["W", "john_ws"], jsonl(RES / "wikipron_with_zupitza" / "john_ws_predictions.jsonl")

    def right(p: dict[str, Any]) -> bool:
        """Whether the prior's pattern on prediction row p is the gold."""
        return p["frequency_prediction"] == p["gold_pattern"]

    for k in ("þes", "gyt", "lytel"):
        z = [t for t in zup if t["key"] == k]
        need(z and all("1" not in t["gold_pattern"] for t in z), f"Zupitza marks {k} short")
        put(f"ZupShort{word_tag(k)}", len(z))
        put(f"DevGain{word_tag(k)}", sum(right(w) - right(x) for w, x in zip(W, Z) if w["key"] == k))
    W, Z = ctx.preds["W", "soliloquies"], jsonl(RES / "wikipron_with_zupitza" / "soliloquies_predictions.jsonl")
    P, lex = ctx.preds["P", "soliloquies"], ctx.lex["W"][0]
    words = ("forþam", "swaswa", "forþi")
    for k in words:
        ix = [i for i, w in enumerate(W) if w["key"] == k]
        need(not lex.get(k), f"{k} is unknown without Zupitza")
        put(f"ZupWord{word_tag(k)}", len(ix))
        put(f"ZupWord{word_tag(k)}Right", sum(right(Z[i]) for i in ix))
        put(f"ZupWord{word_tag(k)}Proposal", sum(right(P[i]) for i in ix))
    put("ZupWords", sum(w["key"] in words for w in W))
    put("ZupWordsRight", sum(w["key"] in words and right(z) for w, z in zip(W, Z)))
    only = [w for w, z in zip(W, Z) if right(z) and not right(w)]
    nocand = [w for w in only if w["gold_pattern"] not in lex.get(w["key"], {})]
    put("ZupOnlyRight", len(only))
    put("ZupOnlyNoCand", len(nocand))
    put("ZupOnlyUnknown", sum(not lex.get(w["key"]) for w in nocand))
    put("ZupOnlyCounts", len(only) - len(nocand))
    put("DThreeSolNet", sum(right(z) - right(w) for w, z in zip(W, Z)))
    # all prose tokens at the chosen setting, with and without Zupitza: means of the runs' two-decimal percentages
    sol = {z: round(statistics.mean(r["sol_all"] for r in by[z, *choice[1:]]), 2) for z in (True, False)}
    put("EncSolAllZup", f"{sol[True]:.2f}")
    put("EncSolAllNoZup", f"{sol[False]:.2f}")
    put("EncDThreeSolCost", f"{sol[True] - sol[False]:.2f}")


def conjoined(put: Numbers, ctx: Context) -> None:
    """The bilateral logistic model plus the four target-word-by-neighbour features, seeds 1 to 3.

    Trains it with the WikiPron lexicon on ctx.train once per set and seed (experiment.predict, arm "conjoined") and
    records its mean and sample SD per population (Conj..., dev: Lex and ClsB only), full-cap local2 minus conjoined
    on the lexicon-ambiguous and class B test tokens (LocalTwoVsConj...), and conjoined minus the prior on the
    ambiguous dev tokens (ConjDevLexVsFreq...). Each contrast is a sentence bootstrap of correctness averaged over runs.
    Asserts that every conjoined run gets both þe of soliloquies:00026 right."""
    lex, counts = ctx.lex["W"]
    out = {}
    pop: str | None  # a population name, then an optional class letter
    for st, s in SETS:
        test = ctx.held[s]
        runs = [E.predict(ctx.train, test, lex, counts, "conjoined", seed) for seed in (1, 2, 3)]
        if s == "soliloquies":  # the worked example: every conjoined run gets the pronoun þē long and the relative short
            ex = [i for i, r in enumerate(test) if r["sentence_id"] == "soliloquies:00026"]
            need(all((p[ex[3]], p[ex[6]]) == ("1", "0") for p in runs), "soliloquies:00026: the conjoined runs")
        ev = [E.evaluate(test, p, lex, counts, ctx.cls) for p in runs]
        for name, pop in POPS if s != "john_ws" else (("Lex", "lexicon_ambiguous"), ("ClsB", "class_B")):
            vals = [e["accuracy"] if pop == "accuracy" else e[pop]["accuracy"] for e in ev]
            put(f"Conj{st}{name}", pct(statistics.mean(vals)))
            put(f"Conj{st}{name}Sd", pct(statistics.stdev(vals)))
        out[s] = {r["id"]: sum(p[i] == r["gold_pattern"] for p in runs) / 3 for i, r in enumerate(test)}
    l2 = runs_of(next(iter(ctx.enc["soliloquies"].values())), "local2", "capfull")
    for st, s in TESTS:
        table = ctx.enc[s]
        for name, pop in (("Lex", None), ("ClsB", "B")):
            ix = [t for t in table.values()
                  if t["capfull:lexicon_ambiguous"] == "1" and pop in (None, t["capfull:class"])]
            b = bootstrap_diff([t["sentence_id"] for t in ix], [correct(t, l2) for t in ix],
                               [out[s][t["id"]] for t in ix])
            for k, v in (("", "diff"), ("Lo", "lo"), ("Hi", "hi")):
                put(f"LocalTwoVsConj{st}{name}{k}", pts(b[v]))
    dev = [t for t in ctx.enc["john_ws"].values() if t["capfull:lexicon_ambiguous"] == "1"]
    b = bootstrap_diff([t["sentence_id"] for t in dev], [out["john_ws"][t["id"]] for t in dev],
                       [float(t["capfull:frequency"] == t["gold"]) for t in dev])
    for k, v in (("", "diff"), ("Lo", "lo"), ("Hi", "hi")):
        put(f"ConjDevLexVsFreq{k}", pts(b[v]))


def editors(put: Numbers, ctx: Context) -> None:
    """Editorial convention in Beowulf: the Wilson interval of Wyatt's agreement with Klaeber (Tauber) on class B
    (PairWTClsBWilson...), the range of alignment rates (Align...), the disagreement counts (PairHWDis, WTClsBDis...),
    the gold swap, which scores full-cap local2 and the word arm against each of Wyatt and Klaeber on their shared class
    B tokens (Swap...), and the article se (Se...). SwapRatio is the prior's swing between the two golds over the
    local2 gain against Klaeber.

    Reads results/agreement/metrics.json and disagreements.tsv and data/ambiguity_classes.csv. Exits when
    disagreements.tsv no longer matches the metrics, the class B pairs no longer match their count, Wyatt or Klaeber
    marks fewer than 95% of the pronouns þē and gē long (found by Tauber's glosses that start with you), or the
    example_se line no longer holds."""
    ag = json.loads((RES / "agreement" / "metrics.json").read_text(encoding="utf8"))
    wt = ag["pairs"]["wyatt-tauber"]["class_B"]
    k = round(wt["agreement"] * wt["n"])
    lo, hi = wilson(k, wt["n"])
    put("PairWTClsBWilsonLo", pct(lo))
    put("PairWTClsBWilsonHi", pct(hi))
    rates = [v for p in ag["pairs"].values() for kk, v in p.items() if kk.startswith("alignment_rate_")]
    put("AlignMin", pct(min(rates), 1))
    put("AlignMax", pct(max(rates), 1))
    with open(RES / "agreement" / "disagreements.tsv", encoding="utf8") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    hw = ag["pairs"]["hs-wyatt"]["all"]
    need(sum(r["pair"] == "hs-wyatt" for r in rows) == hw["n"] - round(hw["agreement"] * hw["n"]),
         "disagreements.tsv matches the Harrison and Sharp and Wyatt agreement")
    put("PairHWDis", num(sum(r["pair"] == "hs-wyatt" for r in rows)))
    cls_b = {s for s, c in ctx.cls.items() if c == "B"}
    dis = [r for r in rows if r["pair"] == "wyatt-tauber" and r["key"] in cls_b]
    need(len(dis) == wt["n"] - k, "disagreements.tsv matches the Wyatt and Tauber class B agreement")
    put("WTClsBDis", len(dis))
    put("WTClsBDisWyattShort", sum("1" not in r["a_pattern"] and "1" in r["b_pattern"] for r in dis))
    for key, c in collections.Counter(r["key"] for r in dis).items():  # over all of them, whichever edition is short
        put(f"WTClsBDis{word_tag(key)}", c)
    # ByT5 local2 and the word arm scored against Wyatt and against Klaeber (Tauber) on the class B tokens they share
    cls = classes(DATA / "ambiguity_classes.csv")
    pairs = [(x, y) for x, y in align(ctx.wy, ctx.tb) if cls.get(x["key"]) == "B"]
    need(len(pairs) == wt["n"], "the Wyatt and Tauber class B tokens")
    glosses = {t["id"]: g for t, (_, g) in zip(ctx.tb, ctx.tb_cols)}
    you = [(x, y) for x, y in pairs if glosses[y["id"]].startswith("you")]  # þē 'thee' and gē 'ye', by Tauber's glosses
    need(you and all(sum(p[i]["gold_pattern"] == "1" for p in you) >= 0.95 * len(you) for i in (0, 1)),
         "Wyatt and Klaeber nearly always mark the pronouns þē and gē long")
    beo = ctx.enc["tauber_beowulf"]
    l2, word = runs_of(next(iter(beo.values())), "local2", "capfull"), runs_of(next(iter(beo.values())), "word", "capfull")

    def agree(cols: Sequence[str], pair: tuple[dict[str, Any], dict[str, Any]], gi: int) -> float:
        """Share of the runs cols that give edition gi's pattern (0 Wyatt, 1 Tauber) for the Tauber token of pair."""
        return sum(beo[pair[1]["id"]][c] == pair[gi]["gold_pattern"] for c in cols) / len(cols)

    gain = {}
    for name, gi in (("Wyatt", 0), ("Tauber", 1)):
        a, b = [agree(l2, p, gi) for p in pairs], [agree(word, p, gi) for p in pairs]
        put(f"SwapLocalTwo{name}", pct(sum(a) / len(a)))
        r = bootstrap_diff([y["sentence_id"] for _, y in pairs], a, b)
        gain[name] = r["diff"]
        for kk, v in (("", "diff"), ("Lo", "lo"), ("Hi", "hi")):
            put(f"SwapGain{name}{kk}", pts(r[v]))
        for key in ("þe", "ne"):
            te = sum(x - y for (_, t), x, y in zip(pairs, a, b) if t["key"] == key)
            put(f"SwapTe{word_tag(key)}{name}", f"{te:+.1f}")
    put("SwapRatio", f"{(wt['prior_vs_wyatt'] - wt['prior_vs_tauber']) / gain['Tauber']:.2f}")
    # the Beowulf article se
    se = example_se(ctx.tb, ctx.tb_cols, ctx.preds["W", "tauber_beowulf"], beo)
    art = [t for t, gloss in se if gloss == "the"]
    pron = [t for t, gloss in se if gloss != "the"]
    put("SeTokens", len(se))
    put("SeArticle", len(art))
    put("SeArticleShort", sum(t["gold_pattern"] == "0" for t in art))
    put("SeOther", len(pron))
    put("SeOtherLong", sum(t["gold_pattern"] == "1" for t in pron))
    train = collections.Counter((t["source"], t["gold_pattern"]) for t in ctx.train if t["key"] == "se")
    put("SeTrainLong", num(sum(v for (_, p), v in train.items() if p == "1")))
    put("SeTrainShort", num(sum(v for (_, p), v in train.items() if p == "0")))
    for name, w in (("Reader", "bright_reader"), ("Matthew", "matthew_ws"), ("Sweet", "sweet_primer"),
                    ("Physiologus", "physiologus")):
        put(f"SeTrain{name}Long", train[w, "1"])
        put(f"SeTrain{name}Short", train[w, "0"])
    short = [t for t in art if t["gold_pattern"] == "0"]
    prior = {p["id"]: p["frequency_prediction"] for p in ctx.preds["W", "tauber_beowulf"]}
    put("SeArticlePrior", sum(prior[t["id"]] == "0" for t in short))
    put("SeArticleBestRun", max(sum(beo[t["id"]][c] == "0" for t in short) for c in runs_of(beo[short[0]["id"]])))
    put("SeLineLongRuns", len(runs_of(beo[short[0]["id"]])))  # all runs: example_se asserts each marks line 2587 long


def errors(put: Numbers, ctx: Context) -> None:
    """The sampled error types of ByT5 (encoder_sentence, the full-cap sentence arm, seed 1) and CharsiuG2P: kappa and
    counts per system, set and type (ErrKappa..., Err...), the homograph share (ErrShare...), two-sided Fisher tests
    per type of class A, B and C (homograph, grammatical, editorial) and a chi-squared test over the three
    (ErrFisher..., ErrChi...), the fixed rule (ErrRule...), label agreement with and without a class (ErrItems...,
    ErrKappa..., ErrAgree...), the words behind some types, the short prose to (ToTokens, ShortTo, ClsCPriorErr...,
    ToSeventyRuns, ErrEncSolEdToHe) and Beowulf line 2796 (LineRelativeLongRuns).

    Reads results/error_typing/summary.json besides ctx.labels. Exits when an error has none of the six types, an se
    error no longer writes sē, the prior or a local2 run (any cap) no longer writes tō on every short prose to,
    soliloquies:00070 no longer has one short to, or example_2796 no longer holds."""
    L = ctx.labels
    systems = (("Enc", "encoder_sentence"), ("Charsiu", "charsiug2p"))
    types = (("Ceiling", "ceiling"), ("GoldInDoubt", "gold_in_doubt"), ("Homograph", "homograph"),
             ("Grammatical", "grammatical"), ("Editorial", "editorial"), ("Other", "other"))
    summary = json.loads((RES / "error_typing" / "summary.json").read_text(encoding="utf8"))
    s: str | None  # a set name, or None for all sets
    for sn, sysname in systems:  # error types per system and test set
        put(f"ErrKappa{sn}", f"{summary['agreement'][sysname]['cohen_kappa']:.3f}")
        for st, s in TESTS:
            sub = [r for r in L if r["system"] == sysname and r["set"] == s]
            put(f"Err{sn}{st}N", len(sub))
            c = collections.Counter(r["final"] for r in sub)
            need(sum(c[t] for _, t in types) == len(sub), f"{sysname} {s}: every error has one of the six types")
            for name, t in types:
                put(f"Err{sn}{st}{name}", c[t])
    for st, s in (("All", None),) + TESTS:  # ByT5 against CharsiuG2P, Fisher's exact test per type
        def tab(t: str) -> list[list[int]]:
            """The 2x2 table, ByT5 and CharsiuG2P by errors of type t and of any other type, in set s (None: all)."""
            return [[sum(r["system"] == sysname and s in (None, r["set"]) and r["final"] == t for r in L),
                     sum(r["system"] == sysname and s in (None, r["set"]) and r["final"] != t for r in L)]
                    for _, sysname in systems]
        for name, t in types[2:5]:
            put(f"ErrFisher{st}{name}", f"{fisher_exact(tab(t))[1]:.2g}")
        abc = [[tab(t)[i][0] for _, t in types[2:5]] for i in (0, 1)]  # systems by type, over A, B and C
        put(f"ErrChi{st}", f"{chi2_contingency(abc)[1]:.2g}")
        if s:
            for sn, sysname in systems:
                sub = [r for r in L if r["system"] == sysname and r["set"] == s]
                put(f"ErrShare{sn}{st}Homograph", pct(share(r["final"] == "homograph" for r in sub), 1))
    rule = [(r, fixed_rule(r)) for r in L]
    put("ErrRuleDecided", sum(t is not None for _, t in rule))
    put("ErrRuleMatch", sum(t == r["final"] for r, t in rule))
    for name, keep in (("NoClass", False), ("Class", True)):
        sub = [r for r in L if bool(r["class"]) == keep]
        put(f"ErrItems{name}", len(sub))
        put(f"ErrKappa{name}", f"{kappa([r['label_a'] for r in sub], [r['label_b'] for r in sub]):.3f}")
        put(f"ErrAgree{name}", pct(rate([r["label_a"] for r in sub], [r["label_b"] for r in sub]), 1))

    def items(system: str, s: str, final: str) -> list[dict[str, str]]:
        """The label rows of one system and set whose adjudicated type is final."""
        return [r for r in L if r["system"] == system and r["set"] == s and r["final"] == final]

    gram = collections.Counter(E.key(r["gold"]) for r in items("encoder_sentence", "soliloquies", "grammatical"))
    for k in ("þe", "ne", "ge"):
        put(f"ErrEncSolGram{word_tag(k)}", gram[k])
    put("ErrEncBeoHomGeata", sum(E.key(r["gold"]) == "geata" for r in items("encoder_sentence", "tauber_beowulf",
                                                                             "homograph")))
    se = [r for r in items("encoder_sentence", "tauber_beowulf", "editorial") if E.key(r["gold"]) == "se"]
    need(all(E.pattern(r["system_form"]) == "1" and E.pattern(r["gold"]) == "0" for r in se), "the se errors write sē")
    put("ErrEncBeoEdSe", len(se))
    cg = [r for r in L if r["system"] == "charsiug2p" and r["final"] == "grammatical"]
    put("ErrCharsiuGram", len(cg))
    put("ErrCharsiuGramTheNe", sum(E.key(r["gold"]) in ("þe", "ne") and E.pattern(r["gold"]) == "0"  # written long
                                   and E.pattern(r["system_form"]) == "1" for r in cg))
    # short to in the prose gold
    sol = ctx.enc["soliloquies"]
    keys = {r["id"]: r["key"] for r in ctx.held["soliloquies"]}
    to = [t for t in sol.values() if keys[t["id"]] == "to"]
    short = [t for t in to if t["gold"] == "0"]
    need(all(t["capfull:frequency"] == "1" and all(t[c] == "1" for c in runs_of(t, "local2")) for t in short),
         "the prior and every local2 run write tō on every short to")
    put("ToTokens", len(to))
    put("ShortTo", len(short))
    c_err = [t for t in sol.values() if t["capfull:lexicon_ambiguous"] == "1" and t["capfull:class"] == "C"
             and t["capfull:frequency"] != t["gold"]]
    put("ClsCPriorErr", len(c_err))
    put("ClsCPriorErrShortTo", sum(keys[t["id"]] == "to" for t in c_err))
    t70 = [t for t in to if t["sentence_id"] == "soliloquies:00070"]
    need(len(t70) == 1 and t70[0]["gold"] == "0", "soliloquies:00070 has one short to")
    put("ToSeventyRuns", sum(t70[0][c] == "1" for c in runs_of(t70[0])))
    put("ErrEncSolEdToHe", sum(E.key(r["gold"]) in ("to", "he") and E.pattern(r["gold"]) == "0"
                               for r in items("encoder_sentence", "soliloquies", "editorial")))
    put("LineRelativeLongRuns", example_2796(ctx.enc["tauber_beowulf"], ctx))


def charsiu(put: Numbers, ctx: Context) -> None:
    """CharsiuG2P: the word types it was run on and its dictionary size (CharsiuTypes, CharsiuDictWords), its length
    errors on lexicon-ambiguous test tokens and how many of them the WikiPron prior and the seed 1 logistic word and
    bilateral models get right (CharsiuLexWrong..., CharsiuFix...), its five commonest prose error words
    (CharsiuErr...), and its prose wrong-length slots and accuracy in and out of its dictionary (CharsiuTooLong,
    CharsiuTooShort, CharsiuInDict..., CharsiuNotInDict...).

    Its forms come from results/charsiu/ipa.tsv through wikipron_to_marked.mark, keyed by the lower-cased bare word. A
    token with no IPA or one mark cannot align is skipped. The dictionary size counts the words of
    data/charsiu_ang_dict_words.txt. Exits unless its error counts match results/charsiu/metrics.json."""
    with open(RES / "charsiu" / "ipa.tsv", encoding="utf8") as f:
        ipa = {r["word"]: r["ipa"] for r in csv.DictReader(f, delimiter="\t")}
    put("CharsiuTypes", num(len(ipa)))
    put("CharsiuDictWords", num(len((DATA / "charsiu_ang_dict_words.txt").read_text(encoding="utf8").split())))
    lex = ctx.lex["W"][0]
    m = json.loads((RES / "charsiu" / "metrics.json").read_text(encoding="utf8"))
    for st, s in TESTS:  # what the WikiPron prior and the logistic models do on CharsiuG2P's length errors
        preds = {p["id"]: p for p in ctx.preds["W", s]}
        wrong = []
        for r in ctx.held[s]:
            form = mark(r["bare"].lower(), ipa.get(r["bare"].lower(), ""))
            if r["gold_pattern"] and form is not None and E.pattern(form) != r["gold_pattern"] \
                    and len(lex.get(r["key"], ())) >= 2:
                wrong.append(r)
        need(len(wrong) == m[s]["wrong_length_on_ambiguous"], f"{s}: CharsiuG2P's length errors")
        put(f"CharsiuLexWrong{st}", num(len(wrong)))
        for name, k in (("Prior", "frequency_prediction"), ("WordSeedOne", "word_seed1_prediction"),
                        ("BilSeedOne", "sentence_seed1_prediction")):
            fixed = sum(preds[r["id"]][k] == r["gold_pattern"] for r in wrong)
            put(f"CharsiuFix{name}{st}", num(fixed))
            if name == "Prior":
                put(f"CharsiuFixPrior{st}Share", pct(fixed / len(wrong)))
        if s == "soliloquies":
            top = collections.Counter(r["key"] for r in wrong).most_common(5)
            for k, c in top:
                put(f"CharsiuErr{word_tag(k)}", c)
            put("CharsiuErrTopFive", num(sum(c for _, c in top)))
    sol = m["soliloquies"]
    put("CharsiuTooLong", num(sol["wrong_length_slots"]["too_long"]))
    put("CharsiuTooShort", num(sol["wrong_length_slots"]["too_short"]))
    d = sol["by_charsiu_dictionary"]
    for name, side in (("InDict", "in_dict"), ("NotInDict", "not_in_dict")):
        n = sum(v for k, v in d.items() if k.startswith(side + "/"))
        put(f"Charsiu{name}N", num(n))
        put(f"Charsiu{name}", pct(d[side + "/correct"] / n))


def figure(ctx: Context) -> None:
    """Figure 1: the prior and the achievable ceiling per candidate source, and ByT5 local2 at + WikiPron.

    All-token accuracy on the Soliloquies and Beowulf from each source's metrics.json, and the full-cap local2 mean
    over three seeds from summary.json. Sets matplotlib's global rcParams and writes report/generated/fig_coverage.pdf
    without a timestamp. The folder must exist (main makes it)."""
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Liberation Serif", "Nimbus Roman", "Times New Roman",
                         "DejaVu Serif"], "font.size": 11, "axes.titlesize": 11, "axes.labelsize": 11,
                         "xtick.labelsize": 10, "ytick.labelsize": 11, "legend.fontsize": 11, "pdf.fonttype": 42})
    blue, orange, green = "#2a78d6", "#eb6834", "#1baf7a"  # validated categorical slots 1-3 (dataviz palette)
    xs, labels = range(4), ["Training\nforms", "+ WikiPron", "+ three\ndictionaries", "+ all\nfour"]
    full = configs(ctx.summary)["capfull"]
    fig, axes = plt.subplots(1, 2, figsize=(15 / 2.54, 6.2 / 2.54), sharey=True)  # one scale: genres compare directly
    for ax, (s, title) in zip(axes, (("soliloquies", "Prose (Soliloquies)"), ("tauber_beowulf", "Verse (Beowulf)"))):
        prior = [100 * ctx.metrics[tag][s]["frequency"]["accuracy"] for tag, _, _ in SETTINGS]
        ceil = [100 * ctx.metrics[tag][s]["oracle"]["accuracy"] for tag, _, _ in SETTINGS]
        ax.plot(xs, ceil, "--o", color=orange, markerfacecolor="white", markeredgewidth=1.5, markersize=6.5,
                linewidth=1.5, label="Achievable ceiling")
        ax.plot(xs, prior, "-o", color=blue, markeredgecolor="white", markeredgewidth=0.8, markersize=7,
                linewidth=1.5, label="Frequency prior")
        ax.plot([1.15], [100 * full["arms"]["local2"]["sets"][s]["accuracy"]["mean"]], "*", color=green, markersize=11,
                markeredgecolor="white", markeredgewidth=0.6, label="ByT5 local2", zorder=3)
        ax.set_xticks(list(xs), labels)
        ax.set_xlim(-0.35, 3.35)
        ax.set_title(title)
        ax.spines[["top", "right"]].set_visible(False)
        ax.yaxis.grid(True, color="#e6e6e6", linewidth=0.6)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("All-token accuracy (%)")
    handles, names = axes[0].get_legend_handles_labels()
    order = [1, 0, 2]
    fig.legend([handles[i] for i in order], [names[i] for i in order], loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, -0.02), handletextpad=0.3, columnspacing=1.2)
    fig.tight_layout(rect=(0, 0.1, 1, 1), w_pad=1.5)
    fig.savefig(GEN / "fig_coverage.pdf", metadata={"CreationDate": None})  # no timestamp: reruns are byte-identical
    plt.close(fig)


def main() -> None:
    """Run every section, then write report/generated/numbers_paper.tex (macros sorted by name) and Figure 1.

    A failed check exits before anything is written. Prints one summary line at the end."""
    ctx = Context()
    put = Numbers()
    for section in (corpus, candidates, coverage, context, selection, conjoined, editors, errors, charsiu):
        section(put, ctx)
    GEN.mkdir(parents=True, exist_ok=True)
    (GEN / "numbers_paper.tex").write_text(
        "% generated by src/oevl/paper_numbers.py from data/ and results/; do not edit\n"
        + "".join(f"\\newcommand{{\\NP{k}}}{{{v}}}\n" for k, v in sorted(put.items())), encoding="utf8")
    figure(ctx)
    print(f"wrote {len(put)} numbers to report/generated/numbers_paper.tex and report/generated/fig_coverage.pdf, "
          "and the worked examples hold")


if __name__ == "__main__":
    main()
