"""Fail the run if any headline value differs from the frozen reference results.

Checks counts exactly, proportions to 0.00005 and McNemar p-values to 0.0005.
Usage: python -m oevl.check_expected
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROP_TOL, P_TOL = 0.00005, 0.0005
POPS = ("accuracy", "lexicon_ambiguous", "attested_ambiguous", "minority_reading")

# (folder, test) -> population -> n, frequency, word, left, bilateral (means over seeds 1-3), oracle
EXPECTED = {
    ("results_wikipron", "john_ws"): {
        "accuracy": (16649, .9589, .9570, .9565, .9568, .9876),
        "lexicon_ambiguous": (7782, .9487, .9447, .9436, .9444, .9994),
        "attested_ambiguous": (6026, .9492, .9493, .9483, .9497, 1.0),
        "minority_reading": (382, .0969, .0262, .0489, .0820, 1.0),
    },
    ("results_wikipron", "soliloquies"): {
        "accuracy": (15431, .8872, .8890, .8895, .8901, .9411),
        "lexicon_ambiguous": (5968, .8814, .8861, .8874, .8890, .9992),
        "attested_ambiguous": (4658, .8718, .8725, .8748, .8780, 1.0),
        "minority_reading": (638, .0047, .0094, .0329, .0617, 1.0),
    },
    ("results_wikipron", "tauber_beowulf"): {
        "accuracy": (17361, .8944, .8940, .8933, .8932, .9273),
        "lexicon_ambiguous": (4434, .8872, .8857, .8832, .8827, 1.0),
        "attested_ambiguous": (3394, .9057, .9067, .9058, .9062, 1.0),
        "minority_reading": (358, .0251, .0372, .0428, .0484, 1.0),
    },
    ("results_training_only", "john_ws"): {
        "accuracy": (16649, .9504, .9487, .9485, .9493, .9773),
        "lexicon_ambiguous": (7447, .9526, .9487, .9482, .9500, .9993),
        "attested_ambiguous": (6026, .9492, .9491, .9483, .9504, 1.0),
        "minority_reading": (382, .0969, .0366, .0663, .1143, 1.0),
    },
    ("results_training_only", "soliloquies"): {
        "accuracy": (15431, .8725, .8728, .8733, .8745, .9211),
        "lexicon_ambiguous": (5569, .8833, .8840, .8854, .8888, .9991),
        "attested_ambiguous": (4658, .8718, .8733, .8762, .8813, 1.0),
        "minority_reading": (638, .0047, .0157, .0439, .0883, 1.0),
    },
    ("results_training_only", "tauber_beowulf"): {
        "accuracy": (17361, .8275, .8281, .8279, .8276, .8532),
        "lexicon_ambiguous": (4085, .9114, .9138, .9130, .9118, 1.0),
        "attested_ambiguous": (3394, .9057, .9072, .9074, .9071, 1.0),
        "minority_reading": (358, .0251, .0363, .0624, .0764, 1.0),
    },
    ("results_proposal", "john_ws"): {
        "accuracy": (16649, .9611, .9581, .9578, .9579, .9904),
        "lexicon_ambiguous": (8951, .9523, .9467, .9461, .9463, .9994),
        "attested_ambiguous": (6026, .9492, .9496, .9482, .9497, 1.0),
        "minority_reading": (382, .0969, .0262, .0340, .0681, 1.0),
    },
    ("results_proposal", "soliloquies"): {
        "accuracy": (15431, .8985, .9008, .9008, .9011, .9626),
        "lexicon_ambiguous": (7703, .8855, .8900, .8902, .8906, .9988),
        "attested_ambiguous": (4658, .8718, .8730, .8746, .8761, 1.0),
        "minority_reading": (638, .0047, .0131, .0303, .0522, 1.0),
    },
    ("results_proposal", "tauber_beowulf"): {
        "accuracy": (17361, .9206, .9188, .9174, .9173, .9640),
        "lexicon_ambiguous": (6015, .8820, .8768, .8728, .8726, .9995),
        "attested_ambiguous": (3394, .9057, .9075, .9058, .9066, 1.0),
        "minority_reading": (358, .0251, .0475, .0466, .0559, 1.0),
    },
    ("results_all", "john_ws"): {
        "accuracy": (16649, .9614, .9581, .9578, .9577, .9906),
        "lexicon_ambiguous": (8953, .9528, .9465, .9459, .9458, .9994),
        "attested_ambiguous": (6026, .9492, .9496, .9484, .9496, 1.0),
        "minority_reading": (382, .0969, .0262, .0358, .0663, 1.0),
    },
    ("results_all", "soliloquies"): {
        "accuracy": (15431, .8995, .9010, .9012, .9013, .9627),
        "lexicon_ambiguous": (7707, .8872, .8902, .8907, .8908, .9988),
        "attested_ambiguous": (4658, .8718, .8733, .8751, .8761, 1.0),
        "minority_reading": (638, .0047, .0157, .0324, .0496, 1.0),
    },
    ("results_all", "tauber_beowulf"): {
        "accuracy": (17361, .9237, .9187, .9176, .9176, .9641),
        "lexicon_ambiguous": (6020, .8907, .8762, .8730, .8730, .9995),
        "attested_ambiguous": (3394, .9057, .9078, .9063, .9064, 1.0),
        "minority_reading": (358, .0251, .0503, .0503, .0521, 1.0),
    },
}

# (folder, test) -> pair -> (a_wrong_b_right, a_right_b_wrong, p)
MCNEMAR = {
    ("results_training_only", "john_ws"): {
        "frequency_vs_word_seed1": (7, 36, 8.96e-06),
        "frequency_vs_left_seed1": (16, 50, 3.33e-05),
        "frequency_vs_sentence_seed1": (28, 51, 0.0128),
        "word_seed1_vs_left_seed1": (10, 15, 0.424),
        "word_seed1_vs_sentence_seed1": (25, 19, 0.451),
        "left_seed1_vs_sentence_seed1": (23, 12, 0.0895),
    },
    ("results_training_only", "soliloquies"): {
        "frequency_vs_word_seed1": (12, 8, 0.503),
        "frequency_vs_left_seed1": (27, 19, 0.302),
        "frequency_vs_sentence_seed1": (47, 25, 0.0128),
        "word_seed1_vs_left_seed1": (20, 16, 0.618),
        "word_seed1_vs_sentence_seed1": (40, 22, 0.03),
        "left_seed1_vs_sentence_seed1": (30, 16, 0.0541),
    },
    ("results_wikipron", "john_ws"): {
        "frequency_vs_word_seed1": (6, 36, 2.83e-06),
        "frequency_vs_left_seed1": (14, 55, 0.0),
        "frequency_vs_sentence_seed1": (29, 59, 0.00182),
        "word_seed1_vs_left_seed1": (8, 19, 0.0522),
        "word_seed1_vs_sentence_seed1": (25, 25, 1.0),
        "left_seed1_vs_sentence_seed1": (23, 12, 0.0895),
    },
    ("results_wikipron", "soliloquies"): {
        "frequency_vs_word_seed1": (43, 14, 0.000154),
        "frequency_vs_left_seed1": (59, 22, 4.77e-05),
        "frequency_vs_sentence_seed1": (77, 31, 1.12e-05),
        "word_seed1_vs_left_seed1": (16, 8, 0.152),
        "word_seed1_vs_sentence_seed1": (35, 18, 0.027),
        "left_seed1_vs_sentence_seed1": (27, 18, 0.233),
    },
    ("results_wikipron", "tauber_beowulf"): {
        "frequency_vs_word_seed1": (53, 64, 0.355),
        "frequency_vs_left_seed1": (52, 71, 0.104),
        "frequency_vs_sentence_seed1": (55, 75, 0.0953),
        "word_seed1_vs_left_seed1": (6, 14, 0.115),
        "word_seed1_vs_sentence_seed1": (15, 24, 0.2),
        "left_seed1_vs_sentence_seed1": (14, 15, 1.0),
    },
    ("results_proposal", "john_ws"): {
        "frequency_vs_word_seed1": (11, 66, 0.0),
        "frequency_vs_left_seed1": (18, 73, 0.0),
        "frequency_vs_sentence_seed1": (30, 80, 2.02e-06),
        "word_seed1_vs_left_seed1": (25, 25, 1.0),
        "word_seed1_vs_sentence_seed1": (32, 27, 0.603),
        "left_seed1_vs_sentence_seed1": (19, 14, 0.487),
    },
    ("results_proposal", "soliloquies"): {
        "frequency_vs_word_seed1": (122, 89, 0.0274),
        "frequency_vs_left_seed1": (147, 111, 0.0291),
        "frequency_vs_sentence_seed1": (156, 117, 0.0213),
        "word_seed1_vs_left_seed1": (34, 31, 0.804),
        "word_seed1_vs_sentence_seed1": (50, 44, 0.606),
        "left_seed1_vs_sentence_seed1": (36, 33, 0.81),
    },
    ("results_proposal", "tauber_beowulf"): {
        "frequency_vs_word_seed1": (175, 210, 0.083),
        "frequency_vs_left_seed1": (172, 227, 0.00679),
        "frequency_vs_sentence_seed1": (172, 230, 0.00441),
        "word_seed1_vs_left_seed1": (22, 42, 0.0169),
        "word_seed1_vs_sentence_seed1": (29, 52, 0.014),
        "left_seed1_vs_sentence_seed1": (21, 24, 0.766),
    },
    ("results_all", "john_ws"): {
        "frequency_vs_word_seed1": (10, 61, 0.0),
        "frequency_vs_left_seed1": (16, 80, 0.0),
        "frequency_vs_sentence_seed1": (24, 89, 0.0),
        "word_seed1_vs_left_seed1": (13, 26, 0.0533),
        "word_seed1_vs_sentence_seed1": (22, 36, 0.0869),
        "left_seed1_vs_sentence_seed1": (18, 19, 1.0),
    },
    ("results_all", "soliloquies"): {
        "frequency_vs_word_seed1": (119, 94, 0.0999),
        "frequency_vs_left_seed1": (139, 114, 0.131),
        "frequency_vs_sentence_seed1": (141, 119, 0.193),
        "word_seed1_vs_left_seed1": (30, 30, 1.0),
        "word_seed1_vs_sentence_seed1": (38, 41, 0.822),
        "left_seed1_vs_sentence_seed1": (36, 39, 0.818),
    },
    ("results_all", "tauber_beowulf"): {
        "frequency_vs_word_seed1": (143, 232, 5.01e-06),
        "frequency_vs_left_seed1": (137, 243, 0.0),
        "frequency_vs_sentence_seed1": (138, 248, 0.0),
        "word_seed1_vs_left_seed1": (18, 35, 0.027),
        "word_seed1_vs_sentence_seed1": (24, 45, 0.0154),
        "left_seed1_vs_sentence_seed1": (23, 27, 0.672),
    },
}
# training-only on Beowulf: no model differs from another at p < 0.12 (min p 0.1433)
MIN_P_TRAINING_ONLY = {"tauber_beowulf": 0.12}

# folder -> (aligned, all agreement, all kappa, ambiguous n, ambiguous agreement, ambiguous kappa)
AGREEMENT = {
    "results_wikipron": (16833, .9773, .9733, 4420, .9604, .9419),
    "results_training_only": (16833, .9773, .9733, 4076, .9585, .9366),
    "results_proposal": (16833, .9773, .9733, 5993, .9605, .9467),
    "results_all": (16833, .9773, .9733, 5998, .9602, .9463),
}

# results/agreement: editor pair -> (aligned, ambiguous n, ambiguous agreement, ambiguous kappa)
AGREEMENT_PAIRS = {
    "hs-wyatt": (16833, 4420, .9604, .9419),
    "hs-tauber": (16573, 4366, .9876, .9818),
    "wyatt-tauber": (16912, 4366, .9611, .9429),
}


def acc(d, pop):
    return d["accuracy"] if pop == "accuracy" else d[pop]["accuracy"]


def main() -> int:
    errors, checked = [], 0

    def close(label, got, want, tol):
        nonlocal checked
        checked += 1
        if got is None or abs(got - want) > tol:
            errors.append(f"{label}: got {got}, expected {want}")

    def equal(label, got, want):
        nonlocal checked
        checked += 1
        if got != want:
            errors.append(f"{label}: got {got}, expected {want}")

    metrics = {}
    for folder in ("results_wikipron", "results_training_only", "results_proposal", "results_all"):
        path = ROOT / "results" / folder.removeprefix("results_") / "metrics.json"  # keys keep the old folder names
        if not path.exists():
            errors.append(f"missing {path}")
            continue
        metrics[folder] = json.loads(path.read_text(encoding="utf8"))

    for (folder, test), pops in EXPECTED.items():
        d = metrics.get(folder, {}).get(test)
        if d is None:
            errors.append(f"{folder}/{test}: missing")
            continue
        for pop, (n, fr, wo, le, bi, orc) in pops.items():
            tag = f"{folder}/{test}/{pop}"
            equal(tag + "/n", d["frequency"]["n"] if pop == "accuracy" else d["frequency"][pop]["n"], n)
            close(tag + "/frequency", acc(d["frequency"], pop), fr, PROP_TOL)
            close(tag + "/oracle", acc(d["oracle"], pop), orc, PROP_TOL)
            for arm, want in (("word", wo), ("left", le), ("sentence", bi)):
                close(f"{tag}/{arm}_mean", d["seed_summary"][arm][pop]["mean"], want, PROP_TOL)

    for (folder, test), pairs in MCNEMAR.items():
        d = metrics.get(folder, {}).get(test, {}).get("mcnemar_seed1", {})
        for pair, (b01, b10, p) in pairs.items():
            r = d.get(pair)
            if r is None:
                errors.append(f"{folder}/{test}/mcnemar/{pair}: missing")
                continue
            equal(f"{folder}/{test}/mcnemar/{pair}/a_wrong_b_right", r["a_wrong_b_right"], b01)
            equal(f"{folder}/{test}/mcnemar/{pair}/a_right_b_wrong", r["a_right_b_wrong"], b10)
            close(f"{folder}/{test}/mcnemar/{pair}/p", r["p_exact"], p, P_TOL * max(1, p))
    for test, floor in MIN_P_TRAINING_ONLY.items():
        d = metrics.get("results_training_only", {}).get(test, {}).get("mcnemar_seed1", {})
        for pair, r in d.items():
            checked += 1
            if r["p_exact"] < floor:
                errors.append(f"results_training_only/{test}/mcnemar/{pair}: p={r['p_exact']} < {floor}")

    for folder, (al, ag, ka, an, aag, aka) in AGREEMENT.items():
        e = metrics.get(folder, {}).get("editor_agreement")
        if e is None:
            errors.append(f"{folder}/editor_agreement: missing")
            continue
        equal(f"{folder}/agreement/aligned", e["aligned"], al)
        close(f"{folder}/agreement/all", e["all"]["agreement"], ag, PROP_TOL)
        close(f"{folder}/agreement/all_kappa", e["all"]["cohen_kappa"], ka, PROP_TOL)
        equal(f"{folder}/agreement/ambiguous_n", e["lexicon_ambiguous"]["n"], an)
        close(f"{folder}/agreement/ambiguous", e["lexicon_ambiguous"]["agreement"], aag, PROP_TOL)
        close(f"{folder}/agreement/ambiguous_kappa", e["lexicon_ambiguous"]["cohen_kappa"], aka, PROP_TOL)

    apath = ROOT / "results" / "agreement" / "metrics.json"
    pairs = json.loads(apath.read_text(encoding="utf8"))["pairs"] if apath.exists() else {}
    for pair, (al, an, aag, aka) in AGREEMENT_PAIRS.items():
        e = pairs.get(pair)
        if e is None:
            errors.append(f"agreement/{pair}: missing")
            continue
        equal(f"agreement/{pair}/aligned", e["aligned"], al)
        equal(f"agreement/{pair}/ambiguous_n", e["lexicon_ambiguous"]["n"], an)
        close(f"agreement/{pair}/ambiguous", e["lexicon_ambiguous"]["agreement"], aag, PROP_TOL)
        close(f"agreement/{pair}/ambiguous_kappa", e["lexicon_ambiguous"]["cohen_kappa"], aka, PROP_TOL)

    if errors:
        print(f"check_expected: FAILED ({len(errors)} of {checked} checks)")
        print("\n".join(errors))
        return 1
    print(f"check_expected: PASSED ({checked} checks)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
