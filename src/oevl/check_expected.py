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
    ("results_wikipron", "soliloquies"): {
        "accuracy": (15480, .8869, .8887, .8892, .8895, .9408),
        "lexicon_ambiguous": (5946, .8806, .8854, .8865, .8874, .9992),
        "attested_ambiguous": (4633, .8707, .8714, .8736, .8759, 1.0),
        "minority_reading": (640, .0047, .0094, .0380, .0635, 1.0),
    },
    ("results_wikipron", "tauber_beowulf"): {
        "accuracy": (17361, .8944, .8944, .8936, .8934, .9273),
        "lexicon_ambiguous": (4434, .8872, .8873, .8841, .8836, 1.0),
        "attested_ambiguous": (3394, .9057, .9067, .9061, .9070, 1.0),
        "minority_reading": (358, .0251, .0317, .0428, .0540, 1.0),
    },
    ("results_training_only", "soliloquies"): {
        "accuracy": (15480, .8723, .8725, .8729, .8737, .9208),
        "lexicon_ambiguous": (5547, .8825, .8829, .8841, .8864, .9991),
        "attested_ambiguous": (4633, .8707, .8719, .8744, .8782, 1.0),
        "minority_reading": (640, .0047, .0135, .0411, .0745, 1.0),
    },
    ("results_training_only", "tauber_beowulf"): {
        "accuracy": (17361, .8275, .8279, .8279, .8276, .8532),
        "lexicon_ambiguous": (4085, .9114, .9128, .9131, .9115, 1.0),
        "attested_ambiguous": (3394, .9057, .9069, .9073, .9071, 1.0),
        "minority_reading": (358, .0251, .0363, .0624, .0773, 1.0),
    },
    ("results_proposal", "soliloquies"): {
        "accuracy": (15480, .8799, .8958, .8958, .8939, .9580),
        "lexicon_ambiguous": (9069, .8655, .8926, .8925, .8894, .9969),
        "attested_ambiguous": (4633, .8707, .8718, .8721, .8722, 1.0),
        "minority_reading": (640, .0047, .0172, .0198, .0266, 1.0),
    },
    ("results_proposal", "tauber_beowulf"): {
        "accuracy": (17361, .8434, .8854, .8837, .8834, .9433),
        "lexicon_ambiguous": (8412, .7902, .8768, .8733, .8727, .9955),
        "attested_ambiguous": (3394, .9057, .9054, .9053, .9051, 1.0),
        "minority_reading": (358, .0251, .0447, .0419, .0410, 1.0),
    },
    ("results_all", "soliloquies"): {
        "accuracy": (15480, .8799, .8995, .8996, .8979, .9623),
        "lexicon_ambiguous": (9149, .8589, .8920, .8923, .8893, .9965),
        "attested_ambiguous": (4633, .8707, .8718, .8723, .8724, 1.0),
        "minority_reading": (640, .0047, .0172, .0219, .0271, 1.0),
    },
    ("results_all", "tauber_beowulf"): {
        "accuracy": (17361, .8434, .8967, .8942, .8936, .9597),
        "lexicon_ambiguous": (8697, .7653, .8717, .8667, .8654, .9968),
        "attested_ambiguous": (3394, .9057, .9054, .9053, .9050, 1.0),
        "minority_reading": (358, .0251, .0447, .0419, .0410, 1.0),
    },
}

# (folder, test) -> pair -> (a_wrong_b_right, a_right_b_wrong, p)
MCNEMAR = {
    ("results_wikipron", "soliloquies"): {
        "frequency_vs_word_seed1": (43, 15, 0.000307),
        "frequency_vs_left_seed1": (64, 33, 0.00215),
        "frequency_vs_sentence_seed1": (80, 46, 0.00313),
        "word_seed1_vs_left_seed1": (22, 19, 0.755),
        "word_seed1_vs_sentence_seed1": (39, 33, 0.556),
        "left_seed1_vs_sentence_seed1": (31, 28, 0.795),
    },
    ("results_wikipron", "tauber_beowulf"): {
        "frequency_vs_word_seed1": (51, 51, 1.0),
        "frequency_vs_left_seed1": (54, 71, 0.152),
        "frequency_vs_sentence_seed1": (59, 78, 0.124),
        "word_seed1_vs_left_seed1": (11, 28, 0.00948),
        "word_seed1_vs_sentence_seed1": (20, 39, 0.0183),
        "left_seed1_vs_sentence_seed1": (14, 16, 0.856),
    },
    ("results_proposal", "soliloquies"): {
        "frequency_vs_word_seed1": (366, 121, 0.0),
        "frequency_vs_left_seed1": (378, 131, 0.0),
        "frequency_vs_sentence_seed1": (397, 173, 0.0),
        "word_seed1_vs_left_seed1": (19, 17, 0.868),
        "word_seed1_vs_sentence_seed1": (40, 61, 0.046),
        "left_seed1_vs_sentence_seed1": (26, 49, 0.0106),
    },
    ("results_proposal", "tauber_beowulf"): {
        "frequency_vs_word_seed1": (1080, 342, 0.0),
        "frequency_vs_left_seed1": (1058, 356, 0.0),
        "frequency_vs_sentence_seed1": (1057, 363, 0.0),
        "word_seed1_vs_left_seed1": (30, 66, 0.000306),
        "word_seed1_vs_sentence_seed1": (52, 96, 0.000373),
        "left_seed1_vs_sentence_seed1": (39, 47, 0.451),
    },
    ("results_all", "soliloquies"): {
        "frequency_vs_word_seed1": (433, 130, 0.0),
        "frequency_vs_left_seed1": (445, 141, 0.0),
        "frequency_vs_sentence_seed1": (455, 178, 0.0),
        "word_seed1_vs_left_seed1": (21, 20, 1.0),
        "word_seed1_vs_sentence_seed1": (34, 60, 0.00955),
        "left_seed1_vs_sentence_seed1": (22, 49, 0.00182),
    },
    ("results_all", "tauber_beowulf"): {
        "frequency_vs_word_seed1": (1289, 350, 0.0),
        "frequency_vs_left_seed1": (1259, 368, 0.0),
        "frequency_vs_sentence_seed1": (1252, 372, 0.0),
        "word_seed1_vs_left_seed1": (37, 85, 1.65e-05),
        "word_seed1_vs_sentence_seed1": (64, 123, 1.91e-05),
        "left_seed1_vs_sentence_seed1": (53, 64, 0.355),
    },
}
MIN_P_TRAINING_ONLY = {"soliloquies": 0.10, "tauber_beowulf": 0.12}

# folder -> (aligned, all agreement, all kappa, ambiguous n, ambiguous agreement, ambiguous kappa)
AGREEMENT = {
    "results_wikipron": (16832, .9773, .9733, 4420, .9604, .9419),
    "results_training_only": (16832, .9773, .9733, 4076, .9585, .9366),
    "results_proposal": (16832, .9773, .9733, 8363, .9635, .9545),
    "results_all": (16832, .9773, .9733, 8649, .9636, .9551),
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

    if errors:
        print(f"check_expected: FAILED ({len(errors)} of {checked} checks)")
        print("\n".join(errors))
        return 1
    print(f"check_expected: PASSED ({checked} checks)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
