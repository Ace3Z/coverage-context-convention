"""Fail the run if any headline value differs from the frozen reference results.

Reads results/<setting>/metrics.json for the four candidate settings and the Zupitza ablation, and
results/agreement/metrics.json. Checks counts exactly, accuracies, agreements and kappas to 0.00005 (the frozen values
are rounded to four decimals) and McNemar p-values to 0.0005 absolute, and that no seed-1 McNemar p-value of the
training-only setting on Beowulf falls below 0.12. That makes 660 checks when every file is complete.
Prints PASSED, or FAILED with one line per mismatch or missing entry, and exits 0 or 1.
Usage: python -m oevl.check_expected
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROP_TOL, P_TOL = 0.00005, 0.0005
POPS = ("accuracy", "lexicon_ambiguous", "attested_ambiguous", "minority_reading")

# (folder, test) -> population -> (n, frequency, word, left, bilateral, oracle): n is the population's token count and
# the rest are accuracies, with word, left and bilateral (the "sentence" arm in metrics.json) as means of seeds 1 to 3.
# The four settings train without Zupitza (decision D3), and results_wikipron_with_zupitza is the D3 ablation with all
# five works.
EXPECTED = {
    ("results_wikipron", "john_ws"): {
        "accuracy": (16649, .9613, .9600, .9596, .9600, .9792),
        "lexicon_ambiguous": (7639, .9560, .9532, .9523, .9531, .9949),
        "attested_ambiguous": (5734, .9569, .9574, .9569, .9581, 1.0),
        "minority_reading": (278, 0.0, .0096, .0767, .0875, 1.0),
    },
    ("results_wikipron", "soliloquies"): {
        "accuracy": (15426, .8785, .8802, .8837, .8831, .9213),
        "lexicon_ambiguous": (5834, .8846, .8891, .8983, .8967, .9978),
        "attested_ambiguous": (4497, .8739, .8739, .8863, .8854, .9998),
        "minority_reading": (604, 0.0, .0017, .1120, .1142, 1.0),
    },
    ("results_wikipron", "tauber_beowulf"): {
        "accuracy": (17361, .8940, .8947, .8940, .8936, .9222),
        "lexicon_ambiguous": (4402, .8885, .8916, .8888, .8869, .9998),
        "attested_ambiguous": (3268, .9067, .9075, .9094, .9080, 1.0),
        "minority_reading": (325, 0.0, .0103, .0646, .0605, 1.0),
    },
    ("results_training_only", "john_ws"): {
        "accuracy": (16649, .9512, .9493, .9494, .9499, .9682),
        "lexicon_ambiguous": (6903, .9583, .9537, .9540, .9550, .9991),
        "attested_ambiguous": (5734, .9569, .9556, .9554, .9567, 1.0),
        "minority_reading": (278, 0.0, .0192, .1019, .1163, 1.0),
    },
    ("results_training_only", "soliloquies"): {
        "accuracy": (15426, .8612, .8611, .8658, .8657, .9008),
        "lexicon_ambiguous": (5197, .8801, .8797, .8938, .8935, .9977),
        "attested_ambiguous": (4497, .8739, .8739, .8904, .8914, .9998),
        "minority_reading": (604, 0.0, 0.0, .1479, .1711, 1.0),
    },
    ("results_training_only", "tauber_beowulf"): {
        "accuracy": (17361, .8226, .8231, .8235, .8231, .8429),
        "lexicon_ambiguous": (3888, .9092, .9114, .9132, .9114, .9997),
        "attested_ambiguous": (3268, .9067, .9063, .9087, .9087, 1.0),
        "minority_reading": (325, 0.0, .0154, .0769, .0769, 1.0),
    },
    ("results_proposal", "john_ws"): {
        "accuracy": (16649, .9640, .9607, .9604, .9607, .9856),
        "lexicon_ambiguous": (8835, .9585, .9523, .9517, .9523, .9992),
        "attested_ambiguous": (5734, .9569, .9571, .9570, .9586, 1.0),
        "minority_reading": (278, 0.0, .0072, .0384, .0743, 1.0),
    },
    ("results_proposal", "soliloquies"): {
        "accuracy": (15426, .8919, .8951, .8981, .8977, .9506),
        "lexicon_ambiguous": (7593, .8782, .8846, .8907, .8899, .9974),
        "attested_ambiguous": (4497, .8739, .8759, .8858, .8838, 1.0),
        "minority_reading": (604, 0.0, .0248, .1021, .1065, 1.0),
    },
    ("results_proposal", "tauber_beowulf"): {
        "accuracy": (17361, .9207, .9166, .9165, .9167, .9612),
        "lexicon_ambiguous": (6002, .8822, .8701, .8699, .8705, .9993),
        "attested_ambiguous": (3268, .9067, .9064, .9067, .9072, 1.0),
        "minority_reading": (325, 0.0, .0154, .0564, .0544, 1.0),
    },
    ("results_all", "john_ws"): {
        "accuracy": (16649, .9643, .9607, .9604, .9606, .9857),
        "lexicon_ambiguous": (8837, .9589, .9522, .9516, .9519, .9992),
        "attested_ambiguous": (5734, .9569, .9572, .9570, .9587, 1.0),
        "minority_reading": (278, 0.0, .0084, .0324, .0695, 1.0),
    },
    ("results_all", "soliloquies"): {
        "accuracy": (15426, .8929, .8953, .8984, .8982, .9507),
        "lexicon_ambiguous": (7597, .8800, .8849, .8911, .8906, .9974),
        "attested_ambiguous": (4497, .8739, .8759, .8862, .8841, 1.0),
        "minority_reading": (604, 0.0, .0248, .1032, .1065, 1.0),
    },
    ("results_all", "tauber_beowulf"): {
        "accuracy": (17361, .9239, .9170, .9169, .9169, .9614),
        "lexicon_ambiguous": (6007, .8910, .8709, .8708, .8707, .9993),
        "attested_ambiguous": (3268, .9067, .9062, .9075, .9065, 1.0),
        "minority_reading": (325, 0.0, .0154, .0615, .0533, 1.0),
    },
    ("results_wikipron_with_zupitza", "john_ws"): {
        "accuracy": (16649, .9589, .9567, .9563, .9567, .9805),
        "lexicon_ambiguous": (7782, .9487, .9441, .9432, .9441, .9950),
        "attested_ambiguous": (6026, .9492, .9486, .9479, .9494, .9998),
        "minority_reading": (345, 0.0, .0058, .0386, .0763, 1.0),
    },
    ("results_wikipron_with_zupitza", "soliloquies"): {
        "accuracy": (15426, .8873, .8891, .8901, .8901, .9322),
        "lexicon_ambiguous": (5965, .8818, .8865, .8890, .8891, .9978),
        "attested_ambiguous": (4655, .8724, .8730, .8769, .8781, .9998),
        "minority_reading": (632, 0.0, .0095, .0527, .0712, 1.0),
    },
    ("results_wikipron_with_zupitza", "tauber_beowulf"): {
        "accuracy": (17361, .8945, .8945, .8938, .8933, .9233),
        "lexicon_ambiguous": (4436, .8873, .8873, .8845, .8828, 1.0),
        "attested_ambiguous": (3394, .9057, .9067, .9073, .9066, 1.0),
        "minority_reading": (349, 0.0, .0258, .0468, .0449, 1.0),
    },
}

# (folder, test) -> pair -> (a_wrong_b_right, a_right_b_wrong, p) for the exact McNemar tests on seed 1 (mcnemar_seed1).
# p is p_exact to three significant figures, frozen as 0.0 when it is below 1e-6. It is checked to 0.0005 absolute, so
# a frozen 0.0 or a tiny p only asserts a p-value of at most about 0.0005.
# results_training_only on tauber_beowulf has no entry here. Only its p-values are checked, against MIN_P_TRAINING_ONLY.
MCNEMAR = {
    ("results_wikipron", "john_ws"): {
        "frequency_vs_word_seed1": (13, 33, 0.00453),
        "frequency_vs_left_seed1": (29, 52, 0.014),
        "frequency_vs_sentence_seed1": (32, 57, 0.0105),
        "word_seed1_vs_left_seed1": (16, 19, 0.736),
        "word_seed1_vs_sentence_seed1": (22, 27, 0.568),
        "left_seed1_vs_sentence_seed1": (14, 16, 0.856),
    },
    ("results_wikipron", "soliloquies"): {
        "frequency_vs_word_seed1": (40, 14, 0.000535),
        "frequency_vs_left_seed1": (99, 22, 0.0),
        "frequency_vs_sentence_seed1": (110, 39, 0.0),
        "word_seed1_vs_left_seed1": (61, 10, 0.0),
        "word_seed1_vs_sentence_seed1": (72, 27, 6.9e-06),
        "left_seed1_vs_sentence_seed1": (26, 32, 0.512),
    },
    ("results_wikipron", "tauber_beowulf"): {
        "frequency_vs_word_seed1": (48, 34, 0.151),
        "frequency_vs_left_seed1": (60, 58, 0.927),
        "frequency_vs_sentence_seed1": (66, 69, 0.863),
        "word_seed1_vs_left_seed1": (24, 36, 0.155),
        "word_seed1_vs_sentence_seed1": (29, 46, 0.0639),
        "left_seed1_vs_sentence_seed1": (18, 23, 0.533),
    },
    ("results_training_only", "john_ws"): {
        "frequency_vs_word_seed1": (8, 40, 3.31e-06),
        "frequency_vs_left_seed1": (29, 65, 0.000262),
        "frequency_vs_sentence_seed1": (33, 51, 0.063),
        "word_seed1_vs_left_seed1": (27, 31, 0.694),
        "word_seed1_vs_sentence_seed1": (37, 23, 0.0925),
        "left_seed1_vs_sentence_seed1": (33, 15, 0.0133),
    },
    ("results_training_only", "soliloquies"): {
        "frequency_vs_word_seed1": (2, 4, 0.688),
        "frequency_vs_left_seed1": (77, 24, 0.0),
        "frequency_vs_sentence_seed1": (91, 33, 0.0),
        "word_seed1_vs_left_seed1": (75, 20, 0.0),
        "word_seed1_vs_sentence_seed1": (89, 29, 0.0),
        "left_seed1_vs_sentence_seed1": (38, 33, 0.635),
    },
    ("results_proposal", "john_ws"): {
        "frequency_vs_word_seed1": (17, 71, 0.0),
        "frequency_vs_left_seed1": (29, 91, 0.0),
        "frequency_vs_sentence_seed1": (43, 95, 1.13e-05),
        "word_seed1_vs_left_seed1": (15, 23, 0.256),
        "word_seed1_vs_sentence_seed1": (28, 26, 0.892),
        "left_seed1_vs_sentence_seed1": (28, 18, 0.184),
    },
    ("results_proposal", "soliloquies"): {
        "frequency_vs_word_seed1": (147, 94, 0.000775),
        "frequency_vs_left_seed1": (208, 102, 0.0),
        "frequency_vs_sentence_seed1": (232, 122, 0.0),
        "word_seed1_vs_left_seed1": (77, 24, 0.0),
        "word_seed1_vs_sentence_seed1": (112, 55, 1.22e-05),
        "left_seed1_vs_sentence_seed1": (51, 47, 0.762),
    },
    ("results_proposal", "tauber_beowulf"): {
        "frequency_vs_word_seed1": (153, 221, 0.000515),
        "frequency_vs_left_seed1": (174, 238, 0.00188),
        "frequency_vs_sentence_seed1": (172, 237, 0.00152),
        "word_seed1_vs_left_seed1": (55, 51, 0.771),
        "word_seed1_vs_sentence_seed1": (60, 57, 0.853),
        "left_seed1_vs_sentence_seed1": (28, 29, 1.0),
    },
    ("results_all", "john_ws"): {
        "frequency_vs_word_seed1": (16, 76, 0.0),
        "frequency_vs_left_seed1": (24, 87, 0.0),
        "frequency_vs_sentence_seed1": (35, 94, 0.0),
        "word_seed1_vs_left_seed1": (12, 15, 0.701),
        "word_seed1_vs_sentence_seed1": (23, 22, 1.0),
        "left_seed1_vs_sentence_seed1": (20, 16, 0.618),
    },
    ("results_all", "soliloquies"): {
        "frequency_vs_word_seed1": (137, 111, 0.112),
        "frequency_vs_left_seed1": (194, 102, 0.0),
        "frequency_vs_sentence_seed1": (202, 122, 1.03e-05),
        "word_seed1_vs_left_seed1": (86, 20, 0.0),
        "word_seed1_vs_sentence_seed1": (102, 48, 1.23e-05),
        "left_seed1_vs_sentence_seed1": (44, 56, 0.271),
    },
    ("results_all", "tauber_beowulf"): {
        "frequency_vs_word_seed1": (118, 242, 0.0),
        "frequency_vs_left_seed1": (132, 254, 0.0),
        "frequency_vs_sentence_seed1": (132, 255, 0.0),
        "word_seed1_vs_left_seed1": (54, 52, 0.923),
        "word_seed1_vs_sentence_seed1": (57, 56, 1.0),
        "left_seed1_vs_sentence_seed1": (31, 32, 1.0),
    },
    ("results_wikipron_with_zupitza", "john_ws"): {
        "frequency_vs_word_seed1": (6, 45, 0.0),
        "frequency_vs_left_seed1": (20, 68, 0.0),
        "frequency_vs_sentence_seed1": (32, 75, 3.92e-05),
        "word_seed1_vs_left_seed1": (21, 30, 0.262),
        "word_seed1_vs_sentence_seed1": (35, 39, 0.728),
        "left_seed1_vs_sentence_seed1": (28, 23, 0.576),
    },
    ("results_wikipron_with_zupitza", "soliloquies"): {
        "frequency_vs_word_seed1": (43, 15, 0.000307),
        "frequency_vs_left_seed1": (94, 35, 0.0),
        "frequency_vs_sentence_seed1": (98, 48, 4.27e-05),
        "word_seed1_vs_left_seed1": (53, 22, 0.00045),
        "word_seed1_vs_sentence_seed1": (57, 35, 0.028),
        "left_seed1_vs_sentence_seed1": (29, 38, 0.328),
    },
    ("results_wikipron_with_zupitza", "tauber_beowulf"): {
        "frequency_vs_word_seed1": (51, 53, 0.922),
        "frequency_vs_left_seed1": (66, 70, 0.797),
        "frequency_vs_sentence_seed1": (61, 78, 0.175),
        "word_seed1_vs_left_seed1": (25, 27, 0.89),
        "word_seed1_vs_sentence_seed1": (25, 40, 0.0817),
        "left_seed1_vs_sentence_seed1": (12, 25, 0.047),
    },
}
# training-only on Beowulf: no seed-1 McNemar pair of the prior and the three arms reaches p < 0.12 (min p 0.1934)
MIN_P_TRAINING_ONLY = {"tauber_beowulf": 0.12}

# folder -> (aligned, all agreement, all kappa, ambiguous n, ambiguous agreement, ambiguous kappa) in editor_agreement:
# Harrison and Sharp against Wyatt, with "ambiguous" meaning lexicon-ambiguous under that setting's lexicon
AGREEMENT = {
    "results_wikipron": (16833, .9773, .9733, 4387, .9608, .9423),
    "results_training_only": (16833, .9773, .9733, 3874, .9566, .9326),
    "results_proposal": (16833, .9773, .9733, 5976, .9605, .9467),
    "results_all": (16833, .9773, .9733, 5981, .9602, .9464),
    "results_wikipron_with_zupitza": (16833, .9773, .9733, 4420, .9604, .9419),
}

# results/agreement: editor pair -> (aligned, ambiguous n, ambiguous agreement, ambiguous kappa), where hs is Harrison
# and Sharp, wyatt is Wyatt and tauber is Tauber's gold, with the lexicon of results_wikipron
AGREEMENT_PAIRS = {
    "hs-wyatt": (16833, 4387, .9608, .9423),
    "hs-tauber": (16577, 4334, .9882, .9827),
    "wyatt-tauber": (16918, 4335, .9608, .9424),
}


def acc(d, pop):
    """Return an arm's accuracy on a population: overall for "accuracy", else d[pop]["accuracy"] (None if empty).

    d is one arm's entry for a test in metrics.json, as experiment.evaluate writes it.
    """
    return d["accuracy"] if pop == "accuracy" else d[pop]["accuracy"]


def main() -> int:
    """Compare the metrics files with the frozen values, print the verdict and return the exit code (0 or 1).

    A missing metrics file, test, McNemar pair or editor entry is recorded as an error but not counted as a check, so
    the FAILED line can report more errors than checks. A key missing inside an entry that is present raises KeyError.
    The p floor in MIN_P_TRAINING_ONLY runs over whatever pairs mcnemar_seed1 holds, so it checks nothing when they are
    absent.
    """
    errors, checked = [], 0

    def close(label, got, want, tol):
        """Count one check and record an error if got is None or differs from want by more than tol."""
        nonlocal checked
        checked += 1
        if got is None or abs(got - want) > tol:
            errors.append(f"{label}: got {got}, expected {want}")

    def equal(label, got, want):
        """Count one check and record an error unless got equals want."""
        nonlocal checked
        checked += 1
        if got != want:
            errors.append(f"{label}: got {got}, expected {want}")

    metrics = {}
    for folder in ("results_wikipron", "results_training_only", "results_proposal", "results_all",
                   "results_wikipron_with_zupitza"):
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
            # max(1, p) is 1 for any p-value, so the tolerance is P_TOL absolute
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
