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
# The four settings train without Zupitza (PLAN D3); results_wikipron_with_zupitza is the D3 ablation with all five works
EXPECTED = {
    ("results_wikipron", "john_ws"): {
        "accuracy": (16649, .9613, .9601, .9591, .9596, .9868),
        "lexicon_ambiguous": (7639, .9560, .9533, .9511, .9523, .9993),
        "attested_ambiguous": (5734, .9569, .9575, .9549, .9572, 1.0),
        "minority_reading": (321, .1340, .0540, .0966, .1236, 1.0),
    },
    ("results_wikipron", "soliloquies"): {
        "accuracy": (15431, .8784, .8800, .8825, .8827, .9307),
        "lexicon_ambiguous": (5837, .8842, .8886, .8952, .8956, .9991),
        "attested_ambiguous": (4500, .8733, .8733, .8823, .8835, 1.0),
        "minority_reading": (612, .0082, .0033, .0975, .1155, 1.0),
    },
    ("results_wikipron", "tauber_beowulf"): {
        "accuracy": (17361, .8938, .8947, .8938, .8935, .9263),
        "lexicon_ambiguous": (4400, .8884, .8919, .8883, .8870, .9998),
        "attested_ambiguous": (3268, .9067, .9080, .9089, .9082, 1.0),
        "minority_reading": (349, .0688, .0716, .1175, .1127, 1.0),
    },
    ("results_training_only", "john_ws"): {
        "accuracy": (16649, .9512, .9493, .9492, .9498, .9747),
        "lexicon_ambiguous": (6903, .9583, .9537, .9535, .9548, .9993),
        "attested_ambiguous": (5734, .9569, .9556, .9548, .9565, 1.0),
        "minority_reading": (321, .1340, .0426, .1059, .1391, 1.0),
    },
    ("results_training_only", "soliloquies"): {
        "accuracy": (15431, .8611, .8609, .8654, .8649, .9082),
        "lexicon_ambiguous": (5200, .8796, .8792, .8924, .8910, .9990),
        "attested_ambiguous": (4500, .8733, .8733, .8889, .8884, 1.0),
        "minority_reading": (612, .0082, .0016, .1329, .1503, 1.0),
    },
    ("results_training_only", "tauber_beowulf"): {
        "accuracy": (17361, .8240, .8244, .8249, .8245, .8490),
        "lexicon_ambiguous": (3886, .9092, .9112, .9133, .9117, .9997),
        "attested_ambiguous": (3268, .9067, .9061, .9087, .9083, 1.0),
        "minority_reading": (349, .0688, .0267, .0850, .0869, 1.0),
    },
    ("results_proposal", "john_ws"): {
        "accuracy": (16649, .9640, .9608, .9605, .9605, .9903),
        "lexicon_ambiguous": (8835, .9585, .9524, .9519, .9519, .9994),
        "attested_ambiguous": (5734, .9569, .9571, .9570, .9579, 1.0),
        "minority_reading": (321, .1340, .0498, .0717, .0955, 1.0),
    },
    ("results_proposal", "soliloquies"): {
        "accuracy": (15431, .8918, .8955, .8976, .8975, .9592),
        "lexicon_ambiguous": (7596, .8778, .8854, .8896, .8894, .9988),
        "attested_ambiguous": (4500, .8733, .8756, .8835, .8822, 1.0),
        "minority_reading": (612, .0082, .0283, .0953, .1040, 1.0),
    },
    ("results_proposal", "tauber_beowulf"): {
        "accuracy": (17361, .9206, .9166, .9165, .9168, .9638),
        "lexicon_ambiguous": (5998, .8821, .8703, .8702, .8710, .9993),
        "attested_ambiguous": (3268, .9067, .9067, .9068, .9068, 1.0),
        "minority_reading": (349, .0688, .0659, .1041, .1012, 1.0),
    },
    ("results_all", "john_ws"): {
        "accuracy": (16649, .9643, .9608, .9605, .9605, .9904),
        "lexicon_ambiguous": (8837, .9589, .9522, .9517, .9518, .9994),
        "attested_ambiguous": (5734, .9569, .9572, .9572, .9582, 1.0),
        "minority_reading": (321, .1340, .0509, .0737, .0966, 1.0),
    },
    ("results_all", "soliloquies"): {
        "accuracy": (15431, .8927, .8954, .8985, .8979, .9594),
        "lexicon_ambiguous": (7600, .8796, .8851, .8913, .8900, .9988),
        "attested_ambiguous": (4500, .8733, .8753, .8858, .8833, 1.0),
        "minority_reading": (612, .0082, .0261, .1084, .1106, 1.0),
    },
    ("results_all", "tauber_beowulf"): {
        "accuracy": (17361, .9238, .9165, .9166, .9169, .9639),
        "lexicon_ambiguous": (6003, .8909, .8697, .8702, .8709, .9993),
        "attested_ambiguous": (3268, .9067, .9062, .9071, .9070, 1.0),
        "minority_reading": (349, .0688, .0630, .1060, .1032, 1.0),
    },
    ("results_wikipron_with_zupitza", "john_ws"): {
        "accuracy": (16649, .9589, .9570, .9565, .9568, .9876),
        "lexicon_ambiguous": (7782, .9487, .9447, .9436, .9444, .9994),
        "attested_ambiguous": (6026, .9492, .9493, .9483, .9497, 1.0),
        "minority_reading": (382, .0969, .0262, .0489, .0820, 1.0),
    },
    ("results_wikipron_with_zupitza", "soliloquies"): {
        "accuracy": (15431, .8872, .8890, .8895, .8901, .9411),
        "lexicon_ambiguous": (5968, .8814, .8861, .8874, .8890, .9992),
        "attested_ambiguous": (4658, .8718, .8725, .8748, .8780, 1.0),
        "minority_reading": (638, .0047, .0094, .0329, .0617, 1.0),
    },
    ("results_wikipron_with_zupitza", "tauber_beowulf"): {
        "accuracy": (17361, .8944, .8940, .8933, .8932, .9273),
        "lexicon_ambiguous": (4434, .8872, .8857, .8832, .8827, 1.0),
        "attested_ambiguous": (3394, .9057, .9067, .9058, .9062, 1.0),
        "minority_reading": (358, .0251, .0372, .0428, .0484, 1.0),
    },
}

# (folder, test) -> pair -> (a_wrong_b_right, a_right_b_wrong, p)
MCNEMAR = {
    ("results_wikipron", "john_ws"): {
        "frequency_vs_word_seed1": (12, 33, 0.00246),
        "frequency_vs_left_seed1": (32, 66, 0.00077),
        "frequency_vs_sentence_seed1": (39, 72, 0.00224),
        "word_seed1_vs_left_seed1": (22, 35, 0.111),
        "word_seed1_vs_sentence_seed1": (31, 43, 0.201),
        "left_seed1_vs_sentence_seed1": (24, 23, 1.0),
    },
    ("results_wikipron", "soliloquies"): {
        "frequency_vs_word_seed1": (40, 14, 0.000535),
        "frequency_vs_left_seed1": (106, 37, 0.0),
        "frequency_vs_sentence_seed1": (139, 54, 0.0),
        "word_seed1_vs_left_seed1": (68, 25, 9.38e-06),
        "word_seed1_vs_sentence_seed1": (101, 42, 0.0),
        "left_seed1_vs_sentence_seed1": (50, 34, 0.101),
    },
    ("results_wikipron", "tauber_beowulf"): {
        "frequency_vs_word_seed1": (48, 32, 0.0929),
        "frequency_vs_left_seed1": (60, 57, 0.853),
        "frequency_vs_sentence_seed1": (69, 68, 1.0),
        "word_seed1_vs_left_seed1": (24, 37, 0.124),
        "word_seed1_vs_sentence_seed1": (35, 50, 0.128),
        "left_seed1_vs_sentence_seed1": (19, 21, 0.875),
    },
    ("results_training_only", "john_ws"): {
        "frequency_vs_word_seed1": (8, 40, 3.31e-06),
        "frequency_vs_left_seed1": (16, 45, 0.000264),
        "frequency_vs_sentence_seed1": (30, 50, 0.033),
        "word_seed1_vs_left_seed1": (14, 11, 0.69),
        "word_seed1_vs_sentence_seed1": (33, 21, 0.134),
        "left_seed1_vs_sentence_seed1": (28, 19, 0.243),
    },
    ("results_training_only", "soliloquies"): {
        "frequency_vs_word_seed1": (2, 4, 0.688),
        "frequency_vs_left_seed1": (62, 11, 0.0),
        "frequency_vs_sentence_seed1": (76, 29, 5.05e-06),
        "word_seed1_vs_left_seed1": (60, 7, 0.0),
        "word_seed1_vs_sentence_seed1": (75, 26, 1.12e-06),
        "left_seed1_vs_sentence_seed1": (28, 32, 0.699),
    },
    ("results_proposal", "john_ws"): {
        "frequency_vs_word_seed1": (17, 71, 0.0),
        "frequency_vs_left_seed1": (27, 85, 0.0),
        "frequency_vs_sentence_seed1": (35, 90, 0.0),
        "word_seed1_vs_left_seed1": (13, 17, 0.585),
        "word_seed1_vs_sentence_seed1": (23, 24, 1.0),
        "left_seed1_vs_sentence_seed1": (22, 19, 0.755),
    },
    ("results_proposal", "soliloquies"): {
        "frequency_vs_word_seed1": (148, 92, 0.000364),
        "frequency_vs_left_seed1": (182, 97, 0.0),
        "frequency_vs_sentence_seed1": (190, 117, 3.67e-05),
        "word_seed1_vs_left_seed1": (53, 24, 0.00126),
        "word_seed1_vs_sentence_seed1": (70, 53, 0.149),
        "left_seed1_vs_sentence_seed1": (36, 48, 0.23),
    },
    ("results_proposal", "tauber_beowulf"): {
        "frequency_vs_word_seed1": (155, 221, 0.00078),
        "frequency_vs_left_seed1": (162, 240, 0.000117),
        "frequency_vs_sentence_seed1": (168, 238, 0.000599),
        "word_seed1_vs_left_seed1": (43, 55, 0.266),
        "word_seed1_vs_sentence_seed1": (52, 56, 0.773),
        "left_seed1_vs_sentence_seed1": (34, 26, 0.366),
    },
    ("results_all", "john_ws"): {
        "frequency_vs_word_seed1": (16, 74, 0.0),
        "frequency_vs_left_seed1": (27, 93, 0.0),
        "frequency_vs_sentence_seed1": (35, 95, 0.0),
        "word_seed1_vs_left_seed1": (15, 23, 0.256),
        "word_seed1_vs_sentence_seed1": (23, 25, 0.885),
        "left_seed1_vs_sentence_seed1": (22, 16, 0.418),
    },
    ("results_all", "soliloquies"): {
        "frequency_vs_word_seed1": (136, 96, 0.0103),
        "frequency_vs_left_seed1": (202, 101, 0.0),
        "frequency_vs_sentence_seed1": (217, 126, 1.02e-06),
        "word_seed1_vs_left_seed1": (80, 19, 0.0),
        "word_seed1_vs_sentence_seed1": (104, 53, 5.74e-05),
        "left_seed1_vs_sentence_seed1": (46, 56, 0.373),
    },
    ("results_all", "tauber_beowulf"): {
        "frequency_vs_word_seed1": (117, 256, 0.0),
        "frequency_vs_left_seed1": (135, 255, 0.0),
        "frequency_vs_sentence_seed1": (135, 250, 0.0),
        "word_seed1_vs_left_seed1": (55, 36, 0.0586),
        "word_seed1_vs_sentence_seed1": (65, 41, 0.025),
        "left_seed1_vs_sentence_seed1": (29, 24, 0.583),
    },
    ("results_wikipron_with_zupitza", "john_ws"): {
        "frequency_vs_word_seed1": (6, 36, 2.83e-06),
        "frequency_vs_left_seed1": (14, 55, 0.0),
        "frequency_vs_sentence_seed1": (29, 59, 0.00182),
        "word_seed1_vs_left_seed1": (8, 19, 0.0522),
        "word_seed1_vs_sentence_seed1": (25, 25, 1.0),
        "left_seed1_vs_sentence_seed1": (23, 12, 0.0895),
    },
    ("results_wikipron_with_zupitza", "soliloquies"): {
        "frequency_vs_word_seed1": (43, 14, 0.000154),
        "frequency_vs_left_seed1": (59, 22, 4.77e-05),
        "frequency_vs_sentence_seed1": (77, 31, 1.12e-05),
        "word_seed1_vs_left_seed1": (16, 8, 0.152),
        "word_seed1_vs_sentence_seed1": (35, 18, 0.027),
        "left_seed1_vs_sentence_seed1": (27, 18, 0.233),
    },
    ("results_wikipron_with_zupitza", "tauber_beowulf"): {
        "frequency_vs_word_seed1": (53, 64, 0.355),
        "frequency_vs_left_seed1": (52, 71, 0.104),
        "frequency_vs_sentence_seed1": (55, 75, 0.0953),
        "word_seed1_vs_left_seed1": (6, 14, 0.115),
        "word_seed1_vs_sentence_seed1": (15, 24, 0.2),
        "left_seed1_vs_sentence_seed1": (14, 15, 1.0),
    },
}
# training-only on Beowulf: no model differs from another at p < 0.12 (min p 0.1597)
MIN_P_TRAINING_ONLY = {"tauber_beowulf": 0.12}

# folder -> (aligned, all agreement, all kappa, ambiguous n, ambiguous agreement, ambiguous kappa)
AGREEMENT = {
    "results_wikipron": (16833, .9773, .9733, 4387, .9608, .9423),
    "results_training_only": (16833, .9773, .9733, 3874, .9566, .9326),
    "results_proposal": (16833, .9773, .9733, 5976, .9605, .9467),
    "results_all": (16833, .9773, .9733, 5981, .9602, .9464),
    "results_wikipron_with_zupitza": (16833, .9773, .9733, 4420, .9604, .9419),
}

# results/agreement: editor pair -> (aligned, ambiguous n, ambiguous agreement, ambiguous kappa)
AGREEMENT_PAIRS = {
    "hs-wyatt": (16833, 4387, .9608, .9423),
    "hs-tauber": (16573, 4332, .9882, .9827),
    "wyatt-tauber": (16912, 4333, .9608, .9424),
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
