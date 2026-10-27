"""Paired comparison of two systems on the same tokens that does not rest on one seed."""
from collections.abc import Sequence

import numpy as np


def bootstrap_diff(sentences: Sequence[str], a: Sequence[float], b: Sequence[float], n: int = 1000,
                   seed: int = 0) -> dict[str, float] | None:
    """Accuracy of a minus accuracy of b, with a 95% percentile interval from resampling whole sentences.

    sentences: a list with the sentence id of each token. a, b: one score per token in the same order, each in [0, 1],
    usually correctness or its mean over seeds (paper_numbers also passes the share of runs that agree with an edition).
    Tokens of one sentence are not independent, so a sentence is drawn with all its tokens or not at all. Each of the
    n draws picks as many sentences as there are distinct ones, with replacement, and scores the difference pooled over
    the drawn tokens, so longer sentences weigh more. The interval is the 2.5th and 97.5th percentile of the draws. The
    point estimate uses every token once. The runs behind a mean are held fixed, and the seed argument fixes the random
    draws, so the result is repeatable. Returns {"diff", "lo", "hi"} as fractions, not points, or None when there are no
    tokens."""
    if not sentences:
        return None
    ids = {s: i for i, s in enumerate(dict.fromkeys(sentences))}
    g = np.array([ids[s] for s in sentences])
    diff = np.bincount(g, weights=np.asarray(a, float) - np.asarray(b, float), minlength=len(ids))
    size = np.bincount(g, minlength=len(ids))
    draw = np.random.default_rng(seed).integers(0, len(ids), size=(n, len(ids)))
    lo, hi = np.percentile(diff[draw].sum(1) / size[draw].sum(1), [2.5, 97.5])
    return {"diff": float(diff.sum() / size.sum()), "lo": float(lo), "hi": float(hi)}


def bootstrap_diff_runs(sentences: Sequence[str], a: Sequence[Sequence[float]], b: Sequence[Sequence[float]],
                        n: int = 4000, seed: int = 0) -> dict[str, float] | None:
    """As bootstrap_diff, but every draw also resamples each system's runs with replacement.

    The interval then includes the variance between training runs, not only between sentences.
    a, b: per system, one list per run of each token's correctness (0 or 1), in the order of sentences. The systems may
    have different numbers of runs, but each needs at least one. In a draw the runs of a and of b are drawn
    independently and apply to every drawn sentence, and the score is the pooled accuracy of a on the drawn tokens,
    averaged over its drawn runs, minus the same for b. The point estimate equals bootstrap_diff on correctness averaged
    over runs, up to rounding. The paper passes n=40,000 (paper_numbers.TWO_LEVEL_DRAWS), so the default of 4,000 is
    not used there. Returns {"diff", "lo", "hi"} as fractions, not points, or None when there are no tokens."""
    if not sentences:
        return None
    ids = {s: i for i, s in enumerate(dict.fromkeys(sentences))}
    g = np.array([ids[s] for s in sentences])
    size = np.bincount(g, minlength=len(ids))
    per = [np.stack([np.bincount(g, weights=np.asarray(r, float), minlength=len(ids)) for r in x], 1) for x in (a, b)]
    rng, out = np.random.default_rng(seed), []
    for _ in range(n):  # the order of the draws (runs of a, runs of b, sentences) fixes the result for a seed
        ra, rb = rng.integers(0, len(a), len(a)), rng.integers(0, len(b), len(b))
        sd = rng.integers(0, len(ids), len(ids))
        out.append((per[0][sd][:, ra].mean(1).sum() - per[1][sd][:, rb].mean(1).sum()) / size[sd].sum())
    lo, hi = np.percentile(out, [2.5, 97.5])
    return {"diff": float((per[0].mean(1).sum() - per[1].mean(1).sum()) / size.sum()), "lo": float(lo), "hi": float(hi)}
