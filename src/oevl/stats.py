"""Paired comparison of two systems on the same tokens that does not rest on one seed."""
import numpy as np


def bootstrap_diff(sentences, a, b, n=1000, seed=0):
    """Accuracy of a minus accuracy of b, with a 95% percentile interval from resampling whole sentences.

    sentences: the sentence id of each token; a, b: each token's correctness in [0, 1] (a mean over seeds is fine).
    Tokens of one sentence are not independent, so a sentence is drawn with all its tokens or not at all.
    Returns None when there are no tokens."""
    if not sentences:
        return None
    ids = {s: i for i, s in enumerate(dict.fromkeys(sentences))}
    g = np.array([ids[s] for s in sentences])
    diff = np.bincount(g, weights=np.asarray(a, float) - np.asarray(b, float), minlength=len(ids))
    size = np.bincount(g, minlength=len(ids))
    draw = np.random.default_rng(seed).integers(0, len(ids), size=(n, len(ids)))
    lo, hi = np.percentile(diff[draw].sum(1) / size[draw].sum(1), [2.5, 97.5])
    return {"diff": float(diff.sum() / size.sum()), "lo": float(lo), "hi": float(hi)}


def bootstrap_diff_runs(sentences, a, b, n=4000, seed=0):
    """As bootstrap_diff, but every draw also resamples each system's runs with replacement, so the interval includes
    the variance between training runs, not only between sentences.

    a, b: per system, one list per run of each token's correctness (0 or 1), in the order of sentences.
    Returns None when there are no tokens."""
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
