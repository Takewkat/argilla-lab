"""Inter-annotator agreement, pure functions on label counts and label lists."""

import numpy as np


def fleiss_kappa(counts):
    """Fleiss' kappa. counts: one row per item, how many raters picked each label, every row summing to the same n."""
    n = sum(counts[0])
    p_bar = sum((sum(c * c for c in row) - n) / (n * (n - 1)) for row in counts) / len(counts)
    p_j = [sum(row[j] for row in counts) / (len(counts) * n) for j in range(len(counts[0]))]
    p_e = sum(p * p for p in p_j)
    return (p_bar - p_e) / (1 - p_e) if p_e < 1 else 1.0


def nominal(a, b):
    """Any two different labels are equally far apart."""
    return (a != b).astype(float)


def interval(a, b):
    """Labels on a scale: 0 vs 3 is nine times further apart than 1 vs 2."""
    return (a - b) ** 2


def krippendorff_alpha(units, distance):
    """Krippendorff's alpha, for items rated by different raters. units: one array of labels per item, NaN allowed."""
    units = [u[~np.isnan(u)] for u in units]
    units = [u for u in units if len(u) >= 2]
    pooled = np.concatenate(units)
    observed = sum(distance(u[:, None], u[None, :]).sum() / (len(u) - 1) for u in units) / len(pooled)
    values, counts = np.unique(pooled, return_counts=True)
    pairs = counts[:, None] * counts[None, :] * distance(values[:, None], values[None, :])
    expected = pairs.sum() / (len(pooled) * (len(pooled) - 1))
    return 1 - observed / expected
