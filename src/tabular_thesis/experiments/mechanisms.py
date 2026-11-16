"""Fold-local interventions; no outer-test selection or new model family."""

import numpy as np


def balanced_class_weights(y, n_classes: int) -> tuple[float, ...]:
    y = np.asarray(y)
    if y.ndim != 1 or not np.issubdtype(y.dtype, np.integer) or np.any(y < 0):
        raise ValueError("Expected nonnegative integer class codes")
    counts = np.bincount(y, minlength=n_classes)
    if len(counts) != n_classes or np.any(counts == 0):
        raise ValueError("Class weighting needs every class in the training fold")
    return tuple(float(len(y) / (n_classes * count)) for count in counts)


def nested_subset(y, fraction: float, seed: int) -> np.ndarray:
    """Round each class half up; preserve input order and nesting for a fixed seed."""
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0, 1]")
    y = np.asarray(y)
    if y.ndim != 1 or not len(y):
        raise ValueError("Expected a nonempty label vector")
    rng = np.random.default_rng(seed)
    chosen = []
    for label in np.unique(y):
        order = rng.permutation(np.flatnonzero(y == label))
        size = int(np.floor(fraction * len(order) + 0.5))
        if size < 1:
            raise ValueError("A training subset would lose a class")
        chosen.append(order[:size])
    return np.sort(np.concatenate(chosen))
