"""Evaluation metrics for classification tasks."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_fscore_support,
    roc_auc_score,
)


@dataclass
class MetricResult:
    """Container for all computed metrics from a single evaluation."""

    accuracy: float = 0.0
    balanced_accuracy: float = 0.0
    auc_roc: float | None = None
    average_precision: float | None = None
    per_class: dict = field(default_factory=dict)
    f1_macro: float = 0.0
    f1_weighted: float = 0.0
    log_loss_val: float = 0.0
    preprocessing_time_sec: float = 0.0
    training_time_sec: float = 0.0
    inference_time_sec: float = 0.0
    total_time_sec: float = 0.0  # preprocessing + training + inference
    confusion_matrix: list[list[int]] | None = None  # Stored as nested list for JSON serialization


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    n_classes: int = 2,
) -> MetricResult:
    """Compute all evaluation metrics for a classification prediction.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted class labels.
        y_proba: Predicted class probabilities (n_samples, n_classes).
        n_classes: Number of classes.

    Returns:
        MetricResult with all metrics populated.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_proba = validate_probabilities(y_proba, len(y_true), n_classes)
    if y_true.ndim != 1 or y_pred.shape != y_true.shape or len(y_true) == 0:
        raise ValueError("Invalid target/prediction shape")
    if not np.isin(y_true, np.arange(n_classes)).all():
        raise ValueError("Target labels disagree with the declared probability class order")
    if not np.isin(y_pred, np.arange(n_classes)).all():
        raise ValueError("Predicted labels disagree with the declared class order")
    result = MetricResult()

    result.accuracy = float(accuracy_score(y_true, y_pred))
    result.balanced_accuracy = float(balanced_accuracy_score(y_true, y_pred))
    result.f1_macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    result.f1_weighted = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    try:
        if n_classes == 2:
            # Binary: use probability of positive class
            proba_pos = y_proba[:, 1] if y_proba.ndim == 2 else y_proba
            result.auc_roc = float(roc_auc_score(y_true, proba_pos))
        else:
            # Multi-class: one-vs-rest AUC
            result.auc_roc = float(
                roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")
            )
    except ValueError:
        # Can happen if a class is missing from predictions
        result.auc_roc = None
    if result.auc_roc is not None and not np.isfinite(result.auc_roc):
        result.auc_roc = None

    try:
        result.log_loss_val = float(log_loss(y_true, y_proba, labels=np.arange(n_classes)))
    except ValueError as error:
        raise ValueError("Invalid log-loss inputs") from error

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=np.arange(n_classes), zero_division=0
    )
    result.per_class = {
        str(c): {
            "precision": float(precision[c]),
            "recall": float(recall[c]),
            "f1": float(f1[c]),
            "support": int(support[c]),
        }
        for c in range(n_classes)
    }
    for c in range(n_classes):
        if 0 < support[c] < len(y_true):
            result.per_class[str(c)]["average_precision"] = float(
                average_precision_score(y_true == c, y_proba[:, c])
            )
    if n_classes == 2 and len(np.unique(y_true)) == 2:
        result.average_precision = float(average_precision_score(y_true, y_proba[:, 1]))
    # Confusion matrix — stored as nested list for JSON serialization
    result.confusion_matrix = confusion_matrix(y_true, y_pred, labels=np.arange(n_classes)).tolist()

    return result


def validate_probabilities(y_proba: np.ndarray, n_rows: int, n_classes: int) -> np.ndarray:
    """Normalize binary shape; reject invalid values rather than inventing scores."""
    probabilities = np.asarray(y_proba)
    if n_classes < 2 or n_rows < 1:
        raise ValueError("Classification needs rows and at least two classes")
    if probabilities.ndim == 1 and n_classes == 2:
        probabilities = np.column_stack((1 - probabilities, probabilities))
    if probabilities.shape != (n_rows, n_classes):
        raise ValueError(
            f"Expected probability shape {(n_rows, n_classes)}, got {probabilities.shape}"
        )
    if not np.isfinite(probabilities).all():
        raise ValueError("Probabilities contain NaN or infinity")
    if (probabilities < 0).any() or (probabilities > 1).any():
        raise ValueError("Probabilities must lie in [0, 1]")
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-5):
        raise ValueError("Probability rows must sum to one")
    # Correct only accepted floating-point roundoff, after rejecting invalid rows.
    probabilities = probabilities.astype(np.float64)
    return probabilities / probabilities.sum(axis=1, keepdims=True)


class Timer:
    """Simple context manager for timing code blocks.

    Usage::

        with Timer() as t:
            model.fit(X, y)
        print(f"Training took {t.elapsed:.2f}s")
    """

    def __init__(self) -> None:
        self.elapsed: float = 0.0
        self._start: float = 0.0

    def __enter__(self) -> Timer:
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args: object) -> None:
        self.elapsed = time.perf_counter() - self._start
