""" "Tests for evaluation metrics."""

from __future__ import annotations

import numpy as np

from tabular_thesis.evaluation.metrics import Timer, compute_metrics


class TestComputeMetrics:
    """Tests for metric computation."""

    def test_perfect_binary(self) -> None:
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1])
        y_proba = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])

        result = compute_metrics(y_true, y_pred, y_proba, n_classes=2)

        assert result.accuracy == 1.0
        assert result.auc_roc == 1.0
        assert result.f1_macro == 1.0

    def test_random_binary(self) -> None:
        rng = np.random.RandomState(42)
        y_true = rng.randint(0, 2, size=100)
        y_pred = rng.randint(0, 2, size=100)
        y_proba = rng.dirichlet([1, 1], size=100)

        result = compute_metrics(y_true, y_pred, y_proba, n_classes=2)

        assert 0.0 <= result.accuracy <= 1.0
        assert 0.0 <= result.auc_roc <= 1.0
        assert 0.0 <= result.f1_macro <= 1.0

    def test_multiclass(self) -> None:
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 1, 2, 0, 2, 1])
        y_proba = np.array(
            [
                [0.8, 0.1, 0.1],
                [0.1, 0.8, 0.1],
                [0.1, 0.1, 0.8],
                [0.7, 0.2, 0.1],
                [0.1, 0.3, 0.6],
                [0.2, 0.6, 0.2],
            ]
        )

        result = compute_metrics(y_true, y_pred, y_proba, n_classes=3)

        assert 0.0 <= result.accuracy <= 1.0
        assert 0.0 <= result.auc_roc <= 1.0


class TestTimer:
    """Tests for the Timer context manager."""

    def test_timer_measures_time(self) -> None:
        import time

        with Timer() as t:
            time.sleep(0.05)

        assert t.elapsed >= 0.04  # Allow small margin
