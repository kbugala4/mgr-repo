"""Statistical significance tests for model comparison."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass
class StatTestResult:
    """Result of a statistical significance test."""

    test_name: str
    statistic: float
    p_value: float
    significant: bool  # at alpha=0.05


def wilcoxon_test(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    alpha: float = 0.05,
) -> StatTestResult:
    """Wilcoxon signed-rank test for paired model comparison.

    Tests whether two models have significantly different performance
    across multiple datasets or seeds.

    Args:
        scores_a: Metric scores for model A (one per dataset/seed).
        scores_b: Metric scores for model B (one per dataset/seed).
        alpha: Significance level.

    Returns:
        StatTestResult with test statistic and p-value.
    """
    stat, p_value = stats.wilcoxon(scores_a, scores_b, alternative="two-sided")
    return StatTestResult(
        test_name="Wilcoxon signed-rank",
        statistic=float(stat),
        p_value=float(p_value),
        significant=p_value < alpha,
    )


def friedman_test(
    score_matrix: np.ndarray,
    alpha: float = 0.05,
) -> StatTestResult:
    """Friedman test for comparing multiple models across datasets.

    Args:
        score_matrix: 2-D array of shape (n_datasets, n_models).
            Each row is a dataset, each column is a model's score.
        alpha: Significance level.

    Returns:
        StatTestResult with test statistic and p-value.
    """
    # scipy.stats.friedmanchisquare expects each model's scores as a separate array
    model_scores = [score_matrix[:, i] for i in range(score_matrix.shape[1])]
    stat, p_value = stats.friedmanchisquare(*model_scores)
    return StatTestResult(
        test_name="Friedman",
        statistic=float(stat),
        p_value=float(p_value),
        significant=p_value < alpha,
    )
