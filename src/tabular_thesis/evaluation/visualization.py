"""Visualization utilities for experiment results."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_model_comparison(
    results: dict[str, dict[str, float]],
    metric: str = "auc_roc",
    title: str = "Model Comparison",
    save_path: Path | None = None,
) -> None:
    """Bar chart comparing models across datasets.

    Args:
        results: Nested dict ``{model_name: {dataset_name: score}}``.
        metric: Name of the metric (for axis label).
        title: Plot title.
        save_path: If provided, save figure to this path.
    """
    models = sorted(results.keys())
    datasets = sorted({ds for scores in results.values() for ds in scores})

    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(datasets))
    width = 0.8 / len(models)

    for i, model in enumerate(models):
        scores = [results[model].get(ds, 0.0) for ds in datasets]
        ax.bar(x + i * width, scores, width, label=model)

    ax.set_xlabel("Dataset")
    ax.set_ylabel(metric)
    ax.set_title(title)
    ax.set_xticks(x + width * len(models) / 2)
    ax.set_xticklabels(datasets, rotation=45, ha="right")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize="small")
    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_scaling_experiment(
    results: dict[str, list[float]],
    sizes: list[int],
    metric: str = "auc_roc",
    title: str = "Data Scaling Experiment (HIGGS)",
    save_path: Path | None = None,
) -> None:
    """Line plot of performance vs. dataset size.

    Args:
        results: ``{model_name: [score_at_size_1, score_at_size_2, ...]}``.
        sizes: Training set sizes corresponding to each score.
        metric: Metric name for y-axis label.
        title: Plot title.
        save_path: If provided, save figure to this path.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    for model, scores in sorted(results.items()):
        ax.plot(sizes, scores, marker="o", label=model)

    ax.set_xlabel("Training Set Size")
    ax.set_ylabel(metric)
    ax.set_title(title)
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize="small")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
