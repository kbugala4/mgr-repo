"""Compatibility holdout splitting with minimal preprocessing.

The main benchmark uses TaskSpec.outer_split and TabArena inner validation,
not this helper's 60/20/20 split. This helper does not enforce patient groups.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split

from tabular_thesis.data.preprocessor import MinimalPreprocessor
from tabular_thesis.data.registry import DatasetMeta, PreparedData

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SplitRatios:
    """Train / validation / test split ratios. Must sum to 1.0."""

    train: float = 0.6
    val: float = 0.2
    test: float = 0.2

    def __post_init__(self) -> None:
        total = self.train + self.val + self.test
        if not (0.99 < total < 1.01):
            raise ValueError(f"Split ratios must sum to 1.0, got {total:.3f}")


def prepare_data(
    X: pd.DataFrame,
    y: pd.Series,
    meta: DatasetMeta,
    preprocessor: MinimalPreprocessor | None = None,
    split: SplitRatios | None = None,
    seed: int = 42,
) -> PreparedData:
    """Split raw data, apply minimal preprocessing, return PreparedData.

    Minimal preprocessing (imputation only) is fitted on training data,
    then applied to all splits. No scaling or encoding — that's model-specific.

    Args:
        X: Raw feature DataFrame.
        y: Raw target Series.
        meta: Dataset metadata.
        preprocessor: Minimal preprocessor. Created with defaults if None.
        split: Split ratios. Defaults to 60/20/20.
        seed: Random seed for reproducible splitting.

    Returns:
        PreparedData with imputed DataFrames and column metadata.
    """
    split = split or SplitRatios()
    preprocessor = preprocessor or MinimalPreprocessor()

    # First split: separate test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X,
        y,
        test_size=split.test,
        stratify=y,
        random_state=seed,
    )

    # Second split: separate validation from training
    val_ratio_adjusted = split.val / (split.train + split.val)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp,
        y_temp,
        test_size=val_ratio_adjusted,
        stratify=y_temp,
        random_state=seed,
    )

    logger.info(
        "Split sizes — train: %d, val: %d, test: %d",
        len(X_train),
        len(X_val),
        len(X_test),
    )

    # Fit minimal preprocessor on training data only
    preprocessor.fit(X_train, y_train)

    # Transform all splits (imputation only — returns DataFrames)
    X_train_clean = preprocessor.transform(X_train)
    X_val_clean = preprocessor.transform(X_val)
    X_test_clean = preprocessor.transform(X_test)

    # Encode target labels
    y_train_enc = preprocessor.transform_target(y_train)
    y_val_enc = preprocessor.transform_target(y_val)
    y_test_enc = preprocessor.transform_target(y_test)

    return PreparedData(
        X_train=X_train_clean,
        X_val=X_val_clean,
        X_test=X_test_clean,
        y_train=y_train_enc,
        y_val=y_val_enc,
        y_test=y_test_enc,
        numerical_columns=preprocessor.numerical_columns,
        categorical_columns=preprocessor.categorical_columns,
        feature_names=preprocessor.feature_names,
        n_classes=meta.n_classes,
        dataset_meta=meta,
    )
