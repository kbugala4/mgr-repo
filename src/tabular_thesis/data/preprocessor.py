"""Compatibility preprocessing for direct holdout workflows.

The main experiment backend does not use this shared preprocessor; each
model fits its own transformations inside the declared training partition.

This preprocessor handles ONLY what all models need:
- Missing value imputation
- Feature type detection (numerical vs. categorical)
- Target label encoding

Model-specific preprocessing (standardization, encoding, embeddings)
is handled inside each model's fit() method. This design:
- Gives tree-based models raw data (their advantage)
- Lets DL models apply their own scaling/embedding
- Attributes preprocessing cost to the model that needs it
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)


@dataclass
class PreprocessorConfig:
    """Configuration for the minimal preprocessing pipeline."""

    missing_numerical: str = "median"  # 'median', 'mean', 'zero'
    missing_categorical: str = "most_frequent"  # 'most_frequent', 'constant'


class MinimalPreprocessor:
    """Minimal shared preprocessing: imputation + type detection.

    Does NOT standardize or encode features — that's model-specific.
    Follows the fit/transform pattern: learn from training data only.

    Usage::

        preprocessor = MinimalPreprocessor()
        preprocessor.fit(X_train, y_train)
        X_train_clean = preprocessor.transform(X_train)
        X_val_clean = preprocessor.transform(X_val)
    """

    def __init__(self, config: PreprocessorConfig | None = None):
        self.config = config or PreprocessorConfig()
        self._fitted = False

        # Learned components
        self._numerical_cols: list[str] = []
        self._categorical_cols: list[str] = []
        self._num_imputer: SimpleImputer | None = None
        self._cat_imputer: SimpleImputer | None = None
        self._label_encoder: LabelEncoder | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> MinimalPreprocessor:
        """Learn imputation parameters and detect feature types.

        Args:
            X: Training feature DataFrame.
            y: Training target Series (for label encoding).

        Returns:
            self, for method chaining.
        """
        self._detect_feature_types(X)

        # Fit numerical imputer
        if self._numerical_cols:
            self._num_imputer = SimpleImputer(
                strategy="constant"
                if self.config.missing_numerical == "zero"
                else self.config.missing_numerical,
                fill_value=0,
                keep_empty_features=True,
            )
            self._num_imputer.fit(X[self._numerical_cols])

        # Fit categorical imputer
        if self._categorical_cols:
            self._cat_imputer = SimpleImputer(
                strategy=self.config.missing_categorical,
                fill_value="__MISSING__",
                keep_empty_features=True,
            )
            self._cat_imputer.fit(
                X[self._categorical_cols]
                .astype(object)
                .map(lambda v: str(v) if pd.notna(v) else np.nan)
            )

        # Fit label encoder for target
        if y is not None:
            self._label_encoder = LabelEncoder()
            self._label_encoder.fit(y)

        self._fitted = True
        logger.info(
            "MinimalPreprocessor fitted: %d numerical, %d categorical features",
            len(self._numerical_cols),
            len(self._categorical_cols),
        )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply imputation, returning a clean DataFrame (not numpy array).

        Numerical columns have missing values imputed.
        Categorical columns have missing values imputed.
        No scaling or encoding is applied — that's model-specific.

        Args:
            X: Feature DataFrame.

        Returns:
            DataFrame with imputed values and the original index/column order.
            Categorical values are converted to strings; dtypes may change.
        """
        if not self._fitted:
            raise RuntimeError("MinimalPreprocessor must be fitted before transform()")

        result = X.copy()

        if self._numerical_cols and self._num_imputer is not None:
            result[self._numerical_cols] = self._num_imputer.transform(X[self._numerical_cols])

        if self._categorical_cols and self._cat_imputer is not None:
            result[self._categorical_cols] = self._cat_imputer.transform(
                X[self._categorical_cols]
                .astype(object)
                .map(lambda v: str(v) if pd.notna(v) else np.nan)
            )

        return result

    def transform_target(self, y: pd.Series) -> np.ndarray:
        """Encode target labels as integers."""
        if self._label_encoder is None:
            raise RuntimeError("MinimalPreprocessor was not fitted with a target (y)")
        return self._label_encoder.transform(y)

    def fit_transform(self, X: pd.DataFrame, y: pd.Series | None = None) -> pd.DataFrame:
        """Fit and return transformed DataFrame."""
        self.fit(X, y)
        return self.transform(X)

    @property
    def numerical_columns(self) -> list[str]:
        """Names of numerical columns."""
        return list(self._numerical_cols)

    @property
    def categorical_columns(self) -> list[str]:
        """Names of categorical columns."""
        return list(self._categorical_cols)

    @property
    def feature_names(self) -> list[str]:
        """All feature names (numerical first, then categorical)."""
        return self._numerical_cols + self._categorical_cols

    @property
    def n_features(self) -> int:
        """Total number of features."""
        return len(self._numerical_cols) + len(self._categorical_cols)

    def _detect_feature_types(self, X: pd.DataFrame) -> None:
        """Classify columns as numerical or categorical."""
        self._numerical_cols = []
        self._categorical_cols = []

        for col in X.columns:
            if X[col].dtype in ("object", "category", "bool"):
                self._categorical_cols.append(col)
            else:
                self._numerical_cols.append(col)

        logger.debug(
            "Detected — numerical: %s, categorical: %s",
            self._numerical_cols,
            self._categorical_cols,
        )
