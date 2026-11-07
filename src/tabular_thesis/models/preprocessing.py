"""Fold-local preprocessing recipes shared only where their semantics match."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler


def column_roles(X: pd.DataFrame, numerical: list[str] | None, categorical: list[str] | None):
    if numerical is None and categorical is None:
        categorical = X.select_dtypes(
            include=["object", "category", "bool", "string"]
        ).columns.tolist()
        numerical = [c for c in X.columns if c not in categorical]
    numerical, categorical = numerical or [], categorical or []
    if set(numerical) & set(categorical) or set(numerical + categorical) != set(X.columns):
        raise ValueError("Feature roles must cover each predictor exactly once")
    return numerical, categorical


def categorical_values(X: pd.DataFrame) -> pd.DataFrame:
    return X.astype(object).map(lambda value: str(value) if pd.notna(value) else np.nan)


class TabularPreprocessor:
    def __init__(
        self,
        numerical: list[str],
        categorical: list[str],
        *,
        scale=True,
        encoding="ordinal",
        max_categories=None,
    ):
        self.numerical, self.categorical = numerical, categorical
        self.scale, self.encoding = scale, encoding
        self.num_imputer = SimpleImputer(strategy="median", keep_empty_features=True)
        self.cat_imputer = SimpleImputer(
            strategy="constant", fill_value="__MISSING__", keep_empty_features=True
        )
        self.scaler = StandardScaler()
        self.encoder = (
            OneHotEncoder(
                handle_unknown="ignore", sparse_output=True, max_categories=max_categories
            )
            if encoding == "onehot"
            else OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        )

    def fit(self, X: pd.DataFrame):
        if self.numerical:
            numeric = self.num_imputer.fit_transform(X[self.numerical])
            if self.scale:
                self.scaler.fit(numeric)
        if self.categorical:
            categorical = self.cat_imputer.fit_transform(categorical_values(X[self.categorical]))
            self.encoder.fit(categorical)
        return self

    def split(self, X: pd.DataFrame):
        numeric = np.empty((len(X), 0), dtype=np.float32)
        categorical = np.empty((len(X), 0), dtype=np.int64)
        if self.numerical:
            numeric = self.num_imputer.transform(X[self.numerical])
            if self.scale:
                numeric = self.scaler.transform(numeric)
            numeric = numeric.astype(np.float32)
        if self.categorical:
            categorical = self.encoder.transform(
                self.cat_imputer.transform(categorical_values(X[self.categorical]))
            )
            if self.encoding != "onehot":
                # Index zero is reserved for unseen categories, not the first known value.
                categorical = (categorical + 1).astype(np.int64)
        return numeric, categorical

    def transform(self, X: pd.DataFrame, *, dense=False):
        numeric, categorical = self.split(X)
        if sparse.issparse(categorical):
            result = sparse.hstack([sparse.csr_matrix(numeric), categorical], format="csr")
            return result.toarray().astype(np.float32) if dense else result
        return np.hstack([numeric, categorical]).astype(np.float32)

    @property
    def cardinalities(self):
        return [len(values) + 1 for values in self.encoder.categories_] if self.categorical else []
