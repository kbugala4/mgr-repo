"""Regularized logistic regression with fold-local nominal encoding."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression

from tabular_thesis.models.base import BaseModel
from tabular_thesis.models.preprocessing import TabularPreprocessor, column_roles


class LogisticRegressionModel(BaseModel):
    def fit(
        self,
        X_train,
        y_train,
        X_val=None,
        y_val=None,
        numerical_columns=None,
        categorical_columns=None,
    ):
        numerical, categorical = column_roles(X_train, numerical_columns, categorical_columns)
        self._store_columns(numerical, categorical)
        self.preprocessor_ = TabularPreprocessor(numerical, categorical, encoding="onehot").fit(
            X_train
        )
        if len(np.unique(y_train)) > 2 and self.params.get("solver") == "liblinear":
            raise ValueError("liblinear is not a valid multiclass solver")
        weights = self.context.class_weights
        self.model_ = LogisticRegression(
            C=self.params.get("C", 1.0),
            max_iter=self.params.get("max_iter", 1000),
            solver=self.params.get("solver", "lbfgs"),
            random_state=self.context.seed,
            class_weight=None if weights is None else dict(enumerate(weights)),
        )
        self.model_.fit(self.preprocessor_.transform(X_train), y_train)

    @property
    def is_deep_learning(self) -> bool:
        return False

    def predict(self, X):
        return np.asarray(self.model_.predict(self.preprocessor_.transform(X))).reshape(-1)

    def predict_proba(self, X):
        return self.model_.predict_proba(self.preprocessor_.transform(X))

    def get_search_space(self):
        from tabular_thesis.experiments.configuration import (
            ConfigResolver,
            default_config_directory,
        )

        return {
            key: tuple(value)
            for key, value in ConfigResolver(default_config_directory())
            .model(self.name)
            .search_space.items()
        }
