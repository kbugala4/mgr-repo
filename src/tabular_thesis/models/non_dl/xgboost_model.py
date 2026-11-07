"""Task-aware XGBoost with explicit resource limits and fold-local encoding."""

from __future__ import annotations

import numpy as np
from xgboost import XGBClassifier

from tabular_thesis.models.base import BaseModel
from tabular_thesis.models.preprocessing import TabularPreprocessor, column_roles


class XGBoostModel(BaseModel):
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
        self.preprocessor_ = TabularPreprocessor(numerical, categorical, scale=False).fit(X_train)
        multiclass = len(np.unique(y_train)) > 2
        params = {key: value for key, value in self.params.items() if key != "random_state"}
        self.model_ = XGBClassifier(
            **params,
            objective="multi:softprob" if multiclass else "binary:logistic",
            eval_metric="mlogloss" if multiclass else "logloss",
            n_jobs=self.context.num_cpus,
            random_state=self.context.seed,
            device=self.context.device,
            verbosity=0,
        )
        kwargs = {}
        if X_val is not None:
            kwargs.update(eval_set=[(self.preprocessor_.transform(X_val), y_val)], verbose=False)
        if self.context.class_weights:
            kwargs["sample_weight"] = np.asarray(self.context.class_weights)[np.asarray(y_train)]
        self.model_.fit(self.preprocessor_.transform(X_train), y_train, **kwargs)

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
