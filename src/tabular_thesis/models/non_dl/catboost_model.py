"""CatBoost preserves native categorical and missing-numeric handling."""

from __future__ import annotations

import pandas as pd
from catboost import CatBoostClassifier

from tabular_thesis.models.base import BaseModel
from tabular_thesis.models.preprocessing import column_roles


class CatBoostModel(BaseModel):
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
        params = {key: value for key, value in self.params.items() if key != "random_state"}
        self.model_ = CatBoostClassifier(
            **params,
            cat_features=categorical or None,
            thread_count=self.context.num_cpus,
            random_seed=self.context.seed,
            verbose=0,
            allow_writing_files=False,
            task_type="GPU" if self.context.device == "cuda" else "CPU",
            class_weights=self.context.class_weights,
        )
        kwargs = {}
        if X_val is not None:
            kwargs.update(eval_set=(self._prepare_df(X_val), y_val), early_stopping_rounds=50)
        self.model_.fit(self._prepare_df(X_train), y_train, **kwargs)

    @property
    def is_deep_learning(self):
        return False

    def predict(self, X):
        return self.model_.predict(self._prepare_df(X)).reshape(-1)

    def predict_proba(self, X):
        return self.model_.predict_proba(self._prepare_df(X))

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

    def _prepare_df(self, X):
        frame = X[self.numerical_columns + self.categorical_columns].copy()
        for column in self.categorical_columns:
            frame[column] = (
                frame[column].astype(object).map(lambda v: str(v) if pd.notna(v) else "__MISSING__")
            )
        return frame
