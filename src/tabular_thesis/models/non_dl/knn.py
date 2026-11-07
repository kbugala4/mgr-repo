"""Direct adapter to the same native KNN used by the experiment backend."""

from __future__ import annotations

import numpy as np
import pandas as pd

from tabular_thesis.evaluation.metrics import validate_probabilities
from tabular_thesis.models.base import BaseModel
from tabular_thesis.models.preprocessing import column_roles


class KNNModel(BaseModel):
    def fit(
        self,
        X_train,
        y_train,
        X_val=None,
        y_val=None,
        numerical_columns=None,
        categorical_columns=None,
    ):
        if self.context.class_weights:
            raise ValueError("Class-weighted KNN is outside the declared study")
        from pathlib import Path

        from tabular_thesis.experiments.runtime import configure_caches

        configure_caches(Path("data/cache"))
        from tabular_thesis.experiments.upstream_models import StudyKNN

        numerical, categorical = column_roles(X_train, numerical_columns, categorical_columns)
        self._store_columns(numerical, categorical)
        self.n_classes_ = len(np.unique(y_train))
        self.model_ = StudyKNN(
            path=str(Path("data/cache/direct-knn")),
            problem_type="binary" if self.n_classes_ == 2 else "multiclass",
            hyperparameters={k: v for k, v in self.params.items() if k != "random_state"},
            eval_metric="roc_auc" if self.n_classes_ == 2 else "log_loss",
        )
        self.model_.fit(
            X=self._frame(X_train),
            y=pd.Series(y_train, index=X_train.index),
            num_cpus=self.context.num_cpus,
            time_limit=self.context.time_limit,
            random_seed=self.context.seed,
        )

    @property
    def is_deep_learning(self):
        return False

    def _frame(self, X):
        frame = X.copy()
        for column in self.categorical_columns:
            frame[column] = frame[column].astype("category")
        return frame

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)

    def predict_proba(self, X):
        return validate_probabilities(
            self.model_.predict_proba(self._frame(X)), len(X), self.n_classes_
        )

    def get_search_space(self):
        from tabarena.models.knn.hpo import gen_knn

        return dict(gen_knn.search_space)
