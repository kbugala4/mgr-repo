"""Task-aware TabNet adapter with categorical embeddings and explicit resources."""

import time

import numpy as np
import torch
from pytorch_tabnet.callbacks import Callback
from pytorch_tabnet.tab_model import TabNetClassifier

from tabular_thesis.models.base import BaseModel
from tabular_thesis.models.preprocessing import TabularPreprocessor, column_roles


class _Deadline(Callback):
    def __init__(self, deadline):
        self.deadline = deadline

    def on_batch_end(self, batch, logs=None):
        if time.monotonic() >= self.deadline:
            raise TimeoutError("TabNet exceeded its fit budget")


class TabNetModel(BaseModel):
    @property
    def is_deep_learning(self):
        return True

    def fit(
        self,
        X_train,
        y_train,
        X_val=None,
        y_val=None,
        numerical_columns=None,
        categorical_columns=None,
    ):
        self.context.activate()
        if self.context.class_weights:
            raise ValueError("A class-weighted TabNet condition has not been qualified")
        deadline = self.context.deadline()
        start = time.perf_counter()
        numerical, categorical = column_roles(X_train, numerical_columns, categorical_columns)
        self._store_columns(numerical, categorical)
        self.preprocessor_ = TabularPreprocessor(numerical, categorical).fit(X_train)
        train = self.preprocessor_.transform(X_train)
        if len(train) < 2 or self.params["batch_size"] < 2:
            raise ValueError("TabNet BatchNorm needs at least two training rows and batch_size >=2")
        validation = self.preprocessor_.transform(X_val) if X_val is not None else None
        self.preprocessing_time_sec = time.perf_counter() - start
        architecture = {
            k: self.params[k]
            for k in [
                "n_d",
                "n_a",
                "n_steps",
                "gamma",
                "n_independent",
                "n_shared",
                "momentum",
                "mask_type",
            ]
        }
        self.model_ = TabNetClassifier(
            **architecture,
            cat_idxs=list(range(len(numerical), len(numerical) + len(categorical))),
            cat_dims=self.preprocessor_.cardinalities,
            seed=self.context.seed,
            device_name=self.context.device,
            verbose=0,
            optimizer_fn=torch.optim.AdamW,
            optimizer_params={
                "lr": self.params["learning_rate"],
                "weight_decay": self.params.get("weight_decay", 0.0),
            },
        )
        # Avoid a singleton BatchNorm batch without dropping training observations.
        batch_size = self.params["batch_size"]
        while len(train) % batch_size == 1:
            batch_size += 1
        self.effective_batch_size_ = batch_size
        kwargs = dict(
            max_epochs=self.params["epochs"],
            patience=self.params.get("patience", 10),
            batch_size=batch_size,
            virtual_batch_size=min(128, batch_size),
            drop_last=False,
            callbacks=[_Deadline(deadline)],
        )
        if validation is not None:
            kwargs.update(
                eval_set=[(validation, y_val)],
                eval_metric=["auc" if len(np.unique(y_train)) == 2 else "logloss"],
            )
        self.model_.fit(train, y_train, **kwargs)
        self.model_.batch_size = self.context.prediction_batch_size

    def predict(self, X):
        return self.model_.predict(self.preprocessor_.transform(X))

    def predict_proba(self, X):
        return self.model_.predict_proba(self.preprocessor_.transform(X))

    def get_search_space(self):
        from tabular_thesis.experiments.configuration import (
            ConfigResolver,
            default_config_directory,
        )

        return {
            k: tuple(v)
            for k, v in ConfigResolver(default_config_directory())
            .model(self.name)
            .search_space.items()
        }
