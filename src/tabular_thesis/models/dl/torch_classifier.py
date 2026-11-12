"""Shared fit lifecycle for the two local reference-network adapters."""

from __future__ import annotations

import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from tabular_thesis.models.base import BaseModel
from tabular_thesis.models.dl.batching import dense_batch, resolve
from tabular_thesis.models.dl.training import BudgetGuard
from tabular_thesis.models.preprocessing import TabularPreprocessor, column_roles


class TorchClassifier(BaseModel):
    separate_inputs = False

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
        torch.manual_seed(self.context.seed)
        budget = BudgetGuard(self.context.time_limit, self.name)
        start = time.perf_counter()
        numerical, categorical = column_roles(X_train, numerical_columns, categorical_columns)
        self._store_columns(numerical, categorical)
        self.preprocessor_ = TabularPreprocessor(
            numerical,
            categorical,
            encoding="ordinal" if self.separate_inputs else "onehot",
            max_categories=None if self.separate_inputs else 256,
        ).fit(X_train)
        train_inputs = self._arrays(X_train)
        validation_inputs = self._arrays(X_val) if X_val is not None else None
        self.preprocessing_time_sec = time.perf_counter() - start
        self.n_classes_ = len(np.unique(y_train))
        self.device_ = self.context.device
        self.network_ = self._network(self.n_classes_, train_inputs).to(self.device_)
        optimizer = torch.optim.AdamW(
            self.network_.parameters(),
            lr=self.params["learning_rate"],
            weight_decay=self.params["weight_decay"],
        )
        weights = self.context.class_weights
        criterion = nn.CrossEntropyLoss(
            weight=None
            if weights is None
            else torch.tensor(weights, dtype=torch.float32, device=self.device_)
        )
        self.effective_batch_size_ = self._batch_size(X_train)
        train_loader = self._loader(
            train_inputs, y_train, shuffle=True, batch_size=self.effective_batch_size_
        )
        validation_loader = (
            None
            if validation_inputs is None
            else self._loader(
                validation_inputs,
                y_val,
                shuffle=False,
                batch_size=min(self.context.prediction_batch_size, self.effective_batch_size_),
            )
        )
        best_state, best_loss, patience = None, float("inf"), 0
        for _ in range(self.params["epochs"]):
            epoch_started = budget.start_epoch()
            self.network_.train()
            for batch in train_loader:
                if budget.check_batch():
                    break
                inputs, labels = (
                    [x.to(self.device_) for x in batch[:-1]],
                    batch[-1].to(self.device_),
                )
                optimizer.zero_grad()
                criterion(self.network_(*inputs), labels).backward()
                optimizer.step()
            if budget.truncated:
                break
            if validation_loader is not None:
                self.network_.eval()
                loss_sum, count = 0.0, 0
                with torch.no_grad():
                    for batch in validation_loader:
                        inputs, labels = (
                            [x.to(self.device_) for x in batch[:-1]],
                            batch[-1].to(self.device_),
                        )
                        # Validation keeps the original class distribution and no training weights.
                        loss_sum += nn.functional.cross_entropy(
                            self.network_(*inputs), labels, reduction="sum"
                        ).item()
                        count += len(labels)
                loss = loss_sum / count
                if loss < best_loss:
                    best_loss, patience = loss, 0
                    best_state = {
                        k: value.detach().cpu().clone()
                        for k, value in self.network_.state_dict().items()
                    }
                else:
                    patience += 1
            elif self.context.time_limit is not None:
                # A partial later epoch must not replace the last complete unvalidated epoch.
                best_state = {
                    k: value.detach().cpu().clone()
                    for k, value in self.network_.state_dict().items()
                }
            early_stopped = patience >= self.params.get("patience", 10)
            if budget.finish_epoch(
                epoch_started, self.params["epochs"], early_stopped=early_stopped
            ) or early_stopped:
                break
        if best_state is not None:
            self.network_.load_state_dict(best_state)
        self.network_.eval()
        self.completed_epochs_ = budget.completed_epochs
        self.training_info_ = budget.result(self.effective_batch_size_)

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)

    def predict_proba(self, X):
        self.network_.eval()
        predictions = []
        with torch.no_grad():
            for batch in self._loader(
                self._arrays(X),
                None,
                shuffle=False,
                batch_size=min(self.context.prediction_batch_size, self.effective_batch_size_),
            ):
                logits = self.network_(*(x.to(self.device_) for x in batch))
                predictions.append(logits.softmax(dim=1).cpu().numpy())
        return np.vstack(predictions)

    def _arrays(self, X):
        if self.separate_inputs:
            return self.preprocessor_.split(X)
        return (self.preprocessor_.transform(X, dense=True),)

    def _batch_size(self, X):
        return resolve(
            self.params["batch_size"],
            dense_batch(len(X)),
            self.params.get("batch_size_multiplier", 1.0),
        )

    def _loader(self, arrays, y, *, shuffle, batch_size):
        tensors = [torch.as_tensor(array) for array in arrays]
        if y is not None:
            tensors.append(torch.as_tensor(np.asarray(y), dtype=torch.long))
        return DataLoader(
            TensorDataset(*tensors),
            batch_size=batch_size,
            shuffle=shuffle,
            generator=torch.Generator().manual_seed(self.context.seed),
        )

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

    def _network(self, n_classes, arrays):
        raise NotImplementedError
