""" "Integration tests for DL models — train, predict, evaluate on synthetic data."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tabular_thesis.evaluation.metrics import compute_metrics
from tabular_thesis.models.registry import get_model


class TestDLModelsEndToEnd:
    """Test that each DL model can train, predict, and produce valid metrics."""

    @pytest.fixture
    def synthetic_data(
        self,
    ) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray, list[str], list[str]]:
        """Small synthetic dataset with mixed features for DL testing."""
        rng = np.random.RandomState(42)
        n_train, n_test = 300, 60
        n_num = 6
        n_cat = 2

        num_data_train = rng.randn(n_train, n_num)
        num_data_test = rng.randn(n_test, n_num)
        num_cols = [f"num_{i}" for i in range(n_num)]

        cat_choices = ["A", "B", "C"]
        cat_data_train = rng.choice(cat_choices, size=(n_train, n_cat))
        cat_data_test = rng.choice(cat_choices, size=(n_test, n_cat))
        cat_cols = [f"cat_{i}" for i in range(n_cat)]

        X_train = pd.DataFrame(num_data_train, columns=num_cols)
        for i, col in enumerate(cat_cols):
            X_train[col] = cat_data_train[:, i]

        X_test = pd.DataFrame(num_data_test, columns=num_cols)
        for i, col in enumerate(cat_cols):
            X_test[col] = cat_data_test[:, i]

        y_train = (num_data_train[:, 0] + num_data_train[:, 1] > 0).astype(int)
        y_test = (num_data_test[:, 0] + num_data_test[:, 1] > 0).astype(int)

        return X_train, y_train, X_test, y_test, num_cols, cat_cols

    @pytest.mark.parametrize(
        "model_name,extra_params",
        [
            ("mlp", {"epochs": 5, "hidden_dims": [32, 16], "batch_size": 64}),
            (
                "ft_transformer",
                {
                    "epochs": 3,
                    "d_token": 32,
                    "n_blocks": 1,
                    "attention_n_heads": 4,
                    "ffn_d_hidden": 32,
                    "batch_size": 64,
                },
            ),
            ("tabnet", {"epochs": 5, "n_d": 8, "n_a": 8, "n_steps": 3, "batch_size": 64}),
        ],
    )
    def test_dl_train_predict_evaluate(
        self,
        model_name: str,
        extra_params: dict,
        synthetic_data: tuple,
    ) -> None:
        """Each DL model should train, predict, and produce valid output shapes."""
        X_train, y_train, X_test, y_test, num_cols, cat_cols = synthetic_data

        model = get_model(model_name, params=extra_params)
        model.fit(
            X_train,
            y_train,
            X_val=X_test,
            y_val=y_test,
            numerical_columns=num_cols,
            categorical_columns=cat_cols,
        )

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)

        # Shape checks
        assert y_pred.shape == (len(X_test),), f"{model_name}: y_pred shape wrong"
        assert y_proba.shape == (len(X_test), 2), f"{model_name}: y_proba shape wrong"

        # Probabilities should sum to ~1
        assert np.allclose(y_proba.sum(axis=1), 1.0, atol=1e-4), (
            f"{model_name}: proba doesn't sum to 1"
        )

        # Metrics should be computable
        metrics = compute_metrics(y_test, y_pred, y_proba, n_classes=2)
        assert 0.0 <= metrics.auc_roc <= 1.0, f"{model_name}: AUC out of range"
        assert 0.0 <= metrics.accuracy <= 1.0, f"{model_name}: Acc out of range"

    @pytest.mark.parametrize(
        "model_name",
        ["mlp", "ft_transformer", "tabnet"],
    )
    def test_dl_search_space_defined(self, model_name: str) -> None:
        """Each DL model should define a non-empty HP search space."""
        model = get_model(model_name)
        space = model.get_search_space()
        assert isinstance(space, dict)
        assert len(space) > 0, f"{model_name} has empty search space"

    @pytest.mark.parametrize(
        "model_name",
        ["mlp", "ft_transformer", "tabnet"],
    )
    def test_dl_is_deep_learning(self, model_name: str) -> None:
        """All DL models should report is_deep_learning=True."""
        model = get_model(model_name)
        assert model.is_deep_learning is True
