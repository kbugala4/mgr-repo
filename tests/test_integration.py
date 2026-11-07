""" "Integration tests — end-to-end model training and evaluation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tabular_thesis.evaluation.metrics import compute_metrics
from tabular_thesis.models.registry import get_model


class TestNonDLModelsEndToEnd:
    """Test that each non-DL model can train, predict, and produce valid metrics."""

    @pytest.fixture
    def synthetic_data(
        self,
    ) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray, list[str], list[str]]:
        """Generate a synthetic binary classification dataset as DataFrames."""
        rng = np.random.RandomState(42)
        n_train, n_test = 500, 100
        n_num = 8
        n_cat = 2

        # Numerical features
        num_data_train = rng.randn(n_train, n_num)
        num_data_test = rng.randn(n_test, n_num)
        num_cols = [f"num_{i}" for i in range(n_num)]

        # Categorical features
        cat_choices = ["A", "B", "C", "D"]
        cat_data_train = rng.choice(cat_choices, size=(n_train, n_cat))
        cat_data_test = rng.choice(cat_choices, size=(n_test, n_cat))
        cat_cols = [f"cat_{i}" for i in range(n_cat)]

        # Build DataFrames
        X_train = pd.DataFrame(num_data_train, columns=num_cols)
        for i, col in enumerate(cat_cols):
            X_train[col] = cat_data_train[:, i]

        X_test = pd.DataFrame(num_data_test, columns=num_cols)
        for i, col in enumerate(cat_cols):
            X_test[col] = cat_data_test[:, i]

        # Target based on numerical features
        y_train = (
            num_data_train[:, 0] + num_data_train[:, 1] + rng.randn(n_train) * 0.3 > 0
        ).astype(int)
        y_test = (num_data_test[:, 0] + num_data_test[:, 1] + rng.randn(n_test) * 0.3 > 0).astype(
            int
        )

        return X_train, y_train, X_test, y_test, num_cols, cat_cols

    @pytest.mark.parametrize(
        "model_name",
        [
            "logistic_regression",
            "knn",
            "xgboost",
            "catboost",
        ],
    )
    def test_train_predict_evaluate(
        self,
        model_name: str,
        synthetic_data: tuple[
            pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray, list[str], list[str]
        ],
    ) -> None:
        """Each non-DL model should train, predict, and produce valid AUC > 0.5."""
        X_train, y_train, X_test, y_test, num_cols, cat_cols = synthetic_data

        model = get_model(model_name)
        model.fit(
            X_train,
            y_train,
            numerical_columns=num_cols,
            categorical_columns=cat_cols,
        )

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)

        # Basic shape checks
        assert y_pred.shape == (len(X_test),)
        assert y_proba.shape == (len(X_test), 2)

        # Probabilities should sum to ~1
        assert np.allclose(y_proba.sum(axis=1), 1.0, atol=1e-5)

        # Compute metrics
        metrics = compute_metrics(y_test, y_pred, y_proba, n_classes=2)

        # Model should do better than random on this easy dataset
        assert metrics.auc_roc > 0.5, f"{model_name} AUC={metrics.auc_roc:.3f} <= 0.5"
        assert metrics.accuracy > 0.5, f"{model_name} Acc={metrics.accuracy:.3f} <= 0.5"

    @pytest.mark.parametrize(
        "model_name",
        [
            "logistic_regression",
            "knn",
            "xgboost",
            "catboost",
        ],
    )
    def test_search_space_defined(self, model_name: str) -> None:
        """Each model should define a non-empty HP search space."""
        model = get_model(model_name)
        space = model.get_search_space()
        assert isinstance(space, dict)
        assert len(space) > 0, f"{model_name} has empty search space"
