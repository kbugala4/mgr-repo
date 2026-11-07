""" "Tests for model registry and base class."""

from __future__ import annotations

from tabular_thesis.models.registry import is_registered, list_models


class TestModelRegistry:
    """Tests for the model registry."""

    def test_list_models_not_empty(self) -> None:
        models = list_models()
        assert len(models) > 0

    def test_all_expected_models_registered(self) -> None:
        expected = [
            "logistic_regression",
            "knn",
            "xgboost",
            "catboost",
            "mlp",
            "ft_transformer",
            "tabnet",
        ]
        for name in expected:
            assert is_registered(name), f"Model '{name}' not registered"

    def test_unknown_model_not_registered(self) -> None:
        assert not is_registered("nonexistent_model")
