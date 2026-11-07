"""Abstract base class for all models (DL and non-DL).

Every model in the benchmark implements this interface, enabling
uniform training, evaluation, and hyperparameter tuning.

Models receive raw DataFrames and own fold-local imputation, scaling,
and encoding in fit().
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from tabular_thesis.experiments.runtime import FitContext
from tabular_thesis.models.specification import model_spec


@dataclass
class PredictionResult:
    """Container for model predictions and timing information."""

    y_pred: np.ndarray
    y_proba: np.ndarray
    training_time_sec: float = 0.0
    inference_time_sec: float = 0.0
    model_params: dict[str, Any] = field(default_factory=dict)


class BaseModel(ABC):
    """Abstract base class that all models must implement.

    Models receive raw DataFrames, including missing values.
    Each model applies its own preprocessing in fit() and stores the
    fitted transformers for use in predict()/predict_proba().

    Attributes:
        name: Model identifier string.
        params: Hyperparameters dictionary.
        numerical_columns: Set during fit — names of numerical features.
        categorical_columns: Set during fit — names of categorical features.
    """

    def __init__(self, name: str, params: dict[str, Any] | None = None):
        self.name = name
        self.params = params or {}
        model_spec(name).validate_parameters(self.params)
        self.context = FitContext(seed=self.params.get("random_state", 42))
        self.numerical_columns: list[str] = []
        self.categorical_columns: list[str] = []
        self.preprocessing_time_sec: float = 0.0  # Measured inside fit()

    def set_context(self, context: FitContext) -> BaseModel:
        context.validate()
        self.context = context
        self.params["random_state"] = context.seed
        return self

    @abstractmethod
    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_val: pd.DataFrame | None = None,
        y_val: np.ndarray | None = None,
        numerical_columns: list[str] | None = None,
        categorical_columns: list[str] | None = None,
    ) -> None:
        """Train the model on the given data.

        Models should apply their own preprocessing (scaling, encoding)
        inside this method and store fitted transformers for predict().

        Args:
            X_train: Training features (DataFrame, imputed but not scaled).
            y_train: Training labels (int-encoded).
            X_val: Validation features (for early stopping).
            y_val: Validation labels.
            numerical_columns: Names of numerical columns.
            categorical_columns: Names of categorical columns.
        """

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return hard class predictions.

        Must apply the same preprocessing learned in fit().

        Args:
            X: Feature DataFrame (imputed but not scaled).

        Returns:
            1-D array of predicted class labels.
        """

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return class probability predictions.

        Must apply the same preprocessing learned in fit().

        Args:
            X: Feature DataFrame (imputed but not scaled).

        Returns:
            2-D array of shape (n_samples, n_classes).
        """

    @abstractmethod
    def get_search_space(self) -> dict[str, tuple]:
        """Return the Optuna hyperparameter search space.

        Each entry maps a parameter name to a tuple describing its range.
        Supported formats::

            ("int", low, high)
            ("float", low, high)
            ("float_log", low, high)
            ("categorical", [choice1, choice2, ...])
        """

    @property
    @abstractmethod
    def is_deep_learning(self) -> bool:
        """Whether this model uses deep learning."""

    def _store_columns(
        self,
        numerical_columns: list[str] | None,
        categorical_columns: list[str] | None,
    ) -> None:
        """Store column metadata for use in predict/predict_proba."""
        self.numerical_columns = numerical_columns or []
        self.categorical_columns = categorical_columns or []

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"
