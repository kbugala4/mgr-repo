"""Dataset metadata, types, and containers using dataclasses."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto

import numpy as np
import pandas as pd


class FeatureType(Enum):
    """Supported feature types in tabular data."""

    NUMERICAL = auto()
    CATEGORICAL = auto()
    BINARY = auto()
    ORDINAL = auto()


class TaskType(Enum):
    """Classification task types."""

    BINARY = auto()
    MULTICLASS = auto()


@dataclass(frozen=True)
class DatasetMeta:
    """Immutable metadata describing a dataset.

    Attributes:
        name: Human-readable dataset name (e.g. 'adult').
        source: Configured route ('openml', 'uci_zip', 'local_csv').
        source_id: OpenML or UCI dataset ID, if applicable.
        task: Classification task type.
        target_column: Name of the target column.
        n_classes: Number of distinct classes.
        description: Optional human-readable description.
    """

    name: str
    source: str
    source_id: int | None = None
    task: TaskType = TaskType.BINARY
    target_column: str = "target"
    n_classes: int = 2
    description: str = ""


@dataclass
class PreparedData:
    """Container for preprocessed, split data ready for model training.

    Contains DataFrames (minimally preprocessed — imputed only) so that
    models can apply their own scaling/encoding. Also provides column
    metadata so models know which features are numerical vs. categorical.
    """

    # DataFrames — imputed but NOT scaled/encoded
    X_train: pd.DataFrame
    X_val: pd.DataFrame
    X_test: pd.DataFrame
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray

    # Column metadata
    numerical_columns: list[str] = field(default_factory=list)
    categorical_columns: list[str] = field(default_factory=list)
    feature_names: list[str] = field(default_factory=list)

    # Task metadata
    n_classes: int = 2
    dataset_meta: DatasetMeta | None = None
