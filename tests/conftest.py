""" "Shared test fixtures."""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture
def sample_binary_data() -> tuple[np.ndarray, np.ndarray]:
    """Small synthetic binary classification dataset for testing."""
    rng = np.random.RandomState(42)
    X = rng.randn(200, 5)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    return X, y


@pytest.fixture
def sample_multiclass_data() -> tuple[np.ndarray, np.ndarray]:
    """Small synthetic multi-class dataset for testing."""
    rng = np.random.RandomState(42)
    X = rng.randn(300, 5)
    y = np.digitize(X[:, 0], bins=[-1, 0, 1])  # 4 classes
    return X, y
