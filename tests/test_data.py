""" "Tests for the data pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tabular_thesis.data.preprocessor import MinimalPreprocessor
from tabular_thesis.data.registry import DatasetMeta, TaskType
from tabular_thesis.data.splitter import SplitRatios


class TestDatasetMeta:
    """Tests for DatasetMeta dataclass."""

    def test_create_binary(self) -> None:
        meta = DatasetMeta(name="test", source="openml", task=TaskType.BINARY, n_classes=2)
        assert meta.name == "test"
        assert meta.task == TaskType.BINARY
        assert meta.n_classes == 2

    def test_frozen(self) -> None:
        meta = DatasetMeta(name="test", source="openml")
        with pytest.raises(AttributeError):
            meta.name = "changed"  # type: ignore[misc]


class TestSplitRatios:
    """Tests for SplitRatios validation."""

    def test_valid_ratios(self) -> None:
        split = SplitRatios(train=0.6, val=0.2, test=0.2)
        assert split.train == 0.6

    def test_invalid_ratios(self) -> None:
        with pytest.raises(ValueError, match="sum to 1.0"):
            SplitRatios(train=0.5, val=0.1, test=0.1)


class TestMinimalPreprocessor:
    """Tests for the minimal preprocessing pipeline."""

    def test_numerical_only(self) -> None:
        X = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "b": [10.0, 20.0, 30.0, 40.0]})
        y = pd.Series([0, 1, 0, 1])

        preprocessor = MinimalPreprocessor()
        preprocessor.fit(X, y)
        result = preprocessor.transform(X)

        # Returns DataFrame (not numpy array)
        assert isinstance(result, pd.DataFrame)
        assert result.shape == (4, 2)
        assert len(preprocessor.numerical_columns) == 2
        assert len(preprocessor.categorical_columns) == 0

    def test_categorical_only(self) -> None:
        X = pd.DataFrame({"color": ["red", "blue", "red", "green"], "size": ["S", "M", "L", "M"]})
        y = pd.Series([0, 1, 0, 1])

        preprocessor = MinimalPreprocessor()
        preprocessor.fit(X, y)
        result = preprocessor.transform(X)

        assert isinstance(result, pd.DataFrame)
        assert result.shape == (4, 2)
        assert len(preprocessor.numerical_columns) == 0
        assert len(preprocessor.categorical_columns) == 2

    def test_mixed_features(self) -> None:
        X = pd.DataFrame(
            {
                "age": [25.0, 30.0, 35.0, 40.0],
                "color": ["red", "blue", "red", "green"],
            }
        )
        y = pd.Series([0, 1, 0, 1])

        preprocessor = MinimalPreprocessor()
        preprocessor.fit(X, y)
        result = preprocessor.transform(X)

        assert isinstance(result, pd.DataFrame)
        assert result.shape == (4, 2)
        assert len(preprocessor.numerical_columns) == 1
        assert len(preprocessor.categorical_columns) == 1

    def test_target_encoding(self) -> None:
        X = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
        y = pd.Series(["cat", "dog", "cat"])

        preprocessor = MinimalPreprocessor()
        preprocessor.fit(X, y)
        y_encoded = preprocessor.transform_target(y)

        assert y_encoded.dtype in (np.int64, np.intp)
        assert set(y_encoded) == {0, 1}

    def test_not_fitted_raises(self) -> None:
        preprocessor = MinimalPreprocessor()
        X = pd.DataFrame({"a": [1.0]})
        with pytest.raises(RuntimeError, match="fitted"):
            preprocessor.transform(X)

    def test_imputes_missing_numerical(self) -> None:
        X = pd.DataFrame({"a": [1.0, np.nan, 3.0, 4.0]})
        preprocessor = MinimalPreprocessor()
        preprocessor.fit(X)
        result = preprocessor.transform(X)
        assert not result["a"].isna().any()

    def test_imputes_missing_categorical(self) -> None:
        X = pd.DataFrame({"color": ["red", None, "red", "blue"]})
        preprocessor = MinimalPreprocessor()
        preprocessor.fit(X)
        result = preprocessor.transform(X)
        assert not result["color"].isna().any()
