""" "Tests for the data pipeline."""

from __future__ import annotations

from collections import Counter
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from tabular_thesis.data.preprocessor import MinimalPreprocessor
from tabular_thesis.data.registry import DatasetMeta, TaskType
from tabular_thesis.data.splitter import SplitRatios
from tabular_thesis.data.tasks import DatasetRepository
from tabular_thesis.experiments.configuration import ConfigResolver, default_config_directory
from tabular_thesis.experiments.runtime import configure_caches


def test_all_accepted_datasets_have_automatic_sources():
    resolver = ConfigResolver(default_config_directory())
    sources = Counter(resolver.dataset(name).source for name in resolver.study().datasets)
    assert sources == {"openml": 8, "uci_zip": 4}
    santander = resolver.dataset("santander")
    assert (santander.source_id, santander.version, santander.target) == (46859, 2, "TARGET")
    assert santander.exclude == []
    assert santander.local_file is None


def test_santander_uses_openml_schema_and_version(tmp_path, monkeypatch):
    configure_caches(tmp_path / "cache")
    import openml

    config = ConfigResolver(default_config_directory()).dataset("santander")
    X = pd.DataFrame({"var3": range(60), "var15": range(60, 120)})
    y = pd.Series(pd.Categorical(["0.0", "1.0"] * 30), name="TARGET")

    def get_data(**kwargs):
        assert kwargs == {
            "dataset_format": "dataframe", "target": "TARGET", "include_row_id": True
        }
        return X.copy(), y.copy(), [False, False], X.columns.tolist()

    dataset = SimpleNamespace(version=2, get_data=get_data)
    calls = []

    def get_dataset(source_id, *, download_data):
        calls.append(source_id)
        assert source_id == 46859 and download_data is True
        return dataset

    monkeypatch.setattr(openml.datasets, "get_dataset", get_dataset)
    repository = DatasetRepository(tmp_path)
    task = repository.prepare(config)
    assert calls == [46859]
    assert task.X.columns.tolist() == ["var3", "var15"]
    assert task.classes == ["0.0", "1.0"]
    np.testing.assert_array_equal(task.y, np.tile([0, 1], 30))
    assert task.source_identity["source"]["source_id"] == 46859
    assert task.source_identity["source"]["version"] == 2
    assert repository.prepare(config).source_identity == task.source_identity
    assert calls == [46859]
    assert not (tmp_path / "raw").exists()
    dataset.version = 1
    with pytest.raises(ValueError, match="Unexpected OpenML version"):
        repository._download(config)


def test_old_santander_csv_cache_is_preserved_and_not_reused(tmp_path, monkeypatch):
    config = ConfigResolver(default_config_directory()).dataset("santander")
    legacy = replace(
        config, source="local_csv", source_id=0, version=1,
        local_file="santander/train.csv", exclude=["ID"],
    )
    raw = tmp_path / "raw" / "santander" / "train.csv"
    raw.parent.mkdir(parents=True)
    pd.DataFrame({"ID": range(60), "var3": range(60), "TARGET": [0, 1] * 30}).to_csv(
        raw, index=False
    )
    repository = DatasetRepository(tmp_path)
    repository.prepare(legacy)
    directory = tmp_path / "prepared" / "santander"
    saved = {path: path.read_bytes() for path in [raw, *directory.iterdir()]}
    monkeypatch.setattr(
        repository, "_download", lambda _: pytest.fail("Incompatible cache was overwritten")
    )
    with pytest.raises(ValueError, match="Cached source definition differs"):
        repository.prepare(config)
    for path, content in saved.items():
        assert path.read_bytes() == content


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
