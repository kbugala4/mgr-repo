"""Fold-local automatic batches, explicit overrides and revised search spaces."""

import numpy as np
import pandas as pd
import pytest

from tabular_thesis.experiments.catalog import ModelCatalog
from tabular_thesis.experiments.configuration import ConfigResolver, default_config_directory
from tabular_thesis.models.dl.batching import dense_batch, resolve, tabnet_batch
from tabular_thesis.models.registry import get_model
from tabular_thesis.models.specification import model_spec


@pytest.mark.parametrize(
    "rows,expected",
    [(1999, 64), (2000, 128), (7999, 128), (8000, 256), (31999, 256),
     (32000, 512), (127999, 512), (128000, 1024), (381000, 1024)],
)
def test_dense_thresholds(rows, expected):
    assert dense_batch(rows) == expected


@pytest.mark.parametrize(
    "rows,cardinality,expected",
    [(28500, 100, (1024, 128)), (19000, 15600, (2048, 256)),
     (57000, 60, (2048, 256)), (381000, 0, (4096, 512)), (230, 0, (128, 128)),
     (1999, 0, (128, 128)), (2000, 0, (256, 128)), (7999, 0, (256, 128)),
     (8000, 0, (1024, 128)), (31999, 0, (1024, 128)), (32000, 0, (2048, 256)),
     (127999, 0, (2048, 256)), (128000, 0, (4096, 512))],
)
def test_tabnet_thresholds(rows, cardinality, expected):
    assert tabnet_batch(rows, cardinality) == expected


@pytest.mark.parametrize(
    "configured,base,multiplier,expected",
    [("auto", 256, 0.5, 128), ("auto", 256, 1, 256), ("auto", 256, 2, 512),
     ("auto", 1024, 2, 2048), ("auto", 64, 0.5, 32), (64, 256, 2, 64),
     ("auto", 5, 1, 8), ("auto", 64, 0.001, 8), ("auto", 256, 0.75, 128)],
)
def test_batch_resolution(configured, base, multiplier, expected):
    assert resolve(configured, base, multiplier) == expected


@pytest.mark.parametrize("name,expected", [("mlp", 256), ("tabnet", 512)])
def test_auto_batch_fit_records_realised_values(name, expected):
    X = pd.DataFrame(np.random.default_rng(4).normal(size=(3000, 3)), columns=list("abc"))
    y = np.tile([0, 1], 1500)
    architecture = {"hidden_dims": [8]} if name == "mlp" else {"n_d": 4, "n_a": 4, "n_steps": 3}
    model = get_model(
        name, {**architecture, "batch_size": "auto", "batch_size_multiplier": 2, "epochs": 1}
    )
    model.fit(X, y, X.iloc[:32], y[:32])
    assert model.training_info_["effective_batch_size"] == expected
    if name == "tabnet":
        assert model.training_info_["effective_virtual_batch_size"] == 128
    assert model.predict_proba(X.iloc[:32]).shape == (32, 2)


@pytest.mark.parametrize("virtual,expected", [("auto", 64), (8, 8), (512, 64)])
def test_explicit_tabnet_batch_and_clamped_virtual_metadata(virtual, expected):
    X = pd.DataFrame(np.random.default_rng(4).normal(size=(128, 3)), columns=list("abc"))
    y = np.tile([0, 1], 64)
    model = get_model(
        "tabnet",
        {"n_d": 4, "n_a": 4, "n_steps": 3, "epochs": 1, "batch_size": 64,
         "batch_size_multiplier": 2, "virtual_batch_size": virtual},
    )
    model.fit(X, y)
    assert model.training_info_["effective_batch_size"] == 64
    assert model.training_info_["effective_virtual_batch_size"] == expected
    assert model.model_.virtual_batch_size == expected


@pytest.mark.parametrize("rows", [7, 15])
def test_tabnet_virtual_chunks_never_drop_or_isolate_training_rows(rows, monkeypatch):
    import torch

    X = pd.DataFrame(np.random.default_rng(5).normal(size=(rows, 3)), columns=list("abc"))
    y = np.arange(rows) % 2
    sizes = []
    original = torch.nn.BatchNorm1d.forward

    def check_chunks(module, values):
        if module.training:
            sizes.append(len(values))
            assert len(values) >= 2
        return original(module, values)

    monkeypatch.setattr(torch.nn.BatchNorm1d, "forward", check_chunks)
    model = get_model(
        "tabnet", {"n_d": 4, "n_a": 4, "n_steps": 3, "epochs": 1,
                   "batch_size": 8, "virtual_batch_size": 3},
    )
    model.fit(X, y)
    assert model.training_info_["effective_batch_size"] == 8
    assert model.training_info_["effective_virtual_batch_size"] == 4
    assert sizes
    assert model.predict_proba(X).shape == (rows, 2)


def test_ft_cap_follows_auto_multiplier_resolution():
    model = get_model(
        "ft_transformer", {"batch_size": "auto", "batch_size_multiplier": 2, "attention_n_heads": 8}
    )
    X = pd.DataFrame(np.empty((3000, 1776), dtype=np.float32))
    assert model._batch_size(X) == 32


@pytest.mark.parametrize("name", ["mlp", "ft_transformer", "tabnet"])
def test_revised_configuration_catalog(name):
    resolver = ConfigResolver(default_config_directory())
    study = resolver.study(["models=[mlp,ft_transformer,tabnet]"])
    config = resolver.resolved_models(study)[name]
    assert config.params["batch_size"] == "auto"
    assert config.params["batch_size_multiplier"] == 1
    _, candidates = ModelCatalog().configurations(config, 25, study.search_seed)
    assert len(candidates) == 26
    for candidate in candidates:
        assert candidate["batch_size"] == "auto"
        assert candidate["batch_size_multiplier"] in {0.5, 1, 2}
        model_spec(name).validate_parameters(candidate)


@pytest.mark.parametrize("value", [-1, 0, True, "2", float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize("name", ["mlp", "ft_transformer", "tabnet"])
def test_invalid_multipliers_fail_at_configuration_boundary(name, value):
    with pytest.raises(ValueError, match="finite positive number"):
        model_spec(name).validate_parameters({"batch_size_multiplier": value})


@pytest.mark.parametrize("value", [0, 1, -1, True, 2.5])
def test_invalid_virtual_batches_fail_at_configuration_boundary(value):
    with pytest.raises(ValueError, match="integer >=2"):
        model_spec("tabnet").validate_parameters({"virtual_batch_size": value})


def test_virtual_batch_is_tabnet_only_and_other_methods_reject_multiplier():
    model_spec("tabnet").validate_parameters({"virtual_batch_size": "auto"})
    with pytest.raises(ValueError, match="Unsupported parameters"):
        model_spec("mlp").validate_parameters({"virtual_batch_size": "auto"})
    for name in ["realmlp", "modernnca", "tabm", "knn", "logistic_regression", "xgboost",
                 "catboost", "tabpfn"]:
        with pytest.raises(ValueError, match="Unsupported parameters"):
            model_spec(name).validate_parameters({"batch_size_multiplier": 2})
