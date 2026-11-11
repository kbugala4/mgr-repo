"""Deterministic regressions for advisory local-wrapper fit budgets."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from tabular_thesis.experiments.runtime import FitContext
from tabular_thesis.models.dl import training
from tabular_thesis.models.registry import get_model

LOCAL_MODELS = [
    ("mlp", {"hidden_dims": [8], "dropout": 0.0}),
    (
        "ft_transformer",
        {"n_blocks": 1, "d_token": 8, "attention_n_heads": 2, "ffn_d_hidden": 8},
    ),
    ("tabnet", {"n_d": 4, "n_a": 4, "n_steps": 3}),
]


@pytest.fixture
def binary_task():
    X = pd.DataFrame(np.random.default_rng(8).normal(size=(32, 3)), columns=list("abc"))
    return X, np.tile([0, 1], 16)


@pytest.mark.parametrize("name,parameters", LOCAL_MODELS)
def test_predictive_budget_stop_restores_best_weights(
    name, parameters, binary_task, tmp_path, monkeypatch
):
    from tabular_thesis.experiments.tabarena_models import LOCAL_ADAPTERS

    X, y = binary_task
    params = {**parameters, "batch_size": 16, "epochs": 1}
    reference = get_model(name, params)
    reference.fit(X, y, X, y)
    expected = reference.predict_proba(X)

    # First epoch lasts six clock units; another would exceed the ten-unit grant.
    clock = iter([0, 0, 1, 2, 6, 6])
    monkeypatch.setattr(training, "time", SimpleNamespace(monotonic=lambda: next(clock)))
    adapter = LOCAL_ADAPTERS[name](
        path=str(tmp_path), problem_type="binary", hyperparameters={**params, "epochs": 5}
    )
    # Exercise the same metadata bridge used by AutoGluon's fold fits.
    adapter.initialize(X=X, y=y)
    adapter.random_seed = 42
    adapter._fit(X, y, X, y, num_cpus=1, time_limit=10)
    info = adapter.params_trained
    assert info["budget_truncated"] is True
    assert info["completed_epochs"] == 1
    assert info["granted_seconds"] == 10
    assert info["fit_seconds_wrapper"] == 6
    assert info["effective_batch_size"] == 16
    np.testing.assert_allclose(adapter.model.predict_proba(X), expected, atol=1e-7)


@pytest.mark.parametrize("name,parameters", LOCAL_MODELS)
def test_no_completed_epoch_still_fails(name, parameters, binary_task, monkeypatch):
    X, y = binary_task
    clock = iter([0, 0, 11])
    monkeypatch.setattr(training, "time", SimpleNamespace(monotonic=lambda: next(clock)))
    model = get_model(name, {**parameters, "batch_size": 16, "epochs": 5})
    model.set_context(FitContext(time_limit=10))
    with pytest.raises(TimeoutError, match="before completing an epoch"):
        model.fit(X, y, X, y)


@pytest.mark.parametrize("name,parameters", LOCAL_MODELS)
def test_unlimited_budget_records_normal_completion(name, parameters, binary_task):
    X, y = binary_task
    model = get_model(name, {**parameters, "batch_size": 16, "epochs": 2})
    model.fit(X, y, X, y)
    assert model.training_info_["budget_truncated"] is False
    assert model.training_info_["granted_seconds"] is None
    assert model.completed_epochs_ == 2
    assert model.predict_proba(X).shape == (32, 2)


@pytest.mark.parametrize("with_validation", [False, True])
def test_partial_torch_epoch_restores_complete_checkpoint(
    binary_task, monkeypatch, with_validation
):
    X, y = binary_task
    validation = (X, y) if with_validation else ()
    params = {"hidden_dims": [8], "dropout": 0.0, "batch_size": 16, "epochs": 1}
    reference = get_model("mlp", params)
    reference.fit(X, y, *validation)
    expected = reference.predict_proba(X)
    # Epoch one completes at t=2. Epoch two updates one batch, then hits t=11.
    clock = iter([0, 0, 0, 1, 2, 2, 3, 11, 11])
    monkeypatch.setattr(training, "time", SimpleNamespace(monotonic=lambda: next(clock)))
    model = get_model("mlp", {**params, "epochs": 5})
    model.set_context(FitContext(time_limit=10))
    model.fit(X, y, *validation)
    assert model.training_info_["budget_truncated"] is True
    assert model.completed_epochs_ == 1
    np.testing.assert_allclose(model.predict_proba(X), expected)


def test_tabnet_partial_epoch_finishes_and_restores_best(binary_task, monkeypatch):
    from pytorch_tabnet.callbacks import EarlyStopping

    X, y = binary_task
    clock = iter([0, 0, 0, 1, 2, 2, 11, 12, 13, 13])
    monkeypatch.setattr(training, "time", SimpleNamespace(monotonic=lambda: next(clock)))
    model = get_model(
        "tabnet", {"n_d": 4, "n_a": 4, "n_steps": 3, "batch_size": 16, "epochs": 5}
    )
    model.set_context(FitContext(time_limit=10))
    model.fit(X, y, X, y)
    assert model.completed_epochs_ == 2
    assert model.training_info_["budget_truncated"] is True
    callback = next(
        c for c in model.model_._callback_container.callbacks if isinstance(c, EarlyStopping)
    )
    for key, value in model.model_.network.state_dict().items():
        np.testing.assert_array_equal(value.cpu(), callback.best_weights[key].cpu())


@pytest.mark.parametrize("features,heads,expected", [(1776, 8, 32), (1776, 4, 64), (50, 8, 256)])
def test_ft_attention_batch_cap(features, heads, expected):
    model = get_model("ft_transformer", {"batch_size": 256, "attention_n_heads": heads})
    assert model._batch_size(pd.DataFrame(columns=range(features))) == expected


def test_ft_attention_cap_applies_to_fit_validation_and_prediction(binary_task, monkeypatch):
    from tabular_thesis.models.dl import ft_transformer

    monkeypatch.setattr(ft_transformer, "ATTENTION_BUDGET_BYTES", 1024)
    X, y = binary_task
    model = get_model("ft_transformer", {**LOCAL_MODELS[1][1], "batch_size": 16, "epochs": 1})
    original = model._network
    observed = []

    def network_with_batch_check(*args):
        network = original(*args)

        def check_batch(module, inputs):
            observed.append(len(inputs[0]))
            assert len(inputs[0]) <= 8

        network.register_forward_pre_hook(check_batch)
        return network

    model._network = network_with_batch_check
    model.fit(X, y, X, y)
    assert model.predict_proba(X).shape == (32, 2)
    assert model.training_info_["effective_batch_size"] == 8
    assert model.params["batch_size"] == 16
    assert len(observed) == 12


@pytest.mark.parametrize("early_stopped,max_epochs", [(True, 5), (False, 1)])
def test_natural_stopping_does_not_mark_budget_truncation(monkeypatch, early_stopped, max_epochs):
    clock = iter([0, 6])
    monkeypatch.setattr(training, "time", SimpleNamespace(monotonic=lambda: next(clock)))
    budget = training.BudgetGuard(10, "mlp")
    assert not budget.finish_epoch(0, max_epochs, early_stopped=early_stopped)
    assert budget.completed_epochs == 1
