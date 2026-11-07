"""Scientific protocol and recovery regressions, using small offline tasks."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tabular_thesis.data.tasks import DatasetRepository, TaskSpec
from tabular_thesis.evaluation.metrics import compute_metrics
from tabular_thesis.experiments.artifacts import ArtifactStore
from tabular_thesis.experiments.configuration import (
    ConfigResolver,
    DatasetConfig,
    default_config_directory,
)
from tabular_thesis.experiments.runtime import FitContext, configure_caches
from tabular_thesis.models.preprocessing import TabularPreprocessor, column_roles
from tabular_thesis.models.registry import get_model


@pytest.fixture
def resolver():
    return ConfigResolver(default_config_directory())


@pytest.fixture
def task():
    rng = np.random.RandomState(17)
    X = pd.DataFrame(rng.normal(size=(180, 3)), columns=list("abc"))
    X["category"] = pd.Categorical(np.tile(["a", "b", "c"], 60))
    X["constant"] = 1.0
    X["constant_category"] = pd.Categorical(["constant"] * len(X))
    y = np.tile([0, 1, 2], 60)
    return TaskSpec("synthetic", X, y, ["0", "1", "2"], {"fixture": "three-classes-v1"})


def test_config_rejects_unknown_options_and_parameters(resolver):
    with pytest.raises(Exception, match="not in"):
        resolver.study(["runtime.gpu_typo=true"])
    with pytest.raises(ValueError, match="Unsupported parameters"):
        resolver.study(["model_overrides.mlp.dropuot=0.9"])
    with pytest.raises(ValueError, match="positive integer"):
        resolver.study(["model_overrides.mlp.epochs=0"])
    with pytest.raises(ValueError, match="outside"):
        resolver.study(["models=[mlp]", "model_overrides.tabnet.epochs=1"])


def test_checkpoint_follows_configured_data_directory(resolver, tmp_path):
    study = resolver.study([f"paths.data={tmp_path}"])
    params = resolver.resolved_models(study)["tabpfn"].params
    assert Path(params["checkpoint"]) == tmp_path / "weights" / "tabpfn-v3.5-20260909.safetensors"


def test_foundation_package_import_and_checkpoint_api_use_writable_caches(tmp_path):
    import subprocess
    import sys

    program = """
import inspect, sys
from pathlib import Path
from tabular_thesis.experiments.runtime import configure_caches
configure_caches(Path(sys.argv[1]))
from tabpfn import TabPFNClassifier
from tabular_thesis.experiments.tabpfn_model import TabPFN35Model
for parameter in ('model_path', 'device', 'n_jobs', 'categorical_features_indices',
                  'random_state', 'n_estimators', 'ignore_pretraining_limits'):
    assert parameter in inspect.signature(TabPFNClassifier).parameters
model = TabPFN35Model(path=str(Path(sys.argv[1]) / "model"), problem_type='binary')
checkpoint = model._get_model_checkpoint({'classification': '/tmp/pinned.safetensors'})
assert str(checkpoint) == '/tmp/pinned.safetensors'
"""
    subprocess.run([sys.executable, "-c", program, str(tmp_path)], check=True, timeout=60)


def test_plan_and_reproducible_random_search(resolver, tmp_path):
    configure_caches(tmp_path)
    from tabular_thesis.experiments.catalog import ModelCatalog
    from tabular_thesis.experiments.runner import planned_jobs

    assert planned_jobs(resolver.study())["configuration_jobs"] == 3132
    for name in ("mlp", "knn", "tabm", "modernnca", "realmlp"):
        config = resolver.model(name)
        _, a = ModelCatalog().configurations(config, 2, 42)
        _, b = ModelCatalog().configurations(config, 2, 42)
        _, c = ModelCatalog().configurations(config, 2, 43)
        assert a == b and a != c


def test_fold_local_preprocessing_preserves_empty_columns_and_unknowns():
    X = pd.DataFrame({"count": [1, 2, 3], "empty": [np.nan] * 3, "cat": ["a", None, "a"]})
    numeric, categorical = column_roles(X, None, None)
    assert "count" in numeric
    preprocessor = TabularPreprocessor(numeric, categorical).fit(X)
    statistics = preprocessor.num_imputer.statistics_.copy()
    held_out = pd.DataFrame({"count": [1000], "empty": [7.0], "cat": ["unseen"]})
    transformed = preprocessor.transform(held_out)
    assert transformed.shape == (1, 3) and transformed[0, -1] == 0
    np.testing.assert_array_equal(preprocessor.num_imputer.statistics_, statistics)
    assert np.isfinite(preprocessor.transform(X)).all()


@pytest.mark.parametrize(
    "probabilities", [np.array([[np.nan, np.nan]]), np.array([[1.1, -0.1]]), np.array([[0.2, 0.2]])]
)
def test_invalid_probabilities_cannot_be_successful(probabilities):
    with pytest.raises(ValueError):
        compute_metrics(np.array([0]), np.array([0]), probabilities)


def test_dataset_cache_provenance_and_corruption(tmp_path, monkeypatch):
    config = DatasetConfig(name="fixture", target="target", categorical=["category"])
    repository = DatasetRepository(tmp_path)
    raw = pd.DataFrame({"category": ["a", "b"] * 20, "value": range(40), "target": [0, 1] * 20})
    monkeypatch.setattr(repository, "_download", lambda _: (raw, []))
    first = repository.prepare(config)
    monkeypatch.setattr(
        repository, "_download", lambda _: pytest.fail("Cache hit downloaded again")
    )
    assert repository.prepare(config).source_identity == first.source_identity
    frozen = replace(config, expected_sha256=first.source_identity["data_sha256"])
    assert repository.load(frozen).source_identity == first.source_identity
    with pytest.raises(ValueError, match="differs"):
        repository.load(replace(config, version=2))
    (tmp_path / "prepared" / "fixture" / "data.parquet").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="corrupt"):
        repository.load(config)


def test_immutable_identity_and_completed_checksum(tmp_path):
    store = ArtifactStore(tmp_path)
    identity = {"seed": 42, "dropout": 0.1}
    job = store.run_directory("study", identity)
    assert job != store.run_directory("study", {**identity, "dropout": 0.2})
    store.initialize(job, identity)
    assert not store.complete(job)
    store.prediction_record(job, "result", {"model": "mlp"}, y=np.array([0, 1]))
    store.status(job, "completed")
    assert store.complete(job)
    (job / "predictions" / "result.npz").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="Corrupt"):
        store.complete(job)


def test_gpu_request_never_silently_falls_back(monkeypatch):
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA was requested"):
        FitContext(device="cuda").validate()


def test_neural_adapters_request_gpu_resources(tmp_path):
    configure_caches(tmp_path)
    from tabular_thesis.experiments.tabarena_models import LOCAL_ADAPTERS

    for name in ("mlp", "ft_transformer", "tabnet"):
        assert LOCAL_ADAPTERS[name].default_num_gpus == 1


def test_modernnca_quantile_seed_is_scoped_and_restored(tmp_path, monkeypatch):
    configure_caches(tmp_path)
    from tabarena.models.modernnca import model as upstream

    from tabular_thesis.experiments.upstream_models import FoldLocalCategories, StudyModernNCA

    original = upstream.RTDLQuantileTransformer
    model = StudyModernNCA(path=str(tmp_path / "model"), problem_type="binary")
    model.random_seed = 42

    def fail(*args, **kwargs):
        assert upstream.RTDLQuantileTransformer().random_state == 42
        raise RuntimeError("deliberate failure")

    monkeypatch.setattr(FoldLocalCategories, "_fit", fail)
    with pytest.raises(RuntimeError, match="deliberate"):
        model._fit(None, None)
    assert upstream.RTDLQuantileTransformer is original


def test_knn_nominal_distances_ignore_category_spelling():
    categories = [f"category_{i}" for i in range(30)]
    X = pd.DataFrame({"category": pd.Categorical(categories * 4)})
    y = np.tile(np.arange(30) % 2, 4)
    held_out = pd.DataFrame({"category": pd.Categorical(["unseen", *categories[:4]])})
    first = get_model("knn", {"n_neighbors": 3})
    first.fit(X, y)
    mapping = {value: f"renamed_{(i * 11) % 30}" for i, value in enumerate(categories)}
    second = get_model("knn", {"n_neighbors": 3})
    second.fit(pd.DataFrame({"category": X.category.map(mapping)}), y)
    renamed = pd.DataFrame(
        {"category": held_out.category.astype(object).map(lambda x: mapping.get(x, x))}
    )
    np.testing.assert_allclose(first.predict_proba(held_out), second.predict_proba(renamed))


def test_knn_seed_controls_quantile_subsampling_and_records_neighbor_cap():
    X = pd.DataFrame(np.random.default_rng(1).normal(size=(12000, 3)), columns=list("abc"))
    y = (X.a > 0).astype(int).to_numpy()
    parameters = {"random_state": 123, "scaler": "quantile", "n_neighbors": 20000}
    first = get_model("knn", parameters)
    first.fit(X, y)
    np.random.seed(999)
    second = get_model("knn", parameters)
    second.fit(X, y)
    assert first.model_.random_seed == second.model_.random_seed == 123
    assert first.model_.params_trained["n_neighbors"] == len(X) - 1
    np.testing.assert_array_equal(
        first.model_.knn_preprocessor.final_scaler_.quantiles_,
        second.model_.knn_preprocessor.final_scaler_.quantiles_,
    )


@pytest.mark.parametrize(
    "name,params",
    [
        ("mlp", {"epochs": 1, "batch_size": 64, "hidden_dims": [16]}),
        ("tabnet", {"epochs": 1, "batch_size": 64, "n_d": 8, "n_a": 8}),
    ],
)
def test_singleton_final_batch_keeps_all_observations(name, params):
    X = pd.DataFrame(np.random.RandomState(4).normal(size=(65, 3)), columns=list("abc"))
    y = np.tile([0, 1], 33)[:65]
    model = get_model(name, params)
    model.fit(X, y, X.iloc[:10], y[:10])
    assert model.predict_proba(X).shape == (65, 2)


@pytest.mark.parametrize(
    "name,parameters",
    [
        ("mlp", {"hidden_dims": [16]}),
        (
            "ft_transformer",
            {"n_blocks": 1, "d_token": 16, "attention_n_heads": 4, "ffn_d_hidden": 32},
        ),
    ],
)
def test_neural_validation_and_inference_respect_fit_batch_memory_ceiling(name, parameters):
    model = get_model(name, {**parameters, "epochs": 1, "batch_size": 16})
    original = model._network

    def bounded_network(*args):
        network = original(*args)

        def check_batch(module, inputs):
            assert len(inputs[0]) <= 16, "Validation/inference exceeded the fit memory ceiling"

        network.register_forward_pre_hook(check_batch)
        return network

    model._network = bounded_network
    X = pd.DataFrame(np.random.default_rng(1).normal(size=(65, 3)), columns=list("abc"))
    y = np.tile([0, 1], 33)[:65]
    model.fit(X, y, X, y)
    assert model.predict_proba(X).shape == (65, 2)


def test_ft_transformer_distinguishes_numerical_feature_identity():
    import torch

    from tabular_thesis.models.dl.ft_transformer import _FTTransformerNetwork

    network = _FTTransformerNetwork(
        3,
        0,
        [],
        2,
        n_blocks=1,
        d_token=16,
        attention_n_heads=4,
        attention_dropout=0.0,
        ffn_d_hidden=16,
        ffn_dropout=0.0,
        residual_dropout=0.0,
    ).eval()
    with torch.no_grad():
        empty = torch.empty((2, 0), dtype=torch.long)
        X = torch.tensor([[1.0, 2.0, 7.0], [5.0, 0.0, 3.0]])
        assert not torch.allclose(network(X, empty), network(X[:, [2, 0, 1]], empty))


@pytest.mark.parametrize(
    "name,parameters",
    [
        ("mlp", {"epochs": 1, "hidden_dims": [16]}),
        (
            "ft_transformer",
            {"epochs": 1, "n_blocks": 1, "d_token": 16, "attention_n_heads": 4, "ffn_d_hidden": 32},
        ),
        ("tabnet", {"epochs": 1, "n_d": 8, "n_a": 8}),
        ("xgboost", {"n_estimators": 3}),
        ("catboost", {"iterations": 3}),
        ("logistic_regression", {}),
        ("knn", {}),
    ],
)
def test_local_adapters_multiclass_with_categorical_signal(task, name, parameters):
    model = get_model(name, parameters)
    model.fit(task.X.iloc[:120], task.y[:120], task.X.iloc[120:], task.y[120:])
    probabilities = model.predict_proba(task.X.iloc[120:])
    metrics = compute_metrics(task.y[120:], probabilities.argmax(axis=1), probabilities, 3)
    assert probabilities.shape == (60, 3)
    assert np.isfinite(metrics.log_loss_val)
    assert len(metrics.per_class) == 3


def test_actual_eight_fold_run_resume_and_report(resolver, task, tmp_path, monkeypatch):
    configure_caches(tmp_path / "cache")
    from tabular_thesis.experiments.backend import TabArenaBackend
    from tabular_thesis.experiments.reporting import report
    from tabular_thesis.experiments.runner import ExperimentRunner

    study = resolver.study(
        [
            "models=[logistic_regression]",
            "runtime.device=cpu",
            "runtime.num_cpus=1",
            "random_configurations=1",
        ]
    )
    study.datasets = [task.name]
    study.paths.data, study.paths.results = str(tmp_path / "data"), str(tmp_path / "results")
    runner = ExperimentRunner(resolver, study)
    directory = runner.run(tasks=[task])
    records = list(directory.glob("jobs/*/records/result.json"))
    assert len(records) == 2
    for path in records:
        record = json.loads(path.read_text())
        assert record["model_info"]["bagged_info"]["num_child_models"] == 8
        assert record["metric"] == "log_loss"
    frame = report(directory)
    assert set(frame.condition) == {"default", "tuned", "tuned_ensemble"}
    monkeypatch.setattr(
        TabArenaBackend, "execute", lambda *a, **k: pytest.fail("Completed job was retrained")
    )
    assert runner.run(tasks=[task]) == directory
    assert len(report(directory)) == 3


def test_failed_job_cannot_be_resumed_as_complete(resolver, task, tmp_path, monkeypatch):
    configure_caches(tmp_path / "cache")
    from tabular_thesis.experiments.backend import TabArenaBackend
    from tabular_thesis.experiments.runner import ExperimentRunner

    study = resolver.study(
        ["models=[logistic_regression]", "runtime.device=cpu", "random_configurations=0"]
    )
    study.datasets = [task.name]
    study.paths.results = str(tmp_path / "results")

    def fail(*args, **kwargs):
        work = Path(args[-1])
        work.mkdir(parents=True)
        (work / "partial-model").write_text("temporary fit state")
        raise RuntimeError("Deliberate model failure")

    monkeypatch.setattr(TabArenaBackend, "execute", fail)
    with pytest.raises(RuntimeError, match="Deliberate"):
        ExperimentRunner(resolver, study).run(tasks=[task])
    status = json.loads(
        next(Path(study.paths.results).glob("runs/*/*/jobs/*/status.json")).read_text()
    )
    assert status["state"] == "failed"
    assert not list(Path(study.paths.results).glob("runs/*/*/jobs/*/work"))


def test_failed_random_configuration_does_not_replace_successful_default(
    resolver, task, tmp_path, monkeypatch
):
    configure_caches(tmp_path / "cache")
    from tabular_thesis.experiments.backend import TabArenaBackend
    from tabular_thesis.experiments.reporting import report
    from tabular_thesis.experiments.runner import ExperimentRunner

    study = resolver.study(
        [
            "models=[logistic_regression]",
            "runtime.device=cpu",
            "runtime.num_cpus=1",
            "random_configurations=1",
        ]
    )
    study.datasets = [task.name]
    study.paths.results = str(tmp_path / "results")
    original = TabArenaBackend.execute

    def execute(self, task, name, model, parameters, directory):
        if parameters["C"] != 1.0:
            raise RuntimeError("Deliberate search failure")
        return original(self, task, name, model, parameters, directory)

    monkeypatch.setattr(TabArenaBackend, "execute", execute)
    with pytest.raises(RuntimeError, match="Deliberate search failure"):
        ExperimentRunner(resolver, study).run(tasks=[task])
    directory = next(Path(study.paths.results).glob("runs/*/*"))
    frame = report(directory)
    assert (frame.successful_configurations == 1).all()
    assert (frame.failed_configurations == 1).all()
    assert all(weights == [1.0] for weights in frame.weights)


@pytest.mark.parametrize("model", ["logistic_regression", "knn"])
def test_patient_groups_stay_disjoint_in_outer_and_inner(resolver, task, tmp_path, model):
    configure_caches(tmp_path / "cache")
    from tabular_thesis.experiments.runner import ExperimentRunner

    task.groups = np.repeat(np.arange(60), 3).astype(str)
    study = resolver.study(
        [
            f"models=[{model}]",
            "runtime.device=cpu",
            "runtime.num_cpus=1",
            "random_configurations=0",
        ]
    )
    study.datasets = [task.name]
    study.paths.results = str(tmp_path / "results")
    train, test = task.outer_split(study.outer_seed, study.inner_folds)
    assert not set(task.groups[train]) & set(task.groups[test])
    directory = ExperimentRunner(resolver, study).run(tasks=[task])
    path = next(directory.glob("jobs/*/predictions/*.npz"))
    with np.load(path) as arrays:
        for fold in range(8):
            validation = arrays[f"inner_validation_{fold}"]
            training = np.setdiff1d(arrays["oof_indices"], validation)
            assert not set(task.groups[training]) & set(task.groups[validation])


@pytest.mark.parametrize(
    "name,params",
    [
        ("realmlp", {"n_epochs": 1, "hidden_sizes": [16]}),
        (
            "modernnca",
            {"n_epochs": 1, "dim": 8, "num_emb_n_frequencies": 4, "num_emb_d_embedding": 4},
        ),
        ("tabm", {"n_epochs": 1, "d_block": 16, "n_blocks": 1, "tabm_k": 2}),
        ("knn", {}),
    ],
)
def test_native_methods_fit_multiclass_with_unseen_category(resolver, task, tmp_path, name, params):
    configure_caches(tmp_path / "cache")
    from tabular_thesis.experiments.backend import TabArenaBackend
    from tabular_thesis.experiments.catalog import ModelCatalog

    config = resolver.model(name)
    config.params.update(params)
    # Native TabM estimates at least ~5 GB even for this one-epoch, 60-row tiny model.
    # Disable that heuristic only in this compatibility test; production keeps its guard.
    if name == "tabm":
        config.params["ag_args_fit"] = {"max_memory_usage_ratio": None}
    model, configurations = ModelCatalog().configurations(config, 0, 42)
    study = resolver.study(
        [f"models=[{name}]", "runtime.device=cpu", "runtime.num_cpus=1", "inner_folds=2"]
    )
    _, test = task.outer_split(study.outer_seed, study.inner_folds)
    task.X["category"] = task.X["category"].cat.add_categories(["held-out"])
    task.X.loc[test[0], "category"] = "held-out"
    # Reduced folds/epochs are a package compatibility check, not a thesis result.
    out = TabArenaBackend(study).execute(task, name, model, configurations[0], tmp_path / "work")
    assert np.isfinite(out["metric_error"])
