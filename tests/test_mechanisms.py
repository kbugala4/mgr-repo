"""Offline qualification for paired interventions and frozen analysis edge cases."""

import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import log_loss

from tabular_thesis.experiments.artifacts import canonical
from tabular_thesis.experiments.configuration import ConfigResolver, default_config_directory
from tabular_thesis.experiments.mechanisms import balanced_class_weights, nested_subset
from tabular_thesis.experiments.runtime import FitContext, configure_caches
from tabular_thesis.models.registry import get_model
from tabular_thesis.models.specification import model_spec


@pytest.fixture(scope="module")
def analysis():
    scripts = str(Path(__file__).resolve().parents[1] / "scripts")
    sys.path.insert(0, scripts)
    yield importlib.import_module("mechanism_analysis")
    sys.path.remove(scripts)


@pytest.fixture
def binary():
    X = pd.DataFrame(np.random.default_rng(123).normal(size=(96, 3)), columns=list("abc"))
    y = np.tile([0, 0, 0, 1], 24)
    return X, y


def test_weights_and_nested_subsets(binary):
    X, y = binary
    assert balanced_class_weights(y, 2) == (2 / 3, 2.0)
    with pytest.raises(ValueError):
        balanced_class_weights(y, 3)
    subsets = [nested_subset(y, f, 10) for f in [0.25, 0.5, 1.0]]
    assert set(subsets[0]) < set(subsets[1]) < set(subsets[2])
    np.testing.assert_array_equal(subsets[2], np.arange(len(y)))
    np.testing.assert_array_equal(subsets[0], nested_subset(y, 0.25, 10))
    assert not np.array_equal(subsets[0], nested_subset(y, 0.25, 20))
    assert np.bincount(y[subsets[0]]).tolist() == [18, 6]
    with pytest.raises(ValueError, match="lose a class"):
        nested_subset(np.array([0, 0, 1]), 0.25, 0)


@pytest.mark.parametrize(
    "name,params",
    [
        ("tabm", {"_record_history": True}),
        ("catboost", {"_train_fraction": 0.5}),
        ("mlp", {"_train_fraction": float("nan")}),
        ("mlp", {"_train_fraction": True}),
        ("mlp", {"_class_weighting": "automatic"}),
        ("mlp", {"_record_history": 1}),
        ("mlp", {"_selection_loss": "unweighted"}),
    ],
)
def test_reject_invalid_interventions(name, params):
    with pytest.raises(ValueError):
        model_spec(name).validate_parameters(params)


def test_training_seed_default_snapshot():
    resolver = ConfigResolver(default_config_directory())
    assert "training_seed" not in resolver.snapshot(resolver.study())["study"]
    assert resolver.snapshot(resolver.study(["training_seed=0"]))["study"]["training_seed"] == 0
    with pytest.raises(ValueError):
        resolver.study(["training_seed=-1"])
    with pytest.raises(Exception):
        resolver.study(["training_seed=1.5"])


@pytest.mark.parametrize("weighted,control", [(False, False), (True, False), (True, True)])
def test_mlp_history_selection_and_losses(binary, weighted, control):
    import torch

    X, y = binary
    weights = balanced_class_weights(y[:64], 2) if weighted else None
    model = get_model("mlp", {"epochs": 5, "patience": 5, "batch_size": 16, "hidden_dims": [8]})
    model.set_context(
        FitContext(
            seed=10,
            class_weights=weights,
            record_history=True,
            weighted_validation=weighted and not control,
            unweighted_selection=control,
        )
    )
    model.fit(X.iloc[:64], y[:64], X.iloc[64:], y[64:])
    info = model.training_info_
    history = info["history"]
    criterion = "val_loss_weighted" if weighted and not control else "val_loss"
    assert len(history) == 5 and info["optimizer_steps"] == 20
    assert info["best_epoch"] == np.argmin([h[criterion] for h in history]) + 1
    best = history[info["best_epoch"] - 1]
    p = model.predict_proba(X.iloc[64:])
    assert log_loss(y[64:], p) == pytest.approx(best["val_loss"], abs=1e-6)
    if weighted:
        assert log_loss(y[64:], p, sample_weight=np.asarray(weights)[y[64:]]) == pytest.approx(
            best["val_loss_weighted"], abs=1e-6
        )
    norm = torch.sqrt(sum(p.pow(2).sum() for p in model.network_.parameters())).item()
    assert norm == pytest.approx(best["param_norm"])
    assert log_loss(y[:64], model.predict_proba(X.iloc[:64])) == pytest.approx(
        best["train_loss"], abs=1e-6
    )
    canonical(info)


def test_history_does_not_change_mlp_or_catboost_predictions(binary):
    X, y = binary
    for name, params in [
        ("mlp", {"epochs": 4, "hidden_dims": [8]}),
        ("catboost", {"iterations": 15}),
    ]:
        predictions = []
        for record in [False, True]:
            model = get_model(name, params)
            model.set_context(FitContext(class_weights=(2 / 3, 2.0), record_history=record))
            model.fit(X.iloc[:64], y[:64], X.iloc[64:], y[64:])
            predictions.append(model.predict_proba(X.iloc[64:]))
            if not record:
                assert "history" not in getattr(model, "training_info_", {})
            elif name == "catboost":
                assert "Logloss:use_weights=false" in model.training_info_["validation_curves"]
        np.testing.assert_array_equal(*predictions)


def test_catboost_weighted_and_control_selection(binary):
    X, y = binary
    for control in [False, True]:
        model = get_model("catboost", {"iterations": 20})
        model.set_context(
            FitContext(
                class_weights=(2 / 3, 2.0),
                record_history=True,
                weighted_validation=not control,
                unweighted_selection=control,
            )
        )
        model.fit(X.iloc[:64], y[:64], X.iloc[64:], y[64:])
        curve = model.training_info_["validation_curves"][
            "Logloss:use_weights=false" if control else "Logloss"
        ]
        best = model.model_.get_best_iteration()
        assert best == int(np.argmin(curve))
        assert curve[best] == pytest.approx(
            log_loss(
                y[64:],
                model.predict_proba(X.iloc[64:]),
                sample_weight=None if control else np.array([2 / 3, 2.0])[y[64:]],
            ),
            abs=1e-7,
        )


def test_nonfinite_history_is_json_null(binary, monkeypatch):
    X, y = binary
    model = get_model("mlp", {"epochs": 1, "hidden_dims": [8]})
    model.set_context(FitContext(record_history=True))
    monkeypatch.setattr(model, "_evaluation_loss", lambda *args: float("inf"))
    model.fit(X.iloc[:64], y[:64], X.iloc[64:], y[64:])
    assert model.training_info_["history"][0]["train_loss"] is None
    canonical(model.training_info_)


def test_threshold_family_boundaries_and_ties(analysis):
    assert analysis.select_thresholds([1, 0], [0.1, 0.2])["f1"] == 0.1
    for value in [0, 0.5, 1]:
        thresholds = analysis.select_thresholds([0, 1], [value, value])
        assert thresholds["f1"] == value
        assert thresholds["fpr01"] > value
    # F1 tie: 2/3 at one TP and at two TP/two FP; higher midpoint wins.
    assert analysis.select_thresholds([1, 0, 0, 1], [0.9, 0.7, 0.5, 0.3])["f1"] == 0.8
    metric = analysis.threshold_metrics([1, 0], [0, 0], 1)
    assert metric["precision"] is None and metric["f1"] == 0 and metric["recall"] == 0
    a = np.nextafter(0.5, 1)
    t, tp, fp, _, _ = analysis.operating_points([1, 0], [a, 0.5])
    assert t[1] > 0.5 and tp[1] == 1 and fp[1] == 0


def test_fpr_is_discrete_and_tie_safe(analysis):
    y = np.r_[np.zeros(100), 1]
    p = np.r_[np.zeros(98), 0.8, 0.8, 0.9]
    t = analysis.select_thresholds(y, p)["fpr01"]
    assert t == pytest.approx(0.85)
    assert analysis.threshold_metrics(y, p, t)["fp"] == 0


def test_comparator_boundaries_and_cross_class_ranks(analysis):
    rng = np.random.default_rng(10)
    for _ in range(30):
        y = np.tile([0, 1], 7)
        a, b = rng.integers(0, 5, size=(2, len(y))) / 4
        expected = []
        for i in np.flatnonzero(y):
            for j in np.flatnonzero(1 - y):
                s1, s2 = np.sign(a[i] - a[j]), np.sign(b[i] - b[j])
                expected.append(abs(s1 - s2) / 2)
        assert analysis.rank_changes(y, a, b)["discordance"] == pytest.approx(np.mean(expected))
    p = np.array([0.0, 0.1, 0.4, 0.8, 1.0])
    q = analysis.shifted_probability(p, 16.3)
    assert q[0] == 0 and q[-1] == 1 and np.all(np.diff(q) > 0)
    assert analysis.rank_changes([0, 1], [0.5, 0.5], [0.5, 0.5])["spearman"] is None
    assert analysis.finite_logit(0)["value"] is None
    assert analysis.finite_logit(1)["boundary"] == "upper"
    assert analysis.logit_histogram(p)["zero"] == 1


def test_bootstrap_pairing_missing_replicates_and_determinism(analysis):
    y = np.tile([0, 1], 10)
    p = np.arange(20) / 20
    left, right = [p] * 3, [p] * 3

    def metric(y, p, i):
        return float(np.mean(p[i]))

    a = analysis.paired_bootstrap(y, left, right, metric, 30)
    assert a == analysis.paired_bootstrap(y, left, right, metric, 30)
    assert a["interval"] == [0, 0]
    assert (
        analysis.paired_bootstrap(y, left, right, lambda *args: None, 5)["undefined_replicates"]
        == 5
    )
    with pytest.raises(ValueError, match="three seeds"):
        analysis.paired_bootstrap(y, left[:2], right, metric, 5)


def test_control_complete_denominator_ties_and_censoring(analysis):
    control = importlib.import_module("analyze_study_a").control_rule
    child = {"best_epoch": 1, "history": [{"val_loss": 1.0}, {"val_loss": 2.0}]}
    assert control([child] * 23)["trigger"] is None
    assert control([child] * 24)["trigger"] is False
    censored = {"best_epoch": 1, "history": [{"val_loss": 2.0}, {"val_loss": 1.0}]}
    assert control([child] * 12 + [censored] * 12)["trigger"] is True


def test_plan_counts_and_german_1002(analysis):
    module = importlib.import_module("run_mechanism_studies")
    root = default_config_directory() / "studies"
    a = module.expand(module.load_plan(root / "study-a.yaml"))
    b = module.expand(module.load_plan(root / "study-b.yaml"))
    assert len(a) == 36 and sum(bool(r["blocked"]) for r in a) == 12
    assert len(b) == 108
    y = np.r_[np.zeros(408, dtype=int), np.ones(175, dtype=int)]
    folds = [(np.arange(len(y)), np.array([], dtype=int))] * 8
    evidence = module.budget_from_folds(y, folds, 0.25, 0, 64, 1000)
    assert {r["optimizer_steps"] for r in evidence} == {1002}
    assert {r["epochs"] for r in evidence} == {334}


def test_eight_fold_seed_subset_history_resume_and_deferred_test(tmp_path, monkeypatch, analysis):
    from tabular_thesis.data.tasks import TaskSpec
    from tabular_thesis.experiments.runner import ExperimentRunner

    configure_caches(tmp_path / "cache")
    from tabular_thesis.experiments.backend import DeferredTestRunner, TabArenaBackend

    # Any accidental outer-test metric computation is a regression.
    monkeypatch.setattr(DeferredTestRunner, "evaluate", lambda *args, **kwargs: None)
    resolver = ConfigResolver(default_config_directory())
    X = pd.DataFrame(np.random.default_rng(2).normal(size=(180, 3)), columns=list("abc"))
    task = TaskSpec("synthetic", X, np.tile([0, 0, 1], 60), ["0", "1"], {"fixture": "ab-v2"})
    validation_sets, subsets = [], []
    for seed, fraction in [(0, 0.5), (10, 0.5), (0, 1.0)]:
        study = resolver.study(
            [
                "models=[mlp,catboost]",
                "runtime.device=cpu",
                "runtime.num_cpus=1",
                "random_configurations=0",
                f"training_seed={seed}",
                "model_overrides.mlp.epochs=2",
                "model_overrides.mlp.patience=2",
                "model_overrides.mlp.hidden_dims=[8]",
                "model_overrides.mlp._record_history=true",
                f"model_overrides.mlp._train_fraction={fraction}",
                "model_overrides.catboost.iterations=3",
                "model_overrides.catboost._record_history=true",
                "model_overrides.catboost._class_weighting=balanced",
                f"paths.data={tmp_path / 'data'}",
                f"paths.results={tmp_path / 'results'}",
            ]
        )
        study.datasets = [task.name]
        runner = ExperimentRunner(resolver, study)
        directory = runner.run([task])
        assert not list(directory.glob("jobs/*/work"))
        assert not (directory / "report.json").exists()
        record = json.loads((directory / "jobs/synthetic-mlp-000/records/result.json").read_text())
        assert record["metric_error"] is None
        children = list(record["model_info"]["children_info"].values())
        info = [c["hyperparameters_fit"] for c in children]
        assert [c["random_state"] for c in info] == list(range(seed, seed + 8))
        assert all(len(c["history"]) == 2 for c in info)
        subsets.append([c["training_row_ids"] for c in info])
        with np.load(directory / "jobs/synthetic-mlp-000/predictions/oof.npz") as arrays:
            assert not any(k.startswith("test_") for k in arrays)
            folds = [arrays[f"inner_validation_{i}"].copy() for i in range(8)]
            validation_sets.append(folds)
            from autogluon.common.utils.cv_splitter import CVSplitter

            outer, _ = task.outer_split(42, 8)
            expected = CVSplitter(n_splits=8, n_repeats=1, random_state=1, stratify=True).split(
                X=task.X.iloc[outer], y=task.y[outer]
            )
            for (_, val), actual in zip(expected, folds, strict=True):
                np.testing.assert_array_equal(outer[val], actual)
            for i, c in enumerate(info):
                assert not set(c["training_row_ids"]) & set(folds[i])
        with monkeypatch.context() as m:
            m.setattr(TabArenaBackend, "execute", lambda *a, **k: pytest.fail("Retrained"))
            assert runner.run([task]) == directory
    for left, right in zip(validation_sets[0], validation_sets[1], strict=True):
        np.testing.assert_array_equal(left, right)
    assert any(a != b for a, b in zip(subsets[0], subsets[1], strict=True))
    assert all(set(a) < set(b) for a, b in zip(subsets[0], subsets[2], strict=True))


def test_adapter_subset_precedes_scaling_and_weights(binary, tmp_path):
    configure_caches(tmp_path / "cache")
    from tabular_thesis.experiments.tabarena_models import MLPAdapter

    X, y = binary
    X = X.copy()
    selected = nested_subset(y[:64], 0.5, 10)
    X["subset_constant"] = 100.0
    X.loc[selected, "subset_constant"] = 0.0
    adapter = MLPAdapter(
        path=str(tmp_path / "model"),
        problem_type="binary",
        hyperparameters={
            "epochs": 1,
            "hidden_dims": [8],
            "_train_fraction": 0.5,
            "_class_weighting": "balanced",
            "_record_history": True,
        },
    )
    adapter.initialize(X=X.iloc[:64], y=y[:64])
    adapter.random_seed = 10
    adapter._fit(X.iloc[:64], y[:64], X.iloc[64:], y[64:])
    selected = nested_subset(y[:64], 0.5, 10)
    np.testing.assert_allclose(
        adapter.model.preprocessor_.scaler.mean_, X.iloc[selected][list("abc")].mean()
    )
    assert "subset_constant" not in adapter.model.numerical_columns
    assert adapter.params_trained["class_weights"] == balanced_class_weights(y[selected], 2)
    assert adapter.params_trained["training_row_ids"] == selected.tolist()


def test_history_survives_failure_after_fit(tmp_path, monkeypatch):
    from tabular_thesis.data.tasks import TaskSpec
    from tabular_thesis.experiments.runner import ExperimentRunner

    configure_caches(tmp_path / "cache")
    from tabular_thesis.experiments.backend import DeferredTestRunner

    def fail(self, out):
        raise ValueError("Synthetic post-fit failure")

    monkeypatch.setattr(DeferredTestRunner, "post_fit", fail)
    resolver = ConfigResolver(default_config_directory())
    study = resolver.study(
        [
            "models=[mlp]",
            "runtime.device=cpu",
            "runtime.num_cpus=1",
            "random_configurations=0",
            "model_overrides.mlp.epochs=1",
            "model_overrides.mlp.hidden_dims=[8]",
            "model_overrides.mlp._record_history=true",
            f"paths.results={tmp_path / 'results'}",
            f"paths.data={tmp_path / 'data'}",
        ]
    )
    study.datasets = ["synthetic"]
    task = TaskSpec(
        "synthetic",
        pd.DataFrame({"a": np.arange(96, dtype=float)}),
        np.tile([0, 1], 48),
        ["0", "1"],
        {"fixture": "post-fit-failure"},
    )
    with pytest.raises(RuntimeError, match="Synthetic post-fit failure"):
        ExperimentRunner(resolver, study).run([task])
    job = next((tmp_path / "results").glob("runs/*/*/jobs/*"))
    records = list((job / "records").glob("fold-*.json"))
    assert len(records) == 8 and not (job / "work").exists()
    assert all(len(json.loads(p.read_text())["history"]) == 1 for p in records)


def synthetic_registry(tmp_path):
    from tabular_thesis.experiments.artifacts import ArtifactStore

    store = ArtifactStore(tmp_path)
    rows = {}
    y = np.tile([0, 0, 1], 8)
    for variant in ["unweighted", "weighted"]:
        for seed in [0, 10, 20]:
            directory = tmp_path / f"{variant}-{seed}"
            job = directory / "jobs/amazon-mlp-000"
            params = {"epochs": 2}
            children = {
                f"S1F{i + 1}": {
                    "hyperparameters_fit": {
                        "random_state": seed + i,
                        "budget_truncated": False,
                        "best_epoch": 1,
                        "history": [{"val_loss": 0.5}, {"val_loss": 0.6}],
                    }
                }
                for i in range(8)
            }
            p = np.linspace(0.05, 0.95, len(y))
            arrays = {
                "oof_y": y,
                "oof_indices": np.arange(len(y)),
                "oof_probabilities": np.column_stack((1 - p, p)),
            }
            record = {
                "parameters": params,
                "classes": ["0", "1"],
                "model_info": {"children_info": children},
            }
            store.prediction_record(job, "oof", {}, **arrays)
            store.prediction_record(
                job,
                "result",
                record,
                **arrays,
                test_y=y,
                test_indices=np.arange(len(y), 2 * len(y)),
                test_probabilities=np.column_stack((1 - p, p)),
            )
            store.status(job, "completed")
            key = f"{variant}-{seed}"
            rows[key] = dict(
                dataset="amazon",
                model="mlp",
                variant=variant,
                seed=seed,
                params=params,
                directory=str(directory),
                state="completed",
                rare_code=0,
            )
    return {"conditions": rows}


def test_control_never_opens_predictions_and_diagnostics_only_oof(tmp_path, monkeypatch, analysis):
    module = importlib.import_module("analyze_study_a")
    registry = synthetic_registry(tmp_path)
    original = Path.open

    def no_predictions(path, *args, **kwargs):
        assert "predictions" not in path.parts
        return original(path, *args, **kwargs)

    with monkeypatch.context() as m:
        m.setattr(Path, "open", no_predictions)
        assert module.control_check(registry)["amazon/mlp"]["trigger"] is False
    original_load = np.load

    def only_oof(path, *args, **kwargs):
        assert Path(path).name == "oof.npz"
        return original_load(path, *args, **kwargs)

    monkeypatch.setattr(np, "load", only_oof)
    diagnostics = module.diagnostics(registry)
    assert diagnostics["amazon/mlp"]["complete"] is True
    assert diagnostics["amazon/mlp"]["ap_range"] == {"weighted": 0, "unweighted": 0}


def test_synthetic_a_analysis_determinism_and_missing_seeds(tmp_path, monkeypatch, analysis):
    module = importlib.import_module("analyze_study_a")
    registry = synthetic_registry(tmp_path)
    original = analysis.paired_metric_bootstrap
    monkeypatch.setattr(
        module,
        "paired_metric_bootstrap",
        lambda y, a, b, keys: original(y, a, b, keys, replicates=8),
    )
    decisions = module.control_check(registry)
    first = module.test_analysis(registry, decisions)
    assert canonical(first) == canonical(module.test_analysis(registry, decisions))
    assert len(first["conditions"]) == 9
    assert any(r["variant"] == "comparator" for r in first["conditions"])
    registry["conditions"].pop("weighted-20")
    missing = module.test_analysis(registry, decisions)
    assert not missing["conditions"] and missing["missing"]


def test_b_plateau_zero_and_relative_change(analysis):
    plateau = importlib.import_module("analyze_study_b").plateau
    history = [{"optimizer_steps": i * 10, "train_loss": 0.5} for i in range(1, 11)]
    assert plateau(history)["plateau"] is True
    history[-1]["train_loss"] = 0.51
    assert plateau(history)["plateau"] is False
    for row in history:
        row["train_loss"] = 0.0
    assert plateau(history)["relative_change"] == 0
    history[-1]["train_loss"] = 1.0
    assert plateau(history)["plateau"] is False


def test_freeze_rejects_unreviewed_or_changed_analysis(tmp_path, analysis):
    from tabular_thesis.experiments.artifacts import file_sha256

    script = tmp_path / "script.py"
    script.write_text("pass\n")
    registry = tmp_path / "registry.json"
    registry.write_text("{}")
    approval = {
        "analysis_version": analysis.ANALYSIS_VERSION,
        "reviewer": "synthetic-test",
        "frozen_at": "synthetic",
        "prior_test_access": False,
        "script_sha256": {script.name: file_sha256(script)},
        "registry_sha256": file_sha256(registry),
    }
    path = tmp_path / "freeze.json"
    path.write_text(json.dumps(approval))
    assert analysis.authorize_test(path, registry, [script]) == approval
    script.write_text("# changed\n")
    with pytest.raises(ValueError, match="changed"):
        analysis.authorize_test(path, registry, [script])


def test_synthetic_b_history_and_paired_analysis(tmp_path, monkeypatch, analysis):
    from tabular_thesis.experiments.artifacts import ArtifactStore

    module = importlib.import_module("analyze_study_b")
    store = ArtifactStore(tmp_path)
    rows = {}
    for variant in ["full", "no_dropout", "no_weight_decay", "no_both"]:
        for seed in [0, 10, 20]:
            directory = tmp_path / f"{variant}-{seed}"
            job = directory / "jobs/ionosphere-mlp-000"
            params = {"epochs": 2, "batch_size": 2}
            children = {
                f"S1F{i + 1}": {
                    "hyperparameters_fit": {
                        "random_state": seed + i,
                        "budget_truncated": False,
                        "best_epoch": 2,
                        "training_row_ids": [0, 1, 2, 3],
                        "validation_row_ids": [4, 5],
                        "training_rows": 4,
                        "steps_per_epoch": 2,
                        "stop_reason": "epochs",
                        "learning_rate": 0.001,
                        "weight_decay": 0.01,
                        "history": [
                            {
                                "epoch": 1,
                                "optimizer_steps": 2,
                                "train_loss": 0.8,
                                "val_loss": 0.7,
                                "param_norm": 1.1,
                            },
                            {
                                "epoch": 2,
                                "optimizer_steps": 4,
                                "train_loss": 0.4,
                                "val_loss": 0.5,
                                "param_norm": 1.0,
                            },
                        ],
                    }
                }
                for i in range(8)
            }
            y = np.tile([0, 1], 8)
            p = np.linspace(0.1, 0.9, len(y))
            store.prediction_record(
                job,
                "result",
                {
                    "parameters": params,
                    "classes": ["0", "1"],
                    "model_info": {"children_info": children},
                },
                test_y=y,
                test_indices=np.arange(len(y)),
                test_probabilities=np.column_stack((1 - p, p)),
            )
            store.status(job, "completed")
            key = f"{variant}-{seed}"
            rows[key] = dict(
                dataset="ionosphere",
                model="mlp",
                variant=variant,
                seed=seed,
                fraction=1.0,
                params=params,
                target_steps=4,
                directory=str(directory),
                state="completed",
            )
    registry = {"conditions": rows}
    result = module.history_analysis(registry)
    assert not result["missing"] and len(result["folds"]) == 96
    assert len(result["norm_ratios"]) == 96 and len(result["curves"]) == 24
    assert result["summaries"][0]["final_gap"]["median"] == pytest.approx(0.1)
    original = analysis.paired_bootstrap
    monkeypatch.setattr(
        module, "paired_bootstrap", lambda y, a, b, metric: original(y, a, b, metric, replicates=8)
    )
    result = module.test_analysis(registry)
    assert len(result["metrics"]) == 12 and len(result["contrasts"]) == 6
    assert all(row["interval"] == [0, 0] for row in result["contrasts"])


def test_source_ranking_reconstruction_without_test(tmp_path, monkeypatch, analysis):
    module = importlib.import_module("select_mechanism_sources")

    def candidates(path, model, allow_failed=False):
        assert model == "mlp"
        rows = [
            dict(
                index=i,
                error=float(i != 21),
                params={
                    "dropout": 0.3,
                    "weight_decay": 0.002,
                },
            )
            for i in range(26)
        ]
        return path.name, dict(candidates=rows)

    monkeypatch.setattr(module, "candidates", candidates)
    output = tmp_path / "source.json"
    monkeypatch.setattr(
        sys, "argv", ["select", "b", *sorted(module.B_SOURCES), "--output", str(output)]
    )
    module.main()
    evidence = json.loads(output.read_text())["evidence"]
    assert evidence["selected_index"] == 21
    assert evidence["mean_ranks"][21] == 1.0
    assert evidence["mean_ranks"][0] == 14.0


def test_duplicate_registry_rejected(tmp_path, analysis):
    path = tmp_path / "registry.json"
    path.write_text('{"conditions":{"same":{},"same":{}}}')
    with pytest.raises(ValueError, match="Duplicate registry"):
        analysis.load_registry(path)


@pytest.mark.parametrize(
    "dataset,counts",
    [
        ("german_credit", (700, 300)),
        ("ionosphere", (225, 126)),
        ("bioresponse", (2034, 1717)),
    ],
)
@pytest.mark.parametrize("fraction", [0.25, 0.5, 1.0])
def test_b_preflight_native_api_and_expected_budgets(analysis, dataset, counts, fraction):
    from tabular_thesis.data.tasks import TaskSpec

    module = importlib.import_module("run_mechanism_studies")
    resolver = ConfigResolver(default_config_directory())
    plan = module.load_plan(default_config_directory() / "studies/study-b.yaml")
    row = next(
        r
        for r in module.expand(plan)
        if r["dataset"] == dataset
        and r["fraction"] == fraction
        and r["seed"] == 0
        and r["variant"] == "full"
    )
    # These are synthetic class-count fixtures, not a raw-data qualification claim.
    y = np.repeat([0, 1], counts)
    task = TaskSpec(
        dataset, pd.DataFrame({"x": np.arange(len(y))}), y, ["0", "1"], {"fixture": "class-counts"}
    )
    evidence = module.preflight_b(row, task, resolver.study())
    assert len(evidence) == 8
    assert all(e["epochs"] == row["params"]["epochs"] for e in evidence)
    target = 1002 if dataset == "german_credit" and fraction == 0.25 else row["target_steps"]
    assert all(e["optimizer_steps"] == target for e in evidence)


def test_source_a_allows_failed_random_but_not_failed_default(tmp_path, analysis):
    from tabular_thesis.experiments.artifacts import ArtifactStore, atomic_json

    module = importlib.import_module("select_mechanism_sources")
    store = ArtifactStore(tmp_path)
    atomic_json(tmp_path / "manifest.json", {"identity": {"sources": {"amazon": {}}}})
    for index in range(26):
        job = tmp_path / "jobs" / f"amazon-mlp-{index:03d}"
        if index:
            store.status(job, "failed", error="synthetic failure")
        else:
            atomic_json(job / "manifest.json", {"identity": {"parameters": {}}})
            store.prediction_record(
                job,
                "result",
                {"parameters": {}},
                oof_y=np.array([0, 1]),
                oof_probabilities=np.array([[0.8, 0.2], [0.2, 0.8]]),
            )
            store.status(job, "completed")
    _, evidence = module.candidates(tmp_path, "mlp", allow_failed=True)
    assert len(evidence["candidates"]) == 26
    assert evidence["candidates"][0]["error"] == 0
    assert sum(c["error"] is None for c in evidence["candidates"]) == 25
    with pytest.raises(ValueError, match="admissible"):
        module.candidates(tmp_path, "mlp", allow_failed=False)
    store.status(tmp_path / "jobs/amazon-mlp-000", "failed", error="synthetic default failure")
    with pytest.raises(ValueError, match="admissible"):
        module.candidates(tmp_path, "mlp", allow_failed=True)
