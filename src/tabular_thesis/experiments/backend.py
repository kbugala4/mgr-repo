"""TabArena execution on the thesis' own immutable outer partitions."""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
from autogluon.features.generators import IdentityFeatureGenerator
from tabarena.benchmark.exec_models.autogluon import AGModelWrapper, AGSingleBagWrapper
from tabarena.benchmark.experiment import (
    AGModelBagExperiment,
    AGModelOuterExperiment,
    OOFExperimentRunner,
)
from tabarena.benchmark.task import UserTask
from tabarena.benchmark.task.metadata import GroupLabelTypes

from tabular_thesis.data.tasks import TaskSpec
from tabular_thesis.evaluation.metrics import validate_probabilities
from tabular_thesis.experiments.configuration import StudyConfig
from tabular_thesis.models.specification import model_spec


class SynchronizedExecution:
    def _synchronize(self):
        if self.fit_kwargs.get("num_gpus", 0):
            import torch

            torch.cuda.synchronize()

    def fit(self, *args, **kwargs):
        self._synchronize()
        try:
            return super().fit(*args, **kwargs)
        finally:
            self._synchronize()

    def predict_proba(self, *args, **kwargs):
        self._synchronize()
        try:
            return super().predict_proba(*args, **kwargs)
        finally:
            self._synchronize()


class TimedBagWrapper(SynchronizedExecution, AGSingleBagWrapper):
    pass


class TimedOuterWrapper(SynchronizedExecution, AGModelWrapper):
    pass


class ValidatedRunner(OOFExperimentRunner):
    def post_fit(self, out):
        validate_probabilities(
            np.asarray(out["probabilities"]),
            len(self._load_y_test()),
            len(self.label_cleaner.ordered_class_labels),
        )
        return super().post_fit(out)

    def post_evaluate(self, out):
        probabilities = np.asarray(out["probabilities"])
        out = super().post_evaluate(out)
        if self.model.can_get_per_child_val_idx:
            out["inner_validation_indices"] = self.model.get_per_child_val_idx()
        if out["simulation_artifacts"] is None:
            out["simulation_artifacts"] = {
                "pred_proba_dict_test": {self.method: probabilities},
                "y_test": np.asarray(self._load_y_test()),
                "y_test_idx": np.asarray(self._load_y_test().index),
            }
        return out


class TabArenaBackend:
    def __init__(self, study: StudyConfig):
        self.study = study

    def execute(
        self,
        task: TaskSpec,
        model_name: str,
        model_cls: type,
        parameters: dict,
        work_directory: Path,
    ) -> dict:
        study, spec = self.study, model_spec(model_name)
        train, test = task.outer_split(study.outer_seed, study.inner_folds)
        if {"__target__", "__group__"} & set(task.X):
            raise ValueError("Input contains a reserved task column")
        frame = task.X.copy()
        frame["__target__"] = task.y
        group_args = {}
        if task.groups is not None and spec.evaluation != "fixed":
            frame["__group__"] = task.groups
            group_args = {"group_on": "__group__", "group_labels": GroupLabelTypes.PER_SAMPLE}
        user_task = UserTask(task_name=task.name, task_cache_path=work_directory / "task")
        wrapper = user_task.create_task(
            dataset=frame,
            target_feature="__target__",
            problem_type="classification",
            splits={0: {0: (train.tolist(), test.tolist())}},
            eval_metric="roc_auc" if len(task.classes) == 2 else "log_loss",
            **group_args,
        )
        parameters = copy.deepcopy(parameters)
        if not spec.upstream and spec.evaluation != "fixed":
            parameters["_prediction_batch_size"] = study.runtime.prediction_batch_size
            parameters["_requested_device"] = study.runtime.device if spec.neural else "cpu"
        resources = {
            "num_cpus": study.runtime.num_cpus,
            "num_gpus": int(spec.neural and study.runtime.device == "cuda"),
        }
        if spec.evaluation == "fixed":
            experiment = AGModelOuterExperiment(
                name=model_name,
                model_cls=model_cls,
                model_hyperparameters=parameters,
                method_kwargs={
                    "fit_kwargs": {
                        **resources,
                        "time_limit": study.runtime.configuration_time_limit,
                    }
                },
            )
        else:
            parameters["ag_args_ensemble"] = {
                "fold_fitting_strategy": "sequential_local",
                "model_random_seed": study.outer_seed,
                "vary_seed_across_folds": True,
                "use_child_oof": spec.evaluation == "loo" and task.groups is None,
            }
            experiment = AGModelBagExperiment(
                name=model_name,
                model_cls=model_cls,
                model_hyperparameters=parameters,
                num_bag_folds=study.inner_folds,
                num_bag_sets=1,
                time_limit=study.runtime.configuration_time_limit,
                time_limit_with_preprocessing=True,
                raise_on_model_failure=True,
                dynamic_tabarena_validation_protocol=task.groups is not None,
                method_kwargs={
                    "validation_metadata": {
                        "default_num_folds": study.inner_folds,
                        "default_num_repeats": 1,
                        "max_samples_for_tiny_data": 0,
                    },
                    "init_kwargs": {"path": str(work_directory / "model"), "verbosity": 0},
                    "fit_kwargs": {
                        **resources,
                        "feature_generator": IdentityFeatureGenerator(
                            features_in=list(task.X.columns),
                            pre_enforce_types=False,
                            pre_drop_useless=False,
                        ),
                    },
                },
            )
        experiment.experiment_cls = ValidatedRunner
        experiment.method_cls = TimedOuterWrapper if spec.evaluation == "fixed" else TimedBagWrapper
        # Per-child test predictions would repeat inference outside the timed stage.
        experiment.experiment_kwargs.update(compute_bag_info=False)
        return experiment.run(
            task=wrapper,
            fold=0,
            task_name=task.name,
            cache_task_key=user_task.cache_key,
            debug_mode=True,
            raise_on_failure=True,
        )
