"""Importable AutoGluon adapters; TabArena owns validation and bagging."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from autogluon.core.models import AbstractModel

from tabular_thesis.experiments.artifacts import atomic_json
from tabular_thesis.experiments.mechanisms import balanced_class_weights, nested_subset
from tabular_thesis.experiments.runtime import FitContext
from tabular_thesis.models.preprocessing import column_roles
from tabular_thesis.models.registry import get_model


class ThesisModelAdapter(AbstractModel):
    model_name = ""

    @classmethod
    def warmup(cls, *, num_gpus=0, **kwargs):
        model = get_model(cls.model_name)
        if model.is_deep_learning:
            from tabarena.models.warmup import warmup_torch

            warmup_torch(cuda=bool(num_gpus))

    def _fit(
        self,
        X,
        y,
        X_val=None,
        y_val=None,
        num_cpus=1,
        num_gpus=0,
        time_limit=None,
        sample_weight=None,
        **kwargs,
    ):
        if sample_weight is not None:
            raise ValueError("Arbitrary sample weights require a separately declared condition")
        params = self._get_model_params()
        weighting = params.pop("_class_weighting", None)
        fraction = params.pop("_train_fraction", None)
        selection = params.pop("_selection_loss", None)
        record = params.pop("_record_history", False)
        history_prefix = params.pop("_history_prefix", None)
        seed = int(self.random_seed or 0)
        y = np.asarray(y)
        if fraction is not None:
            rows = nested_subset(y, fraction, seed)
            X, y = X.iloc[rows], y[rows]
            # Recompute native constant-column selection on the subset, not the full fold.
            self._preprocess_set_features(X)
        # AutoGluon selects fold-local features; fit and inference must use that same schema.
        X = self.preprocess(X)
        if X_val is not None:
            X_val = self.preprocess(X_val)
        weights = None if weighting is None else balanced_class_weights(y, self.num_classes)
        prediction_batch_size = params.pop("_prediction_batch_size", 4096)
        requested_device = params.pop("_requested_device", "cuda" if num_gpus else "cpu")
        if requested_device == "cuda" and not num_gpus:
            raise RuntimeError("CUDA was requested but the fold received no GPU allocation")
        self.model = get_model(self.model_name, params=params)
        context = FitContext(
            device="cuda" if num_gpus else "cpu",
            num_cpus=num_cpus,
            time_limit=time_limit,
            seed=seed,
            prediction_batch_size=prediction_batch_size,
            class_weights=weights,
            weighted_validation=weights is not None and selection is None,
            unweighted_selection=selection == "unweighted",
            record_history=record,
        )
        self.model.set_context(context)
        numerical, categorical = column_roles(X, None, None)
        self.model.fit(
            X,
            np.asarray(y),
            X_val,
            None if y_val is None else np.asarray(y_val),
            numerical_columns=numerical,
            categorical_columns=categorical,
        )
        self.params_trained = dict(self.model.params)
        if self.model.is_deep_learning or record:
            self.params_trained.update(getattr(self.model, "training_info_", {}))
        if record:
            self.params_trained.update(
                class_weights=weights,
                training_rows=len(y),
                training_class_counts=np.bincount(y, minlength=self.num_classes).tolist(),
                training_row_ids=[int(i) for i in X.index],
                validation_row_ids=[int(i) for i in X_val.index],
                row_id_space="task",
            )
            if history_prefix is not None:
                atomic_json(Path(f"{history_prefix}-{seed}.json"), self.params_trained)

    @classmethod
    def supported_problem_types(cls):
        return ["binary", "multiclass"]

    def _get_default_auxiliary_params(self):
        return {
            **super()._get_default_auxiliary_params(),
            "valid_raw_types": ["int", "float", "category", "bool"],
            "ignored_type_group_special": [],
        }


class NeuralModelAdapter(ThesisModelAdapter):
    default_num_gpus = 1


class MLPAdapter(NeuralModelAdapter):
    ag_key, ag_name, model_name = "THESIS-MLP", "ThesisMLP", "mlp"


class FTTransformerAdapter(NeuralModelAdapter):
    ag_key, ag_name, model_name = "THESIS-FT", "ThesisFT", "ft_transformer"


class TabNetAdapter(NeuralModelAdapter):
    ag_key, ag_name, model_name = "THESIS-TABNET", "ThesisTabNet", "tabnet"


class LogisticAdapter(ThesisModelAdapter):
    ag_key, ag_name, model_name = "THESIS-LR", "ThesisLR", "logistic_regression"


class XGBoostAdapter(ThesisModelAdapter):
    ag_key, ag_name, model_name = "THESIS-XGB", "ThesisXGB", "xgboost"


class CatBoostAdapter(ThesisModelAdapter):
    ag_key, ag_name, model_name = "THESIS-CAT", "ThesisCatBoost", "catboost"


LOCAL_ADAPTERS = {
    "mlp": MLPAdapter,
    "ft_transformer": FTTransformerAdapter,
    "tabnet": TabNetAdapter,
    "logistic_regression": LogisticAdapter,
    "xgboost": XGBoostAdapter,
    "catboost": CatBoostAdapter,
}
