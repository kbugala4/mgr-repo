"""Evaluation capabilities and parameter validation for the accepted methods."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True)
class ModelSpec:
    name: str
    neural: bool
    evaluation: str = "bagged"
    upstream: str | None = None
    parameters: frozenset[str] = frozenset()

    def validate_parameters(self, params: dict) -> None:
        unknown = set(params) - self.parameters
        if unknown:
            raise ValueError(f"Unsupported parameters for {self.name}: {sorted(unknown)}")
        for name in {
            "epochs",
            "n_epochs",
            "patience",
            "batch_size",
            "n_estimators",
            "iterations",
            "max_iter",
            "n_neighbors",
            "n_blocks",
            "d_token",
            "attention_n_heads",
        } & params.keys():
            if params[name] == "auto":
                continue
            minimum = 0 if self.name == "modernnca" and name == "n_blocks" else 1
            if (
                not isinstance(params[name], int)
                or isinstance(params[name], bool)
                or params[name] < minimum
            ):
                raise ValueError(f"{self.name}.{name} must be a positive integer")
        if "virtual_batch_size" in params and params["virtual_batch_size"] != "auto":
            value = params["virtual_batch_size"]
            if not isinstance(value, int) or isinstance(value, bool) or value < 2:
                raise ValueError(f"{self.name}.virtual_batch_size must be an integer >=2 or 'auto'")
        if "batch_size_multiplier" in params:
            value = params["batch_size_multiplier"]
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not isfinite(value)
                or value <= 0
            ):
                raise ValueError(
                    f"{self.name}.batch_size_multiplier must be a finite positive number"
                )
        for name in {
            "dropout",
            "attention_dropout",
            "ffn_dropout",
            "residual_dropout",
            "p_drop",
        } & params.keys():
            if not 0 <= params[name] < 1:
                raise ValueError(f"{self.name}.{name} must be in [0, 1)")
        if "hidden_dims" in params and (
            not params["hidden_dims"] or any(v < 1 for v in params["hidden_dims"])
        ):
            raise ValueError("hidden_dims must contain positive layer widths")


_TRAINING = {
    "random_state",
    "epochs",
    "patience",
    "batch_size",
    "batch_size_multiplier",
    "learning_rate",
    "weight_decay",
}
_SPECS = {
    "mlp": ModelSpec("mlp", True, parameters=frozenset(_TRAINING | {"hidden_dims", "dropout"})),
    "ft_transformer": ModelSpec(
        "ft_transformer",
        True,
        parameters=frozenset(
            _TRAINING
            | {
                "n_blocks",
                "d_token",
                "attention_n_heads",
                "attention_dropout",
                "ffn_d_hidden",
                "ffn_dropout",
                "residual_dropout",
            }
        ),
    ),
    "tabnet": ModelSpec(
        "tabnet",
        True,
        parameters=frozenset(
            _TRAINING
            | {
                "n_d",
                "n_a",
                "n_steps",
                "gamma",
                "n_independent",
                "n_shared",
                "momentum",
                "mask_type",
                "virtual_batch_size",
            }
        ),
    ),
    "logistic_regression": ModelSpec(
        "logistic_regression",
        False,
        parameters=frozenset(
            {
                "random_state",
                "C",
                "max_iter",
                "solver",
            }
        ),
    ),
    "knn": ModelSpec(
        "knn",
        False,
        "loo",
        "KNN",
        frozenset(
            {
                "random_state",
                "n_neighbors",
                "weights",
                "metric",
                "p",
                "scaler",
                "cat_threshold",
            }
        ),
    ),
    "xgboost": ModelSpec(
        "xgboost",
        False,
        parameters=frozenset(
            {
                "random_state",
                "n_estimators",
                "max_depth",
                "learning_rate",
                "subsample",
                "colsample_bytree",
                "reg_alpha",
                "reg_lambda",
            }
        ),
    ),
    "catboost": ModelSpec(
        "catboost",
        False,
        parameters=frozenset(
            {
                "random_state",
                "iterations",
                "depth",
                "learning_rate",
                "l2_leaf_reg",
                "random_strength",
                "bagging_temperature",
            }
        ),
    ),
    "realmlp": ModelSpec(
        "realmlp",
        True,
        upstream="RealMLP",
        parameters=frozenset(
            {
                "n_epochs",
                "n_ens",
                "hidden_sizes",
                "n_hidden_layers",
                "hidden_width",
                "lr",
                "wd",
                "p_drop",
                "act",
                "plr_sigma",
                "sq_mom",
                "plr_lr_factor",
                "scale_lr_factor",
                "first_layer_lr_factor",
                "ls_eps_sched",
                "ls_eps",
                "p_drop_sched",
                "use_ls",
                "max_one_hot_cat_size",
                "embedding_size",
                "ens_av_before_softmax",
                "plr_hidden_1",
                "plr_hidden_2",
                "use_early_stopping",
                "default_hyperparameters",
                "batch_size",
            }
        ),
    ),
    "modernnca": ModelSpec(
        "modernnca",
        True,
        upstream="ModernNCA",
        parameters=frozenset(
            {
                "n_epochs",
                "batch_size",
                "lr",
                "weight_decay",
                "dim",
                "dropout",
                "d_block",
                "n_blocks",
                "temperature",
                "sample_rate",
                "num_emb_type",
                "num_emb_n_frequencies",
                "num_emb_frequency_scale",
                "num_emb_d_embedding",
                "num_emb_lite",
            }
        ),
    ),
    "tabm": ModelSpec(
        "tabm",
        True,
        upstream="TabM",
        parameters=frozenset(
            {
                "n_epochs",
                "batch_size",
                "patience",
                "amp",
                "arch_type",
                "tabm_k",
                "gradient_clipping_norm",
                "share_training_batches",
                "lr",
                "weight_decay",
                "n_blocks",
                "d_block",
                "dropout",
                "num_emb_type",
                "d_embedding",
                "num_emb_n_bins",
                "bool_to_cat",
            }
        ),
    ),
    "tabpfn": ModelSpec(
        "tabpfn",
        True,
        "fixed",
        parameters=frozenset(
            {
                "checkpoint",
                "checkpoint_sha256",
                "n_estimators",
            }
        ),
    ),
}


def model_spec(name: str) -> ModelSpec:
    try:
        return _SPECS[name]
    except KeyError as error:
        raise ValueError(f"Unknown accepted model: {name!r}") from error


def accepted_models() -> list[str]:
    return list(_SPECS)
