"""Local adapter registry; ModelCatalog supplies all eleven study methods.

Usage::

    from tabular_thesis.models.registry import get_model, list_models

    model = get_model("xgboost", params={"n_estimators": 300})
    print(list_models())
"""

from __future__ import annotations

from typing import Any

from tabular_thesis.models.base import BaseModel

# Registry is populated lazily to avoid import errors when optional
# packages are not installed.  Each entry is (module_path, class_name).
_REGISTRY: dict[str, tuple[str, str]] = {
    # Non-DL models
    "logistic_regression": (
        "tabular_thesis.models.non_dl.logistic_regression",
        "LogisticRegressionModel",
    ),
    "knn": ("tabular_thesis.models.non_dl.knn", "KNNModel"),
    "xgboost": ("tabular_thesis.models.non_dl.xgboost_model", "XGBoostModel"),
    "catboost": ("tabular_thesis.models.non_dl.catboost_model", "CatBoostModel"),
    # DL models
    "mlp": ("tabular_thesis.models.dl.mlp", "MLPModel"),
    "ft_transformer": ("tabular_thesis.models.dl.ft_transformer", "FTTransformerModel"),
    "tabnet": ("tabular_thesis.models.dl.tabnet_model", "TabNetModel"),
}


def get_model(name: str, params: dict[str, Any] | None = None) -> BaseModel:
    """Instantiate a model by its registered name.

    Args:
        name: Model key (e.g. ``'xgboost'``, ``'ft_transformer'``).
        params: Hyperparameters to pass to the model constructor.

    Returns:
        An instance of the requested model.

    Raises:
        ValueError: If *name* is not in the registry.
    """
    if name not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY))
        raise ValueError(f"Unknown model '{name}'. Available: {available}")

    module_path, class_name = _REGISTRY[name]

    # Lazy import to avoid loading all frameworks at startup
    import importlib

    module = importlib.import_module(module_path)
    model_cls = getattr(module, class_name)
    from tabular_thesis.experiments.configuration import ConfigResolver, default_config_directory

    defaults = ConfigResolver(default_config_directory()).model(name).params
    return model_cls(name=name, params={**defaults, **(params or {})})


def list_models() -> list[str]:
    """Return sorted list of all registered model names."""
    return sorted(_REGISTRY.keys())


def is_registered(name: str) -> bool:
    """Check if a model name is in the registry."""
    return name in _REGISTRY
