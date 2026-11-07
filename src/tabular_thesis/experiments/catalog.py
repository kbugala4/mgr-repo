"""One parameter/search-space service for local and upstream model objects."""

from __future__ import annotations

import copy
import importlib

from tabular_thesis.experiments.configuration import ModelConfig
from tabular_thesis.models.specification import model_spec


class ModelCatalog:
    def configurations(self, config: ModelConfig, count: int, seed: int):
        from autogluon.common.space import Categorical, Int, Real
        from autogluon.core.searcher.local_random_searcher import LocalRandomSearcher

        spec = model_spec(config.name)
        if spec.upstream:
            from tabular_thesis.experiments.upstream_models import UPSTREAM_MODELS

            # Import only the chosen model, not the whole auto-discovered registry.
            modules = {"knn": "knn", "realmlp": "realmlp", "modernnca": "modernnca", "tabm": "tabm"}
            module = importlib.import_module(f"tabarena.models.{modules[config.name]}.hpo")
            generator = getattr(module, f"gen_{config.name}")
            default = {**copy.deepcopy(generator.manual_configs[0]), **config.params}
            if hasattr(generator, "search_space_func"):
                random = generator.search_space_func(count, seed=seed) if count else []
            else:
                searcher = LocalRandomSearcher(
                    search_space={
                        key: value
                        for key, value in generator.search_space.items()
                        if key not in config.params
                    },
                    first_is_default=False,
                    random_seed=seed,
                )
                random = [searcher.get_config() for _ in range(count)]
            return UPSTREAM_MODELS[config.name], [
                default,
                *[{**p, **config.params} for p in random],
            ]
        if spec.evaluation == "fixed":
            from pathlib import Path

            from tabular_thesis.experiments.artifacts import file_sha256
            from tabular_thesis.experiments.tabpfn_model import TabPFN35Model

            checkpoint = Path(config.params["checkpoint"])
            if not checkpoint.is_file():
                raise FileNotFoundError(f"Missing checkpoint: {checkpoint}; run 'thesis weights'")
            if file_sha256(checkpoint) != config.params["checkpoint_sha256"]:
                raise ValueError("TabPFN checkpoint checksum mismatch")
            return TabPFN35Model, [dict(config.params)]
        from tabular_thesis.experiments.tabarena_models import LOCAL_ADAPTERS

        search_space = {}
        for name, definition in config.search_space.items():
            kind, *values = definition
            if kind == "categorical":
                search_space[name] = Categorical(*values[0])
            elif kind == "int":
                search_space[name] = Int(*values)
            elif kind in {"float", "float_log"}:
                search_space[name] = Real(*values, log=kind == "float_log")
            else:
                raise ValueError(f"Unknown search-space type: {kind}")
        searcher = LocalRandomSearcher(
            search_space=search_space, first_is_default=False, random_seed=seed
        )
        random = [searcher.get_config() for _ in range(count)]
        return LOCAL_ADAPTERS[config.name], [
            dict(config.params),
            *[{**config.params, **p} for p in random],
        ]
