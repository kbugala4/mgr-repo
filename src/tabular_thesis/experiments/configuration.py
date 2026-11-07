"""Typed YAML resolution shared by planning, downloads, and execution."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf


@dataclass
class RuntimeConfig:
    device: str = "cuda"
    num_cpus: int = 4
    configuration_time_limit: int = 3600
    prediction_batch_size: int = 4096


@dataclass
class PathsConfig:
    data: str = "data"
    results: str = "results"


@dataclass
class StudyConfig:
    name: str = "thesis-lite"
    models: list[str] = field(default_factory=list)
    datasets: list[str] = field(default_factory=list)
    random_configurations: int = 25
    inner_folds: int = 8
    outer_seed: int = 42
    search_seed: int = 42
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    model_overrides: dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelConfig:
    name: str = ""
    type: str = ""
    params: dict[str, Any] = field(default_factory=dict)
    search_space: dict[str, Any] = field(default_factory=dict)


@dataclass
class DatasetConfig:
    name: str = ""
    source: str = "openml"
    source_id: int = 0
    version: int = 1
    target: str = "class"
    n_classes: int = 2
    categorical: list[str] = field(default_factory=list)
    exclude: list[str] = field(default_factory=list)
    group: str | None = None
    time: str | None = None
    expected_sha256: str | None = None
    local_file: str | None = None


class ConfigResolver:
    def __init__(self, directory: Path):
        self.directory = directory.resolve()

    @staticmethod
    def _load(schema: type, path: Path, overrides: list[str] | None = None):
        if not path.is_file():
            raise ValueError(f"Configuration not found: {path}")
        merged = OmegaConf.merge(
            OmegaConf.structured(schema),
            OmegaConf.load(path),
            OmegaConf.from_dotlist(overrides or []),
        )
        OmegaConf.resolve(merged)
        return OmegaConf.to_object(merged)

    def study(self, overrides: list[str] | None = None) -> StudyConfig:
        config = self._load(StudyConfig, self.directory / "config.yaml", overrides)
        if not config.models or not config.datasets:
            raise ValueError("A study must declare models and datasets")
        for values in (config.models, config.datasets):
            if len(values) != len(set(values)):
                raise ValueError("Duplicate models/datasets would repeat the same experiment")
        if config.inner_folds < 2 or config.random_configurations < 0:
            raise ValueError("inner_folds must be >=2 and random_configurations >=0")
        if config.runtime.device not in {"cpu", "cuda"}:
            raise ValueError("runtime.device must be cpu or cuda")
        if (
            min(
                config.runtime.num_cpus,
                config.runtime.configuration_time_limit,
                config.runtime.prediction_batch_size,
            )
            < 1
        ):
            raise ValueError("Runtime resource limits must be positive")
        if not config.name or any(
            c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
            for c in config.name
        ):
            raise ValueError("Study name must contain only letters, numbers, '_' or '-'")
        for name in config.models:
            self.model(name)
        for name in config.datasets:
            self.dataset(name)
        if set(config.model_overrides) - set(config.models):
            raise ValueError("Overrides refer to a model outside the selected study")
        self.resolved_models(config)
        return config

    def model(self, name: str) -> ModelConfig:
        return self._named(ModelConfig, "model", name)

    def dataset(self, name: str) -> DatasetConfig:
        return self._named(DatasetConfig, "dataset", name)

    def _named(self, schema: type, group: str, name: str):
        if not name or Path(name).name != name or name in {".", ".."}:
            raise ValueError(f"Invalid configuration name: {name!r}")
        config = self._load(schema, self.directory / group / f"{name}.yaml")
        if config.name != name:
            raise ValueError(f"Configuration name mismatch: {name!r} != {config.name!r}")
        return config

    def resolved_models(self, study: StudyConfig) -> dict[str, ModelConfig]:
        from tabular_thesis.models.specification import model_spec

        models = {}
        for name in study.models:
            config = self.model(name)
            config.params.update(study.model_overrides.get(name, {}))
            model_spec(name).validate_parameters(config.params)
            expected_type = "dl" if model_spec(name).neural else "non_dl"
            if config.type != expected_type:
                raise ValueError(f"Model type mismatch for {name}: expected {expected_type}")
            if name == "tabpfn":
                checkpoint = Path(config.params["checkpoint"])
                if not checkpoint.is_absolute():
                    config.params["checkpoint"] = str(
                        Path(study.paths.data) / "weights" / checkpoint
                    )
            if set(config.search_space) - model_spec(name).parameters:
                raise ValueError(f"Search space contains unsupported parameters for {name}")
            models[name] = config
        return models

    def snapshot(self, study: StudyConfig) -> dict:
        return {
            "study": asdict(study),
            "models": {name: asdict(c) for name, c in self.resolved_models(study).items()},
            "datasets": {name: asdict(self.dataset(name)) for name in study.datasets},
        }


def default_config_directory() -> Path:
    checkout = Path(__file__).resolve().parents[3] / "configs"
    return checkout if checkout.is_dir() else Path(__file__).resolve().parents[1] / "configs"
