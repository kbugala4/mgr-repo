"""Compatibility loader backed by the authoritative YAML dataset catalogue."""

from pathlib import Path

import pandas as pd

from tabular_thesis.data.registry import DatasetMeta, TaskType
from tabular_thesis.data.tasks import DatasetRepository
from tabular_thesis.experiments.configuration import ConfigResolver, default_config_directory


def load_dataset(name: str, data_dir: Path | None = None, max_samples: int | None = None):
    if max_samples is not None:
        raise ValueError(
            "Subsets require an explicit study condition; implicit truncation is refused"
        )
    config = ConfigResolver(default_config_directory()).dataset(name)
    task = DatasetRepository(data_dir or Path("data")).prepare(config)
    meta = DatasetMeta(
        name=name,
        source=config.source,
        source_id=config.source_id,
        task=TaskType.BINARY if config.n_classes == 2 else TaskType.MULTICLASS,
        target_column=config.target,
        n_classes=config.n_classes,
    )
    return task.X, pd.Series(task.y, index=task.X.index), meta
