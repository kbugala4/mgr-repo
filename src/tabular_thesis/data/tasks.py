"""Download once, verify cached inputs, and preserve task/split identity."""

from __future__ import annotations

import io
import json
import os
import tempfile
import urllib.request
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from filelock import FileLock
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.preprocessing import LabelEncoder

from tabular_thesis.experiments.artifacts import atomic_json, file_sha256
from tabular_thesis.experiments.configuration import DatasetConfig
from tabular_thesis.experiments.runtime import configure_caches


@dataclass
class TaskSpec:
    name: str
    X: pd.DataFrame
    y: np.ndarray
    classes: list[str]
    source_identity: dict
    groups: np.ndarray | None = None

    def outer_split(self, seed: int, inner_folds: int) -> tuple[np.ndarray, np.ndarray]:
        if len(self.y) != len(self.X) or len(self.classes) < 2:
            raise ValueError(f"Invalid task dimensions: {self.name}")
        if self.groups is None:
            splitter = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
            train, test = next(splitter.split(self.X, self.y))
        else:
            splitter = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=seed)
            train, test = next(splitter.split(self.X, self.y, self.groups))
            if set(self.groups[train]) & set(self.groups[test]):
                raise ValueError("Outer patient/group leakage")
            if len(np.unique(self.groups[train])) < inner_folds:
                raise ValueError("Insufficient training groups for inner validation")
        for partition, required in ((train, inner_folds), (test, 1)):
            counts = np.bincount(self.y[partition], minlength=len(self.classes))
            if np.any(counts < required):
                raise ValueError(
                    f"{self.name}: insufficient class counts {counts.tolist()} "
                    "for the declared split"
                )
        return train, test

    def split_identity(self, seed: int, inner_folds: int) -> dict:
        train, test = self.outer_split(seed, inner_folds)
        return {
            "outer_seed": seed,
            "inner_folds": inner_folds,
            "train_indices": train.tolist(),
            "test_indices": test.tolist(),
            "grouped": self.groups is not None,
        }


class DatasetRepository:
    def __init__(self, root: Path):
        self.root = root.resolve()

    def prepare(self, config: DatasetConfig) -> TaskSpec:
        """Reuse verified inputs. Never silently refresh an existing source version."""
        directory = self.root / "prepared" / config.name
        directory.parent.mkdir(parents=True, exist_ok=True)
        with FileLock(str(directory) + ".lock"):
            if (directory / "manifest.json").exists():
                return self.load(config)
            frame, categories = self._download(config)
            if config.target not in frame:
                raise ValueError(
                    f"Target {config.target!r} missing from {config.name}; columns: {list(frame)}"
                )
            if frame[config.target].isna().any():
                raise ValueError(
                    f"Missing target labels in {config.name}; declare an audited policy"
                )
            frame = frame.reset_index(drop=True)
            for column in set(categories) | set(config.categorical):
                if column not in frame:
                    raise ValueError(f"Declared categorical feature missing: {column}")
                frame[column] = frame[column].astype("category")
            for column in frame:
                if column != config.target and frame[column].dtype == object:
                    frame[column] = frame[column].astype("category")
            directory.mkdir(parents=True, exist_ok=True)
            handle, temporary = tempfile.mkstemp(
                prefix=".dataset-", suffix=".parquet", dir=directory
            )
            os.close(handle)
            try:
                frame.to_parquet(temporary, index=False)
                digest = file_sha256(Path(temporary))
                if config.expected_sha256 and digest != config.expected_sha256:
                    raise ValueError(f"Dataset checksum mismatch: {config.name}")
                os.replace(temporary, directory / "data.parquet")
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)
            atomic_json(
                directory / "manifest.json",
                {
                    "schema_version": 1,
                    "source": asdict(config),
                    "data_sha256": digest,
                    "rows": len(frame),
                    "columns": list(frame),
                },
            )
        return self.load(config)

    def verified_manifest(self, config: DatasetConfig) -> dict:
        directory = self.root / "prepared" / config.name
        manifest_path = directory / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"Dataset {config.name} is not prepared. Run 'thesis prepare' first."
            )
        manifest = json.loads(manifest_path.read_text())
        stored_source = {**manifest["source"], "expected_sha256": None}
        requested_source = {**asdict(config), "expected_sha256": None}
        if manifest.get("schema_version") != 1 or stored_source != requested_source:
            raise ValueError(
                f"Cached source definition differs for {config.name}; use a new data directory"
            )
        path = directory / "data.parquet"
        if file_sha256(path) != manifest["data_sha256"]:
            raise ValueError(f"Cached dataset is corrupt: {path}")
        if config.expected_sha256 and manifest["data_sha256"] != config.expected_sha256:
            raise ValueError(f"Dataset checksum mismatch: {config.name}")
        return manifest

    def load(self, config: DatasetConfig) -> TaskSpec:
        manifest = self.verified_manifest(config)
        path = self.root / "prepared" / config.name / "data.parquet"
        frame = pd.read_parquet(path)
        encoder = LabelEncoder().fit(frame[config.target].astype(str))
        if len(encoder.classes_) != config.n_classes:
            raise ValueError(f"Unexpected class count for {config.name}: {len(encoder.classes_)}")
        groups = None
        if config.group:
            if config.group not in frame or frame[config.group].isna().any():
                raise ValueError(f"Missing group identifiers in {config.name}")
            groups = frame[config.group].astype(str).to_numpy()
        if config.time:
            raise NotImplementedError(
                "Temporal tasks need an audited forward split; IID splitting is refused"
            )
        excluded = [config.target, *config.exclude, *([config.group] if config.group else [])]
        X = frame.drop(columns=excluded)
        if X.shape[1] == 0 or X.columns.duplicated().any():
            raise ValueError(f"Invalid predictor schema: {config.name}")
        return TaskSpec(
            name=config.name,
            X=X,
            y=encoder.transform(frame[config.target].astype(str)),
            classes=encoder.classes_.tolist(),
            groups=groups,
            source_identity={"source": manifest["source"], "data_sha256": manifest["data_sha256"]},
        )

    def _download(self, config: DatasetConfig) -> tuple[pd.DataFrame, list[str]]:
        if config.source == "openml":
            configure_caches(self.root / "cache")
            import openml

            openml.config.set_root_cache_directory(self.root / "cache" / "openml")
            dataset = openml.datasets.get_dataset(config.source_id, download_data=True)
            if dataset.version != config.version:
                raise ValueError(f"Unexpected OpenML version for {config.name}: {dataset.version}")
            X, y, categorical, _ = dataset.get_data(
                dataset_format="dataframe",
                target=config.target,
                include_row_id=True,
            )
            categories = [str(c) for c, flag in zip(X.columns, categorical, strict=True) if flag]
            return X.assign(**{config.target: y}), categories
        if config.source == "local_csv":
            if not config.local_file:
                raise ValueError("A local source requires local_file")
            path = (self.root / "raw" / config.local_file).resolve()
            if not path.is_relative_to((self.root / "raw").resolve()):
                raise ValueError("local_file must stay inside data/raw")
            if not path.is_file():
                raise FileNotFoundError(
                    f"Place the officially acquired {config.name} training CSV at {path}. "
                    "Santander competition access may require accepting Kaggle's rules."
                )
            return pd.read_csv(path), []
        if config.source != "uci_zip":
            raise ValueError(f"Unsupported source: {config.source}")
        slugs = {
            52: "ionosphere",
            144: "statlog+german+credit+data",
            296: "diabetes+130-us+hospitals+for+years+1999-2008",
            468: "online+shoppers+purchasing+intention+dataset",
        }
        if config.source_id not in slugs:
            raise ValueError(f"Unsupported UCI source: {config.source_id}")
        directory = self.root / "raw" / config.name
        directory.mkdir(parents=True, exist_ok=True)
        archive_path = directory / "source.zip"
        if not archive_path.exists():
            url = f"https://archive.ics.uci.edu/static/public/{config.source_id}/{slugs[config.source_id]}.zip"
            request = urllib.request.Request(url, headers={"User-Agent": "tabular-thesis/0.2"})
            with urllib.request.urlopen(request, timeout=120) as response:
                payload = response.read()
            # Validate the archive before committing it to the cache.
            with zipfile.ZipFile(io.BytesIO(payload)) as archive:
                if archive.testzip() is not None:
                    raise ValueError("Corrupt UCI download")
            handle, temporary = tempfile.mkstemp(dir=directory)
            with os.fdopen(handle, "wb") as stream:
                stream.write(payload)
            os.replace(temporary, archive_path)
        with zipfile.ZipFile(archive_path) as archive:
            wanted = {
                52: "ionosphere.data",
                144: "german.data",
                296: "diabetic_data.csv",
                468: "online_shoppers_intention.csv",
            }[config.source_id]
            members = [name for name in archive.namelist() if Path(name).name == wanted]
            if len(members) != 1:
                raise ValueError(f"Expected one {wanted} in {archive_path}")
            with archive.open(members[0]) as stream:
                if config.source_id in {52, 144}:
                    frame = pd.read_csv(
                        stream, header=None, sep=r"\s+" if config.source_id == 144 else ","
                    )
                    frame.columns = [
                        *[f"feature_{i}" for i in range(frame.shape[1] - 1)],
                        config.target,
                    ]
                else:
                    frame = pd.read_csv(
                        stream,
                        na_values=["?"],
                        low_memory=False,
                        dtype={column: "string" for column in config.categorical},
                    )
        return frame, []
