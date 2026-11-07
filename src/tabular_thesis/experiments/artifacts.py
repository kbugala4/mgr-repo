"""Atomic, versioned experiment artifacts; CSV is a derived export."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import tempfile
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
from filelock import FileLock

SCHEMA_VERSION = 1


def canonical(value: Any) -> str:
    def convert(item):
        if is_dataclass(item):
            return asdict(item)
        if isinstance(item, Path):
            return str(item)
        if isinstance(item, np.generic):
            return item.item()
        if isinstance(item, np.ndarray):
            return item.tolist()
        raise TypeError(f"Unsupported artifact value: {type(item).__name__}")

    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False, default=convert
    )


def fingerprint(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    payload = canonical(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(handle, "w") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def code_identity(root: Path) -> dict:
    if not (root / "src").is_dir():
        root = Path(__file__).resolve().parents[1]
    files = [*sorted((root / "src" if (root / "src").is_dir() else root).rglob("*.py"))]
    files += [p for p in (root / "pyproject.toml", root / "uv.lock") if p.exists()]
    versions = {}
    for name in (
        "tabular-thesis",
        "numpy",
        "pandas",
        "scipy",
        "xgboost",
        "catboost",
        "pyarrow",
        "openml",
        "data-foundry",
        "rtdl-num-embeddings",
        "category-encoders",
        "tabarena",
        "autogluon.core",
        "autogluon.tabular",
        "torch",
        "scikit-learn",
        "pytorch-tabnet",
        "rtdl-revisiting-models",
        "tabpfn",
        "pytabkit",
        "tabm",
    ):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {
        "source": fingerprint({str(p.relative_to(root)): file_sha256(p) for p in files}),
        "packages": versions,
    }


class ArtifactStore:
    def __init__(self, root: Path):
        self.root = root.resolve()

    def run_directory(self, study: str, identity: dict) -> Path:
        return self.root / "runs" / study / fingerprint(identity)

    def initialize(self, directory: Path, identity: dict) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        with FileLock(str(directory / ".manifest.lock")):
            path = directory / "manifest.json"
            manifest = {"schema_version": SCHEMA_VERSION, "identity": identity}
            if path.exists() and json.loads(path.read_text()) != manifest:
                raise ValueError(f"Artifact identity mismatch: {directory}")
            if not path.exists():
                atomic_json(path, manifest)

    def lock(self, directory: Path) -> FileLock:
        directory.mkdir(parents=True, exist_ok=True)
        return FileLock(str(directory / ".execution.lock"), timeout=0)

    def complete(self, directory: Path) -> bool:
        path = directory / "status.json"
        if not path.exists():
            return False
        status = json.loads(path.read_text())
        if status.get("state") != "completed":
            return False
        for name, digest in status.get("artifacts", {}).items():
            candidate = directory / name
            if not candidate.is_file() or file_sha256(candidate) != digest:
                raise ValueError(f"Corrupt completed artifact: {candidate}")
        if not status.get("artifacts"):
            raise ValueError(f"Completed run has no artifact checksums: {directory}")
        return True

    def status(self, directory: Path, state: str, *, error: str | None = None) -> None:
        if state not in {"running", "failed", "completed"}:
            raise ValueError(f"Unknown execution state: {state}")
        payload = {"schema_version": SCHEMA_VERSION, "state": state, "error": error}
        if state == "completed":
            paths = sorted((directory / "records").glob("*.json"))
            if not paths:
                raise ValueError("Cannot complete an experiment without result records")
            payload["artifacts"] = {
                str(p.relative_to(directory)): file_sha256(p)
                for p in [
                    *paths,
                    *sorted((directory / "predictions").glob("*.npz")),
                    *sorted((directory / "attempts").glob("*.json")),
                ]
            }
        atomic_json(directory / "status.json", payload)

    def prediction_record(self, directory: Path, key: str, metadata: dict, **arrays) -> None:
        if not key or Path(key).name != key:
            raise ValueError("Invalid artifact key")
        path = directory / "predictions" / f"{key}.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        handle, temporary = tempfile.mkstemp(prefix=".probabilities-", dir=path.parent)
        try:
            with os.fdopen(handle, "wb") as stream:
                np.savez_compressed(stream, **arrays)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        atomic_json(
            directory / "records" / f"{key}.json",
            {
                "schema_version": SCHEMA_VERSION,
                **metadata,
                "predictions": str(path.relative_to(directory)),
                "prediction_sha256": file_sha256(path),
            },
        )
