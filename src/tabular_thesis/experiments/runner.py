"""Sequential, resumable jobs; TabArena owns each configuration's fold fits."""

from __future__ import annotations

import gc
import platform
import shutil
import time
import traceback
import uuid
from dataclasses import asdict
from pathlib import Path

import numpy as np

from tabular_thesis.data.tasks import DatasetRepository, TaskSpec
from tabular_thesis.evaluation.metrics import validate_probabilities
from tabular_thesis.experiments.artifacts import (
    ArtifactStore,
    atomic_json,
    code_identity,
    fingerprint,
)
from tabular_thesis.experiments.catalog import ModelCatalog
from tabular_thesis.experiments.configuration import ConfigResolver, StudyConfig
from tabular_thesis.experiments.runtime import FitContext, configure_caches
from tabular_thesis.models.specification import model_spec


class ExperimentRunner:
    def __init__(self, resolver: ConfigResolver, study: StudyConfig):
        self.resolver, self.study = resolver, study
        self.repository = DatasetRepository(Path(study.paths.data))
        self.store = ArtifactStore(Path(study.paths.results))

    def run(self, tasks: list[TaskSpec] | None = None) -> Path:
        study = self.study
        configure_caches(Path(study.paths.data) / "cache")
        FitContext(device=study.runtime.device, num_cpus=study.runtime.num_cpus).activate()
        from tabular_thesis.experiments.backend import TabArenaBackend

        if "tabpfn" in study.models and study.runtime.device != "cuda":
            raise ValueError("The accepted TabPFN checkpoint requires a CUDA worker")
        configs = self.resolver.resolved_models(study)
        catalog = {
            name: ModelCatalog().configurations(
                config, study.random_configurations, study.search_seed
            )
            for name, config in configs.items()
        }
        # Inspect cached manifests without retaining twelve datasets in memory.
        sources = (
            {t.name: t.source_identity for t in tasks}
            if tasks is not None
            else {
                name: self.repository.verified_manifest(self.resolver.dataset(name))
                for name in study.datasets
            }
        )
        snapshot = (
            self.resolver.snapshot(study)
            if tasks is None
            else {
                "study": asdict(study),
                "models": {n: asdict(c) for n, c in configs.items()},
                "datasets": sources,
            }
        )
        identity = {
            "configuration": snapshot,
            "sources": sources,
            "code": code_identity(Path(__file__).resolve().parents[3]),
            "host": self._host(),
            "protocol": "thesis-tabarena-lite-v1",
        }
        directory = self.store.run_directory(study.name, identity)
        self.store.initialize(directory, identity)
        index = {
            "jobs": [
                f"jobs/{dataset}-{name}-{i:03d}"
                for dataset in study.datasets
                for name, (_, params) in catalog.items()
                for i in range(len(params))
            ]
        }
        atomic_json(directory / "index.json", index)
        failures = []
        for name in study.datasets:
            task = (
                next(t for t in tasks if t.name == name)
                if tasks is not None
                else self.repository.load(self.resolver.dataset(name))
            )
            split = task.split_identity(study.outer_seed, study.inner_folds)
            atomic_json(directory / "splits" / f"{name}.json", split)
            for model, (model_cls, configurations) in catalog.items():
                for i, parameters in enumerate(configurations):
                    job = directory / "jobs" / f"{name}-{model}-{i:03d}"
                    job_identity = {
                        "study": fingerprint(identity),
                        "dataset": name,
                        "model": model,
                        "configuration_index": i,
                        "parameters": parameters,
                        "source": task.source_identity,
                        "split": fingerprint(split),
                        "classes": task.classes,
                    }
                    self.store.initialize(job, job_identity)
                    with self.store.lock(job):
                        if self.store.complete(job):
                            continue
                        self.store.status(job, "running")
                        print(
                            f"{name}/{model}: configuration {i + 1}/{len(configurations)}",
                            flush=True,
                        )
                        start = time.monotonic()
                        attempt = job / "attempts" / f"{uuid.uuid4().hex}.json"
                        attempt_info = {"started_at_unix": time.time(), "state": "running"}
                        atomic_json(attempt, attempt_info)
                        try:
                            # Failed attempts cannot leave stale native models in a retry.
                            shutil.rmtree(job / "work", ignore_errors=True)
                            out = TabArenaBackend(study).execute(
                                task, model, model_cls, parameters, job / "work"
                            )
                            self._save(job, job_identity, task, out, time.monotonic() - start)
                            atomic_json(
                                attempt,
                                {
                                    **attempt_info,
                                    "state": "completed",
                                    "wall_seconds": time.monotonic() - start,
                                },
                            )
                            self.store.status(job, "completed")
                        except BaseException as error:
                            atomic_json(
                                job / "failure.json",
                                {
                                    "type": type(error).__name__,
                                    "message": str(error),
                                    "traceback": traceback.format_exc(),
                                    "wall_seconds": time.monotonic() - start,
                                },
                            )
                            self.store.status(job, "failed", error=str(error))
                            atomic_json(
                                attempt,
                                {
                                    **attempt_info,
                                    "state": "failed",
                                    "type": type(error).__name__,
                                    "message": str(error),
                                    "traceback": traceback.format_exc(),
                                    "wall_seconds": time.monotonic() - start,
                                },
                            )
                            if not isinstance(error, Exception):
                                raise
                            failures.append({"job": str(job), "error": str(error)})
                        finally:
                            # Cached probabilities/metadata are authoritative;
                            # partial fit files are disposable.
                            shutil.rmtree(job / "work", ignore_errors=True)
                            gc.collect()
                            if study.runtime.device == "cuda":
                                import torch

                                torch.cuda.empty_cache()
        if failures:
            atomic_json(directory / "failures.json", failures)
            raise RuntimeError(
                f"{len(failures)} configuration(s) failed; first: {failures[0]['error']}. "
                f"Successful jobs are preserved. Artifacts: {directory}"
            )
        (directory / "failures.json").unlink(missing_ok=True)
        return directory

    def _save(self, directory: Path, identity: dict, task: TaskSpec, out: dict, elapsed: float):
        artifact = out["simulation_artifacts"]
        arrays = {}
        for source, label in (("test", "test"), ("val", "oof")):
            if f"y_{source}" not in artifact:
                continue
            y = np.asarray(artifact[f"y_{source}"])
            probabilities = next(iter(artifact[f"pred_proba_dict_{source}"].values()))
            arrays[f"{label}_probabilities"] = validate_probabilities(
                probabilities, len(y), len(task.classes)
            )
            indices = np.asarray(artifact[f"y_{source}_idx"])
            if not np.array_equal(task.y[indices], y):
                raise ValueError(
                    "Prediction labels/row identifiers do not align with the source task"
                )
            arrays[f"{label}_y"], arrays[f"{label}_indices"] = y, indices
        metadata = out.get("method_metadata", {})
        info = metadata.get("info", {})
        bag = info.get("bagged_info", {})
        folds = out.get("inner_validation_indices", [])
        if model_spec(identity["model"]).evaluation != "fixed":
            expected = (
                1
                if model_spec(identity["model"]).evaluation == "loo" and task.groups is None
                else self.study.inner_folds
            )
            if bag.get("num_child_models") != expected:
                raise ValueError(
                    f"Incomplete fold ensemble: expected {expected}, "
                    f"found {bag.get('num_child_models')}"
                )
            train_indices = arrays["oof_indices"]
            counts = np.zeros(len(train_indices), dtype=int)
            for i, validation in enumerate(folds):
                validation = np.asarray(validation)
                counts[validation] += 1
                training = np.setdiff1d(np.arange(len(train_indices)), validation)
                if expected > 1:
                    if len(np.unique(task.y[train_indices[training]])) != len(task.classes):
                        raise ValueError("An inner training fold is missing a class")
                    if len(np.unique(task.y[train_indices[validation]])) != len(task.classes):
                        raise ValueError("An inner validation fold is missing a class")
                if task.groups is not None and set(task.groups[train_indices[training]]) & set(
                    task.groups[train_indices[validation]]
                ):
                    raise ValueError("Inner patient/group leakage")
                arrays[f"inner_validation_{i}"] = train_indices[validation]
            if not np.all(counts == 1):
                raise ValueError(
                    "Inner validation predictions do not cover each development row once"
                )
        # Model info contains effective parameters and child seeds; runtime objects are excluded.
        self.store.prediction_record(
            directory,
            "result",
            {
                **identity,
                "metric": out["metric"],
                "metric_error": out["metric_error"],
                "metric_error_val": out.get("metric_error_val"),
                "fit_seconds": out["time_train_s"],
                "inference_seconds": out["time_infer_s"],
                "wall_seconds": elapsed,
                "model_info": info,
                "memory_usage": out["memory_usage"],
                "experiment_metadata": out["experiment_metadata"],
            },
            **arrays,
        )

    @staticmethod
    def _host():
        import os
        import subprocess

        import psutil
        import torch

        gpu = torch.cuda.get_device_properties(0) if torch.cuda.is_available() else None
        driver = None
        if gpu is not None:
            try:
                result = subprocess.run(
                    ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                    check=False,
                )
                driver = result.stdout.strip() or None
            except (OSError, subprocess.TimeoutExpired):
                pass
        return {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "gpu_memory_bytes": gpu.total_memory if gpu is not None else None,
            "driver": driver,
            "cpu": platform.processor(),
            "cpu_count": os.cpu_count(),
            "ram_bytes": psutil.virtual_memory().total,
        }


def planned_jobs(study: StudyConfig) -> dict:
    tunable = sum(model_spec(name).evaluation != "fixed" for name in study.models)
    fixed = len(study.models) - tunable
    return {
        "datasets": len(study.datasets),
        "methods": len(study.models),
        "configuration_jobs": len(study.datasets)
        * (tunable * (study.random_configurations + 1) + fixed),
        "inner_folds": study.inner_folds,
        "random_configurations": study.random_configurations,
        "protocol": "TabArena Lite adaptation on custom datasets; one outer split",
        "resources": asdict(study.runtime),
        "note": (
            "KNN uses child LOO on IID tasks; grouped tasks use the declared folds. "
            "TabPFN is fixed."
        ),
    }
