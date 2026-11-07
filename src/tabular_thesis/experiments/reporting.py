"""D/T/T+E selection from aligned cached OOF predictions only."""

from __future__ import annotations

import json
import os
import tempfile
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from tabular_thesis.evaluation.metrics import compute_metrics
from tabular_thesis.experiments.artifacts import ArtifactStore, atomic_json, file_sha256


def attempt_seconds(job: Path, fallback: float) -> float | None:
    attempts = [json.loads(path.read_text()) for path in (job / "attempts").glob("*.json")]
    if not attempts:
        return fallback
    if any(attempt["state"] == "running" for attempt in attempts):
        return None  # An interrupted attempt has no trustworthy ending timestamp.
    return sum(attempt["wall_seconds"] for attempt in attempts)


def report(directory: Path) -> pd.DataFrame:
    store, groups, failures = ArtifactStore(directory), {}, []
    manifest = json.loads((directory / "manifest.json").read_text())
    for relative in json.loads((directory / "index.json").read_text())["jobs"]:
        job = directory / relative
        if not store.complete(job):
            status = job / "status.json"
            if status.exists() and json.loads(status.read_text())["state"] == "failed":
                failures.append(
                    {
                        "job": relative,
                        **json.loads(status.read_text()),
                        **json.loads((job / "failure.json").read_text()),
                    }
                )
                failures[-1]["attempt_wall_seconds"] = attempt_seconds(
                    job, failures[-1]["wall_seconds"]
                )
                identity = json.loads((job / "manifest.json").read_text())["identity"]
                if identity["configuration_index"] == 0:
                    raise ValueError(f"Required default configuration failed: {job}")
                continue
            raise ValueError(f"Study is incomplete: {job}")
        record = json.loads((job / "records" / "result.json").read_text())
        record["attempt_wall_seconds"] = attempt_seconds(job, record["wall_seconds"])
        path = job / record["predictions"]
        if file_sha256(path) != record["prediction_sha256"]:
            raise ValueError(f"Corrupt predictions: {path}")
        groups.setdefault((record["dataset"], record["model"]), []).append((record, path))
    rows = []
    for (dataset, model), entries in groups.items():
        # Bound prediction memory by one model/dataset search, not the whole study.
        results = []
        for record, path in entries:
            with np.load(path, allow_pickle=False) as cached:
                results.append((record, {name: cached[name] for name in cached.files}))
        results.sort(key=lambda result: result[0]["configuration_index"])
        default, first = results[0]
        selection_start = time.monotonic()
        predictions = [("default", first["test_probabilities"], [1.0] + [0.0] * (len(results) - 1))]
        if "oof_y" in first:
            for _, arrays in results:
                for name in ("oof_y", "oof_indices", "test_y", "test_indices"):
                    if not np.array_equal(first[name], arrays[name]):
                        raise ValueError(
                            "Cannot combine configurations with different validation/test rows"
                        )
            from autogluon.core.metrics import get_metric
            from autogluon.core.models.greedy_ensemble.ensemble_selection import EnsembleSelection

            binary = len(default["classes"]) == 2
            scorer = get_metric(default["metric"])
            oof = [
                arrays["oof_probabilities"][:, 1] if binary else arrays["oof_probabilities"]
                for _, arrays in results
            ]
            scores = [float(scorer(first["oof_y"], values)) for values in oof]
            best = int(np.argmax(scores))
            best_weights = np.eye(len(results))[best].tolist()
            predictions.append(("tuned", results[best][1]["test_probabilities"], best_weights))
            ensemble = EnsembleSelection(
                ensemble_size=100,
                problem_type="binary" if binary else "multiclass",
                metric=scorer,
                random_state=np.random.RandomState(42),
            )
            ensemble.fit(predictions=oof, labels=first["oof_y"])
            weights = np.asarray(ensemble.weights_)
            combined = sum(
                weight * arrays["test_probabilities"]
                for weight, (_, arrays) in zip(weights, results, strict=True)
            )
            predictions.append(("tuned_ensemble", combined, weights.tolist()))
        selection_seconds = time.monotonic() - selection_start
        method_failures = [f for f in failures if f["job"].startswith(f"jobs/{dataset}-{model}-")]
        costs = [r["attempt_wall_seconds"] for r, _ in results] + [
            f["attempt_wall_seconds"] for f in method_failures
        ]
        for condition, probabilities, weights in predictions:
            metrics = asdict(
                compute_metrics(
                    first["test_y"],
                    probabilities.argmax(axis=1),
                    probabilities,
                    len(default["classes"]),
                )
            )
            # These legacy fields have no measured value for cached selection.
            metrics = {key: value for key, value in metrics.items() if not key.endswith("time_sec")}
            row = {
                "dataset": dataset,
                "model": model,
                "condition": condition,
                "classes": default["classes"],
                "primary_metric": default["metric"],
                "weights": weights,
                "configuration_indices": [r["configuration_index"] for r, _ in results],
                "successful_configurations": len(results),
                "failed_configurations": len(method_failures),
                "search_fit_seconds": sum(r["fit_seconds"] for r, _ in results),
                "search_wall_seconds": None if any(cost is None for cost in costs) else sum(costs),
                "selection_seconds": selection_seconds,
                "selected_fit_seconds": sum(
                    r["fit_seconds"] for w, (r, _) in zip(weights, results, strict=True) if w > 0
                ),
                "selected_inference_seconds": sum(
                    r["inference_seconds"]
                    for w, (r, _) in zip(weights, results, strict=True)
                    if w > 0
                ),
                **metrics,
            }
            rows.append(row)
        if "oof_y" in first:
            del oof
    atomic_json(
        directory / "report.json",
        {
            "study_identity": manifest["identity"],
            "rows": rows,
            "failed_configurations": failures,
            "inference_unit": "dataset; inner folds are dependent and are not independent samples",
            "selection": "OOF only; outer test used only for final metrics",
        },
    )
    frame = pd.DataFrame(rows)
    handle, temporary = tempfile.mkstemp(dir=directory, suffix=".csv")
    os.close(handle)
    try:
        frame.to_csv(temporary, index=False)
        os.replace(temporary, directory / "report.csv")
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return frame
