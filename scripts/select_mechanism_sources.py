"""Audit A's T or B's configuration-21 selection using only cached OOF arrays."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import log_loss, roc_auc_score

from tabular_thesis.experiments.artifacts import atomic_json, file_sha256

B_SOURCES = {
    "adult",
    "credit_fraud",
    "higgs",
    "amazon",
    "santander",
    "online_shoppers",
    "diabetes",
    "helena",
}


def candidates(directory, model, allow_failed=False):
    manifest = json.loads((directory / "manifest.json").read_text())
    sources = manifest["identity"]["sources"]
    if len(sources) != 1:
        raise ValueError("Supply one dataset run directory per source")
    dataset = next(iter(sources))
    values = []
    for index in range(26):
        job = directory / "jobs" / f"{dataset}-{model}-{index:03d}"
        status = json.loads((job / "status.json").read_text())
        if status["state"] == "failed" and allow_failed and index != 0:
            values.append(dict(index=index, error=None, state="failed"))
            continue
        if status["state"] != "completed":
            raise ValueError("Source audit requires completed candidates or admissible A failures")
        record_path = job / "records/result.json"
        if status["artifacts"]["records/result.json"] != file_sha256(record_path):
            raise ValueError("Corrupt source record")
        record = json.loads(record_path.read_text())
        prediction_path = job / record["predictions"]
        if record["prediction_sha256"] != file_sha256(prediction_path):
            raise ValueError("Corrupt source predictions")
        with np.load(prediction_path, allow_pickle=False) as arrays:
            # The source archive also contains test arrays; never request those members.
            y, p = arrays["oof_y"], arrays["oof_probabilities"]
        error = 1 - roc_auc_score(y, p[:, 1]) if p.shape[1] == 2 else log_loss(y, p)
        values.append(
            dict(
                index=index,
                error=float(error),
                params=record["parameters"],
                manifest_sha256=file_sha256(job / "manifest.json"),
            )
        )
    return dataset, dict(run=directory.name, candidates=values, source=sources[dataset])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["a", "b"])
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--model", choices=["mlp", "catboost"], default="mlp")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = [candidates(path, args.model, allow_failed=args.mode == "a") for path in args.runs]
    evidence = dict(rows)
    if len(rows) != len(evidence):
        raise ValueError("Duplicate source dataset")
    if args.mode == "a":
        result = {
            name: {
                **item,
                "selected_index": min(
                    [c for c in item["candidates"] if c["error"] is not None],
                    key=lambda c: (c["error"], c["index"]),
                )["index"],
            }
            for name, item in evidence.items()
        }
    else:
        if args.model != "mlp" or set(evidence) != B_SOURCES:
            raise ValueError("B needs the eight declared non-B, non-Covertype MLP datasets")
        reference = rows[0][1]["candidates"]
        if any(
            [c["params"] for c in item["candidates"]] != [c["params"] for c in reference]
            for item in evidence.values()
        ):
            raise ValueError("Source candidates differ across datasets")
        ranks = {
            name: rankdata([c["error"] for c in item["candidates"]], method="average")
            for name, item in evidence.items()
        }
        mean = np.mean(list(ranks.values()), axis=0)
        minimum_decay = 0.001
        eligible = [
            c["index"]
            for c in reference
            if c["params"]["dropout"] >= 0.2 and c["params"]["weight_decay"] >= minimum_decay
        ]
        if len(eligible) < 3:
            minimum_decay = 0.0005
            eligible = [
                c["index"]
                for c in reference
                if c["params"]["dropout"] >= 0.2 and c["params"]["weight_decay"] >= minimum_decay
            ]
        if not eligible:
            raise ValueError("No admissible B candidate")
        selected = min(eligible, key=lambda i: (mean[i], i))
        result = dict(
            sources=evidence,
            per_dataset_ranks={k: v.tolist() for k, v in ranks.items()},
            mean_ranks=mean.tolist(),
            minimum_decay=minimum_decay,
            eligible=eligible,
            selected_index=selected,
            agrees_with_configuration_21=selected == 21,
            limitation="Reconstruction cannot establish when the historical rule was frozen",
        )
    atomic_json(args.output, dict(script_sha256=file_sha256(Path(__file__)), evidence=result))


if __name__ == "__main__":
    main()
