"""Build the final method x dataset report from the freshest run of every cell.

python scripts/aggregate_final.py [--runs results/runs/thesis-lite] [--out results/final]

Each (dataset, method) cell is taken from the newest study directory that contains it,
ordered by the first recorded job attempt. The newest run must be complete with a
successful default; older runs are never used as a fallback. Selection is the unchanged
reporting logic applied to that cell's jobs, so a cell's rows equal `thesis report`.
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from collections import Counter
from pathlib import Path

import pandas as pd

from tabular_thesis.experiments.artifacts import atomic_json
from tabular_thesis.experiments.configuration import ConfigResolver, default_config_directory
from tabular_thesis.experiments.reporting import report


def started(directory: Path) -> float:
    times = [
        json.loads(path.read_text())["started_at_unix"]
        for path in directory.glob("jobs/*/attempts/*.json")
    ]
    if not times:
        raise ValueError(f"No job attempts: {directory}")
    return min(times)


def cell_jobs(directory: Path, dataset: str, model: str) -> list[str]:
    jobs = json.loads((directory / "index.json").read_text())["jobs"]
    return [job for job in jobs if job.rsplit("-", 1)[0] == f"jobs/{dataset}-{model}"]


def truncated_folds(job: Path) -> int:
    record = job / "records" / "result.json"
    if not record.exists():
        return 0
    children = json.loads(record.read_text())["model_info"].get("children_info", {})
    return sum(
        bool(child.get("hyperparameters_fit", {}).get("budget_truncated"))
        for child in children.values()
    )


def cell_report(directory: Path, dataset: str, model: str, jobs: list[str]) -> dict:
    """Run the study reporter on a view of one cell; job directories are symlinked."""
    with tempfile.TemporaryDirectory() as temporary:
        view = Path(temporary)
        (view / "jobs").mkdir()
        shutil.copy(directory / "manifest.json", view / "manifest.json")
        atomic_json(view / "index.json", {"jobs": jobs})
        for job in jobs:
            (view / job).symlink_to((directory / job).resolve(), target_is_directory=True)
        report(view)
        return json.loads((view / "report.json").read_text())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs", type=Path, default=Path("results/runs/thesis-lite"))
    parser.add_argument("--out", type=Path, default=Path("results/final"))
    args = parser.parse_args()

    study = ConfigResolver(default_config_directory()).study()
    directories = []
    for directory in args.runs.iterdir():
        if (directory / "manifest.json").exists() and (directory / "index.json").exists():
            identity = json.loads((directory / "manifest.json").read_text())["identity"]
            directories.append((started(directory), directory, identity))
    directories.sort(key=lambda item: item[0], reverse=True)

    rows, failures, sources, superseded = [], [], [], []
    for dataset in study.datasets:
        for model in study.models:
            candidates = [
                (time, directory, identity)
                for time, directory, identity in directories
                if dataset in identity["configuration"]["study"]["datasets"]
                and model in identity["configuration"]["study"]["models"]
            ]
            if not candidates:
                raise ValueError(f"No run contains {dataset}/{model}")
            time, directory, identity = candidates[0]
            superseded += [
                {"dataset": dataset, "model": model, "directory": other.name}
                for _, other, _ in candidates[1:]
            ]
            jobs = cell_jobs(directory, dataset, model)
            # Raises on incomplete cells or a failed default; no silent fallback to older runs.
            cell = cell_report(directory, dataset, model, jobs)
            truncation = {job: truncated_folds(directory / job) for job in jobs}
            states = Counter(
                json.loads((directory / job / "status.json").read_text())["state"] for job in jobs
            )
            source = {
                "dataset": dataset,
                "model": model,
                "directory": directory.name,
                "code_source": identity["code"]["source"],
                "data_sha256": identity["sources"][dataset]["data_sha256"],
                "split": json.loads((directory / jobs[0] / "manifest.json").read_text())[
                    "identity"
                ]["split"],
                "started_at_unix": time,
                "jobs": len(jobs),
                "completed": states["completed"],
                "failed": states["failed"],
                "truncated_configurations": sum(n > 0 for n in truncation.values()),
                "truncated_folds": sum(truncation.values()),
            }
            sources.append(source)
            for row in cell["rows"]:
                rows.append(
                    {
                        **row,
                        "source_directory": directory.name,
                        "code_source": source["code_source"],
                        "truncated_configurations": source["truncated_configurations"],
                    }
                )
            failures += [
                {**failure, "directory": directory.name}
                for failure in cell["failed_configurations"]
            ]

    for dataset in study.datasets:
        cells = [s for s in sources if s["dataset"] == dataset]
        if len({s["data_sha256"] for s in cells}) != 1 or len({s["split"] for s in cells}) != 1:
            raise ValueError(f"Methods on {dataset} do not share one data/split identity")

    args.out.mkdir(parents=True, exist_ok=True)
    atomic_json(
        args.out / "final_report.json",
        {
            "rows": rows,
            "sources": sources,
            "superseded_runs": superseded,
            "failed_configurations": failures,
            "selection": "freshest complete run per cell; OOF-only tuning; outer test for metrics",
            "inference_unit": "dataset; inner folds are dependent and are not independent samples",
        },
    )
    pd.DataFrame(rows).to_csv(args.out / "final_report.csv", index=False)
    pd.DataFrame(sources).to_csv(args.out / "final_sources.csv", index=False)
    print(
        f"{len(rows)} rows from {len(sources)} cells "
        f"({len({s['directory'] for s in sources})} directories): {args.out / 'final_report.csv'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
