"""Small, shared command surface for local and external experiment workers."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import sys
from pathlib import Path

from tabular_thesis.experiments.configuration import ConfigResolver, default_config_directory
from tabular_thesis.experiments.runtime import FitContext, configure_caches


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-dir", type=Path, default=default_config_directory())
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Typed YAML override; repeat for multiple values",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("doctor", "prepare", "weights", "plan", "run", "smoke"):
        commands.add_parser(command)
    reporting = commands.add_parser("report")
    reporting.add_argument("directory", type=Path, help="Study directory printed by run")
    args = parser.parse_args(argv)
    try:
        resolver = ConfigResolver(args.config_dir)
        study = resolver.study(args.set)
        resolver.resolved_models(study)
        configure_caches(Path(study.paths.data) / "cache")
        if args.command == "doctor":
            FitContext(device=study.runtime.device, num_cpus=study.runtime.num_cpus).activate()
            import torch

            print(
                json.dumps(
                    {
                        "python": platform.python_version(),
                        "torch": torch.__version__,
                        "cuda_runtime": torch.version.cuda,
                        "device": study.runtime.device,
                        "gpu": torch.cuda.get_device_name(0)
                        if study.runtime.device == "cuda"
                        else None,
                        "tabarena": importlib.metadata.version("tabarena"),
                    },
                    indent=2,
                )
            )
        elif args.command == "prepare":
            from tabular_thesis.data.tasks import DatasetRepository

            repository = DatasetRepository(Path(study.paths.data))
            for name in study.datasets:
                task = repository.prepare(resolver.dataset(name))
                task.outer_split(study.outer_seed, study.inner_folds)
                print(
                    f"{name}: {len(task.y):,} rows, {task.X.shape[1]} features, "
                    f"{len(task.classes)} classes; verified"
                )
        elif args.command == "weights":
            from tabular_thesis.experiments.weights import prepare_weights

            models = resolver.resolved_models(study)
            if "tabpfn" not in models:
                raise ValueError("Select tabpfn in study.models to prepare its checkpoint")
            print(prepare_weights(Path(models["tabpfn"].params["checkpoint"])))
        elif args.command == "plan":
            from tabular_thesis.experiments.runner import planned_jobs

            print(
                json.dumps({**planned_jobs(study), "resolved": resolver.snapshot(study)}, indent=2)
            )
        elif args.command == "report":
            from tabular_thesis.experiments.reporting import report

            print(f"{len(report(args.directory))} result rows: {args.directory / 'report.csv'}")
        else:
            from tabular_thesis.experiments.runner import ExperimentRunner

            tasks = None
            if args.command == "smoke":
                import numpy as np
                import pandas as pd

                from tabular_thesis.data.tasks import TaskSpec

                study.name = "smoke"
                study.models, study.datasets = ["logistic_regression", "mlp"], ["synthetic"]
                study.model_overrides = {"mlp": {"epochs": 2, "hidden_dims": [16]}}
                study.random_configurations = 1
                # Keep all eight folds: smoke exercises the actual validation path.
                rng = np.random.RandomState(42)
                X = pd.DataFrame(rng.normal(size=(180, 4)), columns=list("abcd"))
                X["category"] = pd.Categorical(np.where(X.a > 0, "yes", "no"))
                y = (X.a + X.b > 0).astype(int).to_numpy()
                tasks = [
                    TaskSpec("synthetic", X, y, ["0", "1"], {"generator": "normal-v1", "seed": 42})
                ]
            directory = ExperimentRunner(resolver, study).run(tasks=tasks)
            from tabular_thesis.experiments.reporting import report

            report(directory)
            print(directory)
        return 0
    except (Exception, KeyboardInterrupt) as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
