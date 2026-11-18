"""Expand and execute approved A/B plans without automatic test reporting."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from filelock import FileLock
from omegaconf import OmegaConf

from tabular_thesis.data.tasks import DatasetRepository
from tabular_thesis.experiments.artifacts import atomic_json, file_sha256
from tabular_thesis.experiments.configuration import ConfigResolver, default_config_directory
from tabular_thesis.experiments.mechanisms import nested_subset
from tabular_thesis.experiments.runtime import configure_caches
from tabular_thesis.models.specification import model_spec


@dataclass
class Method:
    dataset: str = ""
    model: str = ""
    rare_code: int = 1
    source: dict[str, Any] = field(default_factory=dict)
    params: dict[str, Any] | None = None
    blocked_reason: str = ""


@dataclass
class Plan:
    name: str = ""
    seeds: list[int] = field(default_factory=list)
    time_limit: int = 1000000
    conditions: list[str] = field(default_factory=list)
    control_condition: str = "weighted_unweighted_selection"
    methods: list[Method] = field(default_factory=list)
    fractions: list[float] = field(default_factory=list)
    base: dict[str, Any] = field(default_factory=dict)
    source_configuration_index: int = 21
    variants: dict[str, Any] = field(default_factory=dict)
    batch_size: dict[str, int] = field(default_factory=dict)
    step_budget: dict[str, int] = field(default_factory=dict)
    expected_epochs: dict[str, list[int]] = field(default_factory=dict)


def load_plan(path: Path) -> Plan:
    plan = OmegaConf.to_object(OmegaConf.merge(OmegaConf.structured(Plan), OmegaConf.load(path)))
    if plan.name not in {"study-a", "study-b"} or plan.seeds != [0, 10, 20]:
        raise ValueError("Expected study-a/b and paired seeds 0/10/20")
    if plan.time_limit != 1000000:
        raise ValueError("A/B requires the declared 1000000-second technical limit")
    if plan.name == "study-a":
        pairs = [(m.dataset, m.model) for m in plan.methods]
        if len(pairs) != len(set(pairs)) or plan.conditions != ["unweighted", "weighted"]:
            raise ValueError("Duplicate methods or unexpected A conditions")
        for method in plan.methods:
            if (
                method.model not in {"mlp", "catboost"}
                or method.dataset not in {"amazon", "credit_fraud", "santander"}
                or method.rare_code != (0 if method.dataset == "amazon" else 1)
            ):
                raise ValueError("Invalid A method, dataset or semantic rare class")
            if method.params is not None:
                model_spec(method.model).validate_parameters(method.params)
                if not method.source or method.blocked_reason:
                    raise ValueError("Available A configuration needs unambiguous provenance")
    else:
        if plan.fractions != [0.25, 0.5, 1.0] or set(plan.variants) != {
            "full",
            "no_dropout",
            "no_weight_decay",
            "no_both",
        }:
            raise ValueError("B requires the accepted fractions and four variants")
        model_spec("mlp").validate_parameters(plan.base)
        if set(plan.batch_size) != {"german_credit", "ionosphere", "bioresponse"}:
            raise ValueError("Unexpected B datasets")
    return plan


def expand(plan: Plan, only: str | None = None, control: str | None = None) -> list[dict]:
    rows = []
    if plan.name == "study-a":
        for method in plan.methods:
            pair = f"{method.dataset}/{method.model}"
            if (only and pair != only) or (control and pair != control):
                continue
            variants = [plan.control_condition] if control else plan.conditions
            for variant in variants:
                for seed in plan.seeds:
                    params = dict(method.params or {})
                    params["_record_history"] = True
                    if variant != "unweighted":
                        params["_class_weighting"] = "balanced"
                    if variant == plan.control_condition:
                        params["_selection_loss"] = "unweighted"
                    rows.append(
                        dict(
                            dataset=method.dataset,
                            model=method.model,
                            variant=variant,
                            seed=seed,
                            fraction=1.0,
                            rare_code=method.rare_code,
                            params=params,
                            source=method.source,
                            blocked=method.blocked_reason if method.params is None else None,
                        )
                    )
    else:
        if control:
            raise ValueError("Control is defined only for A")
        for dataset, batch in plan.batch_size.items():
            if only and only != f"{dataset}/mlp":
                continue
            for i, fraction in enumerate(plan.fractions):
                for variant, changes in plan.variants.items():
                    for seed in plan.seeds:
                        epochs = plan.expected_epochs[dataset][i]
                        rows.append(
                            dict(
                                dataset=dataset,
                                model="mlp",
                                variant=variant,
                                seed=seed,
                                fraction=fraction,
                                blocked=None,
                                params={
                                    **plan.base,
                                    **changes,
                                    "batch_size": batch,
                                    "epochs": epochs,
                                    "patience": epochs,
                                    "_train_fraction": fraction,
                                    "_record_history": True,
                                },
                                target_steps=plan.step_budget[dataset],
                            )
                        )
    if not rows:
        raise ValueError("Selection contains no conditions")
    for row in rows:
        row["key"] = "/".join(
            str(row[k]) for k in ("dataset", "model", "variant", "fraction", "seed")
        )
    return rows


def verify_source(row: dict, root: Path) -> None:
    source = row.get("source")
    if not source:
        raise ValueError("No source run declared; B selection evidence requires separate audit")
    job = (
        root
        / source["run"]
        / "jobs"
        / (f"{row['dataset']}-{row['model']}-{source['configuration_index']:03d}")
    )
    identity = json.loads((job / "manifest.json").read_text())["identity"]
    expected = {k: v for k, v in row["params"].items() if not k.startswith("_")}
    if identity["parameters"] != expected or identity["model"] != row["model"]:
        raise ValueError(f"Source parameters differ: {job}")


def budget_from_folds(y, folds, fraction, seed, batch, target) -> list[dict]:
    evidence = []
    for i, (train, validation) in enumerate(folds):
        selected = np.asarray(train)[nested_subset(y[train], fraction, seed + i)]
        steps = math.ceil(len(selected) / batch)
        epochs = math.ceil(target / steps)
        evidence.append(
            dict(
                fold=i,
                seed=seed + i,
                rows=len(selected),
                class_counts=np.bincount(y[selected]).tolist(),
                steps_per_epoch=steps,
                epochs=epochs,
                optimizer_steps=steps * epochs,
            )
        )
    if len({r["steps_per_epoch"] for r in evidence}) != 1:
        raise ValueError("Steps per epoch differ between folds; qualify the plan before training")
    return evidence


def preflight_b(row, task, study):
    from autogluon.common.utils.cv_splitter import CVSplitter

    outer, _ = task.outer_split(study.outer_seed, study.inner_folds)
    # Pinned AutoGluon trainer: level 1 + trainer seed 0. Actual folds are checked after fit.
    folds = CVSplitter(n_splits=8, n_repeats=1, random_state=1, stratify=True).split(
        X=task.X.iloc[outer], y=task.y[outer]
    )
    evidence = budget_from_folds(
        task.y[outer],
        folds,
        row["fraction"],
        row["seed"],
        row["params"]["batch_size"],
        row["target_steps"],
    )
    if any(r["epochs"] != row["params"]["epochs"] for r in evidence):
        raise ValueError("Actual fold counts disagree with the declared epoch plan")
    return evidence


def overrides(row, plan, data, results, device):
    values = dict(
        name=plan.name,
        models=[row["model"]],
        datasets=[row["dataset"]],
        random_configurations=0,
        inner_folds=8,
        outer_seed=42,
        training_seed=row["seed"],
    )
    values.update(
        {
            "runtime.configuration_time_limit": plan.time_limit,
            "runtime.device": device,
            "paths.data": str(data),
            "paths.results": str(results),
        }
    )
    values.update({f"model_overrides.{row['model']}.{k}": v for k, v in row["params"].items()})
    return [f"{k}={json.dumps(v)}" for k, v in values.items()]


def check_control(path, pair, registry):
    decision = json.loads(path.read_text())
    if decision["registry_sha256"] != file_sha256(registry):
        # Adding control runs changes the registry, but its weighted sources must remain identical.
        current = json.loads(registry.read_text())["conditions"]
        if any(current.get(k) != v for k, v in decision["weighted_sources"].items()):
            raise ValueError("Control decision is stale")
    if decision["decisions"].get(pair, {}).get("trigger") is not True:
        raise ValueError("Control requires a complete, positive 12-of-24 decision")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--only", help="DATASET/MODEL")
    parser.add_argument("--seed", type=int, choices=[0, 10, 20])
    parser.add_argument("--variant", help="Restrict to one declared variant for qualification")
    parser.add_argument("--fraction", type=float, choices=[0.25, 0.5, 1.0])
    parser.add_argument("--control", help="DATASET/MODEL")
    parser.add_argument("--control-decision", type=Path)
    parser.add_argument(
        "--control-source",
        type=Path,
        help="Registry the control decision was made on (default: this results root)",
    )
    parser.add_argument(
        "--verify-source", type=Path, help="Main study directory containing run hashes"
    )
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    args = parser.parse_args()
    plan = load_plan(args.plan)
    rows = expand(plan, args.only, args.control)
    rows = [
        row
        for row in rows
        if (args.seed is None or row["seed"] == args.seed)
        and (args.variant is None or row["variant"] == args.variant)
        and (args.fraction is None or row["fraction"] == args.fraction)
    ]
    if not rows:
        parser.error("Filters select no declared conditions")
    if args.dry_run:
        print(
            json.dumps(
                {
                    "conditions": rows,
                    "nominal_fits": len(rows) * 8,
                    "blocked_conditions": sum(bool(r["blocked"]) for r in rows),
                },
                indent=2,
            )
        )
        return
    configure_caches(args.data / "cache")
    from tabular_thesis.experiments.runner import ExperimentRunner

    resolver = ConfigResolver(default_config_directory())
    registry = args.results / "runs" / plan.name / "conditions.json"
    provenance = {
        "plan_sha256": file_sha256(args.plan),
        "launcher_sha256": file_sha256(Path(__file__)),
        "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    if args.control:
        if args.control_decision is None:
            parser.error("--control requires --control-decision")
        check_control(args.control_decision, args.control, args.control_source or registry)
    registry.parent.mkdir(parents=True, exist_ok=True)
    failures = []
    with FileLock(str(registry) + ".lock", timeout=0):
        saved = (
            json.loads(registry.read_text())
            if registry.exists()
            else {"name": plan.name, "provenance": provenance, "conditions": {}}
        )
        if saved["provenance"] != provenance:
            raise ValueError("Registry provenance changed; use a separate results directory")
        for row in rows:
            if row["blocked"]:
                saved["conditions"][row["key"]] = {**row, "state": "blocked"}
                failures.append(row["key"])
                atomic_json(registry, saved)
                continue
            runner = None
            entry = {**row, "state": "running"}
            previous = saved["conditions"].get(row["key"])
            try:
                if args.verify_source:
                    verify_source(row, args.verify_source)
                study = resolver.study(overrides(row, plan, args.data, args.results, args.device))
                task = DatasetRepository(args.data).load(resolver.dataset(row["dataset"]))
                if plan.name == "study-b":
                    entry["budget_evidence"] = preflight_b(row, task, study)
                # elif task.classes[row["rare_code"]] != str(row["rare_code"]):
                elif float(task.classes[row["rare_code"]]) != row["rare_code"]:
                    raise ValueError("Semantic rare-class mapping differs from the approved plan")
                runner = ExperimentRunner(resolver, study)
                if previous and previous.get("directory"):
                    runner.expected_directory = previous["directory"]
                directory = runner.run()
                if plan.name == "study-b":
                    record = json.loads(
                        (
                            directory / "jobs" / f"{row['dataset']}-mlp-000" / "records/result.json"
                        ).read_text()
                    )
                    actual = [
                        child["hyperparameters_fit"]
                        for child in record["model_info"]["children_info"].values()
                    ]
                    for child, expected in zip(actual, entry["budget_evidence"], strict=True):
                        if any(
                            child[key] != expected[key]
                            for key in ["steps_per_epoch", "optimizer_steps"]
                        ):
                            raise ValueError("Actual folds differ from the budget preflight")
                if previous and previous.get("directory") not in {None, str(directory)}:
                    raise ValueError("Condition resolves to a different run; use separate results")
                entry.update(state="completed", directory=str(directory))
            except Exception as error:
                entry.update(state="failed", error=str(error))
                if runner is not None and hasattr(runner, "directory"):
                    entry["directory"] = str(runner.directory)
                failures.append(row["key"])
            saved["conditions"][row["key"]] = entry
            atomic_json(registry, saved)
    print(registry)
    if failures:
        raise SystemExit(
            f"{len(failures)} blocked/failed conditions; independent work was preserved"
        )


if __name__ == "__main__":
    main()
