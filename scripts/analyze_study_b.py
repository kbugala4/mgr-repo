"""B: fold learning diagnostics and explicitly frozen paired test contrasts."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
from mechanism_analysis import (
    ANALYSIS_VERSION,
    SEEDS,
    aligned,
    authorize_test,
    histories,
    load_registry,
    paired_bootstrap,
    read_predictions,
    scores,
)

from tabular_thesis.experiments.artifacts import atomic_json, file_sha256

CONTRASTS = [("full", "no_dropout"), ("full", "no_weight_decay"), ("no_dropout", "no_both")]


def plateau(history):
    """Endpoint relative absolute change over the final >=10% of recorded updates."""
    if len(history) < 2:
        return None
    end = history[-1]
    candidates = [row for row in history if row["optimizer_steps"] <= 0.9 * end["optimizer_steps"]]
    start = candidates[-1] if candidates else history[0]
    a, b = start["train_loss"], end["train_loss"]
    if a is None or b is None:
        return None
    change = abs(b - a) / abs(a) if a else (0.0 if b == 0 else None)
    return dict(
        start_step=start["optimizer_steps"],
        end_step=end["optimizer_steps"],
        relative_change=change,
        plateau=change < 0.01 if change is not None else False,
    )


def summarize(values):
    if any(v is None for v in values):
        return dict(median=None, range=None, undefined=sum(v is None for v in values))
    return dict(median=float(np.median(values)), range=[min(values), max(values)], undefined=0)


def history_analysis(registry):
    result = {"folds": [], "summaries": [], "curves": [], "norm_ratios": [], "missing": []}
    cached = {}
    for key, row in registry["conditions"].items():
        try:
            children = histories(row)
            gaps = []
            for fold, child in enumerate(children):
                history = child["history"]
                if len(history) != row["params"]["epochs"] or child["stop_reason"] != "epochs":
                    raise ValueError("B requires the complete epoch budget")
                expected_steps = math.ceil(child["training_rows"] / row["params"]["batch_size"])
                if child["steps_per_epoch"] != expected_steps or any(
                    h["optimizer_steps"] != h["epoch"] * expected_steps for h in history
                ):
                    raise ValueError("Actual steps disagree with fold/batch sizes")
                if len(history) != math.ceil(row["target_steps"] / expected_steps):
                    raise ValueError("Actual epoch budget differs from rounded target")
                best, last = history[child["best_epoch"] - 1], history[-1]
                if any(h["val_loss"] is None for h in history):
                    raise ValueError("Nonfinite validation history")
                if child["best_epoch"] != int(np.argmin([h["val_loss"] for h in history])) + 1:
                    raise ValueError("Selected epoch differs from the first minimum")

                def gap(h):
                    if h["val_loss"] is None or h["train_loss"] is None:
                        return None
                    return h["val_loss"] - h["train_loss"]

                gaps.append(gap(last))
                result["folds"].append(
                    dict(
                        key=key,
                        fold=fold,
                        final_gap=gap(last),
                        selected_gap=gap(best),
                        plateau=plateau(history),
                        **child,
                    )
                )
            cached[(row["dataset"], row["fraction"], row["variant"], row["seed"])] = children
            result["summaries"].append(
                dict(
                    key=key,
                    final_gap=summarize(gaps),
                    selected_gap=summarize(
                        [gap(c["history"][c["best_epoch"] - 1]) for c in children]
                    ),
                    selected_epoch=summarize([c["best_epoch"] for c in children]),
                    final_train_loss=summarize([c["history"][-1]["train_loss"] for c in children]),
                )
            )
            for epoch in range(row["params"]["epochs"]):
                snapshots = [c["history"][epoch] for c in children]
                result["curves"].append(
                    dict(
                        key=key,
                        epoch=epoch + 1,
                        optimizer_steps=snapshots[0]["optimizer_steps"],
                        **{
                            field: summarize([h[field] for h in snapshots])
                            for field in ["train_loss", "val_loss", "param_norm"]
                        },
                        gap=summarize([gap(h) for h in snapshots]),
                    )
                )
        except (ValueError, KeyError, OSError) as error:
            result["missing"].append(dict(key=key, reason=str(error)))
    for (dataset, fraction, variant, seed), left in cached.items():
        right_variant = {"full": "no_weight_decay", "no_dropout": "no_both"}.get(variant)
        if right_variant is None:
            continue
        right = cached.get((dataset, fraction, right_variant, seed))
        if right is None:
            result["missing"].append(
                dict(
                    dataset=dataset,
                    fraction=fraction,
                    seed=seed,
                    reason=f"Missing norm pair {variant}/{right_variant}",
                )
            )
            continue
        for fold, (a, b) in enumerate(zip(left, right, strict=True)):
            if (
                a["training_row_ids"] != b["training_row_ids"]
                or a["validation_row_ids"] != b["validation_row_ids"]
            ):
                raise ValueError("Matched variants use different populations")
            for state in ["selected", "final"]:
                ha = a["history"][a["best_epoch"] - 1] if state == "selected" else a["history"][-1]
                hb = b["history"][b["best_epoch"] - 1] if state == "selected" else b["history"][-1]
                ratio = (
                    ha["param_norm"] / hb["param_norm"]
                    if (ha["param_norm"] is not None and hb["param_norm"] not in {None, 0})
                    else None
                )
                result["norm_ratios"].append(
                    dict(
                        dataset=dataset,
                        fraction=fraction,
                        seed=seed,
                        fold=fold,
                        state=state,
                        contrast=f"{variant}/{right_variant}",
                        ratio=ratio,
                        diagnostic_interval=0.98 <= ratio <= 1.0 if ratio is not None else None,
                        left_epoch=ha["epoch"],
                        right_epoch=hb["epoch"],
                        no_gradient_factor=(1 - a["learning_rate"] * a["weight_decay"])
                        ** ha["optimizer_steps"],
                    )
                )
    # All four variants share subsets; fractions nest within each seed/fold.
    for (dataset, fraction, variant, seed), children in cached.items():
        full = cached.get((dataset, fraction, "full", seed))
        larger = [
            v
            for k, v in cached.items()
            if k[0] == dataset and k[1] > fraction and k[2:] == (variant, seed)
        ]
        for i, child in enumerate(children):
            if full and child["training_row_ids"] != full[i]["training_row_ids"]:
                raise ValueError("Variants have different training subsets")
            if any(
                not set(child["training_row_ids"]) <= set(v[i]["training_row_ids"]) for v in larger
            ):
                raise ValueError("Training subsets are not nested")
    return result


def test_analysis(registry):
    result = {"metrics": [], "contrasts": [], "missing": []}
    rows = {
        (r["dataset"], r["fraction"], r["variant"], r["seed"]): r
        for r in registry["conditions"].values()
    }
    if len(rows) != len(registry["conditions"]):
        raise ValueError("Duplicate B condition")
    for dataset, fraction in sorted({key[:2] for key in rows}):
        reference, cached = None, {}
        for variant in ["full", "no_dropout", "no_weight_decay", "no_both"]:
            for seed in SEEDS:
                row = rows.get((dataset, fraction, variant, seed))
                if row is None or row["state"] != "completed":
                    result["missing"].append(
                        dict(dataset=dataset, fraction=fraction, variant=variant, seed=seed)
                    )
                    continue
                data = read_predictions(row, "test")
                if reference is not None:
                    aligned(reference, data)
                reference = data
                metrics = scores(data[1], data[2])
                result["metrics"].append(
                    dict(
                        dataset=dataset,
                        fraction=fraction,
                        variant=variant,
                        seed=seed,
                        auc=metrics["auc"],
                        log_loss=metrics["log_loss"],
                    )
                )
                cached[(variant, seed)] = data[2]
        for left, right in CONTRASTS:
            if any((v, s) not in cached for v in [left, right] for s in SEEDS):
                continue
            y = reference[1]
            for key in ["auc", "log_loss"]:
                a, b = [cached[(left, s)] for s in SEEDS], [cached[(right, s)] for s in SEEDS]
                differences = [
                    scores(y, x)[key] - scores(y, z)[key] for x, z in zip(a, b, strict=True)
                ]
                result["contrasts"].append(
                    dict(
                        dataset=dataset,
                        fraction=fraction,
                        contrast=f"{left}-{right}",
                        metric=key,
                        seed_differences=differences,
                        seed_range=[min(differences), max(differences)],
                        **paired_bootstrap(y, a, b, lambda labels, p, i: scores(labels, p[i])[key]),
                    )
                )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("registry", type=Path)
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--freeze", type=Path)
    parser.add_argument("--output", type=Path, default=Path("results/analysis/study-b"))
    args = parser.parse_args()
    registry = load_registry(args.registry)
    if registry.get("name") != "study-b":
        raise ValueError("Expected a study-b registry")
    scripts = [Path(__file__), Path(__file__).with_name("mechanism_analysis.py")]
    result = dict(
        analysis_version=ANALYSIS_VERSION,
        registry_sha256=file_sha256(args.registry),
        script_sha256={p.name: file_sha256(p) for p in scripts},
        diagnostics=history_analysis(registry),
    )
    if args.test:
        if not args.freeze:
            parser.error("--test requires --freeze")
        result["freeze"] = authorize_test(args.freeze, args.registry, scripts)
        invalid = {item["key"] for item in result["diagnostics"]["missing"] if "key" in item}
        selected = {
            **registry,
            "conditions": {k: v for k, v in registry["conditions"].items() if k not in invalid},
        }
        result.update(test_analysis(selected))
    args.output.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output / ("test.json" if args.test else "diagnostics.json"), result)
    for key, value in result["diagnostics"].items():
        pd.DataFrame(value).to_csv(args.output / f"{key}.csv", index=False)
    if args.test:
        for key in ["metrics", "contrasts"]:
            pd.DataFrame(result[key]).to_csv(args.output / f"{key}.csv", index=False)


if __name__ == "__main__":
    main()
