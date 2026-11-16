"""A: metadata-only control, separate OOF diagnostics, explicitly frozen test analysis."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from mechanism_analysis import (
    ANALYSIS_VERSION,
    SEEDS,
    aligned,
    authorize_test,
    finite_logit,
    histories,
    load_registry,
    logit_histogram,
    paired_metric_bootstrap,
    rank_changes,
    read_predictions,
    scores,
    select_thresholds,
    shifted_probability,
    threshold_metrics,
)
from sklearn.metrics import average_precision_score, precision_recall_curve

from tabular_thesis.experiments.artifacts import atomic_json, file_sha256


def unweighted_curve(child):
    if "history" in child:
        return [epoch["val_loss"] for epoch in child["history"]]
    curves = child["validation_curves"]
    return curves.get("Logloss:use_weights=false", curves.get("Logloss"))


def selected_epoch(child):
    return child["best_epoch"] if "best_epoch" in child else child["best_iteration"] + 1


def loss_values(child):
    if "history" in child:
        return [
            h[key]
            for h in child["history"]
            for key in ["train_loss", "val_loss", "val_loss_weighted"]
            if key in h
        ]
    return [value for curve in child["validation_curves"].values() for value in curve]


def control_rule(children):
    if len(children) != 24:
        return dict(trigger=None, reason="Requires all 24 folds; denominator is never reduced")
    details = []
    for child in children:
        curve = unweighted_curve(child)
        if not curve or any(v is None or not np.isfinite(v) for v in curve):
            return dict(trigger=None, reason="Incomplete/nonfinite validation curve")
        kw, ku = selected_epoch(child), int(np.argmin(curve)) + 1
        if kw is None or not 1 <= kw <= len(curve):
            return dict(trigger=None, reason="Invalid selected state")
        diverged = abs(kw - ku) > max(2, 0.2 * kw) or ku == len(curve)
        details.append(
            dict(
                weighted_epoch=kw,
                unweighted_epoch=ku,
                last_minimum=ku == len(curve),
                divergent=diverged,
            )
        )
    count = sum(r["divergent"] for r in details)
    return dict(trigger=count >= 12, divergent_folds=count, total_folds=24, folds=details)


def group_rows(registry):
    grouped = {}
    for row in registry["conditions"].values():
        pair = f"{row['dataset']}/{row['model']}"
        group = grouped.setdefault(pair, {})
        key = (row["variant"], row["seed"])
        if key in group:
            raise ValueError("Duplicate condition")
        group[key] = row
    return grouped


def control_check(registry):
    decisions = {}
    for pair, rows in group_rows(registry).items():
        children = []
        errors = []
        for seed in SEEDS:
            row = rows.get(("weighted", seed))
            try:
                if row is None:
                    raise ValueError("Missing weighted seed")
                children.extend(histories(row))
            except (ValueError, OSError, KeyError) as error:
                errors.append(str(error))
        decisions[pair] = (
            control_rule(children)
            if not errors
            else dict(trigger=None, reason=errors, total_folds=len(children))
        )
    return decisions


def diagnostics(registry):
    result = {}
    for pair, rows in group_rows(registry).items():
        try:
            by_variant, ap, reference = {}, {}, None
            for variant in ["unweighted", "weighted"]:
                by_variant[variant], ap[variant] = [], []
                for seed in SEEDS:
                    row = rows[(variant, seed)]
                    by_variant[variant].extend(histories(row))
                    data = read_predictions(row, "oof")
                    if reference is not None:
                        aligned(reference, data)
                    reference = data
                    ap[variant].append(float(average_precision_score(data[1], data[2])))
            curves = [unweighted_curve(c) for c in by_variant["weighted"]]
            nonfinite = any(
                v is None or not np.isfinite(v)
                for variant in by_variant.values()
                for c in variant
                for v in loss_values(c)
            )
            weighted_median = float(np.median([selected_epoch(c) for c in by_variant["weighted"]]))
            unweighted_median = float(
                np.median([selected_epoch(c) for c in by_variant["unweighted"]])
            )
            growing = sum(
                curve[0] is not None
                and curve[selected_epoch(c) - 1] is not None
                and curve[selected_epoch(c) - 1] > curve[0]
                for curve, c in zip(curves, by_variant["weighted"], strict=True)
            )
            spreads = {k: max(v) - min(v) for k, v in ap.items()}
            signals = [
                nonfinite,
                weighted_median < 0.5 * unweighted_median and growing > 12,
                spreads["weighted"] > 3 * spreads["unweighted"],
            ]
            result[pair] = dict(
                complete=True,
                oof_ap=ap,
                ap_range=spreads,
                selected_epoch_medians=[weighted_median, unweighted_median],
                growing_folds=growing,
                instability_signals=signals,
                interpretation="Signals do not identify a cause",
            )
        except (ValueError, OSError, KeyError) as error:
            result[pair] = dict(
                complete=False,
                reason=str(error),
                failed_conditions=[r for r in rows.values() if r["state"] == "failed"],
            )
    return result


def test_analysis(registry, decisions):
    output = {"conditions": [], "contrasts": [], "rank_changes": [], "curves": [], "missing": []}
    for pair, rows in group_rows(registry).items():
        decision = decisions.get(pair, {})
        if decision.get("trigger") is None:
            output["missing"].append(dict(pair=pair, reason="Control decision incomplete"))
            continue
        variants = ["unweighted", "weighted"]
        if decision["trigger"]:
            variants.append("weighted_unweighted_selection")
        if any(rows.get((v, s), {}).get("state") != "completed" for v in variants for s in SEEDS):
            output["missing"].append(dict(pair=pair, reason="All three paired seeds required"))
            continue
        cached, reference = {}, None
        for variant in variants:
            for seed in SEEDS:
                row = rows[(variant, seed)]
                oof = read_predictions(row, "oof")
                data = read_predictions(row, "test")
                if reference is not None:
                    aligned(reference, data)
                reference = data
                thresholds = {**select_thresholds(oof[1], oof[2]), "half": 0.5}
                value = scores(data[1], data[2], thresholds)
                cached[(variant, seed)] = dict(p=data[2], thresholds=thresholds, metrics=value)
                output["conditions"].append(
                    dict(
                        pair=pair,
                        variant=variant,
                        seed=seed,
                        **value,
                        thresholds=thresholds,
                        oof_alarm_rate=threshold_metrics(oof[1], oof[2], thresholds["f1"])[
                            "alarm_rate"
                        ],
                        threshold_logit=finite_logit(thresholds["f1"]),
                        histories=histories(row),
                    )
                )
                precision, recall, thresholds_pr = precision_recall_curve(data[1], data[2])
                output["curves"].append(
                    dict(
                        pair=pair,
                        variant=variant,
                        seed=seed,
                        precision=precision.tolist(),
                        recall=recall.tolist(),
                        thresholds=thresholds_pr.tolist(),
                        oof_logits=logit_histogram(oof[2]),
                        test_logits=logit_histogram(data[2]),
                    )
                )
                if variant == "unweighted":
                    ratio = float((oof[1] == 0).sum() / (oof[1] == 1).sum())
                    q = shifted_probability(data[2], ratio)
                    comparator = scores(data[1], q, {"half": 0.5})
                    cached[("comparator", seed)] = dict(
                        p=q,
                        thresholds={"half": 0.5},
                        metrics=comparator,
                        log_ratio=float(np.log(ratio)),
                    )
                    output["conditions"].append(
                        dict(
                            pair=pair,
                            variant="comparator",
                            seed=seed,
                            ratio=ratio,
                            log_ratio=float(np.log(ratio)),
                            **comparator,
                            ranking_roundoff_changed=not np.array_equal(
                                np.unique(q, return_inverse=True)[1],
                                np.unique(data[2], return_inverse=True)[1],
                            ),
                        )
                    )
        y = reference[1]
        for seed in SEEDS:
            u, w = cached[("unweighted", seed)], cached[("weighted", seed)]
            lu, lw = finite_logit(u["thresholds"]["f1"]), finite_logit(w["thresholds"]["f1"])
            output["rank_changes"].append(
                dict(
                    pair=pair,
                    seed=seed,
                    comparison="weighted-unweighted",
                    expected_logit_shift=cached[("comparator", seed)]["log_ratio"],
                    threshold_logit_difference=lw["value"] - lu["value"]
                    if lw["value"] is not None and lu["value"] is not None
                    else None,
                    **rank_changes(y, u["p"], w["p"]),
                )
            )
        for s1, s2 in itertools.combinations(SEEDS, 2):
            output["rank_changes"].append(
                dict(
                    pair=pair,
                    seeds=[s1, s2],
                    comparison="unweighted-between-seeds",
                    **rank_changes(
                        y, cached[("unweighted", s1)]["p"], cached[("unweighted", s2)]["p"]
                    ),
                )
            )
        contrasts = [("weighted", "unweighted"), ("weighted", "comparator")]
        if decision["trigger"]:
            contrasts += [
                ("weighted", "weighted_unweighted_selection"),
                ("weighted_unweighted_selection", "unweighted"),
            ]
        for left, right in contrasts:
            keys = cached[(left, 0)]["metrics"].keys() & cached[(right, 0)]["metrics"].keys()
            a = [cached[(left, seed)] for seed in SEEDS]
            b = [cached[(right, seed)] for seed in SEEDS]
            keys = [key for key in keys if not key.endswith(("_fp", "_fn"))]
            for key, uncertainty in paired_metric_bootstrap(y, a, b, keys).items():
                output["contrasts"].append(
                    dict(
                        pair=pair,
                        contrast=f"{left}-{right}",
                        metric=key,
                        **uncertainty,
                    )
                )
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("registry", type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--control-check", action="store_true")
    mode.add_argument("--diagnostics", action="store_true")
    mode.add_argument("--test", action="store_true")
    parser.add_argument("--freeze", type=Path)
    parser.add_argument("--diagnostics-file", type=Path)
    parser.add_argument("--control-decision", type=Path)
    parser.add_argument("--output", type=Path, default=Path("results/analysis/study-a"))
    args = parser.parse_args()
    registry = load_registry(args.registry)
    if registry.get("name") != "study-a":
        raise ValueError("Expected a study-a registry")
    result = {"analysis_version": ANALYSIS_VERSION, "registry_sha256": file_sha256(args.registry)}
    scripts = [Path(__file__), Path(__file__).with_name("mechanism_analysis.py")]
    result["script_sha256"] = {p.name: file_sha256(p) for p in scripts}
    if args.control_check:
        result["decisions"] = control_check(registry)
        result["weighted_sources"] = {
            k: v for k, v in registry["conditions"].items() if v["variant"] == "weighted"
        }
        name = "control"
    elif args.diagnostics:
        result["diagnostics"] = diagnostics(registry)
        name = "diagnostics"
    else:
        if not args.freeze or not args.diagnostics_file or not args.control_decision:
            parser.error("--test requires --freeze, --diagnostics-file and --control-decision")
        result["freeze"] = authorize_test(args.freeze, args.registry, scripts)
        diagnostic = json.loads(args.diagnostics_file.read_text())
        decision = json.loads(args.control_decision.read_text())
        for prior in (diagnostic, decision):
            if prior["script_sha256"] != result["script_sha256"]:
                raise ValueError("Diagnostics/control use different analysis rules")
        if diagnostic["registry_sha256"] != result["registry_sha256"]:
            raise ValueError("Run OOF diagnostics on the final registry before test analysis")
        for key, value in decision["weighted_sources"].items():
            if registry["conditions"].get(key) != value:
                raise ValueError("Stale control decision")
        complete_pairs = {k for k, v in diagnostic["diagnostics"].items() if v["complete"]}
        selected = {
            **registry,
            "conditions": {
                k: v
                for k, v in registry["conditions"].items()
                if f"{v['dataset']}/{v['model']}" in complete_pairs
            },
        }
        result.update(test_analysis(selected, decision["decisions"]))
        result["diagnostics"] = diagnostic["diagnostics"]
        name = "test"
    args.output.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output / f"{name}.json", result)
    for table in ["conditions", "contrasts", "rank_changes"]:
        if table in result:
            pd.DataFrame(result[table]).to_csv(args.output / f"{table}.csv", index=False)


if __name__ == "__main__":
    main()
