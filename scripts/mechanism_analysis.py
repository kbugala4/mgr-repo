"""Frozen numerical rules shared by the offline A/B analysis entry points."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score

from tabular_thesis.experiments.artifacts import file_sha256

ANALYSIS_VERSION = "ab-v1"
SEEDS = [0, 10, 20]


def binary_inputs(y, p):
    y, p = np.asarray(y), np.asarray(p, dtype=np.float64)
    if y.ndim != 1 or p.shape != y.shape or not len(y):
        raise ValueError("Expected matching nonempty label and probability vectors")
    if not np.isin(y, [0, 1]).all() or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("Invalid binary labels or probabilities")
    return y.astype(bool), p


def operating_points(y, p):
    """All tied-score decisions: no alarms, midpoints, all alarms, in descending order."""
    y, p = binary_inputs(y, p)
    order = np.argsort(-p, kind="stable")
    scores, labels = p[order], y[order]
    ends = np.r_[np.flatnonzero(scores[:-1] != scores[1:]), len(scores) - 1]
    unique = scores[ends]
    midpoints = unique[:-1] / 2 + unique[1:] / 2
    midpoints = np.maximum(midpoints, np.nextafter(unique[1:], unique[:-1]))
    thresholds = np.r_[np.nextafter(unique[0], np.inf), midpoints, unique[-1]]
    tp = np.r_[0, np.cumsum(labels)[ends]]
    fp = np.r_[0, ends + 1 - np.cumsum(labels)[ends]]
    return thresholds, tp, fp, int(y.sum()), int((~y).sum())


def select_thresholds(y, p):
    thresholds, tp, fp, positives, negatives = operating_points(y, p)
    if not positives or not negatives:
        raise ValueError("OOF threshold selection requires both classes")
    f1 = 2 * tp / (tp + fp + positives)
    # Descending candidates and argmax implement the highest-threshold tie rule.
    best = int(np.argmax(f1))
    eligible = np.flatnonzero(fp <= negatives // 100)
    return {"f1": float(thresholds[best]), "fpr01": float(thresholds[eligible[-1]])}


def threshold_metrics(y, p, threshold):
    y, p = binary_inputs(y, p)
    alarm = p >= threshold
    tp, fp = int((alarm & y).sum()), int((alarm & ~y).sum())
    fn, tn = int((~alarm & y).sum()), int((~alarm & ~y).sum())
    denominator = 2 * tp + fp + fn
    return dict(
        precision=tp / (tp + fp) if tp + fp else None,
        recall=tp / (tp + fn) if tp + fn else None,
        f1=2 * tp / denominator if denominator else None,
        fp=fp,
        fn=fn,
        alarm_rate=float(alarm.mean()),
        fpr=fp / (fp + tn) if fp + tn else None,
    )


def scores(y, p, thresholds=None):
    y, p = binary_inputs(y, p)
    both = len(np.unique(y)) == 2
    result = dict(
        ap=float(average_precision_score(y, p)) if both else None,
        auc=float(roc_auc_score(y, p)) if both else None,
        log_loss=float(log_loss(y, np.column_stack((1 - p, p)), labels=[False, True])),
    )
    for name, threshold in (thresholds or {}).items():
        result.update({f"{name}_{k}": v for k, v in threshold_metrics(y, p, threshold).items()})
    return result


def shifted_probability(p, ratio):
    p = np.asarray(p, dtype=np.float64)
    if not math.isfinite(ratio) or ratio <= 0 or np.any((p < 0) | (p > 1)):
        raise ValueError("Invalid prior ratio or probabilities")
    return ratio * p / (1 - p + ratio * p)


def finite_logit(p):
    if not 0 < p < 1:
        return {"value": None, "boundary": "lower" if p <= 0 else "upper"}
    return {"value": float(np.log(p) - np.log1p(-p)), "boundary": None}


def logit_histogram(p):
    p = np.asarray(p, dtype=np.float64)
    interior = p[(p > 0) & (p < 1)]
    logits = np.log(interior) - np.log1p(-interior)
    edges = np.linspace(-36, 36, 73)
    counts, _ = np.histogram(logits, bins=edges)
    return dict(
        edges=edges.tolist(),
        counts=counts.tolist(),
        zero=int((p == 0).sum()),
        one=int((p == 1).sum()),
        below=int((logits < -36).sum()),
        above=int((logits > 36).sum()),
    )


def rank_changes(y, a, b):
    """Cross-class reversal rate; changing a strict order to/from a tie costs one half."""
    y, a = binary_inputs(y, a)
    _, b = binary_inputs(y, b)
    total = int(y.sum()) * int((~y).sum())
    if not total:
        raise ValueError("Rank comparison requires both classes")
    order = np.argsort(a, kind="stable")
    a, b, y = a[order], b[order], y[order]
    ranks = np.unique(b, return_inverse=True)[1] + 1
    tree = np.zeros((2, int(ranks.max()) + 1), dtype=np.int64)
    seen = np.zeros(2, dtype=np.int64)

    def prefix(label, rank):
        value = 0
        while rank:
            value += int(tree[label, rank])
            rank -= rank & -rank
        return value

    discordant = tie_a_only = joint_ties = 0
    start = 0
    for end in np.r_[np.flatnonzero(a[:-1] != a[1:]) + 1, len(a)]:
        group = y[start:end]
        group_pairs = int(group.sum()) * int((~group).sum())
        ties = 0
        _, inverse = np.unique(b[start:end], return_inverse=True)
        counts0 = np.bincount(inverse[~group], minlength=int(inverse.max()) + 1)
        counts1 = np.bincount(inverse[group], minlength=len(counts0))
        ties = int(np.dot(counts0, counts1))
        joint_ties += ties
        tie_a_only += group_pairs - ties
        for i in range(start, end):
            other = 1 - int(y[i])
            discordant += int(seen[other]) - prefix(other, int(ranks[i]))
        for i in range(start, end):
            label, rank = int(y[i]), int(ranks[i])
            seen[label] += 1
            while rank < tree.shape[1]:
                tree[label, rank] += 1
                rank += rank & -rank
        start = end
    counts0 = np.bincount(ranks[~y], minlength=tree.shape[1])
    counts1 = np.bincount(ranks[y], minlength=tree.shape[1])
    tie_b_only = int(np.dot(counts0, counts1)) - joint_ties
    correlation = (
        None if len(np.unique(a)) < 2 or len(np.unique(b)) < 2 else float(spearmanr(a, b).statistic)
    )
    return dict(
        spearman=correlation,
        reversed_pairs=discordant,
        tie_changes=tie_a_only + tie_b_only,
        cross_class_pairs=total,
        discordance=(discordant + 0.5 * (tie_a_only + tie_b_only)) / total,
    )


def paired_bootstrap(y, left, right, metric, replicates=2000):
    """Same stratified row draws for every contrast; never silently omit a seed or replicate."""
    if len(left) != 3 or len(right) != 3:
        raise ValueError("A paired contrast requires all three seeds")
    y = np.asarray(y)
    groups = [np.flatnonzero(y == c) for c in [0, 1]]
    if any(not len(group) for group in groups):
        raise ValueError("Bootstrap requires both classes")
    rng = np.random.default_rng(42)

    def difference(indices):
        differences = []
        for seed in range(3):
            a = metric(y[indices], left[seed], indices)
            b = metric(y[indices], right[seed], indices)
            if a is None or b is None or not np.isfinite([a, b]).all():
                return None
            differences.append(a - b)
        return float(np.mean(differences))

    draws, undefined = [], 0
    for _ in range(replicates):
        indices = np.concatenate([rng.choice(group, len(group), replace=True) for group in groups])
        value = difference(indices)
        if value is None:
            undefined += 1
        else:
            draws.append(value)
    return dict(
        mean_difference=difference(np.arange(len(y))),
        interval=np.percentile(draws, [2.5, 97.5]).tolist() if not undefined else None,
        undefined_replicates=undefined,
        replicates=replicates,
        seed=42,
        confidence=0.95,
        method="stratified percentile; fixed trained models",
    )


def paired_metric_bootstrap(y, left, right, keys, replicates=2000):
    """Evaluate all metrics once per draw, sharing draws across seeds and contrasts."""
    if len(left) != 3 or len(right) != 3:
        raise ValueError("A paired contrast requires all three seeds")
    y = np.asarray(y)
    groups = [np.flatnonzero(y == c) for c in [0, 1]]
    if any(not len(group) for group in groups):
        raise ValueError("Bootstrap requires both classes")
    keys = sorted(keys)
    draws = {key: [] for key in keys}
    undefined = dict.fromkeys(keys, 0)
    rng = np.random.default_rng(42)
    for _ in range(replicates):
        indices = np.concatenate([rng.choice(g, len(g), replace=True) for g in groups])
        differences = {key: [] for key in keys}
        for a, b in zip(left, right, strict=True):
            sa = scores(y[indices], a["p"][indices], a.get("thresholds"))
            sb = scores(y[indices], b["p"][indices], b.get("thresholds"))
            for key in keys:
                differences[key].append(
                    None if sa[key] is None or sb[key] is None else sa[key] - sb[key]
                )
        for key, values in differences.items():
            if None in values:
                undefined[key] += 1
            else:
                draws[key].append(float(np.mean(values)))
    output = {}
    for key in keys:
        differences = [
            a["metrics"][key] - b["metrics"][key]
            if a["metrics"][key] is not None and b["metrics"][key] is not None
            else None
            for a, b in zip(left, right, strict=True)
        ]
        output[key] = dict(
            seed_differences=differences,
            seed_range=[min(differences), max(differences)] if None not in differences else None,
            mean_difference=float(np.mean(differences)) if None not in differences else None,
            interval=np.percentile(draws[key], [2.5, 97.5]).tolist()
            if not undefined[key]
            else None,
            undefined_replicates=undefined[key],
            replicates=replicates,
            seed=42,
            confidence=0.95,
            method="stratified percentile; fixed trained models",
        )
    return output


def load_registry(path):
    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate registry key: {key}")
            result[key] = value
        return result

    registry = json.loads(Path(path).read_text(), object_pairs_hook=unique_object)
    rows = registry["conditions"]
    if len({r.get("directory") for r in rows.values() if r.get("directory")}) != sum(
        bool(r.get("directory")) for r in rows.values()
    ):
        raise ValueError("Multiple conditions point to the same run")
    return registry


def job_path(row):
    return Path(row["directory"]) / "jobs" / f"{row['dataset']}-{row['model']}-000"


def read_record(row):
    job = job_path(row)
    path = job / "records" / "result.json"
    if row["state"] != "completed":
        raise ValueError("Incomplete condition")
    status = json.loads((job / "status.json").read_text())
    if status["state"] != "completed" or status["artifacts"].get(
        "records/result.json"
    ) != file_sha256(path):
        raise ValueError("Incomplete or corrupt record")
    record = json.loads(path.read_text())
    if record["parameters"] != row["params"]:
        raise ValueError("Registry parameters differ from result")
    return record


def histories(row):
    record = read_record(row)
    children = record["model_info"]["children_info"]
    if len(children) != 8:
        raise ValueError("Expected eight fold histories")
    values = list(children.values())
    trained = [child["hyperparameters_fit"] for child in values]
    if [c["random_state"] for c in trained] != list(range(row["seed"], row["seed"] + 8)):
        raise ValueError("Unexpected fold seeds")
    if any(c["budget_truncated"] for c in trained):
        raise ValueError("A/B time truncation is not admissible")
    if any("best_epoch" in c and c["best_epoch"] is None for c in trained):
        raise ValueError("Missing selected checkpoint")
    return trained


def read_predictions(row, partition):
    record = read_record(row)
    job = job_path(row)
    name = "oof" if partition == "oof" else "result"
    path = job / "predictions" / f"{name}.npz"
    status = json.loads((job / "status.json").read_text())
    if status["artifacts"].get(f"predictions/{name}.npz") != file_sha256(path):
        raise ValueError("Prediction checksum mismatch")
    with np.load(path, allow_pickle=False) as arrays:
        indices = arrays[f"{partition}_indices"]
        order = np.argsort(indices)
        y = arrays[f"{partition}_y"][order]
        p = arrays[f"{partition}_probabilities"][order]
    if len(np.unique(indices)) != len(indices) or p.shape != (len(y), 2):
        raise ValueError("Invalid prediction population")
    rare = row.get("rare_code", 1)
    if "rare_code" in row and float(record["classes"][rare]) != str(rare):
        raise ValueError("Rare-class mapping mismatch")
    return indices[order], (y == rare).astype(int), p[:, rare].astype(np.float64)


def aligned(reference, current):
    if not np.array_equal(reference[0], current[0]) or not np.array_equal(reference[1], current[1]):
        raise ValueError("Paired prediction populations differ")


def authorize_test(path, registry_path, scripts):
    """Require reviewer provenance, not an implicit approval generated by training."""
    approval = json.loads(Path(path).read_text())
    if approval.get("analysis_version") != ANALYSIS_VERSION or not approval.get("reviewer"):
        raise ValueError("Missing reviewer/analysis freeze")
    if not approval.get("frozen_at") or approval.get("prior_test_access") is not False:
        raise ValueError("Prior test access requires an explicit author/reviewer protocol decision")
    for script in scripts:
        if approval.get("script_sha256", {}).get(script.name) != file_sha256(script):
            raise ValueError(f"Analysis changed since freeze: {script.name}")
    if approval.get("registry_sha256") != file_sha256(Path(registry_path)):
        raise ValueError("Freeze does not identify this registry")
    return approval
