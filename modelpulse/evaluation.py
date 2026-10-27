"""Classification metrics, calibration and paired bootstrap comparisons.

Metrics are implemented directly (rather than via a metrics library) so the
definitions are visible and testable. ROC-AUC uses the Mann-Whitney rank
formulation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import rankdata

METRIC_NAMES = ["accuracy", "precision", "recall", "f1", "roc_auc", "brier"]


def confusion_counts(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    return {
        "tp": int(np.sum((y_true == 1) & (y_pred == 1))),
        "fp": int(np.sum((y_true == 0) & (y_pred == 1))),
        "tn": int(np.sum((y_true == 0) & (y_pred == 0))),
        "fn": int(np.sum((y_true == 1) & (y_pred == 0))),
    }


def roc_auc(y_true, prob) -> float:
    y_true = np.asarray(y_true).astype(int)
    prob = np.asarray(prob, dtype=float)
    n_pos = int(y_true.sum())
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata(prob, method="average")
    return float((ranks[y_true == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def roc_curve_points(y_true, prob) -> pd.DataFrame:
    """False/true positive rates at every distinct score threshold."""
    y_true = np.asarray(y_true).astype(int)
    prob = np.asarray(prob, dtype=float)
    order = np.argsort(-prob, kind="mergesort")
    y_sorted, p_sorted = y_true[order], prob[order]
    distinct = np.r_[np.where(np.diff(p_sorted))[0], len(p_sorted) - 1]
    tps = np.cumsum(y_sorted)[distinct]
    fps = (distinct + 1) - tps
    n_pos, n_neg = max(y_true.sum(), 1), max(len(y_true) - y_true.sum(), 1)
    return pd.DataFrame({"fpr": np.r_[0.0, fps / n_neg], "tpr": np.r_[0.0, tps / n_pos]})


def classification_metrics(y_true, prob, threshold: float) -> dict:
    y_true = np.asarray(y_true).astype(int)
    prob = np.asarray(prob, dtype=float)
    y_pred = (prob >= threshold).astype(int)
    c = confusion_counts(y_true, y_pred)
    n = len(y_true)
    precision = c["tp"] / (c["tp"] + c["fp"]) if (c["tp"] + c["fp"]) else float("nan")
    recall = c["tp"] / (c["tp"] + c["fn"]) if (c["tp"] + c["fn"]) else float("nan")
    f1 = 2 * c["tp"] / (2 * c["tp"] + c["fp"] + c["fn"]) if c["tp"] else 0.0
    return {
        "accuracy": (c["tp"] + c["tn"]) / n if n else float("nan"),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc(y_true, prob),
        "brier": float(np.mean((prob - y_true) ** 2)) if n else float("nan"),
        "positive_rate": float(np.mean(y_pred)) if n else float("nan"),
        "mean_probability": float(np.mean(prob)) if n else float("nan"),
        **c,
    }


def best_f1_threshold(y_true, prob, grid=None) -> float:
    """Threshold that maximises F1 (used on validation data only)."""
    grid = np.round(np.arange(0.05, 0.951, 0.01), 2) if grid is None else grid
    scores = [classification_metrics(y_true, prob, t)["f1"] for t in grid]
    return float(grid[int(np.nanargmax(scores))])


def _fast_metrics(y, prob, threshold) -> np.ndarray:
    """Vectorised subset of metrics used inside the bootstrap loop."""
    pred = prob >= threshold
    tp = np.sum(pred & (y == 1))
    fp = np.sum(pred & (y == 0))
    fn = np.sum(~pred & (y == 1))
    tn = len(y) - tp - fp - fn
    precision = tp / (tp + fp) if (tp + fp) else np.nan
    recall = tp / (tp + fn) if (tp + fn) else np.nan
    f1 = 2 * tp / (2 * tp + fp + fn) if tp else 0.0
    return np.array([
        (tp + tn) / len(y), precision, recall, f1, roc_auc(y, prob), np.mean((prob - y) ** 2)
    ])


def paired_bootstrap(
    y_true, prob_baseline, prob_current, threshold: float, reps: int = 300, seed: int = 0,
    level: float = 0.95,
) -> pd.DataFrame:
    """Percentile CIs for baseline, current and (current - baseline).

    Baseline and current predictions are for the *same* sessions, so each
    replicate resamples session indices once and evaluates both, which makes
    the CI for the difference much tighter than two independent intervals.
    """
    y = np.asarray(y_true).astype(int)
    pb = np.asarray(prob_baseline, dtype=float)
    pc = np.asarray(prob_current, dtype=float)
    point_b = _fast_metrics(y, pb, threshold)
    point_c = _fast_metrics(y, pc, threshold)
    rows = {"metric": METRIC_NAMES, "baseline": point_b, "current": point_c,
            "delta": point_c - point_b}
    if reps > 0:
        rng = np.random.default_rng(seed)
        boot_b = np.empty((reps, len(METRIC_NAMES)))
        boot_c = np.empty_like(boot_b)
        n = len(y)
        for r in range(reps):
            idx = rng.integers(0, n, n)
            boot_b[r] = _fast_metrics(y[idx], pb[idx], threshold)
            boot_c[r] = _fast_metrics(y[idx], pc[idx], threshold)
        lo, hi = (1 - level) / 2 * 100, (1 + level) / 2 * 100
        rows["current_low"] = np.nanpercentile(boot_c, lo, axis=0)
        rows["current_high"] = np.nanpercentile(boot_c, hi, axis=0)
        diff = boot_c - boot_b
        rows["delta_low"] = np.nanpercentile(diff, lo, axis=0)
        rows["delta_high"] = np.nanpercentile(diff, hi, axis=0)
    return pd.DataFrame(rows)


def calibration_table(y_true, prob, bins: int = 10) -> pd.DataFrame:
    """Observed purchase rate by equal-width probability bin."""
    y = np.asarray(y_true).astype(int)
    p = np.asarray(prob, dtype=float)
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1], right=False), 0, bins - 1)
    df = pd.DataFrame({"bin": idx, "prob": p, "y": y})
    out = df.groupby("bin").agg(mean_predicted=("prob", "mean"),
                                observed_rate=("y", "mean"), n=("y", "size")).reset_index()
    out["bin_low"] = edges[out["bin"].to_numpy()]
    out["bin_high"] = edges[out["bin"].to_numpy() + 1]
    return out


def expected_calibration_error(y_true, prob, bins: int = 10) -> float:
    table = calibration_table(y_true, prob, bins)
    weights = table["n"] / table["n"].sum()
    return float(np.sum(weights * np.abs(table["mean_predicted"] - table["observed_rate"])))


def mcnemar_test(y_true, pred_a, pred_b) -> dict:
    """Exact McNemar test on paired correctness of two prediction vectors."""
    from scipy.stats import binomtest

    y = np.asarray(y_true).astype(int)
    ca = np.asarray(pred_a).astype(int) == y
    cb = np.asarray(pred_b).astype(int) == y
    b = int(np.sum(ca & ~cb))   # baseline right, current wrong
    c = int(np.sum(~ca & cb))   # baseline wrong, current right
    p = binomtest(b, b + c, 0.5).pvalue if (b + c) else 1.0
    return {"baseline_only_correct": b, "current_only_correct": c, "p_value": float(p)}
