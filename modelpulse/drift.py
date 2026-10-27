"""Feature-level drift detection between a reference and a current batch.

* Numerical features: two-sample Kolmogorov-Smirnov test. The KS D statistic
  doubles as the effect size; the standardised mean difference and the
  population stability index (PSI) are reported alongside it.
* Categorical features: chi-square test of homogeneity when the expected-count
  assumption holds, otherwise a Monte-Carlo permutation p-value. Effect size is
  the total variation distance (TVD) between category proportions, with
  Cramer's V for reference.
* All feature p-values are adjusted with Benjamini-Hochberg. A feature is only
  flagged when it is significant *and* its effect size passes a pre-registered
  minimum, because large samples make trivial changes significant.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from . import config
from .config import Thresholds, DEFAULT_THRESHOLDS


def benjamini_hochberg(p_values) -> np.ndarray:
    """BH-adjusted p-values (NaNs are passed through)."""
    p = np.asarray(p_values, dtype=float)
    out = np.full_like(p, np.nan)
    ok = ~np.isnan(p)
    pv = p[ok]
    m = len(pv)
    if m == 0:
        return out
    order = np.argsort(pv)
    ranked = pv[order] * m / np.arange(1, m + 1)
    adjusted = np.minimum.accumulate(ranked[::-1])[::-1].clip(max=1.0)
    res = np.empty(m)
    res[order] = adjusted
    out[ok] = res
    return out


def population_stability_index(ref, cur, bins: int = 10, eps: float = 1e-4) -> float:
    ref = np.asarray(ref, dtype=float)
    cur = np.asarray(cur, dtype=float)
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 2:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    r = np.histogram(ref, edges)[0] / len(ref) + eps
    c = np.histogram(cur, edges)[0] / len(cur) + eps
    return float(np.sum((c - r) * np.log(c / r)))


def numeric_drift(ref: pd.Series, cur: pd.Series) -> dict:
    r = ref.dropna().to_numpy(dtype=float)
    c = cur.dropna().to_numpy(dtype=float)
    if len(r) < 2 or len(c) < 2:
        return {"test": "KS", "statistic": np.nan, "p_value": np.nan, "effect": np.nan,
                "secondary": np.nan, "psi": np.nan, "detail": "too few non-missing values"}
    ks = stats.ks_2samp(r, c, method="asymp")
    pooled = np.sqrt((r.var(ddof=1) + c.var(ddof=1)) / 2)
    smd = (c.mean() - r.mean()) / pooled if pooled > 0 else 0.0
    return {
        "test": "Kolmogorov-Smirnov",
        "statistic": float(ks.statistic),
        "p_value": float(ks.pvalue),
        "effect": float(ks.statistic),
        "secondary": float(smd),
        "psi": population_stability_index(r, c),
        "detail": f"mean {r.mean():,.3g} → {c.mean():,.3g}; median {np.median(r):,.3g} → {np.median(c):,.3g}",
    }


def categorical_drift(ref: pd.Series, cur: pd.Series, min_expected: float = 5.0,
                      seed: int = config.SEED) -> dict:
    r = ref.dropna().astype(str)
    c = cur.dropna().astype(str)
    levels = sorted(set(r) | set(c))
    table = np.vstack([
        r.value_counts().reindex(levels, fill_value=0).to_numpy(),
        c.value_counts().reindex(levels, fill_value=0).to_numpy(),
    ]).astype(float)
    if len(r) == 0 or len(c) == 0 or len(levels) < 2:
        return {"test": "Chi-square", "statistic": np.nan, "p_value": np.nan, "effect": np.nan,
                "secondary": np.nan, "psi": np.nan, "detail": "not enough categories"}

    expected = np.outer(table.sum(axis=1), table.sum(axis=0)) / table.sum()
    assumption_ok = (np.mean(expected >= min_expected) >= 0.8) and np.all(expected >= 1)
    chi2 = float(np.sum((table - expected) ** 2 / expected))
    if assumption_ok:
        p_value = float(stats.chi2.sf(chi2, df=len(levels) - 1))
        test = "Chi-square"
    else:
        p_value = _monte_carlo_chi2_p(table, chi2, seed)
        test = "Chi-square (Monte-Carlo)"

    p_ref = table[0] / table[0].sum()
    p_cur = table[1] / table[1].sum()
    tvd = 0.5 * float(np.abs(p_cur - p_ref).sum())
    cramers_v = float(np.sqrt(chi2 / table.sum()))  # 2 x k table: min(dim) - 1 = 1
    biggest = int(np.argmax(np.abs(p_cur - p_ref)))
    sr, sc = p_ref + 1e-4, p_cur + 1e-4
    psi = float(np.sum((sc - sr) * np.log(sc / sr)))
    return {
        "test": test,
        "statistic": chi2,
        "p_value": p_value,
        "effect": tvd,
        "secondary": cramers_v,
        "psi": psi,
        "detail": f"'{levels[biggest]}' {p_ref[biggest]:.1%} → {p_cur[biggest]:.1%}",
    }


def _monte_carlo_chi2_p(table: np.ndarray, observed: float, seed: int, reps: int = 2000) -> float:
    """Permutation p-value with both margins fixed.

    Randomly reassigning batch membership is equivalent to drawing the
    reference row from a multivariate hypergeometric distribution over the
    pooled category counts, which lets all replicates be drawn at once.
    """
    rng = np.random.default_rng(seed)
    col_totals = table.sum(axis=0).astype(np.int64)
    n_ref = int(table[0].sum())
    ref_rows = rng.multivariate_hypergeometric(col_totals, n_ref, size=reps)
    cur_rows = col_totals - ref_rows
    expected = np.outer(table.sum(axis=1), col_totals) / table.sum()
    keep = col_totals > 0
    stat = (((ref_rows[:, keep] - expected[0, keep]) ** 2 / expected[0, keep]).sum(axis=1)
            + ((cur_rows[:, keep] - expected[1, keep]) ** 2 / expected[1, keep]).sum(axis=1))
    exceed = int(np.sum(stat >= observed - 1e-9))
    return (exceed + 1) / (reps + 1)


def detect_drift(reference: pd.DataFrame, current: pd.DataFrame,
                 thresholds: Thresholds = DEFAULT_THRESHOLDS) -> pd.DataFrame:
    """One row per feature, ranked by effect size relative to its threshold."""
    rows = []
    for col in config.FEATURES:
        if col in config.NUMERIC_FEATURES:
            res = numeric_drift(reference[col], current[col])
            kind, min_effect, effect_name, secondary_name = "numerical", thresholds.ks_min_effect, "KS D", "SMD"
        else:
            res = categorical_drift(reference[col], current[col], thresholds.min_expected_count)
            kind, min_effect, effect_name, secondary_name = "categorical", thresholds.tvd_min_effect, "TVD", "Cramer's V"
        rows.append({"feature": col, "type": kind, "effect_name": effect_name,
                     "secondary_name": secondary_name, "min_effect": min_effect, **res})
    df = pd.DataFrame(rows)
    df["p_adjusted"] = benjamini_hochberg(df["p_value"])
    df["significant"] = df["p_adjusted"] < thresholds.alpha
    df["material"] = df["effect"] >= df["min_effect"]
    df["drifted"] = df["significant"] & df["material"]
    df["effect_ratio"] = df["effect"] / df["min_effect"]
    return df.sort_values("effect_ratio", ascending=False, na_position="last").reset_index(drop=True)


def drift_status(drift: pd.DataFrame, thresholds: Thresholds = DEFAULT_THRESHOLDS) -> dict:
    n_drifted = int(drift["drifted"].sum())
    n_signif_only = int((drift["significant"] & ~drift["material"]).sum())
    share = n_drifted / len(drift) if len(drift) else 0.0
    if share >= thresholds.drift_share_alert:
        status = "Drift detected"
    elif n_drifted > 0:
        status = "Warning"
    else:
        status = "Stable"
    return {
        "status": status,
        "n_drifted": n_drifted,
        "n_features": len(drift),
        "share": share,
        "n_significant_but_small": n_signif_only,
        "drifted_features": drift.loc[drift["drifted"], "feature"].tolist(),
    }


def feature_summary(reference: pd.DataFrame, current: pd.DataFrame, feature: str) -> pd.DataFrame:
    """Descriptive statistics for the Drift Explorer."""
    if feature in config.NUMERIC_FEATURES:
        def describe(x: pd.Series) -> dict:
            v = x.dropna()
            return {"n": len(x), "missing": int(x.isna().sum()), "mean": v.mean(),
                    "median": v.median(), "variance": v.var(), "std_dev": v.std(),
                    "min": v.min(), "max": v.max()}
        return pd.DataFrame({"reference": describe(reference[feature]),
                             "current": describe(current[feature])})
    r = reference[feature].astype("object").fillna("<missing>").value_counts(normalize=True)
    c = current[feature].astype("object").fillna("<missing>").value_counts(normalize=True)
    out = pd.concat([r.rename("reference"), c.rename("current")], axis=1).fillna(0.0)
    out["difference"] = out["current"] - out["reference"]
    return out.sort_values("reference", ascending=False)
