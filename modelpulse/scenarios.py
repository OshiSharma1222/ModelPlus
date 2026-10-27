"""Controlled, reproducible drift scenarios.

Every scenario takes the untouched held-out test set and returns a modified
copy plus a log of exactly what was changed. Rows are never added, removed or
reordered and labels are never modified, so the shifted batch can be compared
session-by-session with the baseline.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config


@dataclass(frozen=True)
class Scenario:
    key: str
    label: str
    description: str
    eligible: tuple
    default_features: tuple


SCENARIOS: dict[str, Scenario] = {
    s.key: s
    for s in [
        Scenario(
            "control", "Unchanged control",
            "The held-out test set with no modification. Any drift flagged here is a "
            "false alarm, so this scenario checks the detector itself.",
            (), (),
        ),
        Scenario(
            "numeric_shift", "Numerical shift",
            "Multiplies the selected numerical features by up to 3x (severity 1). "
            "Counts are rounded and bounded rates are clipped to their valid range. "
            "Simulates longer browsing sessions or a change in how time is logged.",
            tuple(config.NUMERIC_FEATURES),
            ("Administrative_Duration", "Informational_Duration", "ProductRelated_Duration"),
        ),
        Scenario(
            "categorical_shift", "Categorical shift",
            "Reassigns up to 60% of rows (severity 1) of each selected categorical feature "
            "to its second most common level. Simulates a change in the visitor mix.",
            tuple(config.CATEGORICAL_FEATURES),
            ("VisitorType",),
        ),
        Scenario(
            "missing_values", "Missing-value injection",
            "Blanks out up to 50% of values (severity 1) of the selected features at "
            "random (MCAR). The model sees training-median / mode imputations instead. "
            "Simulates a broken upstream pipeline.",
            tuple(config.FEATURES),
            ("PageValues", "ExitRates"),
        ),
        Scenario(
            "relationship_shift", "Feature-outcome relationship shift",
            "Shuffles the selected features among a random fraction of sessions (severity). "
            "Each feature's marginal distribution is unchanged, but its link to the "
            "purchase outcome is broken. Simulates concept drift that marginal drift "
            "tests cannot see.",
            tuple(config.FEATURES),
            ("PageValues",),
        ),
    ]
}


def _second_most_common(x: pd.Series) -> str:
    counts = x.dropna().astype(str).value_counts()
    return str(counts.index[1] if len(counts) > 1 else counts.index[0])


def simulate_batch(
    base: pd.DataFrame,
    scenario: str,
    severity: float = 0.5,
    features=None,
    seed: int = config.SEED,
) -> tuple[pd.DataFrame, list[str]]:
    """Return ``(shifted_copy, transformation_log)``."""
    if scenario not in SCENARIOS:
        raise ValueError(f"Unknown scenario '{scenario}'. Choose from {list(SCENARIOS)}")
    if not 0 <= severity <= 1:
        raise ValueError("severity must be between 0 and 1")
    spec = SCENARIOS[scenario]
    features = list(spec.default_features if features is None else features)
    bad = [f for f in features if f not in spec.eligible]
    if bad:
        raise ValueError(f"Features {bad} are not eligible for scenario '{scenario}'")

    rng = np.random.default_rng(seed)
    df = base.copy()
    n = len(df)
    log = [f"Scenario: {spec.label}; severity {severity:.2f}; seed {seed}; rows {n}."]

    if scenario == "control" or severity == 0 or not features:
        log.append("No transformation applied.")
        return df, log

    if scenario == "numeric_shift":
        factor = 1 + 2 * severity
        for col in features:
            x = df[col] * factor
            if col in config.COUNT_FEATURES:
                x = x.round()
            lo, hi = config.DOMAIN_RANGES[col]
            if hi is not None:
                x = x.clip(upper=hi)
            df[col] = x
            log.append(f"{col}: multiplied by {factor:.2f}"
                       + (" and rounded" if col in config.COUNT_FEATURES else "")
                       + (f", clipped at {hi}" if hi is not None else "") + ".")

    elif scenario == "categorical_shift":
        frac = 0.6 * severity
        for col in features:
            target = _second_most_common(base[col])
            eligible = np.flatnonzero(df[col].astype(str).to_numpy() != target)
            k = int(round(frac * len(eligible)))
            chosen = rng.choice(eligible, size=k, replace=False)
            df.loc[df.index[chosen], col] = target
            log.append(f"{col}: {k} rows ({frac:.0%} of non-'{target}' rows) set to '{target}'.")

    elif scenario == "missing_values":
        frac = 0.5 * severity
        for col in features:
            k = int(round(frac * n))
            chosen = rng.choice(n, size=k, replace=False)
            df.loc[df.index[chosen], col] = np.nan
            log.append(f"{col}: {k} values ({frac:.0%}) set to missing completely at random.")

    elif scenario == "relationship_shift":
        k = int(round(severity * n))
        chosen = np.sort(rng.choice(n, size=k, replace=False))
        permuted = rng.permutation(chosen)
        idx = df.index
        df.loc[idx[chosen], features] = base.loc[idx[permuted], features].to_numpy()
        log.append(f"{', '.join(features)}: values shuffled jointly among {k} sessions "
                   f"({severity:.0%}); marginal distributions preserved exactly.")

    return df, log
