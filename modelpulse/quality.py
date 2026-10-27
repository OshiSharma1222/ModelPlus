"""Schema, missing-value, domain-range and unseen-category checks."""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config
from .data import Preprocessor


def check_schema(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    missing_cols = [c for c in config.FEATURES if c not in df.columns]
    rows.append({"check": "All 17 input features present",
                 "passed": not missing_cols,
                 "detail": "missing: " + ", ".join(missing_cols) if missing_cols else "ok"})
    extra = [c for c in df.columns if c not in config.FEATURES + [config.TARGET]]
    rows.append({"check": "No unexpected columns", "passed": not extra,
                 "detail": ", ".join(extra) if extra else "ok"})
    bad_types = [c for c in config.NUMERIC_FEATURES
                 if c in df.columns and not pd.api.types.is_numeric_dtype(df[c])]
    rows.append({"check": "Numerical features are numeric", "passed": not bad_types,
                 "detail": ", ".join(bad_types) if bad_types else "ok"})
    rows.append({"check": "Labels available", "passed": config.TARGET in df.columns,
                 "detail": "performance can be evaluated" if config.TARGET in df.columns
                 else "drift-only monitoring"})
    return pd.DataFrame(rows)


def quality_report(df: pd.DataFrame, prep: Preprocessor) -> dict:
    """Per-feature counts of missing, invalid and out-of-training-range values.

    *Invalid* means the value violates a domain rule (e.g. a negative duration
    or a bounce rate above 1). *Out of training range* means the value is valid
    but outside what the model saw during training - the model must
    extrapolate there.
    """
    rows = []
    for col in config.FEATURES:
        x = df[col]
        missing = int(x.isna().sum())
        invalid = out_of_range = unseen = 0
        if col in config.NUMERIC_FEATURES:
            v = x.dropna().astype(float)
            lo, hi = config.DOMAIN_RANGES[col]
            bad = np.zeros(len(v), dtype=bool)
            if lo is not None:
                bad |= v.to_numpy() < lo
            if hi is not None:
                bad |= v.to_numpy() > hi
            invalid = int(bad.sum())
            out_of_range = int(((v < prep.train_min[col]) | (v > prep.train_max[col])).sum()) - invalid
            out_of_range = max(out_of_range, 0)
        else:
            v = x.dropna().astype(str)
            if col in config.VALID_CATEGORIES:
                invalid = int((~v.isin(config.VALID_CATEGORIES[col])).sum())
            unseen = int((~v.isin(prep.seen_levels[col])).sum())
        rows.append({"feature": col, "missing": missing, "invalid": invalid,
                     "out_of_training_range": out_of_range, "unseen_levels": unseen})
    table = pd.DataFrame(rows)
    return {
        "schema": check_schema(df),
        "table": table,
        "n_rows": len(df),
        "missing_cells": int(table["missing"].sum()),
        "rows_with_missing": int(df[config.FEATURES].isna().any(axis=1).sum()),
        "invalid_values": int(table["invalid"].sum()),
        "out_of_training_range": int(table["out_of_training_range"].sum()),
        "unseen_levels": int(table["unseen_levels"].sum()),
    }
