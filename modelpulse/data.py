"""Loading, splitting and preprocessing the Online Shoppers dataset.

Every learned preprocessing step (imputation values, category levels, PCA
scaling) is fitted on training data only and then applied unchanged to the
test set and to every simulated monitoring batch.
"""

from __future__ import annotations

import hashlib
import io
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import config


def download_dataset(dest: Path = config.DATA_FILE, overwrite: bool = False) -> Path:
    """Download the UCI CSV into ``dest`` unless it already exists."""
    dest = Path(dest)
    if dest.exists() and not overwrite:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(config.DATA_URL, timeout=60) as resp:
        payload = resp.read()
    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        dest.write_bytes(zf.read("online_shoppers_intention.csv"))
    return dest


def file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def standardise_types(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce raw columns to the types the rest of the pipeline expects.

    Numerical features become floats (unparseable values become NaN),
    categorical features become strings, and the target becomes 0/1.
    """
    missing = [c for c in config.FEATURES + [config.TARGET] if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    out = pd.DataFrame(index=df.index)
    for col in config.NUMERIC_FEATURES:
        out[col] = pd.to_numeric(df[col], errors="coerce").astype(float)
    for col in config.CATEGORICAL_FEATURES:
        values = df[col].astype("object")
        out[col] = values.where(values.isna(), values.astype(str)).astype("object")
    target = df[config.TARGET]
    if target.dtype == bool:
        out[config.TARGET] = target.astype(int)
    else:
        out[config.TARGET] = (
            target.astype(str).str.strip().str.lower().isin({"true", "1", "yes"}).astype(int)
        )
    return out.reset_index(drop=True)


def load_dataset(path: Path = config.DATA_FILE) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        download_dataset(path)
    return standardise_types(pd.read_csv(path))


def stratified_split(
    df: pd.DataFrame,
    test_size: float = config.TEST_SIZE,
    seed: int = config.SEED,
    target: str = config.TARGET,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split so both parts keep the original class balance."""
    rng = np.random.default_rng(seed)
    test_idx = []
    for _, group in df.groupby(target, sort=True):
        n_test = int(round(len(group) * test_size))
        test_idx.extend(rng.choice(group.index.to_numpy(), size=n_test, replace=False))
    test_mask = df.index.isin(test_idx)
    train = df.loc[~test_mask].reset_index(drop=True)
    test = df.loc[test_mask].reset_index(drop=True)
    return train, test


@dataclass
class Preprocessor:
    """Training-fitted transformations for model inputs."""

    min_level_count: int = 20
    medians: dict = field(default_factory=dict)
    train_min: dict = field(default_factory=dict)
    train_max: dict = field(default_factory=dict)
    modes: dict = field(default_factory=dict)
    levels: dict = field(default_factory=dict)
    seen_levels: dict = field(default_factory=dict)
    fitted_rows: int = 0

    def fit(self, train: pd.DataFrame) -> "Preprocessor":
        for col in config.NUMERIC_FEATURES:
            x = train[col]
            self.medians[col] = float(x.median())
            self.train_min[col] = float(x.min())
            self.train_max[col] = float(x.max())
        for col in config.CATEGORICAL_FEATURES:
            counts = train[col].value_counts(dropna=True)
            self.modes[col] = str(counts.index[0])
            self.seen_levels[col] = sorted(str(v) for v in counts.index)
            kept = [str(v) for v in counts.index[counts >= self.min_level_count]]
            if len(kept) < len(counts):
                kept.append("Other")  # bucket for rare and unseen levels
            self.levels[col] = sorted(set(kept))
        self.fitted_rows = len(train)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Impute, map rare/unseen categories and log-transform skewed features."""
        if not self.levels:
            raise RuntimeError("Preprocessor must be fitted before transform().")
        out = pd.DataFrame(index=df.index)
        for col in config.NUMERIC_FEATURES:
            x = df[col].astype(float).fillna(self.medians[col])
            if col in config.LOG_FEATURES:
                x = np.log1p(x.clip(lower=0))
            out[col] = x
        for col in config.CATEGORICAL_FEATURES:
            x = df[col].astype("object").where(df[col].notna(), self.modes[col]).astype(str)
            fallback = "Other" if "Other" in self.levels[col] else self.modes[col]
            x = x.where(x.isin(self.levels[col]), fallback)
            out[col] = pd.Categorical(x, categories=self.levels[col])
        return out

    def numeric_matrix(self, df: pd.DataFrame) -> np.ndarray:
        """Numerical block of ``transform`` as an array (used for PCA)."""
        return self.transform(df)[config.NUMERIC_FEATURES].to_numpy(dtype=float)
