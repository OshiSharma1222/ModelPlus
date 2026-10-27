import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from modelpulse import config  # noqa: E402


def make_sessions(n: int = 600, seed: int = 0) -> pd.DataFrame:
    """Small synthetic dataset with the real schema (no download needed)."""
    rng = np.random.default_rng(seed)
    page_values = np.where(rng.random(n) < 0.25, rng.gamma(2, 15, n), 0.0)
    revenue = (rng.random(n) < 0.05 + 0.6 * (page_values > 0)).astype(int)
    df = pd.DataFrame({
        "Administrative": rng.poisson(2, n).astype(float),
        "Administrative_Duration": rng.exponential(80, n),
        "Informational": rng.poisson(0.5, n).astype(float),
        "Informational_Duration": rng.exponential(30, n),
        "ProductRelated": rng.poisson(30, n).astype(float),
        "ProductRelated_Duration": rng.exponential(1100, n),
        "BounceRates": rng.uniform(0, 0.2, n),
        "ExitRates": rng.uniform(0, 0.2, n),
        "PageValues": page_values,
        "SpecialDay": rng.choice([0, 0, 0, 0.2, 0.6, 1.0], n),
        "Month": rng.choice(["Mar", "May", "Nov", "Dec"], n),
        "OperatingSystems": rng.choice(["1", "2", "3"], n),
        "Browser": rng.choice(["1", "2", "4"], n),
        "Region": rng.choice(["1", "2", "3", "4"], n),
        "TrafficType": rng.choice(["1", "2", "3"], n),
        "VisitorType": rng.choice(["Returning_Visitor", "New_Visitor"], n, p=[0.85, 0.15]),
        "Weekend": rng.choice(["True", "False"], n, p=[0.25, 0.75]),
        config.TARGET: revenue,
    })
    return df


@pytest.fixture
def sessions() -> pd.DataFrame:
    return make_sessions()


@pytest.fixture(scope="session")
def artifacts():
    """Real artifacts; downloads the UCI dataset on first use."""
    from modelpulse.experiment import load_or_build_artifacts

    try:
        return load_or_build_artifacts()
    except OSError as exc:  # offline
        pytest.skip(f"dataset unavailable: {exc}")
