"""Project-wide constants and pre-registered monitoring thresholds.

The thresholds below were fixed *before* the final scenarios were evaluated so
that alerting rules cannot be tuned to make results look better.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
REPORTS_DIR = ROOT_DIR / "reports"
FIGURES_DIR = ROOT_DIR / "figures"

DATA_FILE = DATA_DIR / "online_shoppers_intention.csv"
DATA_URL = (
    "https://archive.ics.uci.edu/static/public/468/"
    "online+shoppers+purchasing+intention+dataset.zip"
)

SEED = 2026
TEST_SIZE = 0.30
VALIDATION_SIZE = 0.20

TARGET = "Revenue"

NUMERIC_FEATURES = [
    "Administrative",
    "Administrative_Duration",
    "Informational",
    "Informational_Duration",
    "ProductRelated",
    "ProductRelated_Duration",
    "BounceRates",
    "ExitRates",
    "PageValues",
    "SpecialDay",
]

CATEGORICAL_FEATURES = [
    "Month",
    "OperatingSystems",
    "Browser",
    "Region",
    "TrafficType",
    "VisitorType",
    "Weekend",
]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Heavily right-skewed, non-negative features that are log1p-transformed
# before modelling. Trees are invariant to this; Naive Bayes benefits from it.
LOG_FEATURES = [
    "Administrative",
    "Administrative_Duration",
    "Informational",
    "Informational_Duration",
    "ProductRelated",
    "ProductRelated_Duration",
    "PageValues",
]

COUNT_FEATURES = ["Administrative", "Informational", "ProductRelated"]

# Domain constraints used by the data-quality checks: (min, max), None = open.
DOMAIN_RANGES = {
    "Administrative": (0, None),
    "Administrative_Duration": (0, None),
    "Informational": (0, None),
    "Informational_Duration": (0, None),
    "ProductRelated": (0, None),
    "ProductRelated_Duration": (0, None),
    "BounceRates": (0, 1),
    "ExitRates": (0, 1),
    "PageValues": (0, None),
    "SpecialDay": (0, 1),
}

VALID_CATEGORIES = {
    "Month": {"Jan", "Feb", "Mar", "Apr", "May", "Jun", "June",
              "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"},
    "VisitorType": {"Returning_Visitor", "New_Visitor", "Other"},
    "Weekend": {"True", "False"},
}

MODEL_LABELS = {"tree": "Decision Tree", "nb": "Naive Bayes"}


@dataclass(frozen=True)
class Thresholds:
    """Alerting rules, fixed in advance of the experiments."""

    alpha: float = 0.05            # Benjamini-Hochberg adjusted significance level
    ks_min_effect: float = 0.10    # minimum KS D statistic for numerical drift
    tvd_min_effect: float = 0.10   # minimum total variation distance (categorical)
    drift_share_alert: float = 0.20  # share of drifted features => "Drift detected"
    auc_drop: float = 0.03         # ROC-AUC decline that counts as material
    f1_drop: float = 0.05          # F1 decline that counts as material
    bootstrap_reps: int = 300      # paired bootstrap replicates for CIs
    min_expected_count: float = 5.0  # chi-square expected-count rule

    def as_dict(self) -> dict:
        return asdict(self)


DEFAULT_THRESHOLDS = Thresholds()
