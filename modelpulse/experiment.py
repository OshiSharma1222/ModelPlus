"""Experiment-to-evidence workflow.

``build_artifacts`` trains the fixed models once. ``run_experiment`` then:

1. creates a reproducible shifted batch from the untouched test set,
2. runs data-quality checks on it,
3. tests every feature for drift against the training reference,
4. scores the batch with the *same* fitted models,
5. measures the change in performance on identical sessions with known labels,
6. classifies the outcome as drift / decline / both / neither.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from . import config
from .config import Thresholds, DEFAULT_THRESHOLDS
from .data import Preprocessor, file_sha256, load_dataset, stratified_split, download_dataset
from .drift import detect_drift, drift_status
from .evaluation import classification_metrics, paired_bootstrap, mcnemar_test
from .models import FittedModel, train_models
from .quality import quality_report
from .scenarios import SCENARIOS, simulate_batch

ARTIFACT_FILE = config.MODELS_DIR / "modelpulse_artifacts.pkl"
ARTIFACT_VERSION = 1


@dataclass
class Artifacts:
    reference: pd.DataFrame          # training set (raw) = drift reference
    test: pd.DataFrame               # untouched held-out test set (raw)
    prep: Preprocessor
    models: dict[str, FittedModel]
    baseline_prob: dict[str, np.ndarray]
    pca: PCA
    pca_mean: np.ndarray
    pca_scale: np.ndarray
    info: dict = field(default_factory=dict)

    @property
    def y_test(self) -> np.ndarray:
        return self.test[config.TARGET].to_numpy()


def build_artifacts(seed: int = config.SEED) -> Artifacts:
    download_dataset()
    df = load_dataset()
    train, test = stratified_split(df, config.TEST_SIZE, seed)
    fit, val = stratified_split(train, config.VALIDATION_SIZE, seed + 1)

    prep = Preprocessor().fit(train)
    fit_X, val_X, test_X = prep.transform(fit), prep.transform(val), prep.transform(test)
    models, train_info = train_models(fit_X, fit[config.TARGET].to_numpy(),
                                      val_X, val[config.TARGET].to_numpy(), seed)
    baseline_prob = {k: m.predict_proba(test_X) for k, m in models.items()}

    num = prep.numeric_matrix(train)
    mean, scale = num.mean(axis=0), num.std(axis=0)
    scale[scale == 0] = 1.0
    pca = PCA(n_components=2, random_state=seed).fit((num - mean) / scale)

    info = {
        "version": ARTIFACT_VERSION,
        "seed": seed,
        "created": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "data_sha256": file_sha256(config.DATA_FILE),
        "n_total": len(df), "n_train": len(train), "n_fit": len(fit),
        "n_validation": len(val), "n_test": len(test),
        "positive_rate": float(df[config.TARGET].mean()),
        "thresholds": {k: m.threshold for k, m in models.items()},
        **train_info,
    }
    return Artifacts(train, test, prep, models, baseline_prob, pca, mean, scale, info)


def load_or_build_artifacts(path=ARTIFACT_FILE, rebuild: bool = False) -> Artifacts:
    if path.exists() and not rebuild:
        with open(path, "rb") as fh:
            art = pickle.load(fh)
        if isinstance(art, Artifacts) and art.info.get("version") == ARTIFACT_VERSION:
            return art
    art = build_artifacts()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump(art, fh)
    return art


def pca_projection(art: Artifacts, current: pd.DataFrame, n: int = 1500,
                   seed: int = config.SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    parts = []
    for name, df in [("Reference", art.reference), ("Current", current)]:
        take = df.iloc[rng.choice(len(df), size=min(n, len(df)), replace=False)]
        z = art.pca.transform((art.prep.numeric_matrix(take) - art.pca_mean) / art.pca_scale)
        parts.append(pd.DataFrame({"PC1": z[:, 0], "PC2": z[:, 1], "batch": name}))
    return pd.concat(parts, ignore_index=True)


@dataclass
class ExperimentResult:
    scenario: str
    severity: float
    features: list
    seed: int
    current: pd.DataFrame
    log: list
    quality: dict
    drift: pd.DataFrame
    drift_summary: dict
    current_prob: dict
    metrics: dict          # model -> {"baseline": {...}, "current": {...}}
    comparison: dict       # model -> paired bootstrap DataFrame
    mcnemar: dict
    performance: dict      # model -> decline verdict
    verdict: dict
    thresholds: Thresholds


def performance_verdict(comp: pd.DataFrame, th: Thresholds) -> dict:
    """Material decline = drop beyond the pre-registered size whose paired
    95% CI excludes zero."""
    def check(metric: str, limit: float) -> bool:
        row = comp.set_index("metric").loc[metric]
        upper = row.get("delta_high", row["delta"])
        return bool(row["delta"] <= -limit and upper < 0)

    auc_decline = check("roc_auc", th.auc_drop)
    f1_decline = check("f1", th.f1_drop)
    return {"declined": auc_decline or f1_decline,
            "auc_decline": auc_decline, "f1_decline": f1_decline}


VERDICT_TEXT = {
    (True, True): ("Drift and performance decline",
                   "The batch differs from the reference and the model performs measurably "
                   "worse on it. Investigate the drifted features first, then consider "
                   "retraining on recent labelled data."),
    (True, False): ("Drift without measurable decline",
                    "The inputs changed, but on these sessions the model's quality did not "
                    "drop beyond the pre-registered limits. Treat the drift as a warning to "
                    "monitor, not as evidence of failure."),
    (False, True): ("Performance decline without detected drift",
                    "Feature distributions look unchanged, yet the model performs worse. "
                    "This pattern points to a change in the relationship between features "
                    "and outcome (concept drift), which marginal drift tests cannot detect."),
    (False, False): ("Neither drift nor decline",
                     "No material drift and no material performance change were found. This "
                     "does not prove the model is reliable - only that these checks did not "
                     "find a problem in this batch."),
}


def run_experiment(
    art: Artifacts,
    scenario: str = "control",
    severity: float = 0.5,
    features=None,
    seed: int = config.SEED,
    thresholds: Thresholds = DEFAULT_THRESHOLDS,
    bootstrap_reps: int | None = None,
    primary_model: str = "tree",
) -> ExperimentResult:
    reps = thresholds.bootstrap_reps if bootstrap_reps is None else bootstrap_reps
    features = list(SCENARIOS[scenario].default_features if features is None else features)
    current, log = simulate_batch(art.test, scenario, severity, features, seed)

    quality = quality_report(current, art.prep)
    drift = detect_drift(art.reference, current, thresholds)
    summary = drift_status(drift, thresholds)

    cur_X = art.prep.transform(current)
    y = current[config.TARGET].to_numpy()
    current_prob, metrics, comparison, mcn, perf = {}, {}, {}, {}, {}
    for key, model in art.models.items():
        p_cur = model.predict_proba(cur_X)
        p_base = art.baseline_prob[key]
        current_prob[key] = p_cur
        metrics[key] = {
            "baseline": classification_metrics(y, p_base, model.threshold),
            "current": classification_metrics(y, p_cur, model.threshold),
        }
        comparison[key] = paired_bootstrap(y, p_base, p_cur, model.threshold, reps, seed)
        mcn[key] = mcnemar_test(y, p_base >= model.threshold, p_cur >= model.threshold)
        perf[key] = performance_verdict(comparison[key], thresholds)

    drift_flag = summary["n_drifted"] > 0
    decline_flag = perf[primary_model]["declined"]
    quality_flag = quality["rows_with_missing"] > 0 or quality["invalid_values"] > 0
    title, explanation = VERDICT_TEXT[(drift_flag, decline_flag)]
    if quality_flag:
        share = quality["rows_with_missing"] / max(quality["n_rows"], 1)
        title = "Data-quality issue" + (" with performance decline" if decline_flag
                                        else " without measurable decline")
        explanation = (
            f"{share:.1%} of sessions have missing values and {quality['invalid_values']:,} "
            "values break domain rules. The observed values may look like the reference, but "
            "the model is scoring imputed inputs. "
            + ("The performance drop is most plausibly caused by the broken inputs rather "
               "than by a real change in shopper behaviour." if decline_flag else
               "Performance held up within the pre-registered limits, but the pipeline "
               "still needs fixing.")
        )
    verdict = {"title": title, "explanation": explanation, "drift": drift_flag,
               "decline": decline_flag, "quality_issue": quality_flag,
               "primary_model": primary_model,
               "any_model_declined": any(v["declined"] for v in perf.values())}

    return ExperimentResult(scenario, severity, features, seed, current, log, quality, drift,
                            summary, current_prob, metrics, comparison, mcn, perf, verdict,
                            thresholds)


def severity_sweep(art: Artifacts, scenario: str, features=None, severities=None,
                   seed: int = config.SEED) -> pd.DataFrame:
    """Drift and performance across severities (point estimates, no bootstrap)."""
    severities = np.round(np.linspace(0, 1, 6), 2) if severities is None else severities
    rows = []
    for s in severities:
        res = run_experiment(art, scenario, float(s), features, seed, bootstrap_reps=0)
        for key, label in config.MODEL_LABELS.items():
            m = res.metrics[key]["current"]
            rows.append({"severity": float(s), "model": label,
                         "drift_share": res.drift_summary["share"],
                         "max_effect_ratio": float(res.drift["effect_ratio"].max()),
                         "roc_auc": m["roc_auc"], "f1": m["f1"], "recall": m["recall"],
                         "precision": m["precision"]})
    return pd.DataFrame(rows)


def recommendations(res: ExperimentResult, art: Artifacts) -> list[str]:
    """Rule-based investigation suggestions for the audit report."""
    recs = []
    importance = art.models["tree"].feature_importance()
    top = set(importance.head(5).index)
    drifted = res.drift_summary["drifted_features"]
    important_drift = [f for f in drifted if f in top]
    if important_drift:
        recs.append("Prioritise " + ", ".join(important_drift) + ": these drifted features are "
                    "among the Decision Tree's five most important inputs.")
    elif drifted:
        recs.append("The drifted features (" + ", ".join(drifted) + ") carry little weight in the "
                    "Decision Tree, which may explain a limited performance impact. Confirm "
                    "with the business whether the change is expected.")
    if res.quality["missing_cells"]:
        recs.append(f"{res.quality['missing_cells']:,} values were missing and were imputed with "
                    "training medians/modes. Trace the upstream pipeline for the affected "
                    "columns before trusting predictions for those sessions.")
    if res.quality["invalid_values"]:
        recs.append(f"{res.quality['invalid_values']:,} values violate domain rules. Add input "
                    "validation upstream of the model.")
    if res.quality["out_of_training_range"]:
        recs.append(f"{res.quality['out_of_training_range']:,} values lie outside the training "
                    "range, so the model is extrapolating for those sessions.")
    if res.verdict["decline"] and not res.verdict["drift"] and not res.verdict["quality_issue"]:
        recs.append("Performance fell without marginal drift. Investigate whether the "
                    "feature-outcome relationship has changed (e.g. segment-level error "
                    "analysis) and collect fresh labels before retraining.")
    if res.verdict["decline"]:
        recs.append("Retrain or recalibrate only after confirming the shift persists on "
                    "newly labelled data; a single simulated batch is not enough evidence.")
    for key, comp in res.comparison.items():
        row = comp.set_index("metric").loc["brier"]
        if row["delta"] > 0.01:
            recs.append(f"{config.MODEL_LABELS[key]} Brier score worsened by {row['delta']:.3f}: "
                        "predicted probabilities are less trustworthy - review calibration.")
    if not recs:
        recs.append("No action required beyond routine monitoring. Re-run this check on the "
                    "next batch; absence of evidence is not evidence of reliability.")
    return recs
