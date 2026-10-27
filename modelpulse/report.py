"""Audit report generation (Markdown and standalone HTML)."""

from __future__ import annotations

from datetime import datetime, timezone

import markdown as md
import numpy as np
import pandas as pd

from . import __version__, config
from .experiment import Artifacts, ExperimentResult, recommendations
from .scenarios import SCENARIOS

LIMITATIONS = [
    "Data drift does not automatically mean the model is failing.",
    "No detected drift does not prove that the model remains reliable.",
    "A decline in metrics on a simulated batch is evidence about that experiment, "
    "not proof of production performance.",
    "Synthetic experiments are not equivalent to monitoring a real deployed service.",
    "Statistical significance is not the same as business significance.",
    "PCA plots are exploratory visualisations, not a definitive reliability test.",
]

METRIC_LABELS = {"accuracy": "Accuracy", "precision": "Precision", "recall": "Recall",
                 "f1": "F1-score", "roc_auc": "ROC-AUC", "brier": "Brier score (lower = better)"}


def _fmt(x, digits: int = 3) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "n/a"
    return f"{x:.{digits}f}"


def _fmt_p(p) -> str:
    if p is None or np.isnan(p):
        return "n/a"
    return "< 0.001" if p < 0.001 else f"{p:.3f}"


def md_table(df: pd.DataFrame) -> str:
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(v).replace("|", "/") for v in row.values) + " |")
    return "\n".join(lines)


def performance_table(res: ExperimentResult, key: str) -> pd.DataFrame:
    comp = res.comparison[key]
    rows = []
    for _, r in comp.iterrows():
        ci = (f"[{_fmt(r['delta_low'])}, {_fmt(r['delta_high'])}]"
              if "delta_low" in comp.columns else "not computed")
        rows.append({"Metric": METRIC_LABELS[r["metric"]], "Baseline": _fmt(r["baseline"]),
                     "Current": _fmt(r["current"]), "Change": f"{r['delta']:+.3f}",
                     "95% paired CI of change": ci})
    return pd.DataFrame(rows)


def drift_table(res: ExperimentResult, top: int | None = None) -> pd.DataFrame:
    d = res.drift if top is None else res.drift.head(top)
    return pd.DataFrame({
        "Feature": d["feature"],
        "Test": d["test"],
        "Effect": [f"{n} = {_fmt(e)}" for n, e in zip(d["effect_name"], d["effect"])],
        "BH-adjusted p": [_fmt_p(p) for p in d["p_adjusted"]],
        "Drifted": np.where(d["drifted"], "**yes**", "no"),
        "Change": d["detail"],
    })


def build_markdown(res: ExperimentResult, art: Artifacts) -> str:
    th = res.thresholds
    spec = SCENARIOS[res.scenario]
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    q = res.quality
    ds = res.drift_summary

    out = [
        "# ModelPulse audit report",
        "",
        f"Generated {now} by ModelPulse v{__version__}.",
        "",
        "## 1. Verdict",
        "",
        f"**{res.verdict['title']}.** {res.verdict['explanation']}",
        "",
        f"- Drift status: **{ds['status']}** ({ds['n_drifted']} of {ds['n_features']} features "
        f"drifted; {ds['n_significant_but_small']} more were statistically significant but below "
        "the effect-size threshold).",
    ]
    for key, label in config.MODEL_LABELS.items():
        comp = res.comparison[key].set_index("metric")
        out.append(f"- {label}: ROC-AUC {_fmt(comp.loc['roc_auc', 'baseline'])} → "
                   f"{_fmt(comp.loc['roc_auc', 'current'])}, F1 {_fmt(comp.loc['f1', 'baseline'])} → "
                   f"{_fmt(comp.loc['f1', 'current'])}; material decline: "
                   f"**{'yes' if res.performance[key]['declined'] else 'no'}**.")
    out += [
        "",
        "## 2. Experiment settings",
        "",
        md_table(pd.DataFrame([
            {"Setting": "Scenario", "Value": spec.label},
            {"Setting": "Scenario description", "Value": spec.description},
            {"Setting": "Severity", "Value": f"{res.severity:.2f} (0 = none, 1 = most severe)"},
            {"Setting": "Features modified", "Value": ", ".join(res.features) or "none"},
            {"Setting": "Random seed", "Value": res.seed},
            {"Setting": "Reference batch", "Value": f"training set, {len(art.reference):,} sessions"},
            {"Setting": "Current batch", "Value": f"modified held-out test set, {len(res.current):,} sessions"},
            {"Setting": "Dataset SHA-256", "Value": f"`{art.info['data_sha256'][:16]}...`"},
            {"Setting": "Decision thresholds",
             "Value": ", ".join(f"{config.MODEL_LABELS[k]} {v:.2f}" for k, v in art.info["thresholds"].items())
                      + " (chosen on validation data)"},
            {"Setting": "Decision Tree hyper-parameters", "Value": str(art.info.get("tree_params"))},
        ])),
        "",
        "Pre-registered alerting rules:",
        "",
        f"- Feature drift: Benjamini-Hochberg adjusted p < {th.alpha} **and** KS D >= "
        f"{th.ks_min_effect} (numerical) or total variation distance >= {th.tvd_min_effect} "
        "(categorical).",
        f"- Batch status \"Drift detected\" when >= {th.drift_share_alert:.0%} of features drift; "
        "\"Warning\" when at least one does.",
        f"- Material performance decline: ROC-AUC drop >= {th.auc_drop} or F1 drop >= "
        f"{th.f1_drop}, with the paired bootstrap 95% CI ({th.bootstrap_reps} replicates) "
        "excluding zero.",
        "",
        "Transformation log:",
        "",
        *[f"- {line}" for line in res.log],
        "",
        "## 3. Data quality",
        "",
        md_table(pd.DataFrame([
            {"Check": "Rows in current batch", "Result": f"{q['n_rows']:,}"},
            {"Check": "Missing cells", "Result": f"{q['missing_cells']:,} "
                                                 f"({q['rows_with_missing']:,} rows affected)"},
            {"Check": "Values violating domain rules", "Result": f"{q['invalid_values']:,}"},
            {"Check": "Values outside training range", "Result": f"{q['out_of_training_range']:,}"},
            {"Check": "Category levels unseen in training", "Result": f"{q['unseen_levels']:,}"},
        ])),
        "",
        md_table(q["schema"].assign(passed=q["schema"]["passed"].map({True: "pass", False: "FAIL"}))
                 .rename(columns={"check": "Schema check", "passed": "Result", "detail": "Detail"})),
        "",
        "## 4. Detected changes",
        "",
        drift_table(res).pipe(md_table),
        "",
        "## 5. Model performance on identical sessions",
        "",
    ]
    for key, label in config.MODEL_LABELS.items():
        m = res.mcnemar[key]
        out += [
            f"### {label}",
            "",
            md_table(performance_table(res, key)),
            "",
            f"McNemar test on paired correctness: {m['baseline_only_correct']} sessions correct "
            f"only before the shift, {m['current_only_correct']} only after (p = "
            f"{_fmt_p(m['p_value'])}).",
            "",
        ]

    warnings = []
    if ds["status"] != "Stable":
        warnings.append(f"Drift {ds['status'].lower()}: " + ", ".join(ds["drifted_features"]) + ".")
    if q["missing_cells"]:
        warnings.append(f"{q['missing_cells']:,} missing values were imputed.")
    if q["invalid_values"]:
        warnings.append(f"{q['invalid_values']:,} invalid values found.")
    for key, label in config.MODEL_LABELS.items():
        if res.performance[key]["declined"]:
            warnings.append(f"{label} shows a material performance decline.")
    out += ["## 6. Warnings", ""]
    out += [f"- {w}" for w in warnings] or ["- None."]
    out += ["", "## 7. Recommended investigations", ""]
    out += [f"{i}. {r}" for i, r in enumerate(recommendations(res, art), 1)]
    out += ["", "## 8. Limitations", ""]
    out += [f"- {lim}" for lim in LIMITATIONS]
    out += ["", "---", "",
            "Data: Online Shoppers Purchasing Intention Dataset (Sakar & Kastro, UCI Machine "
            "Learning Repository, CC BY 4.0).", ""]
    return "\n".join(out)


HTML_TEMPLATE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ModelPulse audit report</title>
<style>
body {{ font-family: -apple-system, Segoe UI, Roboto, sans-serif; max-width: 960px;
       margin: 2rem auto; padding: 0 1rem; color: #1f2433; line-height: 1.55; }}
h1 {{ color: #4263eb; }} h2 {{ border-bottom: 1px solid #e3e7f0; padding-bottom: .3rem; }}
table {{ border-collapse: collapse; width: 100%; margin: 1rem 0; font-size: .9rem; }}
th, td {{ border: 1px solid #e3e7f0; padding: .4rem .6rem; text-align: left; vertical-align: top; }}
th {{ background: #f4f6fb; }} code {{ background: #f4f6fb; padding: 0 .25rem; }}
</style></head><body>
{body}
</body></html>
"""


def build_html(markdown_text: str) -> str:
    body = md.markdown(markdown_text, extensions=["tables"])
    return HTML_TEMPLATE.format(body=body)
