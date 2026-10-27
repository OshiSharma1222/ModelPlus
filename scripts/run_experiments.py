"""Reproduce the documented experiments, figures and example reports.

Usage:  python scripts/run_experiments.py
Outputs: reports/*.md, reports/results.csv, figures/*.png
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modelpulse import config  # noqa: E402
from modelpulse.evaluation import calibration_table, expected_calibration_error  # noqa: E402
from modelpulse.experiment import load_or_build_artifacts, run_experiment, severity_sweep  # noqa: E402
from modelpulse.report import build_markdown  # noqa: E402
from modelpulse.scenarios import SCENARIOS  # noqa: E402

REF, CUR = "#4263EB", "#F76707"
MODEL_COLORS = {"Decision Tree": "#4263EB", "Naive Bayes": "#0CA678"}

EXPERIMENTS = [
    ("control", "control", 0.0, None),
    ("experiment-a", "numeric_shift", 0.8, None),
    ("experiment-b", "relationship_shift", 0.8, ["PageValues"]),
    ("categorical-shift", "categorical_shift", 0.8, None),
    ("missing-values", "missing_values", 0.6, None),
]


def style(ax, title):
    ax.set_title(title, loc="left", fontsize=11, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.25)


def main():
    config.REPORTS_DIR.mkdir(exist_ok=True)
    config.FIGURES_DIR.mkdir(exist_ok=True)
    art = load_or_build_artifacts()
    print(f"Artifacts ready: {art.info['n_train']} train / {art.info['n_test']} test sessions")

    rows, results = [], {}
    for name, scenario, severity, features in EXPERIMENTS:
        res = run_experiment(art, scenario, severity, features)
        results[name] = res
        for key, label in config.MODEL_LABELS.items():
            comp = res.comparison[key].set_index("metric")
            rows.append({
                "experiment": name, "scenario": SCENARIOS[scenario].label, "severity": severity,
                "features": ", ".join(res.features), "model": label,
                "drift_status": res.drift_summary["status"],
                "drifted_features": ", ".join(res.drift_summary["drifted_features"]),
                "auc_baseline": comp.loc["roc_auc", "baseline"],
                "auc_current": comp.loc["roc_auc", "current"],
                "auc_delta": comp.loc["roc_auc", "delta"],
                "auc_delta_low": comp.loc["roc_auc", "delta_low"],
                "auc_delta_high": comp.loc["roc_auc", "delta_high"],
                "f1_baseline": comp.loc["f1", "baseline"], "f1_current": comp.loc["f1", "current"],
                "f1_delta": comp.loc["f1", "delta"],
                "material_decline": res.performance[key]["declined"],
                "verdict": res.verdict["title"],
            })
        (config.REPORTS_DIR / f"{name}.md").write_text(build_markdown(res, art), encoding="utf-8")
        print(f"{name:18s} -> {res.verdict['title']}")

    summary = pd.DataFrame(rows)
    summary.round(4).to_csv(config.REPORTS_DIR / "results.csv", index=False)
    (config.REPORTS_DIR / "example-report.md").write_text(
        build_markdown(results["experiment-b"], art), encoding="utf-8")

    # Figure 1: the A/B contrast across severities
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for (label, scenario, feats), ls in [(("A: numerical shift", "numeric_shift", None), "-"),
                                         (("B: relationship shift", "relationship_shift", ["PageValues"]), "--")]:
        sw = severity_sweep(art, scenario, feats, np.round(np.linspace(0, 1, 11), 2))
        tree = sw[sw["model"] == "Decision Tree"]
        axes[0].plot(tree["severity"], tree["drift_share"], ls, marker="o", color="#E03131", label=label)
        axes[1].plot(tree["severity"], tree["roc_auc"], ls, marker="o", color=REF, label=label)
    limit = config.DEFAULT_THRESHOLDS.auc_drop
    axes[1].axhline(results["control"].metrics["tree"]["baseline"]["roc_auc"] - limit,
                    color="grey", ls=":", label=f"baseline - {limit}")
    style(axes[0], "Share of features flagged as drifted")
    style(axes[1], "Decision Tree ROC-AUC")
    axes[0].set_ylim(-0.02, 0.5)
    for ax in axes:
        ax.set_xlabel("Severity")
        ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "ab_contrast.png", dpi=150)
    plt.close(fig)

    # Figure 2: drift ranking for experiment A
    d = results["experiment-a"].drift.sort_values("effect_ratio")
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = np.where(d["drifted"], "#E03131", np.where(d["significant"], "#FAB005", "#94A3B8"))
    ax.barh(d["feature"], d["effect_ratio"], color=colors)
    ax.axvline(1, color="#495057", ls="--", lw=1)
    style(ax, "Experiment A: effect size / drift threshold")
    ax.set_xlabel("red = drifted, amber = significant but small")
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "experiment_a_drift_ranking.png", dpi=150)
    plt.close(fig)

    # Figure 3: experiment B - same PageValues distribution, different relationship
    res_b = results["experiment-b"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for s, color, label in [(art.test["PageValues"], REF, "Baseline"),
                            (res_b.current["PageValues"], CUR, "Experiment B")]:
        x = np.sort(np.log1p(s.to_numpy()))
        axes[0].step(x, np.arange(1, len(x) + 1) / len(x), color=color, label=label,
                     ls="-" if label == "Baseline" else "--")
    style(axes[0], "PageValues distribution is unchanged (ECDF, log1p)")
    for p, color, label in [(art.baseline_prob["tree"], REF, "Baseline"),
                            (res_b.current_prob["tree"], CUR, "Experiment B")]:
        y = art.y_test
        axes[1].hist(p[y == 1], bins=20, range=(0, 1), alpha=0.55, color=color,
                     label=f"{label} (purchasers)")
    style(axes[1], "Tree scores for actual purchasers")
    for ax in axes:
        ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "experiment_b_hidden_shift.png", dpi=150)
    plt.close(fig)

    # Figure 4: baseline calibration of both models
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot([0, 1], [0, 1], ":", color="grey")
    for key, label in config.MODEL_LABELS.items():
        p = art.baseline_prob[key]
        t = calibration_table(art.y_test, p)
        ece = expected_calibration_error(art.y_test, p)
        ax.plot(t["mean_predicted"], t["observed_rate"], marker="o", color=MODEL_COLORS[label],
                label=f"{label} (ECE {ece:.3f})")
    style(ax, "Baseline calibration on the test set")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed purchase rate")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "calibration.png", dpi=150)
    plt.close(fig)

    print("\n" + summary[["experiment", "model", "drift_status", "auc_baseline", "auc_current",
                          "auc_delta", "f1_delta", "material_decline"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
