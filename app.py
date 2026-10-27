"""ModelPulse - interactive AI reliability lab (Streamlit app).

Run with:  streamlit run app.py
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from modelpulse import config, plots
from modelpulse.drift import feature_summary
from modelpulse.evaluation import expected_calibration_error
from modelpulse.experiment import (
    load_or_build_artifacts, pca_projection, recommendations, run_experiment, severity_sweep,
)
from modelpulse.report import LIMITATIONS, build_html, build_markdown, drift_table, performance_table
from modelpulse.scenarios import SCENARIOS

st.set_page_config(page_title="ModelPulse", page_icon=":material/monitor_heart:", layout="wide")

PRESETS = {
    "A": dict(scenario="numeric_shift", severity=0.8,
              features=list(SCENARIOS["numeric_shift"].default_features)),
    "B": dict(scenario="relationship_shift", severity=0.8, features=["PageValues"]),
}


@st.cache_resource(show_spinner="Training baseline models (first run only)...")
def get_artifacts():
    return load_or_build_artifacts()


@st.cache_resource(max_entries=48, show_spinner="Running experiment...")
def get_experiment(scenario: str, severity: float, features: tuple, seed: int, primary: str):
    return run_experiment(ART, scenario, severity, list(features), seed, primary_model=primary)


@st.cache_resource(max_entries=16, show_spinner="Sweeping severities...")
def get_sweep(scenario: str, features: tuple, seed: int):
    return severity_sweep(ART, scenario, list(features), seed=seed)


ART = get_artifacts()

# ---------------------------------------------------------------- sidebar state
state = st.session_state
state.setdefault("scenario", "control")
state.setdefault("severity", 0.5)
state.setdefault("features", [])
state.setdefault("seed", config.SEED)


def reset_features():
    state.features = list(SCENARIOS[state.scenario].default_features)


def reset_control():
    state.scenario = "control"
    reset_features()


def apply_preset(name: str):
    p = PRESETS[name]
    state.scenario, state.severity, state.features = p["scenario"], p["severity"], list(p["features"])


with st.sidebar:
    st.title("ModelPulse")
    st.caption("Can we still trust this model under changing conditions?")
    st.subheader("Experiment")
    st.selectbox("Scenario", list(SCENARIOS), key="scenario",
                 format_func=lambda k: SCENARIOS[k].label, on_change=reset_features)
    spec = SCENARIOS[state.scenario]
    st.slider("Severity", 0.0, 1.0, step=0.05, key="severity",
              help="0 = no change, 1 = the most severe version of the scenario",
              disabled=state.scenario == "control")
    st.multiselect("Features to modify", list(spec.eligible), key="features",
                   disabled=state.scenario == "control")
    st.number_input("Random seed", min_value=0, max_value=10_000_000, step=1, key="seed")
    primary = st.radio("Primary model for the verdict", list(config.MODEL_LABELS),
                       format_func=config.MODEL_LABELS.get, horizontal=True)
    st.divider()
    st.subheader("Demonstrations")
    st.button("Experiment A: drift, little impact", on_click=apply_preset, args=("A",),
              width="stretch")
    st.button("Experiment B: impact, no drift", on_click=apply_preset, args=("B",),
              width="stretch")
    st.button("Reset to control", on_click=reset_control, width="stretch")
    st.divider()
    st.caption(f"Reference: training set ({ART.info['n_train']:,} sessions). Current: modified "
               f"held-out test set ({ART.info['n_test']:,} sessions). Labels are never changed.")

res = get_experiment(state.scenario, float(state.severity), tuple(state.features),
                     int(state.seed), primary)
th = res.thresholds
ds = res.drift_summary
q = res.quality


def comp_row(key: str, metric: str) -> pd.Series:
    return res.comparison[key].set_index("metric").loc[metric]


tabs = st.tabs(["Overview", "Data Drift Explorer", "Model Reliability", "Drift Simulator",
                "Audit Report"])

# ---------------------------------------------------------------- 1. Overview
with tabs[0]:
    st.header(f"{SCENARIOS[res.scenario].label}"
              + ("" if res.scenario == "control" else f" - severity {res.severity:.2f}"))
    box = (st.error if res.verdict["decline"] else
           st.warning if (res.verdict["drift"] or res.verdict["quality_issue"]) else st.success)
    box(f"**{res.verdict['title']}.** {res.verdict['explanation']}")

    c = st.columns(6)
    c[0].metric("Reference sessions", f"{len(ART.reference):,}")
    c[1].metric("Current sessions", f"{len(res.current):,}")
    c[2].metric(f"Drift status ({ds['n_drifted']}/{ds['n_features']} features)", ds["status"])
    c[3].metric("Missing / invalid values", f"{q['missing_cells']:,} / {q['invalid_values']:,}")
    auc, f1 = comp_row(primary, "roc_auc"), comp_row(primary, "f1")
    c[4].metric(f"ROC-AUC ({config.MODEL_LABELS[primary]})", f"{auc['current']:.3f}",
                f"{auc['delta']:+.3f}", help=f"Baseline {auc['baseline']:.3f}")
    c[5].metric(f"F1 ({config.MODEL_LABELS[primary]})", f"{f1['current']:.3f}",
                f"{f1['delta']:+.3f}", help=f"Baseline {f1['baseline']:.3f}")

    left, right = st.columns([3, 2])
    with left:
        st.subheader("Findings")
        findings = []
        if ds["drifted_features"]:
            top = res.drift.iloc[0]
            findings.append(f"{ds['n_drifted']} feature(s) drifted: "
                            f"{', '.join(ds['drifted_features'])}. Largest change: **{top['feature']}** "
                            f"({top['effect_name']} = {top['effect']:.3f}; {top['detail']}).")
        else:
            findings.append("No feature passed both the significance and effect-size rules.")
        if ds["n_significant_but_small"]:
            findings.append(f"{ds['n_significant_but_small']} feature(s) were statistically "
                            "significant but too small to matter under the pre-registered threshold.")
        for key, label in config.MODEL_LABELS.items():
            a = comp_row(key, "roc_auc")
            ci = (f" (95% CI {a['delta_low']:+.3f} to {a['delta_high']:+.3f})"
                  if "delta_low" in a else "")
            word = "material decline" if res.performance[key]["declined"] else "no material decline"
            findings.append(f"{label}: ROC-AUC changed by {a['delta']:+.3f}{ci} - {word}.")
        if q["rows_with_missing"]:
            findings.append(f"{q['rows_with_missing']:,} sessions contain missing values that were "
                            "imputed with training statistics.")
        if q["out_of_training_range"]:
            findings.append(f"{q['out_of_training_range']:,} values fall outside the training range.")
        st.markdown("\n".join(f"- {f}" for f in findings))
        st.subheader("Recommended investigations")
        st.markdown("\n".join(f"{i}. {r}" for i, r in enumerate(recommendations(res, ART), 1)))
    with right:
        st.subheader("Performance at a glance")
        rows = []
        for key, label in config.MODEL_LABELS.items():
            for metric in ["roc_auc", "f1", "recall", "precision"]:
                r = comp_row(key, metric)
                rows.append({"Model": label, "Metric": metric.upper().replace("_", "-"),
                             "Baseline": round(r["baseline"], 3), "Current": round(r["current"], 3),
                             "Change": round(r["delta"], 3)})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        st.subheader("Data quality")
        st.dataframe(q["table"][(q["table"].iloc[:, 1:] > 0).any(axis=1)]
                     if (q["table"].iloc[:, 1:] > 0).any().any() else
                     pd.DataFrame({"result": ["No missing, invalid or out-of-range values"]}),
                     hide_index=True, width="stretch")
    with st.expander("What this result does not prove"):
        st.markdown("\n".join(f"- {lim}" for lim in LIMITATIONS))

# ---------------------------------------------------------------- 2. Drift explorer
with tabs[1]:
    st.header("Data Drift Explorer")
    st.caption(f"Feature drift = BH-adjusted p < {th.alpha} and effect size at or above its "
               f"threshold (KS D >= {th.ks_min_effect}, TVD >= {th.tvd_min_effect}).")
    feature = st.selectbox("Feature", res.drift["feature"].tolist(),
                           format_func=lambda f: f"{f}  ({'drifted' if res.drift.set_index('feature').loc[f, 'drifted'] else 'stable'})")
    row = res.drift.set_index("feature").loc[feature]
    left, right = st.columns([3, 2])
    with left:
        st.plotly_chart(plots.distribution_fig(ART.reference[feature], res.current[feature], feature),
                        width="stretch")
    with right:
        m = st.columns(3)
        m[0].metric(f"{row['effect_name']} (threshold {row['min_effect']:.2f})",
                    f"{row['effect']:.3f}")
        p = row["p_adjusted"]
        m[1].metric("BH-adjusted p", "< 0.001" if p < 0.001 else f"{p:.3f}")
        m[2].metric("Verdict", "Drifted" if row["drifted"] else "Stable")
        st.caption(f"Test: {row['test']}. {row['secondary_name']} = {row['secondary']:.3f}; "
                   f"PSI = {row['psi']:.3f}. {row['detail']}")
        summary = feature_summary(ART.reference, res.current, feature)
        if feature in config.NUMERIC_FEATURES:
            fmt = lambda v: f"{v:,.0f}" if abs(v) >= 1000 else f"{v:,.4g}"  # noqa: E731
        else:
            fmt = "{:.1%}".format
        st.dataframe(summary.style.format(fmt), width="stretch")
    if feature in config.NUMERIC_FEATURES:
        st.plotly_chart(plots.ecdf_fig(ART.reference[feature], res.current[feature], feature),
                        width="stretch")

    st.subheader("Features ranked by detected change")
    st.plotly_chart(plots.drift_ranking_fig(res.drift), width="stretch")
    st.dataframe(drift_table(res).assign(Drifted=lambda d: d["Drifted"].str.replace("*", "")),
                 hide_index=True, width="stretch")

    st.subheader("Multivariate view")
    st.plotly_chart(plots.pca_fig(pca_projection(ART, res.current, seed=res.seed)),
                    width="stretch")
    st.caption(f"PCA fitted on the training set's numerical features "
               f"(explained variance: PC1 {ART.pca.explained_variance_ratio_[0]:.0%}, "
               f"PC2 {ART.pca.explained_variance_ratio_[1]:.0%}). Exploratory only - "
               "overlapping clouds do not prove the model is reliable.")

# ---------------------------------------------------------------- 3. Model reliability
with tabs[2]:
    st.header("Model Reliability")
    key = st.radio("Model", list(config.MODEL_LABELS), format_func=config.MODEL_LABELS.get,
                   horizontal=True, key="reliability_model")
    model = ART.models[key]
    y = ART.y_test
    base_p, cur_p = ART.baseline_prob[key], res.current_prob[key]
    st.caption(f"Both batches contain the same {len(y):,} sessions with their original labels. "
               f"Decision threshold {model.threshold:.2f} (chosen on validation data). "
               f"Confidence intervals: paired bootstrap, {th.bootstrap_reps} replicates.")
    verdict = res.performance[key]
    (st.error if verdict["declined"] else st.success)(
        f"{config.MODEL_LABELS[key]}: "
        + ("material performance decline." if verdict["declined"] else
           "no material performance decline under the pre-registered rule."))
    left, right = st.columns([3, 2])
    with left:
        st.dataframe(performance_table(res, key), hide_index=True, width="stretch")
        mc = res.mcnemar[key]
        st.caption(f"McNemar test: {mc['baseline_only_correct']} sessions correct only before the "
                   f"shift vs {mc['current_only_correct']} only after (p = {mc['p_value']:.3g}).")
    with right:
        st.plotly_chart(plots.confusion_fig(res.metrics[key]["baseline"], res.metrics[key]["current"]),
                        width="stretch")
    a, b = st.columns(2)
    a.plotly_chart(plots.probability_fig(base_p, cur_p, model.threshold), width="stretch")
    b.plotly_chart(plots.roc_fig(y, base_p, cur_p), width="stretch")
    a, b = st.columns([3, 2])
    a.plotly_chart(plots.calibration_fig(y, base_p, cur_p), width="stretch")
    with b:
        st.subheader("Calibration")
        st.metric("Expected calibration error", f"{expected_calibration_error(y, cur_p):.3f}",
                  f"{expected_calibration_error(y, cur_p) - expected_calibration_error(y, base_p):+.3f}"
                  " vs baseline", delta_color="inverse")
        st.caption("Calibration is evaluated separately: a model can rank sessions well (high "
                   "AUC) while its probabilities are not meaningful likelihoods. Naive Bayes is "
                   "typically over-confident because its independence assumption is violated.")
        if key == "tree":
            st.subheader("Feature importance")
            st.plotly_chart(plots.importance_fig(model.feature_importance()), width="stretch")
    st.subheader("Classifier comparison")
    rows = []
    for k, label in config.MODEL_LABELS.items():
        for batch in ["baseline", "current"]:
            m = res.metrics[k][batch]
            rows.append({"Model": label, "Batch": batch, "ROC-AUC": m["roc_auc"], "F1": m["f1"],
                         "Precision": m["precision"], "Recall": m["recall"],
                         "Accuracy": m["accuracy"], "Brier": m["brier"]})
    st.dataframe(pd.DataFrame(rows).style.format(precision=3), hide_index=True,
                 width="stretch")

# ---------------------------------------------------------------- 4. Simulator
with tabs[3]:
    st.header("Drift Simulator")
    st.markdown(f"**{SCENARIOS[res.scenario].label}.** {SCENARIOS[res.scenario].description}")
    st.caption("Change the scenario, severity and features in the sidebar. Every run is "
               "reproducible from its seed.")

    st.subheader("Experiment to evidence")
    auc = comp_row(primary, "roc_auc")
    steps = [
        ("Create a reproducible shifted batch", " ".join(res.log)),
        ("Check data quality", f"{q['missing_cells']:,} missing, {q['invalid_values']:,} invalid, "
                               f"{q['out_of_training_range']:,} outside training range."),
        ("Detect which features changed", f"{ds['n_drifted']} drifted: "
                                          f"{', '.join(ds['drifted_features']) or 'none'}."),
        ("Compare distributions", f"Largest effect: {res.drift.iloc[0]['feature']} "
                                  f"({res.drift.iloc[0]['effect_name']} = {res.drift.iloc[0]['effect']:.3f}, "
                                  f"{'above' if res.drift.iloc[0]['material'] else 'below'} the "
                                  f"{res.drift.iloc[0]['min_effect']:.2f} threshold)."),
        ("Score with the same fixed models", "Both models were trained once and never refitted."),
        ("Measure performance on known labels",
         f"{config.MODEL_LABELS[primary]} ROC-AUC {auc['baseline']:.3f} -> {auc['current']:.3f} "
         f"({auc['delta']:+.3f})."),
    ]
    for i, (title, detail) in enumerate(steps, 1):
        st.markdown(f"**{i}. {title}** - {detail}")
    st.info(f"**Conclusion: {res.verdict['title']}.**")

    st.subheader("Severity sweep")
    if res.scenario == "control":
        st.caption("Choose a non-control scenario to sweep its severity.")
    else:
        sweep = get_sweep(res.scenario, tuple(res.features), res.seed)
        st.plotly_chart(plots.sweep_fig(sweep, th.auc_drop), width="stretch")
        st.caption("Dotted lines mark each model's baseline AUC minus the pre-registered "
                   f"{th.auc_drop} decline limit.")

    st.subheader("Contrasting experiments")
    contrast = []
    for name, p in PRESETS.items():
        r = get_experiment(p["scenario"], p["severity"], tuple(p["features"]), config.SEED, "tree")
        a = r.comparison["tree"].set_index("metric").loc["roc_auc"]
        contrast.append({"Experiment": name, "Scenario": SCENARIOS[p["scenario"]].label,
                         "Features": ", ".join(p["features"]),
                         "Drifted features": r.drift_summary["n_drifted"],
                         "Tree AUC change": round(a["delta"], 3), "Verdict": r.verdict["title"]})
    st.dataframe(pd.DataFrame(contrast), hide_index=True, width="stretch")
    st.caption("Experiment A changes inputs the model barely uses: drift is obvious but "
               "performance holds. Experiment B breaks the PageValues-purchase relationship "
               "while leaving its distribution untouched: no drift alarm, large decline. "
               "Monitoring signals must be read in context.")

# ---------------------------------------------------------------- 5. Report
with tabs[4]:
    st.header("Audit Report")
    report_md = build_markdown(res, ART)
    stem = f"modelpulse-{res.scenario}-sev{int(res.severity * 100):03d}-seed{res.seed}"
    a, b, _ = st.columns([1, 1, 3])
    a.download_button("Download Markdown", report_md, file_name=f"{stem}.md",
                      mime="text/markdown", width="stretch")
    b.download_button("Download HTML", build_html(report_md), file_name=f"{stem}.html",
                      mime="text/html", width="stretch")
    with st.container(border=True):
        st.markdown(report_md)
