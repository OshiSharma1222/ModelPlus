"""ModelPulse - interactive AI reliability lab (Streamlit app).

Run with:  streamlit run app.py
"""

from __future__ import annotations

import markdown as md
import numpy as np
import pandas as pd
import streamlit as st

from modelpulse import __version__, config, plots, ui
from modelpulse.drift import feature_summary
from modelpulse.evaluation import expected_calibration_error
from modelpulse.experiment import (
    load_or_build_artifacts, pca_projection, recommendations, run_experiment, severity_sweep,
)
from modelpulse.report import LIMITATIONS, METRIC_LABELS, build_html, build_markdown
from modelpulse.scenarios import SCENARIOS

st.set_page_config(page_title="ModelPulse", page_icon=":material/vital_signs:", layout="wide")
st.markdown(ui.CSS, unsafe_allow_html=True)

PRESETS = {
    "A": dict(scenario="numeric_shift", severity=0.8,
              features=list(SCENARIOS["numeric_shift"].default_features)),
    "B": dict(scenario="relationship_shift", severity=0.8, features=["PageValues"]),
}
PLOT_CONFIG = {"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d"]}


def html(fragment: str) -> None:
    st.markdown(fragment, unsafe_allow_html=True)


def chart(fig) -> None:
    st.plotly_chart(fig, width="stretch", config=PLOT_CONFIG)


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

# --------------------------------------------------------------------------- sidebar
state = st.session_state
state.setdefault("scenario", "control")
state.setdefault("severity", 0.5)
state.setdefault("features", [])
state.setdefault("seed", config.SEED)
state.setdefault("primary", "tree")


def reset_features():
    state.features = list(SCENARIOS[state.scenario].default_features)


def reset_control():
    state.scenario = "control"
    reset_features()


def apply_preset(name: str):
    p = PRESETS[name]
    state.scenario, state.severity, state.features = p["scenario"], p["severity"], list(p["features"])


with st.sidebar:
    html(ui.brand(__version__))
    st.selectbox("Scenario", list(SCENARIOS), key="scenario",
                 format_func=lambda k: SCENARIOS[k].label, on_change=reset_features)
    is_control = state.scenario == "control"
    st.slider("Severity", 0.0, 1.0, step=0.05, key="severity", disabled=is_control,
              help="0 leaves the batch untouched; 1 is the most severe version of the scenario.")
    st.multiselect("Features", list(SCENARIOS[state.scenario].eligible), key="features",
                   disabled=is_control, placeholder="None")
    st.number_input("Seed", min_value=0, max_value=10_000_000, step=1, key="seed")
    st.radio("Verdict model", list(config.MODEL_LABELS), key="primary", horizontal=True,
             format_func=config.MODEL_LABELS.get)

    html(ui.side_label("Demonstrations"))
    st.button("A — drift that doesn't matter", on_click=apply_preset, args=("A",), width="stretch")
    st.button("B — failure that drift tests miss", on_click=apply_preset, args=("B",), width="stretch")
    st.button("Reset to control", on_click=reset_control, width="stretch", type="tertiary")
    html(ui.side_note(
        f"Reference is the training set ({ART.info['n_train']:,} sessions). Each current batch "
        f"is a modified copy of the held-out test set ({ART.info['n_test']:,} sessions); labels "
        "are never touched, so every comparison is on identical sessions."
        "<br><br>Data: UCI Online Shoppers Purchasing Intention, CC BY 4.0."))

primary = state.primary
res = get_experiment(state.scenario, float(state.severity), tuple(state.features),
                     int(state.seed), primary)
th, ds, q = res.thresholds, res.drift_summary, res.quality
spec = SCENARIOS[res.scenario]
SHORT = {"tree": "tree", "nb": "NB"}

# Point the feature inspector at the biggest change whenever a new experiment runs.
run_id = (res.scenario, res.severity, tuple(res.features), res.seed)
if state.get("last_run") != run_id:
    state.last_run = run_id
    state.drift_feature = res.drift["feature"].iloc[0]


def comp(key: str) -> pd.DataFrame:
    return res.comparison[key].set_index("metric")


# --------------------------------------------------------------------------- masthead
meta = [f"seed {res.seed}", res.scenario]
if res.scenario != "control":
    meta += [f"severity {res.severity:.2f}", ", ".join(res.features) or "no features"]
html(ui.masthead(meta, spec.label, spec.description))

tabs = st.tabs(["Overview", "Data drift", "Model reliability", "Drift simulator", "Audit report"])

# --------------------------------------------------------------------------- 1 overview
with tabs[0]:
    html(ui.verdict_panel(res.verdict["title"], res.verdict["explanation"],
                          res.verdict["drift"], res.verdict["decline"], res.verdict["quality_issue"]))

    auc, f1 = comp(primary).loc["roc_auc"], comp(primary).loc["f1"]
    perf_tone = lambda d, limit: "bad" if d <= -limit else ("ok" if d >= 0 else "neutral")  # noqa: E731
    html(ui.readout([
        ("Reference", f"{len(ART.reference):,}", "training sessions", None),
        ("Current", f"{len(res.current):,}", "test sessions", None),
        ("Drifted features", f"{ds['n_drifted']} / {ds['n_features']}", ds["status"].lower(),
         "accent" if ds["n_drifted"] else "ok"),
        ("Missing · invalid", f"{q['missing_cells']:,} · {q['invalid_values']:,}",
         f"{q['rows_with_missing']:,} rows affected" if q["rows_with_missing"] else "clean",
         "bad" if q["rows_with_missing"] or q["invalid_values"] else "ok"),
        (f"ROC-AUC · {SHORT[primary]}", f"{auc['current']:.3f}",
         f"{auc['delta']:+.3f} from {auc['baseline']:.3f}", perf_tone(auc["delta"], th.auc_drop)),
        (f"F1 · {SHORT[primary]}", f"{f1['current']:.3f}",
         f"{f1['delta']:+.3f} from {f1['baseline']:.3f}", perf_tone(f1["delta"], th.f1_drop)),
    ]))

    left, right = st.columns([1.25, 1], gap="large")
    with left:
        html(ui.section("Evidence"))
        rows = []
        if ds["drifted_features"]:
            top = res.drift.iloc[0]
            rows.append(("Input drift",
                         f"{ds['n_drifted']} feature(s) passed both the significance and effect-size "
                         f"rules. Largest: <code>{ui.e(top['feature'])}</code>, "
                         f"{ui.e(top['effect_name'])} {top['effect']:.3f} ({ui.e(top['detail'])})."))
        else:
            rows.append(("Input drift", "No feature passed both the significance and effect-size rules."))
        if ds["n_significant_but_small"]:
            rows.append(("Small changes", f"{ds['n_significant_but_small']} feature(s) were significant "
                                          "but below the minimum effect size, so they are not flagged."))
        for key, label in config.MODEL_LABELS.items():
            a = comp(key).loc["roc_auc"]
            ci = (f" (95% CI {a['delta_low']:+.3f} to {a['delta_high']:+.3f})"
                  if "delta_low" in a else "")
            word = "a material decline" if res.performance[key]["declined"] else "within limits"
            rows.append((label, f"ROC-AUC {a['delta']:+.3f}{ci}, {word}."))
        if q["rows_with_missing"]:
            rows.append(("Data quality", f"{q['rows_with_missing']:,} sessions had missing inputs "
                                         "and were scored on training-median imputations."))
        if q["out_of_training_range"]:
            rows.append(("Extrapolation", f"{q['out_of_training_range']:,} values fall outside the "
                                          "range seen in training."))
        html(ui.ledger(rows))
        html(ui.section("What to investigate"))
        html(ui.ordered(recommendations(res, ART)))
    with right:
        html(ui.section("Performance", "same sessions, before → after"))
        trs = []
        for key, label in config.MODEL_LABELS.items():
            trs.append(f'<tr class="group"><td colspan="4">{label}</td></tr>')
            c = comp(key)
            for metric in ["roc_auc", "f1", "precision", "recall"]:
                r = c.loc[metric]
                trs.append(f"<tr><td>{METRIC_LABELS[metric]}</td><td class='n'>{r['baseline']:.3f}</td>"
                           f"<td class='n'>{r['current']:.3f}</td>{ui.delta_cell(r['delta'])}</tr>")
        html(ui.table([("Metric", False), ("Before", True), ("After", True), ("Change", True)], trs))

        html(ui.section("Data quality"))
        qt = q["table"].set_index("feature")
        issues = qt[(qt > 0).any(axis=1)]
        if issues.empty:
            html('<p class="mp-note">All 17 features passed the schema, missing-value, domain-range '
                 "and category checks.</p>")
        else:
            trs = [f"<tr><td>{ui.e(f)}</td>" + "".join(f"<td class='n'>{int(v):,}</td>" for v in r.values)
                   + "</tr>" for f, r in issues.iterrows()]
            html(ui.table([("Feature", False), ("Missing", True), ("Invalid", True),
                           ("Out of range", True), ("Unseen", True)], trs))

    html(ui.section("What this result does not prove"))
    html(ui.limits(LIMITATIONS))

# --------------------------------------------------------------------------- 2 drift
with tabs[1]:
    left, right = st.columns([1, 1.45], gap="large")
    with left:
        html(ui.section("Features ranked by effect", f"line = threshold · BH α = {th.alpha}"))
        html(ui.effect_rank(res.drift))
        html(f'<p class="mp-note" style="margin-top:.8rem">Numerical: KS D ≥ {th.ks_min_effect}. '
             f"Categorical: total variation distance ≥ {th.tvd_min_effect}. A feature is flagged only "
             "when it is significant after Benjamini–Hochberg correction <i>and</i> passes its "
             "effect-size threshold.</p>")
    with right:
        html(ui.section("Inspect a feature"))
        feature = st.selectbox("Feature", res.drift["feature"].tolist(), key="drift_feature",
                               label_visibility="collapsed")
        row = res.drift.set_index("feature").loc[feature]
        tone, label = (("accent", "drifted") if row["drifted"] else
                       ("warn", "significant, small") if row["significant"] else ("neutral", "stable"))
        html(f'<div class="mp-feature-head"><b>{ui.e(feature)}</b>{ui.pill(label, tone)}</div>'
             f'<div class="mp-stats">{ui.e(row["test"])} · <em>{ui.e(row["effect_name"])} '
             f'{row["effect"]:.3f}</em> (threshold {row["min_effect"]:.2f}) · p<sub>BH</sub> '
             f'<em>{ui.fmt_p(row["p_adjusted"])}</em> · {ui.e(row["secondary_name"])} '
             f'{row["secondary"]:.3f} · PSI {row["psi"]:.3f}<br>{ui.e(row["detail"])}</div>')
        chart(plots.distribution_fig(ART.reference[feature], res.current[feature], feature))
        summary = feature_summary(ART.reference, res.current, feature)
        if feature in config.NUMERIC_FEATURES:
            chart(plots.ecdf_fig(ART.reference[feature], res.current[feature], feature))
            num = lambda v: f"{v:,.0f}" if abs(v) >= 1000 else f"{v:,.4g}"  # noqa: E731
            trs = []
            for stat in ["mean", "median", "variance", "std_dev", "min", "max", "missing"]:
                rv, cv = summary.loc[stat, "reference"], summary.loc[stat, "current"]
                change = (cv - rv) / abs(rv) if rv else np.nan
                trs.append(f"<tr><td>{stat.replace('_', ' ')}</td><td class='n'>{num(rv)}</td>"
                           f"<td class='n'>{num(cv)}</td><td class='n flat'>"
                           f"{'–' if np.isnan(change) else f'{change:+.0%}'}</td></tr>")
            html(ui.table([("Statistic", False), ("Reference", True), ("Current", True),
                           ("Change", True)], trs))
        else:
            trs = [f"<tr><td>{ui.e(lv)}</td><td class='n'>{r['reference']:.1%}</td>"
                   f"<td class='n'>{r['current']:.1%}</td>"
                   f"<td class='n flat'>{r['difference'] * 100:+.1f} pp</td></tr>"
                   for lv, r in summary.iterrows()]
            html(ui.table([("Level", False), ("Reference", True), ("Current", True),
                           ("Change", True)], trs))

    html(ui.section("Multivariate view", "PCA on standardised numerical features"))
    left, right = st.columns([2.2, 1], gap="large")
    with left:
        chart(plots.pca_fig(pca_projection(ART, res.current, seed=res.seed)))
    with right:
        ev = ART.pca.explained_variance_ratio_
        html(f'<p class="mp-note">The projection is fitted on the training set and explains '
             f'{ev[0]:.0%} (PC1) and {ev[1]:.0%} (PC2) of the variance. 1,500 sessions are sampled '
             "from each batch.</p><p class='mp-note'>It is exploratory: overlapping clouds do not "
             "prove the model is reliable, and Experiment B is a case where they overlap "
             "perfectly while performance collapses.</p>")

# --------------------------------------------------------------------------- 3 reliability
with tabs[2]:
    key = st.segmented_control("Model", list(config.MODEL_LABELS), key="reliability_model",
                               format_func=config.MODEL_LABELS.get, default="tree",
                               label_visibility="collapsed") or "tree"
    model = ART.models[key]
    y = ART.y_test
    base_p, cur_p = ART.baseline_prob[key], res.current_prob[key]
    declined = res.performance[key]["declined"]
    html(f'<p class="mp-note" style="margin:.4rem 0 0">'
         f'{ui.pill("material decline" if declined else "within limits", "bad" if declined else "ok")}'
         f'&nbsp; {len(y):,} identical sessions · decision threshold {model.threshold:.2f} '
         f"(chosen on validation data) · paired bootstrap, {th.bootstrap_reps} replicates</p>")

    left, right = st.columns([1.15, 1], gap="large")
    with left:
        html(ui.section("Metrics", "95% CI of the change"))
        trs = []
        for _, r in res.comparison[key].iterrows():
            higher_better = r["metric"] != "brier"
            ci = (f"{r['delta_low']:+.3f} … {r['delta_high']:+.3f}" if "delta_low" in r else "–")
            trs.append(f"<tr><td>{METRIC_LABELS[r['metric']]}</td><td class='n'>{r['baseline']:.3f}</td>"
                       f"<td class='n'>{r['current']:.3f}</td>{ui.delta_cell(r['delta'], higher_better)}"
                       f"<td class='n flat'>{ci}</td></tr>")
        html(ui.table([("Metric", False), ("Before", True), ("After", True), ("Change", True),
                       ("95% CI", True)], trs))
        mc = res.mcnemar[key]
        html(f'<p class="mp-note" style="margin-top:.7rem">McNemar: {mc["baseline_only_correct"]:,} '
             f'sessions were classified correctly only before the shift and {mc["current_only_correct"]:,} '
             f'only after (exact p = {mc["p_value"]:.2g}).</p>')
    with right:
        html(ui.section("Confusion matrices"))
        html(ui.confusion(res.metrics[key]["baseline"], res.metrics[key]["current"]))

    html(ui.section("Scores"))
    cols = st.columns(3, gap="medium")
    with cols[0]:
        chart(plots.probability_fig(base_p, cur_p, model.threshold))
    with cols[1]:
        chart(plots.roc_fig(y, base_p, cur_p))
    with cols[2]:
        chart(plots.calibration_fig(y, base_p, cur_p))
    ece_b, ece_c = expected_calibration_error(y, base_p), expected_calibration_error(y, cur_p)
    html(f'<p class="mp-note">Expected calibration error {ece_b:.3f} → {ece_c:.3f}. Calibration is '
         "assessed separately from ranking: a model can order sessions well (high AUC) while its "
         "probabilities are not trustworthy likelihoods. Naive Bayes is a clear example — its "
         "independence assumption makes it over-confident.</p>")

    left, right = st.columns([1.3, 1], gap="large")
    with left:
        html(ui.section("Classifier comparison"))
        trs = []
        for k, label in config.MODEL_LABELS.items():
            for batch in ["baseline", "current"]:
                m = res.metrics[k][batch]
                trs.append(f"<tr><td>{label if batch == 'baseline' else ''}</td><td>{batch}</td>"
                           + "".join(f"<td class='n'>{ui.fmt(m[x])}</td>"
                                     for x in ["roc_auc", "f1", "precision", "recall", "brier"])
                           + "</tr>")
        html(ui.table([("Model", False), ("Batch", False), ("AUC", True), ("F1", True),
                       ("Precision", True), ("Recall", True), ("Brier", True)], trs))
    with right:
        html(ui.section("Decision Tree importance", "Gini, top 8"))
        chart(plots.importance_fig(ART.models["tree"].feature_importance()))

# --------------------------------------------------------------------------- 4 simulator
with tabs[3]:
    left, right = st.columns([1, 1.25], gap="large")
    with left:
        html(ui.section("Transformation log"))
        html(ui.log_block(res.log))
        html(ui.section("Pre-registered rules"))
        html(ui.ledger([
            ("Feature drift", f"BH-adjusted p &lt; {th.alpha} and KS D ≥ {th.ks_min_effect} / "
                              f"TVD ≥ {th.tvd_min_effect}"),
            ("Batch status", f"“Drift detected” at ≥ {th.drift_share_alert:.0%} of features, "
                             "“Warning” at one or more"),
            ("Material decline", f"AUC −{th.auc_drop} or F1 −{th.f1_drop}, with the paired 95% CI "
                                 "excluding zero"),
        ]))
    with right:
        html(ui.section("Experiment → evidence"))
        auc = comp(primary).loc["roc_auc"]
        top = res.drift.iloc[0]
        html(ui.steps([
            ("Create a reproducible batch",
             f"{spec.label} at severity {res.severity:.2f}, seed {res.seed}; "
             f"{len(res.current):,} sessions, labels unchanged.", "neutral"),
            ("Check data quality",
             f"{q['missing_cells']:,} missing, {q['invalid_values']:,} invalid, "
             f"{q['out_of_training_range']:,} outside the training range.",
             "bad" if res.verdict["quality_issue"] else "ok"),
            ("Test every feature for drift",
             f"{ds['n_drifted']} of {ds['n_features']} flagged"
             + (f": {', '.join(ds['drifted_features'])}." if ds["drifted_features"] else "."),
             "accent" if ds["n_drifted"] else "ok"),
            ("Compare distributions",
             f"Largest effect {top['feature']} ({top['effect_name']} {top['effect']:.3f}, "
             f"{'above' if top['material'] else 'below'} its {top['min_effect']:.2f} threshold).",
             "accent" if top["material"] else "ok"),
            ("Score with the same fixed models", "Both models were trained once and are never refitted.",
             "neutral"),
            ("Measure performance on known labels",
             f"{config.MODEL_LABELS[primary]} ROC-AUC {auc['baseline']:.3f} → {auc['current']:.3f} "
             f"({auc['delta']:+.3f}).", "bad" if res.verdict["decline"] else "ok"),
        ]))
        html(f'<p style="margin:.2rem 0 0 1.7rem;font-weight:600">→ {ui.e(res.verdict["title"])}</p>')

    html(ui.section("Severity sweep", "every severity from 0 to 1, same seed and features"))
    if res.scenario == "control":
        html('<p class="mp-note">Choose a scenario other than the control to sweep its severity.</p>')
    else:
        chart(plots.sweep_fig(get_sweep(res.scenario, tuple(res.features), res.seed), th.auc_drop,
                              res.severity))
        html('<p class="mp-note">Dotted lines mark each model’s baseline AUC minus the '
             f"{th.auc_drop} decline limit; the grey line is the severity currently selected.</p>")

    html(ui.section("Two contrasting experiments", "seed 2026, Decision Tree"))
    cards = []
    for name, p in PRESETS.items():
        r = get_experiment(p["scenario"], p["severity"], tuple(p["features"]), config.SEED, "tree")
        a = r.comparison["tree"].set_index("metric").loc["roc_auc"]
        cards.append({
            "label": f"EXPERIMENT {name} · {SCENARIOS[p['scenario']].label.lower()}",
            "title": r.verdict["title"],
            "rows": [("Modified", ", ".join(p["features"])),
                     ("Features flagged", f"{r.drift_summary['n_drifted']} / 17"),
                     ("ROC-AUC", f"{a['baseline']:.3f} → {a['current']:.3f}"),
                     ("Recall", f"{r.metrics['tree']['baseline']['recall']:.3f} → "
                                f"{r.metrics['tree']['current']['recall']:.3f}")],
        })
    html(ui.contrast_cards(cards))
    html('<p class="mp-note" style="margin-top:.8rem">A triples browsing durations — features the '
         "tree barely uses — so the alarm is loud and harmless. B shuffles PageValues among "
         "sessions: its distribution is identical, no test can fire, and the model fails. "
         "Monitoring signals have to be read in context.</p>")

# --------------------------------------------------------------------------- 5 report
with tabs[4]:
    report_md = build_markdown(res, ART)
    stem = f"modelpulse-{res.scenario}-sev{int(round(res.severity * 100)):03d}-seed{res.seed}"
    left, b1, b2 = st.columns([3, 1, 1])
    left.markdown(f'<div class="mp-filename">{stem}.md · {len(report_md.splitlines())} lines</div>',
                  unsafe_allow_html=True)
    b1.download_button("Download .md", report_md, file_name=f"{stem}.md", mime="text/markdown",
                       width="stretch")
    b2.download_button("Download .html", build_html(report_md), file_name=f"{stem}.html",
                       mime="text/html", width="stretch", type="primary")
    html(ui.report(md.markdown(report_md, extensions=["tables"])))
