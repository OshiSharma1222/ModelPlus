"""Plotly figures used by the Streamlit app."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from . import config
from .evaluation import calibration_table, roc_curve_points

REF_COLOR = "#4263EB"
CUR_COLOR = "#F76707"
OK_COLOR = "#94A3B8"
ALERT_COLOR = "#E03131"
MODEL_COLORS = {"Decision Tree": "#4263EB", "Naive Bayes": "#0CA678"}

LAYOUT = dict(template="plotly_white", margin=dict(l=10, r=10, t=80, b=10), height=380,
              legend=dict(orientation="h", yanchor="bottom", y=1.01, x=0, xanchor="left"))


def titled(text: str) -> dict:
    """Title pinned to the top-left so it never collides with the legend."""
    return dict(text=text, x=0.01, xanchor="left", y=0.98, yanchor="top", font=dict(size=15))


def distribution_fig(ref: pd.Series, cur: pd.Series, feature: str) -> go.Figure:
    if feature in config.NUMERIC_FEATURES:
        r, c = ref.dropna(), cur.dropna()
        hi = float(np.nanpercentile(np.r_[r, c], 99)) if len(r) + len(c) else 1.0
        bins = dict(start=float(min(r.min(), c.min())), end=hi, size=max(hi, 1e-9) / 40)
        fig = go.Figure()
        fig.add_histogram(x=r.clip(upper=hi), name="Reference", histnorm="probability",
                          marker_color=REF_COLOR, opacity=0.6, xbins=bins)
        fig.add_histogram(x=c.clip(upper=hi), name="Current", histnorm="probability",
                          marker_color=CUR_COLOR, opacity=0.6, xbins=bins)
        fig.update_layout(barmode="overlay", title=titled(f"{feature} (values above the 99th "
                          "percentile are grouped in the last bin)"), **LAYOUT)
        fig.update_yaxes(title="Share of sessions", tickformat=".0%")
        return fig
    r = ref.astype("object").fillna("<missing>").astype(str).value_counts(normalize=True)
    c = cur.astype("object").fillna("<missing>").astype(str).value_counts(normalize=True)
    levels = list(r.index.union(c.index))
    levels.sort(key=lambda lv: -r.get(lv, 0))
    fig = go.Figure()
    fig.add_bar(x=levels, y=[r.get(lv, 0) for lv in levels], name="Reference", marker_color=REF_COLOR)
    fig.add_bar(x=levels, y=[c.get(lv, 0) for lv in levels], name="Current", marker_color=CUR_COLOR)
    fig.update_layout(barmode="group", title=titled(f"{feature}: category proportions"), **LAYOUT)
    fig.update_xaxes(type="category")
    fig.update_yaxes(title="Share of sessions", tickformat=".0%")
    return fig


def ecdf_fig(ref: pd.Series, cur: pd.Series, feature: str) -> go.Figure:
    """Empirical CDFs; the KS statistic is their largest vertical gap."""
    fig = go.Figure()
    for name, s, color in [("Reference", ref, REF_COLOR), ("Current", cur, CUR_COLOR)]:
        x = np.sort(s.dropna().to_numpy())
        y = np.arange(1, len(x) + 1) / max(len(x), 1)
        step = max(len(x) // 2000, 1)
        fig.add_scatter(x=x[::step], y=y[::step], mode="lines", name=name,
                        line=dict(color=color, shape="hv"))
    fig.update_layout(title=titled(f"{feature}: cumulative distribution (KS = largest vertical gap)"), **LAYOUT)
    fig.update_xaxes(type="log" if (ref.dropna() > 0).mean() > 0.5 and ref.max() > 100 else "linear")
    fig.update_yaxes(title="Cumulative share", tickformat=".0%")
    return fig


def drift_ranking_fig(drift: pd.DataFrame) -> go.Figure:
    d = drift.sort_values("effect_ratio")
    colors = np.where(d["drifted"], ALERT_COLOR, np.where(d["significant"], "#FAB005", OK_COLOR))
    fig = go.Figure(go.Bar(
        x=d["effect_ratio"], y=d["feature"], orientation="h", marker_color=colors,
        customdata=np.c_[d["effect_name"], d["effect"].round(3), d["p_adjusted"]],
        hovertemplate="%{y}<br>%{customdata[0]} = %{customdata[1]}<br>"
                      "BH-adjusted p = %{customdata[2]:.2g}<extra></extra>",
    ))
    fig.add_vline(x=1, line_dash="dash", line_color="#495057",
                  annotation_text="effect threshold", annotation_position="top")
    fig.update_layout(title=titled("Effect size relative to its drift threshold "
                      "(red = drifted, amber = significant but small)"),
                      **{**LAYOUT, "height": 480})
    fig.update_xaxes(title="Effect / threshold")
    return fig


def pca_fig(proj: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for name, color in [("Reference", REF_COLOR), ("Current", CUR_COLOR)]:
        p = proj[proj["batch"] == name]
        fig.add_scatter(x=p["PC1"], y=p["PC2"], mode="markers", name=name,
                        marker=dict(color=color, size=5, opacity=0.45))
    fig.update_layout(title=titled("PCA of numerical features (exploratory only)"), **{**LAYOUT, "height": 420})
    fig.update_xaxes(title="PC1")
    fig.update_yaxes(title="PC2")
    return fig


def confusion_fig(baseline: dict, current: dict) -> go.Figure:
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Baseline batch", "Current batch"),
                        horizontal_spacing=0.12)
    for col, m in [(1, baseline), (2, current)]:
        z = [[m["tn"], m["fp"]], [m["fn"], m["tp"]]]
        text = [[f"TN<br>{m['tn']:,}", f"FP<br>{m['fp']:,}"], [f"FN<br>{m['fn']:,}", f"TP<br>{m['tp']:,}"]]
        fig.add_heatmap(z=z, x=["No purchase", "Purchase"],
                        y=["No purchase", "Purchase"], text=text,
                        texttemplate="%{text}", colorscale="Blues", showscale=False,
                        row=1, col=col)
    fig.update_yaxes(autorange="reversed")
    fig.update_yaxes(title="Actual", row=1, col=1)
    fig.update_yaxes(showticklabels=False, row=1, col=2)
    fig.update_xaxes(title="Predicted", tickangle=0)
    fig.update_layout(**{**LAYOUT, "height": 330, "margin": dict(l=10, r=10, t=40, b=10)})
    return fig


def probability_fig(base_prob, cur_prob, threshold: float) -> go.Figure:
    fig = go.Figure()
    bins = dict(start=0, end=1, size=0.025)
    fig.add_histogram(x=base_prob, name="Baseline", histnorm="probability", xbins=bins,
                      marker_color=REF_COLOR, opacity=0.6)
    fig.add_histogram(x=cur_prob, name="Current", histnorm="probability", xbins=bins,
                      marker_color=CUR_COLOR, opacity=0.6)
    fig.add_vline(x=threshold, line_dash="dash", annotation_text=f"threshold {threshold:.2f}")
    fig.update_layout(barmode="overlay", title=titled("Predicted purchase probability"), **LAYOUT)
    fig.update_xaxes(title="Predicted probability", range=[0, 1])
    fig.update_yaxes(title="Share of sessions", tickformat=".0%")
    return fig


def roc_fig(y, base_prob, cur_prob) -> go.Figure:
    fig = go.Figure()
    for name, p, color in [("Baseline", base_prob, REF_COLOR), ("Current", cur_prob, CUR_COLOR)]:
        pts = roc_curve_points(y, p)
        fig.add_scatter(x=pts["fpr"], y=pts["tpr"], mode="lines", name=name, line=dict(color=color))
    fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", name="Chance",
                    line=dict(color="#ADB5BD", dash="dot"))
    fig.update_layout(title=titled("ROC curve"), **LAYOUT)
    fig.update_xaxes(title="False positive rate", range=[0, 1])
    fig.update_yaxes(title="True positive rate", range=[0, 1])
    return fig


def calibration_fig(y, base_prob, cur_prob) -> go.Figure:
    fig = go.Figure()
    fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", name="Perfect calibration",
                    line=dict(color="#ADB5BD", dash="dot"))
    for name, p, color in [("Baseline", base_prob, REF_COLOR), ("Current", cur_prob, CUR_COLOR)]:
        t = calibration_table(y, p)
        fig.add_scatter(x=t["mean_predicted"], y=t["observed_rate"], mode="lines+markers",
                        name=name, line=dict(color=color), marker=dict(size=np.sqrt(t["n"]) + 3),
                        customdata=t["n"], hovertemplate="predicted %{x:.2f}<br>observed "
                        "%{y:.2f}<br>n = %{customdata}<extra></extra>")
    fig.update_layout(title=titled("Calibration (marker size ~ sessions in bin)"), **LAYOUT)
    fig.update_xaxes(title="Mean predicted probability", range=[0, 1])
    fig.update_yaxes(title="Observed purchase rate", range=[0, 1])
    return fig


def importance_fig(importance: pd.Series, top: int = 8) -> go.Figure:
    imp = importance.head(top).iloc[::-1]
    fig = go.Figure(go.Bar(x=imp.values, y=imp.index, orientation="h", marker_color=REF_COLOR))
    fig.update_layout(**{**LAYOUT, "height": 300, "showlegend": False})
    fig.update_xaxes(title="Gini importance", tickformat=".0%")
    return fig


def sweep_fig(sweep: pd.DataFrame, auc_drop: float) -> go.Figure:
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Share of drifted features", "ROC-AUC"))
    first = sweep[sweep["model"] == sweep["model"].iloc[0]]
    fig.add_scatter(x=first["severity"], y=first["drift_share"], mode="lines+markers",
                    name="Drifted features", line=dict(color=ALERT_COLOR), row=1, col=1)
    for model, g in sweep.groupby("model"):
        fig.add_scatter(x=g["severity"], y=g["roc_auc"], mode="lines+markers", name=model,
                        line=dict(color=MODEL_COLORS.get(model)), row=1, col=2)
        base = g.loc[g["severity"] == g["severity"].min(), "roc_auc"].iloc[0]
        fig.add_hline(y=base - auc_drop, line_dash="dot", line_color=MODEL_COLORS.get(model),
                      opacity=0.5, row=1, col=2)
    fig.update_yaxes(tickformat=".0%", range=[0, 1], row=1, col=1)
    fig.update_xaxes(title="Severity")
    fig.update_layout(**{**LAYOUT, "height": 400, "margin": dict(l=10, r=10, t=40, b=10),
                         "legend": dict(orientation="h", yanchor="top", y=-0.18, x=0)})
    return fig
