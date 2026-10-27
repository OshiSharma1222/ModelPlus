"""Plotly figures used by the Streamlit app.

Colour is semantic and consistent across every chart: warm grey is the
reference / baseline batch, burnt orange is the current batch.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

from . import config
from .evaluation import calibration_table, roc_curve_points
from .ui import CUR, FAINT, INK, LINE, MUTED, NAVY, REF, TEAL

MODEL_COLORS = {"Decision Tree": NAVY, "Naive Bayes": TEAL}
FONT = "IBM Plex Sans, -apple-system, Segoe UI, sans-serif"
MONO = "IBM Plex Mono, ui-monospace, monospace"

pio.templates["modelpulse"] = go.layout.Template(layout=go.Layout(
    font=dict(family=FONT, size=12, color=INK),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    colorway=[REF, CUR, NAVY, TEAL],
    hoverlabel=dict(bgcolor="#FFFFFF", bordercolor=LINE, font=dict(family=MONO, size=11, color=INK)),
    xaxis=dict(showgrid=False, zeroline=False, linecolor=INK, linewidth=1, ticks="outside",
               tickcolor=LINE, ticklen=4, tickfont=dict(family=MONO, size=10, color=MUTED),
               title=dict(font=dict(size=11, color=MUTED))),
    yaxis=dict(gridcolor="#ECE9E2", zeroline=False, showline=False,
               tickfont=dict(family=MONO, size=10, color=MUTED),
               title=dict(font=dict(size=11, color=MUTED))),
    legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0, xanchor="left",
                font=dict(size=11, color=MUTED), bgcolor="rgba(0,0,0,0)"),
    margin=dict(l=8, r=8, t=56, b=8),
    bargap=0.04,
))
pio.templates.default = "modelpulse"


def _finish(fig: go.Figure, title: str = "", height: int = 340) -> go.Figure:
    fig.update_layout(
        template="modelpulse", height=height,
        title=dict(text=title, x=0, xanchor="left", y=0.985, yanchor="top",
                   font=dict(size=12, color=INK)),
    )
    return fig


def distribution_fig(ref: pd.Series, cur: pd.Series, feature: str) -> go.Figure:
    if feature in config.NUMERIC_FEATURES:
        r, c = ref.dropna(), cur.dropna()
        hi = float(np.nanpercentile(np.r_[r, c], 99)) if len(r) + len(c) else 1.0
        lo = float(min(r.min(), c.min()))
        bins = dict(start=lo, end=hi, size=max(hi - lo, 1e-9) / 40)
        fig = go.Figure()
        fig.add_histogram(x=r.clip(upper=hi), name="Reference", histnorm="probability",
                          marker_color=REF, opacity=0.55, xbins=bins)
        fig.add_histogram(x=c.clip(upper=hi), name="Current", histnorm="probability",
                          marker=dict(color="rgba(0,0,0,0)", line=dict(color=CUR, width=1.4)),
                          xbins=bins)
        fig.update_layout(barmode="overlay")
        fig.update_yaxes(tickformat=".0%")
        fig.update_xaxes(title="value (top 1% grouped in last bin)")
        return _finish(fig, "Distribution")
    r = ref.astype("object").fillna("<missing>").astype(str).value_counts(normalize=True)
    c = cur.astype("object").fillna("<missing>").astype(str).value_counts(normalize=True)
    levels = sorted(r.index.union(c.index), key=lambda lv: -r.get(lv, 0))
    fig = go.Figure()
    fig.add_bar(x=levels, y=[r.get(lv, 0) for lv in levels], name="Reference", marker_color=REF)
    fig.add_bar(x=levels, y=[c.get(lv, 0) for lv in levels], name="Current", marker_color=CUR)
    fig.update_layout(barmode="group", bargap=0.25, bargroupgap=0.05)
    fig.update_xaxes(type="category")
    fig.update_yaxes(tickformat=".0%")
    return _finish(fig, "Category shares")


def ecdf_fig(ref: pd.Series, cur: pd.Series, feature: str) -> go.Figure:
    """Empirical CDFs; the KS statistic is their largest vertical gap."""
    fig = go.Figure()
    xs = {}
    for name, s, color in [("Reference", ref, REF), ("Current", cur, CUR)]:
        x = np.sort(s.dropna().to_numpy())
        y = np.arange(1, len(x) + 1) / max(len(x), 1)
        xs[name] = (x, y)
        step = max(len(x) // 2000, 1)
        fig.add_scatter(x=x[::step], y=y[::step], mode="lines", name=name,
                        line=dict(color=color, shape="hv", width=1.6))
    # Mark the KS gap: the largest vertical distance between the two curves.
    (xr, _), (xc, _) = xs["Reference"], xs["Current"]
    title = "Cumulative distribution"
    if len(xr) and len(xc):
        grid = np.unique(np.r_[xr, xc])
        fr = np.searchsorted(xr, grid, side="right") / len(xr)
        fc = np.searchsorted(xc, grid, side="right") / len(xc)
        i = int(np.argmax(np.abs(fr - fc)))
        fig.add_shape(type="line", x0=grid[i], x1=grid[i], y0=fr[i], y1=fc[i],
                      line=dict(color=INK, width=2))
        title += f" — the black bar is the largest gap, KS D = {abs(fr[i] - fc[i]):.3f}"
    positive = ref.dropna()
    if (positive > 0).mean() > 0.5 and positive.max() > 100:
        fig.update_xaxes(type="log", title="value (log scale)")
    fig.update_yaxes(tickformat=".0%")
    return _finish(fig, title)


def pca_fig(proj: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for name, color in [("Reference", REF), ("Current", CUR)]:
        p = proj[proj["batch"] == name]
        fig.add_scatter(x=p["PC1"], y=p["PC2"], mode="markers", name=name,
                        marker=dict(color=color, size=4, opacity=0.4))
    fig.update_xaxes(title="PC1", showgrid=True, gridcolor="#ECE9E2")
    fig.update_yaxes(title="PC2")
    return _finish(fig, "", height=380)


def probability_fig(base_prob, cur_prob, threshold: float) -> go.Figure:
    fig = go.Figure()
    bins = dict(start=0, end=1, size=0.025)
    fig.add_histogram(x=base_prob, name="Baseline", histnorm="probability", xbins=bins,
                      marker_color=REF, opacity=0.55)
    fig.add_histogram(x=cur_prob, name="Current", histnorm="probability", xbins=bins,
                      marker=dict(color="rgba(0,0,0,0)", line=dict(color=CUR, width=1.4)))
    fig.add_vline(x=threshold, line=dict(color=INK, width=1, dash="dot"))
    fig.add_annotation(x=threshold, y=1, yref="paper", text=f"threshold {threshold:.2f}",
                       showarrow=False, xanchor="left", yanchor="top",
                       font=dict(family=MONO, size=10, color=MUTED))
    fig.update_layout(barmode="overlay")
    fig.update_xaxes(title="predicted probability", range=[0, 1])
    fig.update_yaxes(tickformat=".0%")
    return _finish(fig, "Predicted purchase probability", 300)


def roc_fig(y, base_prob, cur_prob) -> go.Figure:
    fig = go.Figure()
    fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", showlegend=False, hoverinfo="skip",
                    line=dict(color=LINE, width=1))
    for name, p, color in [("Baseline", base_prob, REF), ("Current", cur_prob, CUR)]:
        pts = roc_curve_points(y, p)
        fig.add_scatter(x=pts["fpr"], y=pts["tpr"], mode="lines", name=name,
                        line=dict(color=color, width=1.8))
    fig.update_xaxes(title="false positive rate", range=[0, 1])
    fig.update_yaxes(title="true positive rate", range=[0, 1.02])
    return _finish(fig, "ROC curve", 300)


def calibration_fig(y, base_prob, cur_prob) -> go.Figure:
    fig = go.Figure()
    fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", showlegend=False, hoverinfo="skip",
                    line=dict(color=LINE, width=1))
    for name, p, color in [("Baseline", base_prob, REF), ("Current", cur_prob, CUR)]:
        t = calibration_table(y, p)
        fig.add_scatter(x=t["mean_predicted"], y=t["observed_rate"], mode="lines+markers",
                        name=name, line=dict(color=color, width=1.5),
                        marker=dict(size=np.clip(np.sqrt(t["n"]) / 2, 4, 16), color=color),
                        customdata=t["n"], hovertemplate="predicted %{x:.2f}<br>observed "
                        "%{y:.2f}<br>n = %{customdata}<extra></extra>")
    fig.update_xaxes(title="mean predicted probability", range=[0, 1])
    fig.update_yaxes(title="observed purchase rate", range=[0, 1.02])
    return _finish(fig, "Calibration (marker size = sessions)", 300)


def importance_fig(importance: pd.Series, top: int = 8) -> go.Figure:
    imp = importance.head(top).iloc[::-1]
    fig = go.Figure(go.Bar(x=imp.values, y=imp.index, orientation="h", marker_color=NAVY,
                           text=[f"{v:.0%}" for v in imp.values], textposition="outside",
                           textfont=dict(family=MONO, size=10, color=MUTED), cliponaxis=False))
    fig.update_layout(showlegend=False, bargap=0.35)
    fig.update_xaxes(visible=False, range=[0, imp.max() * 1.15])
    fig.update_yaxes(showgrid=False, tickfont=dict(family=FONT, size=11, color=INK))
    return _finish(fig, "", 280)


def sweep_fig(sweep: pd.DataFrame, auc_drop: float, current_severity: float | None = None) -> go.Figure:
    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.09,
                        subplot_titles=("Share of features flagged as drifted", "ROC-AUC"))
    first = sweep[sweep["model"] == sweep["model"].iloc[0]]
    fig.add_scatter(x=first["severity"], y=first["drift_share"], mode="lines+markers",
                    name="Drifted features", line=dict(color=CUR, width=1.8),
                    marker=dict(size=6), row=1, col=1)
    for model, g in sweep.groupby("model"):
        color = MODEL_COLORS.get(model)
        fig.add_scatter(x=g["severity"], y=g["roc_auc"], mode="lines+markers", name=model,
                        line=dict(color=color, width=1.8), marker=dict(size=6), row=1, col=2)
        base = g.loc[g["severity"] == g["severity"].min(), "roc_auc"].iloc[0]
        fig.add_hline(y=base - auc_drop, line=dict(color=color, width=1, dash="dot"),
                      opacity=0.6, row=1, col=2)
    if current_severity is not None:
        for col in (1, 2):
            fig.add_vline(x=current_severity, line=dict(color=FAINT, width=1), row=1, col=col)
    fig.update_yaxes(tickformat=".0%", range=[0, max(0.5, first["drift_share"].max() * 1.15)],
                     row=1, col=1)
    fig.update_xaxes(title="severity")
    for ann, axis in zip(fig.layout.annotations, ["xaxis", "xaxis2"]):
        ann.update(x=fig.layout[axis].domain[0], xanchor="left", font=dict(size=12, color=INK))
    fig.update_layout(legend=dict(orientation="h", yanchor="top", y=-0.2, x=0))
    return _finish(fig, "", 380)

