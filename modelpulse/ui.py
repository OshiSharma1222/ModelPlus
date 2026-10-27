"""HTML components and stylesheet for the Streamlit app.

Everything here returns an HTML string for ``st.markdown(..., unsafe_allow_html=True)``.
Strings are emitted without blank lines or indentation so the Markdown parser
treats them as a single raw HTML block.
"""

from __future__ import annotations

from html import escape

import numpy as np
import pandas as pd

INK = "#1C1B18"
MUTED = "#6B675E"
FAINT = "#9C988D"
LINE = "#E2DFD7"
PAPER = "#FAF9F6"
REF = "#8C887E"
CUR = "#C2410C"
NAVY = "#1F3A5F"
TEAL = "#2A7F7A"
OK = "#2F6B47"
WARN = "#A66A00"
BAD = "#A8321F"
TONES = {"ok": OK, "warn": WARN, "bad": BAD, "neutral": MUTED, "accent": CUR}

CSS = f"""
<style>
:root {{ --ink:{INK}; --muted:{MUTED}; --faint:{FAINT}; --line:{LINE}; --paper:{PAPER};
  --ref:{REF}; --cur:{CUR}; --navy:{NAVY}; --ok:{OK}; --warn:{WARN}; --bad:{BAD};
  --mono:'IBM Plex Mono', ui-monospace, SFMono-Regular, Menlo, monospace; }}
[data-testid="stMainBlockContainer"] {{ max-width: 1320px; padding-top: 2.2rem; padding-bottom: 4rem; }}
[data-testid="stAppDeployButton"], [data-testid="stDecoration"] {{ display: none; }}
header[data-testid="stHeader"] {{ background: transparent; }}
[data-testid="stTabs"] [role="tablist"] {{ gap: 1.6rem; border-bottom: 1px solid var(--line); }}
[data-testid="stTabs"] button[role="tab"] {{ padding: .55rem 0; }}
[data-testid="stTabs"] button[role="tab"] p {{ font-size: .88rem; font-weight: 500; color: var(--muted); }}
[data-testid="stTabs"] button[role="tab"][aria-selected="true"] p {{ color: var(--ink); }}
[data-testid="stTabs"] [data-baseweb="tab-highlight"] {{ background: var(--ink); height: 2px; }}
[data-testid="stTabs"] [data-baseweb="tab-border"] {{ display: none; }}
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p,
[data-testid="stMain"] [data-testid="stWidgetLabel"] p {{ font-size: .7rem; font-weight: 500;
  letter-spacing: .07em; text-transform: uppercase; color: var(--muted); }}
[data-testid="stSidebarUserContent"] {{ padding-top: .5rem; }}
.mp-brand {{ display:flex; align-items:baseline; gap:.5rem; margin:.2rem 0 1.4rem; }}
.mp-brand b {{ font-size:1.15rem; font-weight:600; letter-spacing:-.01em; color:var(--ink); }}
.mp-brand span {{ font-family:var(--mono); font-size:.7rem; color:var(--faint); }}
.mp-mark {{ width:9px; height:9px; border-radius:50%; background:var(--cur); display:inline-block;
  transform:translateY(-1px); }}
.mp-side-label {{ font-size:.7rem; font-weight:500; letter-spacing:.07em; text-transform:uppercase;
  color:var(--muted); margin:1.3rem 0 .45rem; padding-top:1rem; border-top:1px solid var(--line); }}
.mp-side-note {{ font-size:.78rem; line-height:1.5; color:var(--muted); margin-top:1.2rem;
  padding-top:1rem; border-top:1px solid var(--line); }}
.mp-eyebrow {{ font-family:var(--mono); font-size:.72rem; color:var(--muted); letter-spacing:.02em; }}
.mp-eyebrow i {{ font-style:normal; color:var(--faint); padding:0 .35rem; }}
.mp-title {{ font-size:1.85rem; font-weight:600; letter-spacing:-.02em; color:var(--ink);
  margin:.35rem 0 .3rem; line-height:1.2; }}
.mp-sub {{ color:var(--muted); font-size:.95rem; max-width:62rem; margin:0 0 1.4rem; line-height:1.55; }}
.mp-section {{ display:flex; justify-content:space-between; align-items:baseline;
  border-bottom:1px solid var(--line); margin:1.8rem 0 .8rem; padding-bottom:.4rem; }}
.mp-section h3 {{ font-size:.74rem !important; font-weight:600 !important; letter-spacing:.08em;
  text-transform:uppercase; color:var(--ink) !important; margin:0 !important; padding:0 !important; }}
.mp-section span {{ font-size:.78rem; color:var(--faint); }}
.mp-verdict {{ display:grid; grid-template-columns: minmax(0,1.5fr) minmax(0,1fr); gap:2rem;
  border:1px solid var(--line); border-left:3px solid var(--tone); background:#fff;
  padding:1.25rem 1.4rem; border-radius:4px; }}
.mp-verdict .lbl {{ font-family:var(--mono); font-size:.7rem; color:var(--tone); letter-spacing:.06em; }}
.mp-verdict .head {{ font-size:1.25rem; font-weight:600; color:var(--ink); margin:.3rem 0 .45rem;
  letter-spacing:-.01em; }}
.mp-verdict p {{ margin:0; color:#3D3A34; line-height:1.6; font-size:.93rem; }}
.mp-quad {{ display:grid; grid-template-columns: 5.2rem 1fr 1fr; grid-template-rows: auto 1fr 1fr;
  font-size:.74rem; gap:3px; align-self:center; }}
.mp-quad .ax {{ color:var(--faint); font-family:var(--mono); font-size:.66rem; display:flex;
  align-items:center; }}
.mp-quad .ax.top {{ justify-content:center; padding-bottom:.2rem; }}
.mp-quad .cell {{ border:1px solid var(--line); padding:.55rem .6rem; color:var(--faint);
  background:var(--paper); min-height:2.6rem; display:flex; align-items:center; }}
.mp-quad .cell.on {{ background:var(--tone); border-color:var(--tone); color:#fff; font-weight:500; }}
.mp-tag {{ display:inline-block; margin-top:.7rem; font-family:var(--mono); font-size:.7rem;
  color:var(--bad); border:1px solid currentColor; padding:.1rem .45rem; border-radius:3px; }}
.mp-readout {{ display:grid; grid-template-columns: repeat(var(--n), minmax(0,1fr));
  border-top:1px solid var(--line); border-bottom:1px solid var(--line); margin:1.4rem 0 .4rem; }}
.mp-readout > div {{ padding:.85rem 1rem .9rem; border-left:1px solid var(--line); }}
.mp-readout > div:first-child {{ border-left:none; padding-left:0; }}
.mp-readout .k {{ font-size:.7rem; color:var(--muted); letter-spacing:.06em; text-transform:uppercase; }}
.mp-readout .v {{ font-size:1.6rem; font-weight:500; color:var(--ink); margin-top:.2rem;
  font-variant-numeric: tabular-nums; letter-spacing:-.02em; line-height:1.2; }}
.mp-readout .k {{ white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }}
.mp-readout .s {{ font-family:var(--mono); font-size:.74rem; color:var(--faint); margin-top:.1rem; }}
.mp-ledger {{ border-top:1px solid var(--line); }}
.mp-ledger > div {{ display:grid; grid-template-columns: 9.5rem 1fr; gap:1rem; padding:.6rem 0;
  border-bottom:1px solid var(--line); font-size:.9rem; line-height:1.5; }}
.mp-ledger .k {{ color:var(--muted); font-size:.8rem; padding-top:.1rem; }}
.mp-ledger code, .mp-mono {{ font-family:var(--mono); font-size:.84em; background:none; color:var(--ink); padding:0; }}
.mp-list {{ margin:0; padding-left:1.2rem; }}
.mp-list li {{ margin:.1rem 0 .55rem; line-height:1.55; font-size:.9rem; color:#3D3A34; }}
.mp-list li::marker {{ font-family:var(--mono); color:var(--faint); font-size:.8rem; }}
[data-testid="stMarkdownContainer"] table.mp-table, [data-testid="stMarkdownContainer"] .mp-report table {{
  border:none !important; width:100%; border-collapse:collapse; margin:0; }}
[data-testid="stMarkdownContainer"] .mp-table th, [data-testid="stMarkdownContainer"] .mp-table td,
[data-testid="stMarkdownContainer"] .mp-report th, [data-testid="stMarkdownContainer"] .mp-report td {{
  border:none !important; background:none !important; }}
[data-testid="stMarkdownContainer"] .mp-table th, [data-testid="stMarkdownContainer"] .mp-report th {{
  border-bottom:1px solid var(--ink) !important; }}
[data-testid="stMarkdownContainer"] .mp-table td, [data-testid="stMarkdownContainer"] .mp-report td {{
  border-bottom:1px solid var(--line) !important; }}
.mp-table {{ font-size:.85rem; }}
.mp-table th {{ text-align:left; font-weight:500; font-size:.7rem; letter-spacing:.06em;
  text-transform:uppercase; color:var(--muted); border-bottom:1px solid var(--ink); padding:.45rem .6rem .45rem 0; }}
.mp-table td {{ border-bottom:1px solid var(--line); padding:.48rem .6rem .48rem 0; color:var(--ink);
  vertical-align:top; }}
.mp-table td.n, .mp-table th.n {{ text-align:right; font-family:var(--mono); font-variant-numeric:tabular-nums;
  white-space:nowrap; }}
.mp-table td:last-child, .mp-table th:last-child {{ padding-right:0 !important; }}
.mp-table .unit {{ color:var(--faint); font-size:.72rem; margin-right:.3rem; }}
.mp-table td.n {{ font-size:.82rem; }}
.mp-table tr.group td {{ font-weight:600; padding-top:.9rem; border-bottom:1px solid var(--line); }}
.mp-table .up {{ color:var(--ok); }} .mp-table .down {{ color:var(--bad); }} .mp-table .flat {{ color:var(--faint); }}
.mp-rank td:first-child {{ width:62%; }}
.mp-rank .name {{ white-space:nowrap; }}
.mp-rank .bar {{ position:relative; height:6px; background:#EFEDE7; margin-top:.35rem; }}
.mp-rank .bar i {{ position:absolute; left:0; top:0; bottom:0; background:var(--faint); }}
.mp-rank .bar i.hit {{ background:var(--cur); }}
.mp-rank .bar b {{ position:absolute; top:-3px; bottom:-3px; width:1px; background:var(--ink); }}
.mp-dot {{ display:inline-block; width:7px; height:7px; border-radius:50%; margin-right:.4rem;
  transform:translateY(-1px); }}
.mp-pill {{ display:inline-block; font-family:var(--mono); font-size:.7rem; padding:.12rem .5rem;
  border-radius:3px; border:1px solid currentColor; letter-spacing:.03em; }}
.mp-feature-head {{ display:flex; align-items:center; gap:.8rem; margin:.2rem 0 .3rem; }}
.mp-feature-head b {{ font-size:1.15rem; font-weight:600; }}
.mp-stats {{ font-family:var(--mono); font-size:.78rem; color:var(--muted); line-height:1.7; }}
.mp-stats em {{ font-style:normal; color:var(--ink); }}
.mp-cm {{ display:grid; grid-template-columns: 1fr 1fr; gap:1.4rem; }}
.mp-cm .t {{ font-size:.72rem; letter-spacing:.06em; text-transform:uppercase; color:var(--muted); margin-bottom:.45rem; }}
.mp-cm .g {{ display:grid; grid-template-columns: 4.2rem 1fr 1fr; gap:2px; font-size:.72rem; }}
.mp-cm .h {{ color:var(--faint); font-family:var(--mono); font-size:.65rem; padding:.2rem .3rem; }}
.mp-cm .c {{ padding:.55rem .6rem; font-family:var(--mono); white-space:nowrap; }}
.mp-cm .c b {{ display:block; font-size:1.05rem; font-weight:500; }}
.mp-cm .c span {{ font-size:.66rem; opacity:.75; }}
.mp-steps {{ border-left:1px solid var(--line); margin-left:.4rem; }}
.mp-steps > div {{ position:relative; padding:0 0 1rem 1.3rem; }}
.mp-steps > div::before {{ content:""; position:absolute; left:-5px; top:.35rem; width:9px; height:9px;
  border-radius:50%; background:#fff; border:1.5px solid var(--tone, var(--faint)); }}
.mp-steps .n {{ font-family:var(--mono); font-size:.68rem; color:var(--faint); }}
.mp-steps .t {{ font-weight:500; font-size:.92rem; }}
.mp-steps .d {{ color:var(--muted); font-size:.85rem; line-height:1.5; margin-top:.1rem; }}
.mp-log {{ font-family:var(--mono); font-size:.76rem; line-height:1.65; background:#fff;
  border:1px solid var(--line); padding:.8rem 1rem; color:#3D3A34; white-space:pre-wrap; }}
.mp-cards {{ display:grid; grid-template-columns: 1fr 1fr; gap:1rem; }}
.mp-card {{ border:1px solid var(--line); background:#fff; padding:1rem 1.1rem; border-radius:4px; }}
.mp-card .lbl {{ font-family:var(--mono); font-size:.7rem; color:var(--muted); }}
.mp-card .head {{ font-weight:600; margin:.25rem 0 .7rem; }}
.mp-card .row {{ display:flex; justify-content:space-between; font-size:.85rem; padding:.3rem 0;
  border-top:1px solid var(--line); }}
.mp-card .row span:last-child {{ font-family:var(--mono); }}
[data-testid="stMarkdownContainer"] p.mp-note, .mp-note {{ font-size:.82rem !important;
  color:var(--muted); line-height:1.55; }}
.mp-legend {{ font-size:.75rem; color:var(--muted); margin-top:.7rem; display:flex; gap:1.1rem; flex-wrap:wrap; }}
.mp-limits {{ columns:2; column-gap:2.5rem; font-size:.82rem; color:var(--muted); padding-left:1.1rem; margin:0; }}
.mp-limits li {{ margin-bottom:.35rem; break-inside:avoid; }}
.mp-report {{ background:#fff; border:1px solid var(--line); padding:2.2rem 2.6rem; max-width:62rem;
  font-size:.92rem; line-height:1.6; }}
.mp-report h1 {{ font-size:1.5rem !important; margin-top:0 !important; }}
.mp-report h2 {{ font-size:1rem !important; border-bottom:1px solid var(--line); padding-bottom:.3rem !important;
  margin-top:1.8rem !important; }}
.mp-report h3 {{ font-size:.9rem !important; }}
.mp-report table {{ border-collapse:collapse; width:100%; font-size:.8rem; margin:.6rem 0 1rem; }}
.mp-report th {{ text-align:left; border-bottom:1px solid var(--ink); padding:.35rem .5rem .35rem 0; font-weight:500; }}
.mp-report td {{ border-bottom:1px solid var(--line); padding:.35rem .5rem .35rem 0; vertical-align:top; }}
.mp-filename {{ font-family:var(--mono); font-size:.8rem; color:var(--muted); padding-top:.55rem; }}
@media (max-width: 900px) {{
  .mp-verdict, .mp-cards, .mp-cm {{ grid-template-columns: 1fr; }}
  .mp-readout {{ grid-template-columns: repeat(2, minmax(0,1fr)); }}
  .mp-readout > div {{ border-left:none; padding-left:0; }}
  .mp-limits {{ columns:1; }}
}}
</style>
"""


def _flat(html: str) -> str:
    return "".join(line.strip() for line in html.splitlines())


def e(x) -> str:
    return escape(str(x))


def fmt(x, digits: int = 3) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "–"
    return f"{x:.{digits}f}"


def fmt_p(p) -> str:
    if p is None or np.isnan(p):
        return "–"
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def brand(version: str) -> str:
    return _flat(f'<div class="mp-brand"><span class="mp-mark"></span><b>ModelPulse</b>'
                 f'<span>v{e(version)}</span></div>')


def side_label(text: str) -> str:
    return f'<div class="mp-side-label">{e(text)}</div>'


def side_note(text: str) -> str:
    return f'<div class="mp-side-note">{text}</div>'


def masthead(meta: list[str], title: str, subtitle: str) -> str:
    eyebrow = "<i>/</i>".join(e(m) for m in meta)
    return _flat(f'<div class="mp-eyebrow">{eyebrow}</div><div class="mp-title">{e(title)}</div>'
                 f'<p class="mp-sub">{e(subtitle)}</p>')


def section(title: str, note: str = "") -> str:
    return _flat(f'<div class="mp-section"><h3>{e(title)}</h3><span>{e(note)}</span></div>')


def verdict_panel(title: str, text: str, drift: bool, decline: bool, quality: bool) -> str:
    tone = BAD if decline else (WARN if (drift or quality) else OK)
    names = {(False, False): "Quiet", (True, False): "Alert, no impact",
             (False, True): "Silent failure", (True, True): "Visible failure"}
    cells = ""
    for d in (False, True):
        cells += f'<div class="ax">{"drift" if d else "no drift"}</div>'
        for dec in (False, True):
            on = " on" if (d, dec) == (drift, decline) else ""
            cells += f'<div class="cell{on}">{names[(d, dec)]}</div>'
    tag = '<span class="mp-tag">+ data-quality failure</span>' if quality else ""
    return _flat(f"""
<div class="mp-verdict" style="--tone:{tone}">
<div><div class="lbl">VERDICT</div><div class="head">{e(title)}</div><p>{e(text)}</p>{tag}</div>
<div class="mp-quad"><div></div><div class="ax top">held</div><div class="ax top">declined</div>{cells}</div>
</div>""")


def readout(cells: list[tuple]) -> str:
    """cells: (label, value, sub, tone)"""
    parts = []
    for label, value, sub, tone in cells:
        color = TONES.get(tone, FAINT) if tone else FAINT
        parts.append(f'<div><div class="k">{e(label)}</div><div class="v">{e(value)}</div>'
                     f'<div class="s" style="color:{color}">{e(sub)}</div></div>')
    return _flat(f'<div class="mp-readout" style="--n:{len(cells)}">{"".join(parts)}</div>')


def ledger(rows: list[tuple[str, str]]) -> str:
    """rows: (label, html_value)"""
    body = "".join(f'<div><div class="k">{e(k)}</div><div>{v}</div></div>' for k, v in rows)
    return _flat(f'<div class="mp-ledger">{body}</div>')


def ordered(items: list[str]) -> str:
    return _flat('<ol class="mp-list">' + "".join(f"<li>{e(i)}</li>" for i in items) + "</ol>")


def delta_cell(delta: float, higher_is_better: bool = True, digits: int = 3) -> str:
    if delta is None or np.isnan(delta):
        return '<td class="n flat">–</td>'
    if abs(delta) < 0.5 * 10 ** -digits:
        return f'<td class="n flat">{0:+.{digits}f}</td>'
    good = (delta > 0) == higher_is_better
    return f'<td class="n {"up" if good else "down"}">{delta:+.{digits}f}</td>'


def table(header: list[tuple[str, bool]], rows: list[str]) -> str:
    """header: (label, numeric); rows: pre-rendered <tr> strings."""
    th = "".join(f'<th class="{"n" if num else ""}">{e(h)}</th>' for h, num in header)
    return _flat(f'<table class="mp-table"><thead><tr>{th}</tr></thead><tbody>{"".join(rows)}</tbody></table>')


def effect_rank(drift: pd.DataFrame, max_ratio: float = 3.0) -> str:
    """Ranked features with an inline bar; the tick marks the drift threshold."""
    rows = []
    for _, r in drift.iterrows():
        ratio = 0.0 if np.isnan(r["effect_ratio"]) else min(r["effect_ratio"], max_ratio)
        color = CUR if r["drifted"] else (WARN if r["significant"] else FAINT)
        rows.append(
            f'<tr><td><div class="name"><span class="mp-dot" style="background:{color}"></span>'
            f'{e(r["feature"])}</div><div class="bar"><i class="{"hit" if r["drifted"] else ""}" '
            f'style="width:{ratio / max_ratio * 100:.1f}%"></i><b style="left:{100 / max_ratio:.1f}%"></b>'
            f'</div></td><td class="n"><span class="unit">{e(r["effect_name"])}</span>{fmt(r["effect"])}</td>'
            f'<td class="n">{fmt_p(r["p_adjusted"])}</td></tr>')
    legend = (f'<div class="mp-legend"><span><span class="mp-dot" style="background:{CUR}"></span>drifted</span>'
              f'<span><span class="mp-dot" style="background:{WARN}"></span>significant but small</span>'
              f'<span><span class="mp-dot" style="background:{FAINT}"></span>stable</span></div>')
    return '<div class="mp-rank">' + table(
        [("Feature", False), ("Effect", True), ("p (BH)", True)], rows) + legend + "</div>"


def pill(text: str, tone: str) -> str:
    return f'<span class="mp-pill" style="color:{TONES[tone]}">{e(text)}</span>'


def _rgba(hex_color: str, alpha: float) -> str:
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (1, 3, 5))
    return f"rgba({r},{g},{b},{alpha:.2f})"


def confusion(baseline: dict, current: dict) -> str:
    """Two 2x2 matrices. Correct cells are tinted with the batch colour,
    errors in red; tint strength is the share of the actual-class row."""
    def grid(title, m, color):
        rows = [("actual no", [("TN", m["tn"], True), ("FP", m["fp"], False)]),
                ("actual yes", [("FN", m["fn"], False), ("TP", m["tp"], True)])]
        html = (f'<div><div class="t">{e(title)}</div><div class="g"><div></div>'
                '<div class="h">pred. no</div><div class="h">pred. yes</div>')
        for row_label, cells in rows:
            total = sum(c for _, c, _ in cells) or 1
            html += f'<div class="h">{row_label}</div>'
            for name, count, correct in cells:
                share = count / total
                bg = _rgba(color, 0.10 + 0.45 * share) if correct else _rgba(BAD, 0.04 + 0.30 * share)
                html += (f'<div class="c" style="background:{bg}"><b>{count:,}</b>'
                         f'<span>{name} {share:.0%}</span></div>')
        return html + "</div></div>"

    return _flat('<div class="mp-cm">' + grid("Baseline batch", baseline, REF)
                 + grid("Current batch", current, CUR) + "</div>"
                 + '<p class="mp-note" style="margin-top:.6rem">Percentages are shares of the actual '
                 "class. Correct cells are shaded in the batch colour, errors in red.</p>")


def steps(items: list[tuple[str, str, str]]) -> str:
    """items: (title, detail, tone)"""
    body = "".join(
        f'<div style="--tone:{TONES.get(tone, FAINT)}"><div class="n">{i:02d}</div>'
        f'<div class="t">{e(t)}</div><div class="d">{e(d)}</div></div>'
        for i, (t, d, tone) in enumerate(items, 1))
    return _flat(f'<div class="mp-steps">{body}</div>')


def log_block(lines: list[str]) -> str:
    return '<div class="mp-log">' + "<br>".join(e(line) for line in lines) + "</div>"


def contrast_cards(cards: list[dict]) -> str:
    html = '<div class="mp-cards">'
    for c in cards:
        rows = "".join(f'<div class="row"><span>{e(k)}</span><span>{e(v)}</span></div>'
                       for k, v in c["rows"])
        html += (f'<div class="mp-card"><div class="lbl">{e(c["label"])}</div>'
                 f'<div class="head">{e(c["title"])}</div>{rows}</div>')
    return _flat(html + "</div>")


def limits(items: list[str]) -> str:
    return _flat('<ul class="mp-limits">' + "".join(f"<li>{e(i)}</li>" for i in items) + "</ul>")


def report(html_body: str) -> str:
    return f'<div class="mp-report">{html_body}</div>'
