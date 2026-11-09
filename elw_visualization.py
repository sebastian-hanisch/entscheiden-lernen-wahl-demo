"""Plotly-Abbildungen. Alle Achsen sind gesperrt (fixedrange), damit Touch-Geräte beim Scrollen nicht am Diagramm hängen bleiben."""

from statistics import median

import numpy as np
import plotly.graph_objects as go

import elw_constants as C


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _base(fig, height=360):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=-0.25), plot_bgcolor="rgba(0,0,0,0)")
    return lock_axes(fig)


def build_methods(extras: dict, rule_method: str, best_method: str):
    """Mehrkosten der vier Verfahren auf EINER Instanz; die Wahl der Regel und die beste Wahl sind in der Beschriftung markiert."""
    labels = []
    for m in C.METHODEN:
        tag = []
        if m == rule_method:
            tag.append("Regel")
        if m == best_method:
            tag.append("beste")
        labels.append(C.METHODE_NAMEN[m] + (f" ({', '.join(tag)})" if tag else ""))
    fig = go.Figure(go.Bar(x=labels, y=[extras[m] for m in C.METHODEN], marker_color=[C.METHODE_FARBEN[m] for m in C.METHODEN],
                           marker_line_color=["#b0413e" if m == rule_method else "rgba(0,0,0,0)" for m in C.METHODEN],
                           marker_line_width=[3 if m == rule_method else 0 for m in C.METHODEN], hovertemplate="%{x}: %{y:.1f} %<extra></extra>"))
    fig.update_yaxes(title_text="Mehrkosten gegen das Orakel (%)")
    return _base(fig)


def build_band(b: dict):
    """Median der Mehrkosten über die festen Instanzen mit Fehlerbalken (Minimum bis Maximum): drei feste Verfahren, die Regel, die beste Wahl je Instanz."""
    keys = ["emp", "direkt", "kern", "regel", "beste"]
    names = ["immer KQ + Quantil", "immer direkt", "immer Gewichte", "Regel", "beste Wahl je Instanz"]
    colors = [C.COLOR_EMP, C.COLOR_DIREKT, C.COLOR_GEWICHTE, C.COLOR_REGEL, C.COLOR_BEST]
    med = [median(b[k]) for k in keys]
    fig = go.Figure(go.Bar(x=names, y=med, marker_color=colors,
                           error_y=dict(type="data", symmetric=False, array=[max(b[k]) - m for k, m in zip(keys, med)], arrayminus=[m - min(b[k]) for k, m in zip(keys, med)]),
                           hovertemplate="%{x}: Median %{y:.1f} %<extra></extra>"))
    fig.update_yaxes(title_text="Mehrkosten gegen das Orakel (%)")
    return _base(fig)


def build_rule_vs_fixed(frozen: dict, rebuilt: dict, title_frozen: str):
    """Mittlere Mehrkosten über die Zeilen: Vormessung (Wald, eingefroren) neben dem Nachbau (Gauß-Kern, vorgerechnet).

    `frozen` hat die Schlüssel emp, direkt, wald, regel, beste (Zahlen), `rebuilt` emp, direkt, kern, regel, beste."""
    names = ["immer KQ + Quantil", "immer direkt", "immer Gewichte", "Regel", "beste Wahl je Zeile"]
    fig = go.Figure()
    fig.add_trace(go.Bar(name=title_frozen, x=names, y=[frozen[k] for k in ("emp", "direkt", "wald", "regel", "beste")], marker_color=C.COLOR_EMP,
                         text=[f"{frozen[k]:.1f}".replace(".", ",") for k in ("emp", "direkt", "wald", "regel", "beste")], textposition="outside",
                         hovertemplate="%{x}: %{y:.2f} %<extra>Vormessung</extra>"))
    fig.add_trace(go.Bar(name="Nachbau mit dem Code dieser Demo (Gewichte = Gauß-Kern)", x=names, y=[rebuilt[k] for k in ("emp", "direkt", "kern", "regel", "beste")],
                         marker_color=C.COLOR_GEWICHTE, text=[f"{rebuilt[k]:.1f}".replace(".", ",") for k in ("emp", "direkt", "kern", "regel", "beste")], textposition="outside",
                         hovertemplate="%{x}: %{y:.2f} %<extra>Nachbau</extra>"))
    fig.update_layout(barmode="group")
    fig.update_yaxes(title_text="Mittlere Mehrkosten gegen das Orakel (%)")
    return _base(fig, 400)


def build_abs_residuals(x: np.ndarray, r: np.ndarray, title: str):
    """|Residuum| der linearen Anpassung über x₁ mit der Geraden der Streuungs-Diagnose."""
    xs = np.linspace(float(x.min()), float(x.max()), 2)
    xc = x - x.mean()
    slope = float(xc @ (r - r.mean()) / (xc @ xc))
    line = r.mean() + slope * (xs - x.mean())
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=r, mode="markers", marker=dict(size=5, color="rgba(76,120,168,0.5)"), name="|Residuum|", hovertemplate="x₁ = %{x:.2f}: %{y:.1f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=xs, y=line, mode="lines", line=dict(color=C.COLOR_REGEL, width=3), name="lineare Anpassung (was die Diagnose prüft)"))
    fig.update_xaxes(title_text="Merkmal x₁")
    fig.update_yaxes(title_text="Betrag des Residuums")
    fig.update_layout(title=dict(text=title, font=dict(size=13)), margin=dict(t=40))
    return lock_axes(fig.update_layout(height=320, legend=dict(orientation="h", y=-0.3), plot_bgcolor="rgba(0,0,0,0)"))
