"""Auswertung: Diagnose und Verfahren auf einer Instanz, Live-Band über feste Instanzen, Verdikte als reine Funktionen."""

from dataclasses import dataclass
from statistics import median

import numpy as np

import elw_algorithm as A
import elw_constants as C
import elw_diagnose as D
import elw_scenario as S
import elw_vormessung as V
import elw_weights as W


@dataclass(frozen=True)
class Settings:
    welt: str = "streuung-linear"
    tau: float = C.DEFAULT_TAU
    n: int = C.DEFAULT_N
    p: int = C.DEFAULT_P
    seed: int = C.DEFAULT_SEED


@dataclass(frozen=True)
class Analysis:
    settings: Settings
    diagnosis: D.Diagnosis
    extras: dict          # Mehrkosten gegen das Orakel in % je Verfahren (emp, direkt, kern, knn)
    rule: str             # Wahl der Regel: emp, direkt oder gewichte
    rule_extra: float     # Mehrkosten des gewählten Verfahrens (bei „gewichte“: Gauß-Kern)
    best_method: str      # bestes Verfahren dieser Instanz (im Nachhinein, mit Kenntnis der Welt)
    best_extra: float


def de(x: float, digits: int = 1) -> str:
    """Zahl mit Dezimalkomma."""
    return f"{x:.{digits}f}".replace(".", ",")


def pct(x: float, digits: int = 1) -> str:
    return de(x, digits) + " %"


def tau_text(tau: float) -> str:
    return de(tau, 2) if round(tau * 100) % 10 else de(tau, 1)


def p_text(pv: float) -> str:
    """p-Wert für die Anzeige: sehr kleine Werte als „< 0,0001“."""
    return "< 0,0001" if pv < 1e-4 else de(pv, 4)


def extras_for_taus(data: S.Dataset, taus, methods=C.METHODEN) -> dict:
    """Mehrkosten in % gegen das Orakel je Niveau und Verfahren auf EINER Instanz; Abstände und Diagnose werden von den Niveaus geteilt."""
    dist = W.distances(data.X, data.Xt) if ("kern" in methods or "knn" in methods) else None
    out = {}
    for tau in taus:
        row = {}
        for m in methods:
            if m in ("emp", "direkt"):
                q = A.predict(A.fit_rule(m, data.X, data.y, tau), data.Xt)
            elif m == "kern":
                q = W.kernel_rule(data.X, data.y, data.Xt, tau, dist)[0]
            else:
                q = W.knn_rule(data.X, data.y, data.Xt, tau, dist)[0]
            row[m] = S.extra_cost_percent(data.world, data.Xt, q, tau)
        out[tau] = row
    return out


def pick(extras: dict, rule: str) -> float:
    return extras[C.REGEL_METHODE[rule]]


def analyse(settings: Settings) -> Analysis:
    data = S.make_dataset(settings.welt, settings.p, settings.n, settings.seed)
    diag = D.diagnose(data.X, data.y)
    extras = extras_for_taus(data, (settings.tau,))[settings.tau]
    best = min(C.METHODEN, key=lambda m: extras[m])
    return Analysis(settings, diag, extras, diag.rule, pick(extras, diag.rule), best, extras[best])


def band(welt: str, p: int, n: int, tau: float, seeds=C.SWEEP_SEEDS) -> dict:
    """Live-Band über die festen Instanzen: Mehrkosten je Verfahren, der Regel und der besten Wahl je Instanz; dazu die Wahl der Regel je Instanz."""
    runs = [analyse(Settings(welt, tau, n, p, s)) for s in seeds]
    out = {m: [r.extras[m] for r in runs] for m in C.METHODEN}
    out["regel"] = [r.rule_extra for r in runs]
    out["beste"] = [r.best_extra for r in runs]
    out["wahl"] = [r.rule for r in runs]
    return out


def detection_counts(welt: str, p: int, n: int, seeds=C.DIAG_SEEDS) -> dict:
    """Wie oft schlagen die beiden F-Tests an (p-Wert unter 0,01), über feste Instanzen; nur Diagnose, kein Lösen."""
    diags = [D.diagnose(*_train(welt, p, n, s)) for s in seeds]
    return {"runs": len(diags), "nonlin": sum(d.p_nonlin < D.ALPHA for d in diags), "hetero": sum(d.p_hetero < D.ALPHA for d in diags),
            "wahl": {r: sum(d.rule == r for d in diags) for r in (D.RULE_EMP, D.RULE_DIREKT, D.RULE_GEWICHTE)}}


def _train(welt: str, p: int, n: int, seed: int):
    data = S.make_dataset(welt, p, n, seed)
    return data.X, data.y


def abs_residuals(welt: str, p: int, n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """(x₁, |Residuum| der linearen Anpassung) der Trainingsdaten, für die Abbildung in Experiment 4."""
    X, y = _train(welt, p, n, seed)
    coef = A.ols_fit(X, y)
    return X[:, 0], np.abs(y - A.predict(coef, X))


def slope_of(x: np.ndarray, y: np.ndarray) -> float:
    """Steigung der Geraden von y auf x (Kleinste Quadrate)."""
    xc = x - x.mean()
    return float(xc @ (y - y.mean()) / (xc @ xc))


# --- Verdikte (reine Funktionen: Art, Text) --------------------------------------------------------------------------------------------------------


def verdict_instance(a: Analysis) -> tuple[str, str]:
    """Verdikt von Experiment 2 für die eine Instanz: Wahl der Regel, ihre Mehrkosten, die beste Wahl, immer-KQ und immer-direkt."""
    s, d = a.settings, a.diagnosis
    e = a.extras
    wahl = C.REGEL_NAMEN[a.rule]
    gewicht_hinweis = " (hier mit Gauß-Kerngewichten gerechnet, nicht mit einem Wald)" if a.rule == D.RULE_GEWICHTE else ""
    text = (f"Bei dieser Welt (p = {s.p}, n = {s.n}, τ = {tau_text(s.tau)}, Zufalls-Seed {s.seed}) wählt die Regel **{wahl}**{gewicht_hinweis} mit Mehrkosten von **{pct(a.rule_extra)}**; "
            f"die beste Wahl hätte {pct(a.best_extra)} gehabt ({C.METHODE_NAMEN[a.best_method]}), immer KQ + Quantil {pct(e['emp'])}, immer direkt {pct(e['direkt'])}, immer Gewichte {pct(e['kern'])}.")
    if a.rule_extra <= a.best_extra + 1.0 or a.rule_extra <= 1.05 * a.best_extra:
        return "success", text
    return "warning", text + " Auf dieser Instanz liegt die Regel merklich über der besten Wahl."


def verdict_band(b: dict, welt: str, p: int, n: int, tau: float) -> tuple[str, str]:
    """Verdikt des Live-Bandes (Median über die festen Instanzen)."""
    k = len(b["regel"])
    med = {m: median(b[m]) for m in (*C.METHODEN, "regel", "beste")}
    wahl = {r: b["wahl"].count(r) for r in (D.RULE_EMP, D.RULE_DIREKT, D.RULE_GEWICHTE)}
    bis = [v for v in (med["emp"], med["direkt"], med["kern"])]
    text = (f"Welt „{C.WELT_NAMEN[welt]}“, p = {p}, n = {n}, τ = {tau_text(tau)}, Median über {k} feste Instanzen: Regel {pct(med['regel'])}, beste Wahl je Instanz {pct(med['beste'])}, "
            f"immer KQ + Quantil {pct(med['emp'])}, immer direkt {pct(med['direkt'])}, immer Gewichte {pct(med['kern'])}. "
            f"Die Regel wählte KQ + Quantil in {wahl[D.RULE_EMP]}, direkt in {wahl[D.RULE_DIREKT]} und die Gewichte in {wahl[D.RULE_GEWICHTE]} von {k} Instanzen.")
    if med["regel"] <= min(bis) + 1.0 or med["regel"] <= 1.05 * min(bis):
        return "success", text + " Die Regel liegt im Median bei dem besten der drei festen Verfahren."
    return "warning", text + " Die Regel liegt im Median über dem besten festen Verfahren dieser Welt."


def verdict_blind(counts_lin: dict, counts_abs: dict, n: int, p: int) -> tuple[str, str]:
    """Verdikt von Experiment 4: wie oft die Streuungs-Diagnose bei linearer und bei nicht monotoner Streuung anschlägt."""
    k = counts_lin["runs"]
    return "info", (f"Bei n = {n}, p = {p} schlägt die Diagnose auf die Streuung bei linear wachsender Streuung in **{counts_lin['hetero']} von {k}** festen Instanzen an, "
                    f"bei Streuung in |x₁| in **{counts_abs['hetero']} von {k}**. Die Regel wählt dort in {counts_abs['wahl'][D.RULE_EMP]} von {k} Instanzen KQ + Quantil, "
                    f"obwohl die Streuung von x₁ abhängt: der blinde Fleck der Diagnose.")


def verdict_rule_vs_fixed() -> tuple[str, str]:
    """Verdikt von Experiment 3 aus der eingefrorenen Vormessung und dem Nachbau (Seeds 0 bis 29 und 100 bis 129)."""
    ga, gb = V.frozen_set("a")["gesamt"], V.frozen_set("b")["gesamt"]
    name_a, best_a = V.best_fixed("a")
    name_b, best_b = V.best_fixed("b")
    wq = V.frozen_set("a")["welten"]["nichtlinear"]
    nb = V.summarize(V.rows("a"))
    nb_best = min(nb[k]["mittel"] for k in ("emp", "direkt", "kern"))
    med_name = min(V.FIXED, key=lambda k: ga[k]["median"])
    text = (f"Vormessung mit Zufallswald (nicht live): Die Regel kostet im Mittel über alle Zeilen {de(ga['regel']['mittel'], 2)} %, das beste feste Verfahren nach dem Mittel ({V.FIXED_NAMEN[name_a]}) {de(best_a, 2)} %: "
            f"**{de(V.reduction_vs_best_fixed('a'), 0)} % weniger Mehrkosten** im Mittel über die vier Welten, nicht die Hälfte. "
            f"Gegen immer KQ + Quantil und immer direkt entsteht die Ersparnis in der Welt „Wahrheit quadratisch“ ({de(wq['regel'], 1)} % gegen {de(wq['emp'], 1)} / {de(wq['direkt'], 1)} %); "
            f"gegen {V.FIXED_NAMEN[name_a]}, das Vergleichsverfahren der {de(V.reduction_vs_best_fixed('a'), 0)} %, ist die Regel dort gleich ({de(wq['regel'], 1)} gegen {de(wq[name_a], 1)} %) und gewinnt in den drei übrigen Welten. "
            f"In diesen drei Welten hat je ein anderes festes Verfahren kleinere Mehrkosten als die Regel; kein festes Verfahren war in allen vier Welten besser. "
            f"Im Median kostet die Regel {de(ga['regel']['median'], 2)} % gegen {de(ga[med_name]['median'], 2)} % ({V.FIXED_NAMEN[med_name]}). "
            f"Die beste Wahl je Zeile mit Kenntnis der Welt kostet im Mittel {de(ga['beste']['mittel'], 2)} %. "
            f"Unabhängige Zufalls-Seeds 100 bis 129: {de(gb['regel']['mittel'], 2)} % gegen {de(best_b, 2)} % ({V.FIXED_NAMEN[name_b]}), {de(V.reduction_vs_best_fixed('b'), 0)} % weniger. "
            f"Nachbau mit Gauß-Kerngewichten statt Wald (Seeds 0 bis 29): {de(nb['regel']['mittel'], 1)} % gegen {de(nb_best, 1)} %, "
            f"{de(100 * (nb_best - nb['regel']['mittel']) / nb_best, 0)} % weniger. Gilt für diese vier Welten, nicht für echte Daten.")
    return "success", text
