"""Diagnose auf den Trainingsdaten und die Wahlregel (nur numpy/scipy). Neu geschrieben nach der Messlogik von `vormessung/eo_v11_wahlregel.py`.

Zwei F-Tests, beide nur aus den Trainingsdaten (Merkmale X, beobachtete Nachfragen y):

(a) Nichtlinearität: Verbessern die Quadrate der Merkmale die lineare Anpassung von y? Kleines Modell [1, X], großes Modell [1, X, X²];
    F = ((RSS0 - RSS1) / p) / (RSS1 / (n - 2p - 1)), p-Wert `scipy.stats.f.sf`.
(b) Streuung hängt von x ab: Erklärt [1, X] den Betrag |Residuum| der linearen Anpassung besser als eine Konstante? F-Test des Modells [1, X] gegen [1],
    p-Wert wieder `f.sf`. Erkannt wird praktisch nur eine Streuung mit **linearer** Komponente in den Merkmalen (blinder Fleck: die symmetrisch nicht monotone Streuung in |x₁|).

Regel (vor der Messung festgelegt, Schwellen nicht abgestimmt):
    p-Wert (a) < 0,01                            -> Gewichte (gewichtete Stichprobe)
    sonst p-Wert (b) < 0,01 und n >= 10 (p + 1)  -> direkt (Quantilregression)
    sonst                                        -> KQ plus empirisches Quantil

Randfälle: ist ein Modell nicht definiert (n - Zahl der Spalten <= 0), gilt p-Wert 1 (der Test spricht nicht an). Passt schon das kleine Modell bis auf
Rundung (RSS0 <= n (1e-9 * Skala)²), gibt es nichts zu erklären: p-Wert 1. Passt nur das große Modell bis auf Rundung, aber das kleine nicht: p-Wert 0.
Die Vormessung gab bei RSS1 = 0 immer 1 zurück; bei Daten mit Rauschen tritt der Fall nicht auf (die Messung ändert sich dadurch nicht)."""

from dataclasses import dataclass

import numpy as np
from scipy import stats

ALPHA = 0.01          # Schwelle beider Tests
N_PER_COLUMN = 10     # direkt nur mit n >= 10 (p + 1)
TOL_REL = 1e-9        # Rundungsgrenze relativ zur Skala der Nachfragen

RULE_GEWICHTE, RULE_DIREKT, RULE_EMP = "gewichte", "direkt", "emp"


def _rss(design: np.ndarray, y: np.ndarray) -> float:
    coef = np.linalg.lstsq(design, y, rcond=None)[0]
    r = y - design @ coef
    return float(r @ r)


def ftest_pvalue(small: np.ndarray, big: np.ndarray, y: np.ndarray, scale: float) -> float:
    """p-Wert des F-Tests „das große Modell (Designmatrix `big`) erklärt y besser als das kleine (`small`, im großen enthalten)“.

    `scale` ist die Skala der Nachfragen für die Rundungsgrenze (Skalieren aller Nachfragen ändert den p-Wert nicht)."""
    n = len(y)
    d1 = big.shape[1] - small.shape[1]
    d2 = n - big.shape[1]
    if d1 <= 0 or d2 <= 0:
        return 1.0
    floor = n * (TOL_REL * scale) ** 2
    r0, r1 = _rss(small, y), _rss(big, y)
    if r0 <= floor:
        return 1.0
    if r1 <= floor:
        return 0.0
    f = ((r0 - r1) / d1) / (r1 / d2)
    return float(stats.f.sf(max(f, 0.0), d1, d2))


@dataclass(frozen=True)
class Diagnosis:
    n: int
    p: int
    p_nonlin: float
    p_hetero: float
    enough_data: bool    # n >= 10 (p + 1)
    rule: str


def choose_rule(p_nonlin: float, p_hetero: float, n: int, p: int) -> str:
    """Die Wahlregel aus den beiden p-Werten und der Datenmenge."""
    if p_nonlin < ALPHA:
        return RULE_GEWICHTE
    if p_hetero < ALPHA and n >= N_PER_COLUMN * (p + 1):
        return RULE_DIREKT
    return RULE_EMP


def diagnose(X: np.ndarray, y: np.ndarray) -> Diagnosis:
    """Beide F-Tests auf den Trainingsdaten und die Wahl. Hängt nur von (X, y) ab."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n, p = X.shape
    scale = max(float(np.max(np.abs(y))) if n else 0.0, 1.0)
    xa = np.hstack([np.ones((n, 1)), X])
    p_nonlin = ftest_pvalue(xa, np.hstack([xa, X ** 2]), y, scale)
    coef = np.linalg.lstsq(xa, y, rcond=None)[0]
    res = np.abs(y - xa @ coef)
    res[res <= TOL_REL * scale] = 0.0
    p_hetero = ftest_pvalue(np.ones((n, 1)), xa, res, scale)
    return Diagnosis(n, p, p_nonlin, p_hetero, n >= N_PER_COLUMN * (p + 1), choose_rule(p_nonlin, p_hetero, n, p))
