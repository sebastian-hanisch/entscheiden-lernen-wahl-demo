"""Kern: die zwei parametrischen Bestellregeln des Newsvendors mit Merkmalen (nur numpy/scipy).

Herkunft: gekürzte Kopie aus `newsvendor-direkt-demo/nvd_algorithm.py` (Kleinste Quadrate, Rangregel des Quantils, Quantilregression als lineares Programm);
die L1-Strafe, die Kreuzvalidierung und die Regel „Mittel bestellen“ sind nicht übernommen.

* `emp`:    Kleinste Quadrate (KQ) plus empirisches Quantil der Residuen (Rang ceil(tau * n), unabhängig von x): zweistufig.
* `direkt`: lineare Quantilregression zum Niveau tau (Pinball-Verlust = Newsvendor-Kosten bis auf den Faktor b + h), ein lineares Programm
            (`scipy.optimize.linprog`, HiGHS, dünne Matrizen).

Beide Regeln liefern Koeffizienten (c0, c1, ..., cp) der Bestellmenge q(x) = c0 + x . c."""

import numpy as np
from scipy import sparse
from scipy.optimize import linprog

RULES = ("emp", "direkt")


def add_intercept(X: np.ndarray) -> np.ndarray:
    return np.hstack([np.ones((len(X), 1)), X])


def ols_fit(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Kleinste Quadrate mit Achsenabschnitt; Koeffizienten (c0, c1, ..., cp). Bei Rangmangel die Lösung kleinster Norm."""
    return np.linalg.lstsq(add_intercept(X), y, rcond=None)[0]


def predict(coef: np.ndarray, X: np.ndarray) -> np.ndarray:
    return coef[0] + X @ coef[1:]


def empirical_quantile(values: np.ndarray, tau: float) -> float:
    """Das ceil(tau * n)-kleinste Element (Rangregel, Quantil der empirischen Verteilung)."""
    v = np.sort(np.asarray(values, dtype=float))
    if len(v) == 0:
        raise ValueError("leere Stichprobe")
    rank = min(max(int(np.ceil(tau * len(v) - 1e-12)), 1), len(v))
    return float(v[rank - 1])


def quantreg_fit(X: np.ndarray, y: np.ndarray, tau: float) -> np.ndarray:
    """Lineare Quantilregression zum Niveau tau.

    LP: min  (tau * sum u + (1 - tau) * sum v) / n   mit  y - c0 - X c = u - v,  u, v >= 0,  c0 und c frei.
    Rückgabe (c0, c1, ..., cp). Bei Mehrdeutigkeit (etwa n <= p) liefert HiGHS eine von mehreren gleich guten Lösungen."""
    if not 0.0 < tau < 1.0:
        raise ValueError("tau muss zwischen 0 und 1 liegen (ohne die Grenzen)")
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n, p = X.shape
    cost = np.concatenate([np.zeros(p + 1), tau * np.ones(n) / n, (1.0 - tau) * np.ones(n) / n])
    eye = sparse.identity(n, format="csr")
    A_eq = sparse.hstack([sparse.csr_matrix(np.hstack([np.ones((n, 1)), X])), eye, -eye], format="csr")
    bounds = [(None, None)] * (p + 1) + [(0, None)] * (2 * n)
    res = linprog(cost, A_eq=A_eq, b_eq=y, bounds=bounds, method="highs")
    if res.status != 0:
        raise RuntimeError(f"Quantilregression: LP nicht gelöst ({res.message})")
    return np.asarray(res.x[: p + 1], dtype=float)


def fit_rule(rule: str, X: np.ndarray, y: np.ndarray, tau: float) -> np.ndarray:
    """Koeffizienten (c0, ..., cp) der Regel `emp` oder `direkt`."""
    if rule == "emp":
        coef = ols_fit(X, y)
        shifted = coef.copy()
        shifted[0] += empirical_quantile(y - predict(coef, X), tau)
        return shifted
    if rule == "direkt":
        return quantreg_fit(X, y, tau)
    raise ValueError(f"unbekannte Regel: {rule}")


def pinball_loss(q: np.ndarray, y: np.ndarray, tau: float) -> float:
    """Mittlerer Pinball-Verlust zum Niveau tau: tau * (y - q)+ + (1 - tau) * (q - y)+."""
    r = np.asarray(y, dtype=float) - np.asarray(q, dtype=float)
    return float(np.mean(np.maximum(tau * r, (tau - 1.0) * r)))
