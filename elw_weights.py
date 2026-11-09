"""Gewichtete Stichprobe: Gewichte aus Ähnlichkeit und das gewichtete Quantil (SAA mit Gewichten), reine numpy-Einheiten.

Herkunft: Kopie aus `gewichtete-stichprobe-demo` (`gws_weights.py`: Abstände, Nachbarn, kNN- und Kerngewichte, gewichtetes Quantil, Leave-one-out;
`gws_algorithm.py`: die beiden Regeln `knn_rule` und `kernel_rule`), auf das Nötige gekürzt, die Konstanten (Raster, Toleranz) stehen hier statt in
`gws_constants.py`. Die Regeln liefern nur die Bestellmengen der Testfälle.

Zu einem neuen Fall x bekommt jede vergangene Beobachtung i ein Gewicht w_i(x) >= 0 mit Summe 1; die Bestellmenge ist das kleinste q, dessen kumuliertes
Gewicht der Nachfragen D_i <= q mindestens tau erreicht. Zufallswald-Gewichte werden hier NICHT gerechnet (kein scikit-learn zur Laufzeit)."""

import numpy as np

KGRID = (5, 10, 20, 40, 80, 160)        # Nachbarzahlen k, Wahl per Leave-one-out (nur Werte <= n - 1)
HGRID = (0.3, 0.5, 0.8, 1.2, 2.0, 3.0)  # Bandbreiten, bezogen auf p = 3; gerechnet wird h * sqrt(p / 3)
LOO_TOL = 1e-12                         # Toleranz beim Vergleich der kumulierten Gewichte mit tau


def sqdist(a: np.ndarray, b: np.ndarray, chunk: int = 256) -> np.ndarray:
    """Quadratische euklidische Abstände (len(a), len(b)); elementweise Differenzen statt der Binomialformel, damit gleiche Punkte exakt gleiche Abstände haben."""
    a = np.atleast_2d(np.asarray(a, dtype=float))
    b = np.atleast_2d(np.asarray(b, dtype=float))
    out = np.empty((len(a), len(b)))
    for s in range(0, len(a), chunk):
        diff = a[s:s + chunk, None, :] - b[None, :, :]
        out[s:s + chunk] = (diff * diff).sum(axis=2)
    return out


def neighbour_order(d2: np.ndarray) -> np.ndarray:
    """Spaltenindizes je Zeile, aufsteigend nach Abstand; bei gleichem Abstand kommt die kleinere Nummer zuerst (stabile Sortierung)."""
    return np.argsort(d2, axis=1, kind="stable")


def knn_weights(order: np.ndarray, k: int, n_cols: int | None = None) -> np.ndarray:
    """Gewichte 1/k für die k nächsten Beobachtungen. k wird auf die Zahl der verfügbaren Spalten begrenzt (k > n heißt: alle, gleiche Gewichte)."""
    m, avail = order.shape
    n_cols = avail if n_cols is None else n_cols
    if k < 1:
        raise ValueError("k muss mindestens 1 sein")
    k = min(k, avail)
    w = np.zeros((m, n_cols))
    w[np.arange(m)[:, None], order[:, :k]] = 1.0 / k
    return w


def kernel_weights(d2: np.ndarray, h: float) -> np.ndarray:
    """Gauß-Kerngewichte je Zeile, auf Summe 1 normiert. Der kleinste Abstand der Zeile wird vor dem Exponieren abgezogen (Unterlauf-sicher):
    auch bei winziger Bandbreite behält der nächste Nachbar das Gewicht 1 statt 0 / 0. Ausgeschlossene Spalten stehen als +inf in d2 (Gewicht 0)."""
    if h <= 0:
        raise ValueError("Bandbreite muss positiv sein")
    d2 = np.asarray(d2, dtype=float)
    ref = d2.min(axis=1, keepdims=True)
    w = np.exp(-0.5 * (d2 - ref) / (h * h))
    return w / w.sum(axis=1, keepdims=True)


def weighted_quantile(w: np.ndarray, d: np.ndarray, tau: float) -> np.ndarray:
    """Kleinstes D_(j), dessen kumuliertes Gewicht (nach Nachfrage sortiert, Gleichstände nach Nummer) tau erreicht; je Zeile von `w`."""
    w = np.atleast_2d(w)
    d = np.asarray(d, dtype=float)
    order = np.argsort(d, kind="stable")
    cum = np.cumsum(w[:, order], axis=1)
    pos = (cum >= tau * cum[:, -1:] - LOO_TOL).argmax(axis=1)
    return d[order][pos]


def pinball(q: np.ndarray, d: np.ndarray, tau: float) -> float:
    """Mittlerer Pinball-Verlust max(tau * (D - q), (tau - 1) * (D - q))."""
    r = np.asarray(d, dtype=float) - q
    return float(np.mean(np.maximum(tau * r, (tau - 1.0) * r)))


def k_grid(n: int) -> list[int]:
    """Nachbarzahlen des Rasters, die im Leave-one-out noch Platz haben (k <= n - 1); mindestens der kleinste Wert."""
    ks = [k for k in KGRID if k <= n - 1]
    return ks or [min(KGRID)]


def h_grid(p: int) -> list[float]:
    """Bandbreiten des Rasters, auf die Dimension umgerechnet: h * sqrt(p / 3)."""
    return [h * float(np.sqrt(p / 3.0)) for h in HGRID]


def loo_losses_knn(d2_train: np.ndarray, d: np.ndarray, tau: float, ks: list[int]) -> list[float]:
    """Leave-one-out-Pinball-Verlust je k: jede Trainingsbeobachtung wird aus ihren eigenen Nachbarn (ohne sich selbst) vorhergesagt."""
    n = len(d)
    if n < 2:
        return [0.0 for _ in ks]
    dd = d2_train.copy()
    np.fill_diagonal(dd, np.inf)
    order = neighbour_order(dd)[:, : n - 1]
    return [pinball(weighted_quantile(knn_weights(order, k, n), d, tau), d, tau) for k in ks]


def loo_losses_kernel(d2_train: np.ndarray, d: np.ndarray, tau: float, hs: list[float]) -> list[float]:
    n = len(d)
    if n < 2:
        return [0.0 for _ in hs]
    dd = d2_train.copy()
    np.fill_diagonal(dd, np.inf)
    return [pinball(weighted_quantile(kernel_weights(dd, h), d, tau), d, tau) for h in hs]


def loo_choice(losses: list[float], grid: list) -> tuple:
    """Wert des Rasters mit dem kleinsten Verlust; bei Gleichstand der kleinere Rasterwert (np.argmin nimmt den ersten)."""
    i = int(np.argmin(losses))
    return grid[i], i


def distances(x: np.ndarray, xt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Quadratische Abstände Training-Training und Test-Training; einmal rechnen und an mehrere Regeln und Niveaus weitergeben (`dist=`)."""
    return sqdist(x, x), sqdist(xt, x)


def knn_rule(x: np.ndarray, d: np.ndarray, xt: np.ndarray, tau: float, dist=None) -> tuple[np.ndarray, float]:
    """kNN-Gewichte, k per Leave-one-out aus KGRID (nur k <= n - 1). Rückgabe: Bestellmengen der Testfälle und das gewählte k."""
    d2_tr, d2_te = distances(x, xt) if dist is None else dist
    ks = k_grid(len(d))
    kb = loo_choice(loo_losses_knn(d2_tr, d, tau, ks), ks)[0]
    return weighted_quantile(knn_weights(neighbour_order(d2_te), kb, len(d)), d, tau), float(kb)


def kernel_rule(x: np.ndarray, d: np.ndarray, xt: np.ndarray, tau: float, dist=None) -> tuple[np.ndarray, float]:
    """Gauß-Kerngewichte, Bandbreite per Leave-one-out aus dem Raster (auf p umgerechnet). Rückgabe: Bestellmengen der Testfälle und die gewählte Bandbreite."""
    d2_tr, d2_te = distances(x, xt) if dist is None else dist
    hs = h_grid(np.atleast_2d(x).shape[1])
    hb = loo_choice(loo_losses_kernel(d2_tr, d, tau, hs), hs)[0]
    return weighted_quantile(kernel_weights(d2_te, hb), d, tau), float(hb)
