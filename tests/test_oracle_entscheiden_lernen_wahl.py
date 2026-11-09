"""Orakel-Tests: F-Tests, Wahlregel, Quantilregression, Gewichte, Newsvendor-Kosten und Zufallsstrom gegen unabhängige Rechenwege (DEMO-PLAYBOOK §12, ORACLE-PLAYBOOK §5).

1. F-Test der Krümmung (p = 1) gegen **exakte Bruchrechnung** (`fractions`: Normalgleichungen des Modells [1, x, x²], F = t² des Koeffizienten) und p-Wert über die
   **t-Verteilung** (`scipy.stats.t.sf`, nicht `f.sf` wie der Prüfling); F-Test der Streuung (p = 1) gegen `scipy.stats.linregress` auf den Beträgen der Residuen.
2. Beide F-Tests für p = 3 gegen **Projektionsmatrizen** (`numpy.linalg.pinv`, RSS = y' (I − H) y) und die Verteilung über `scipy.stats.f.sf`.
3. Wahlregel gegen eine Entscheidungstabelle von Hand (`test_diagnose.py`) und gegen die Aufzählung kleiner Fälle: jede der drei Wahlen auf jedem Raster aus p-Werten und Datenmengen.
4. Quantilregression gegen **Aufzählung** der Ecken (p = 1) und gegen `sklearn.linear_model.QuantileRegressor`, falls scikit-learn im Test-venv liegt.
5. Gewichtetes Quantil gegen **exakte Brüche** (kNN) und Gauß-Kerngewichte gegen eine Schleife mit `math.fsum`.
6. Erwartete Kosten gegen **Monte Carlo**, wahres Quantil gegen das empirische Quantil von Ziehungen, Optimalität des Orakels; SplitMix64 gegen eine ganzzahlige Referenz.
Der große Lauf (300 Instanzen je Vergleich) lief im Scratch-Ordner; hier steht die kleine, schnelle Fassung."""

import math
from fractions import Fraction

import numpy as np
import pytest
from scipy import stats

import elw_algorithm as A
import elw_diagnose as D
import elw_scenario as S
import elw_weights as W
from elw_rng import SplitMix64


# --- 1. F-Tests gegen exakte Brüche und die t-Verteilung -------------------------------------------------------------------------------------------


def _solve(a, b):
    """Gauß-Verfahren mit Brüchen: löst a c = b."""
    n = len(b)
    m = [row[:] + [b[i]] for i, row in enumerate(a)]
    for col in range(n):
        piv = next(r for r in range(col, n) if m[r][col] != 0)
        m[col], m[piv] = m[piv], m[col]
        m[col] = [v / m[col][col] for v in m[col]]
        for r in range(n):
            if r != col and m[r][col] != 0:
                f = m[r][col]
                m[r] = [v - f * w for v, w in zip(m[r], m[col])]
    return [m[i][n] for i in range(n)]


def _exact_rss(cols, y):
    """Fehlerquadratsumme der Kleinste-Quadrate-Anpassung von y an die Spalten (alles Fraction), über die Normalgleichungen."""
    k = len(cols)
    a = [[sum(ci * cj for ci, cj in zip(cols[r], cols[c])) for c in range(k)] for r in range(k)]
    b = [sum(ci * yi for ci, yi in zip(cols[r], y)) for r in range(k)]
    coef = _solve(a, b)
    res = [yi - sum(coef[j] * cols[j][i] for j in range(k)) for i, yi in enumerate(y)]
    return sum(r * r for r in res)


def _p_from_t(f_exact, df):
    """p-Wert des F-Tests mit 1 Zähler-Freiheitsgrad über t = sqrt(F): zweiseitiger t-Test."""
    return float(2.0 * stats.t.sf(math.sqrt(float(f_exact)), df))


def test_curvature_test_matches_exact_fractions_and_the_t_distribution_for_one_feature():
    rng = np.random.default_rng(7)
    for trial in range(40):
        n = int(rng.integers(6, 16))
        x = [int(v) for v in rng.integers(-6, 7, n)]
        if len(set(x)) < 3:
            continue
        y = [int(v) for v in np.round(30 + 3 * np.array(x) + (trial % 3) * 0.7 * np.array(x) ** 2 + 4 * rng.standard_normal(n))]
        one, fx = [Fraction(1)] * n, [Fraction(v) for v in x]
        fx2, fy = [v * v for v in fx], [Fraction(v) for v in y]
        rss0, rss1 = _exact_rss([one, fx], fy), _exact_rss([one, fx, fx2], fy)
        if rss1 == 0 or rss0 == rss1 or len(set(x)) < 3:
            continue
        f = (rss0 - rss1) / (rss1 / (n - 3))
        got = D.diagnose(np.array(x, dtype=float)[:, None], np.array(y, dtype=float)).p_nonlin
        assert got == pytest.approx(_p_from_t(f, n - 3), rel=1e-7, abs=1e-12), (trial, n)


def test_spread_test_matches_linregress_on_the_absolute_residuals_for_one_feature():
    rng = np.random.default_rng(8)
    for trial in range(40):
        n = int(rng.integers(8, 40))
        x = rng.standard_normal(n)
        y = 50 + 6 * x + (2 + (trial % 2) * 3 * np.abs(x + 0.3)) * rng.standard_normal(n)
        fit = stats.linregress(x, y)
        res = np.abs(y - (fit.intercept + fit.slope * x))
        ref = stats.linregress(x, res).pvalue
        assert D.diagnose(x[:, None], y).p_hetero == pytest.approx(ref, rel=1e-6, abs=1e-12), trial


# --- 2. Beide F-Tests für p = 3 gegen Projektionsmatrizen --------------------------------------------------------------------------------------------


def _rss_by_projection(design, y):
    h = design @ np.linalg.pinv(design)
    r = y - h @ y
    return float(r @ r)


def test_both_f_tests_match_projection_matrices_for_three_features():
    rng = np.random.default_rng(9)
    for trial in range(30):
        n = int(rng.integers(20, 80))
        x = rng.standard_normal((n, 3))
        y = 40 + 5 * x[:, 0] + (trial % 2) * 4 * x[:, 1] ** 2 + (2 + (trial % 3) * 1.5 * np.abs(x[:, 2])) * rng.standard_normal(n)
        xa = np.hstack([np.ones((n, 1)), x])
        r0, r1 = _rss_by_projection(xa, y), _rss_by_projection(np.hstack([xa, x ** 2]), y)
        p_nonlin = float(stats.f.sf(((r0 - r1) / 3) / (r1 / (n - 7)), 3, n - 7))
        res = np.abs(y - xa @ np.linalg.pinv(xa) @ y)
        s0, s1 = _rss_by_projection(np.ones((n, 1)), res), _rss_by_projection(xa, res)
        p_hetero = float(stats.f.sf(((s0 - s1) / 3) / (s1 / (n - 4)), 3, n - 4))
        d = D.diagnose(x, y)
        assert d.p_nonlin == pytest.approx(p_nonlin, rel=1e-6, abs=1e-12) and d.p_hetero == pytest.approx(p_hetero, rel=1e-6, abs=1e-12), trial


# --- 3. Wahlregel gegen Aufzählung ---------------------------------------------------------------------------------------------------------------------


def test_the_rule_matches_an_enumeration_of_all_cases_on_a_grid():
    """Alle Kombinationen aus p-Werten (unter, an und über der Schwelle), Datenmengen und Merkmalszahlen: die Wahl wird aus den Bedingungen einzeln abgeleitet."""
    grid_p = (0.0, 0.0099, 0.01, 0.0101, 1.0)
    for pn in grid_p:
        for ph in grid_p:
            for p in (1, 3, 10):
                for n in (10 * (p + 1) - 1, 10 * (p + 1), 10 * (p + 1) + 1):
                    curved, spread, enough = pn < 0.01, ph < 0.01, n >= 10 * (p + 1)
                    if curved:
                        expected = "gewichte"
                    elif spread and enough:
                        expected = "direkt"
                    else:
                        expected = "emp"
                    assert D.choose_rule(pn, ph, n, p) == expected, (pn, ph, n, p)
    # jede der drei Wahlen kommt auf dem Raster vor
    assert {D.choose_rule(a, b, 100, 3) for a in grid_p for b in grid_p} == {"gewichte", "direkt", "emp"}


# --- 4. Quantilregression --------------------------------------------------------------------------------------------------------------------------------


def _objective(x, y, tau, c0, c):
    r = y - c0 - c * x
    return float(np.mean(np.maximum(tau * r, (tau - 1.0) * r)))


def _enumerated_minimum(x, y, tau):
    """Kleinster Zielwert über alle Geraden durch zwei Datenpunkte (und waagerechte durch einen): Zielfunktion konvex und stückweise linear."""
    n = len(x)
    candidates = [(float(y[i]), 0.0) for i in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if x[i] != x[j]:
                c = (y[i] - y[j]) / (x[i] - x[j])
                candidates.append((float(y[i] - c * x[i]), float(c)))
    return min(_objective(x, y, tau, c0, c) for c0, c in candidates)


def test_quantile_regression_matches_vertex_enumeration_on_random_small_instances():
    rng = np.random.default_rng(20)
    for trial in range(50):
        n = int(rng.integers(4, 10))
        x = np.round(rng.standard_normal(n), 2) if trial % 3 else rng.integers(0, 3, n).astype(float)
        y = np.round(50 + 6 * x + 5 * rng.standard_normal(n), 1)
        tau = float(rng.choice([0.5, 0.8, 0.95]))
        coef = A.quantreg_fit(x.reshape(-1, 1), y, tau)
        assert _objective(x, y, tau, coef[0], coef[1]) == pytest.approx(_enumerated_minimum(x, y, tau), abs=1e-7), (trial, tau)


def test_quantile_regression_matches_scikit_learn():
    linear_model = pytest.importorskip("sklearn.linear_model")
    rng = np.random.default_rng(21)
    for trial in range(12):
        n, p = int(rng.integers(15, 60)), int(rng.integers(1, 5))
        X = rng.standard_normal((n, p))
        y = 50 + 6 * X[:, 0] + (3 + 2 * np.abs(X[:, 0])) * rng.standard_normal(n)
        tau = float(rng.choice([0.5, 0.8, 0.95]))
        coef = A.quantreg_fit(X, y, tau)
        ref = linear_model.QuantileRegressor(quantile=tau, alpha=0.0, solver="highs").fit(X, y)
        assert A.pinball_loss(A.predict(coef, X), y, tau) == pytest.approx(A.pinball_loss(ref.intercept_ + X @ ref.coef_, y, tau), abs=1e-7), trial


# --- 5. Gewichte -------------------------------------------------------------------------------------------------------------------------------------


def test_weighted_quantile_of_knn_weights_matches_exact_fractions():
    rng = np.random.default_rng(30)
    for trial in range(40):
        n = int(rng.integers(5, 25))
        y = rng.integers(0, 12, n).astype(float)                    # viele Gleichstände in den Nachfragen
        x = rng.integers(0, 6, (n, 2)).astype(float)                # viele Gleichstände in den Abständen
        xt = rng.integers(0, 6, (3, 2)).astype(float)
        k, tau = int(rng.integers(1, n + 1)), float(rng.choice([0.3, 0.5, 0.8, 0.95]))
        d2 = W.sqdist(xt, x)
        got = W.weighted_quantile(W.knn_weights(W.neighbour_order(d2), k, n), y, tau)
        for row in range(3):
            order = sorted(range(n), key=lambda i: (d2[row, i], i))[:k]      # die k nächsten, bei Gleichstand die kleinere Nummer
            weights = {}
            for i in order:
                weights[y[i]] = weights.get(y[i], Fraction(0)) + Fraction(1, k)
            cum, ref = Fraction(0), None
            for value in sorted(weights):
                cum += weights[value]
                if cum >= Fraction(tau).limit_denominator(100):
                    ref = value
                    break
            assert got[row] == ref, (trial, row)


def test_kernel_weights_match_a_loop_with_fsum():
    rng = np.random.default_rng(31)
    x, xt = rng.standard_normal((12, 3)), rng.standard_normal((4, 3))
    for h in (0.3, 1.0, 2.5):
        got = W.kernel_weights(W.sqdist(xt, x), h)
        for r in range(4):
            raw = [math.exp(-sum((xt[r, j] - x[i, j]) ** 2 for j in range(3)) / (2 * h * h)) for i in range(12)]
            total = math.fsum(raw)
            assert got[r] == pytest.approx([v / total for v in raw], rel=1e-9, abs=1e-12)


# --- 6. Welt, Kosten, Zufall ---------------------------------------------------------------------------------------------------------------------------


def _truncated_world():
    """Ein Merkmal, mu = 60 + 10 x, sigma = 8; die Merkmalswerte -6, -7, -5 geben mu = 0, -10, 10: die Stutzung bei 0 greift stark."""
    return S.World("linear", 1, (1.0,)), np.array([[-6.0], [-7.0], [-5.0], [0.0], [1.0]])


def test_expected_cost_matches_monte_carlo():
    world, X = _truncated_world()
    z = np.random.default_rng(22).standard_normal(400_000)
    for tau in (0.5, 0.8, 0.95):
        b, h = S.critical_costs(tau)
        for q0 in (0.0, 3.0, 12.0, 70.0):
            q = np.full(len(X), q0)
            exact = S.expected_cost_at(world, X, q, b, h)
            for i in range(len(X)):
                demand = np.maximum(0.0, S.mean_of(world, X)[i] + S.spread_of(world, X)[i] * z)
                mc = float(np.mean(b * np.maximum(demand - q0, 0.0) + h * np.maximum(q0 - demand, 0.0)))
                assert exact[i] == pytest.approx(mc, rel=0.02, abs=0.03), (tau, q0, i)


def test_true_quantile_matches_the_empirical_quantile_of_draws():
    world, X = _truncated_world()
    z = np.random.default_rng(23).standard_normal(400_000)
    for tau in (0.5, 0.8, 0.95):
        q = S.true_quantile(world, X, tau)
        for i in range(len(X)):
            demand = np.maximum(0.0, S.mean_of(world, X)[i] + S.spread_of(world, X)[i] * z)
            assert q[i] == pytest.approx(float(np.quantile(demand, tau)), abs=0.1), (tau, i)


def test_the_true_quantile_is_the_cost_optimal_order_quantity_in_every_world():
    for kind in S.WORLD_KINDS:
        d = S.make_dataset(kind, 3, 10, 4)
        for tau in (0.5, 0.8, 0.95):
            b, h = S.critical_costs(tau)
            qs = S.true_quantile(d.world, d.Xt, tau)
            best = S.expected_cost_at(d.world, d.Xt, qs, b, h)
            for delta in (-6.0, -1.0, -0.05, 0.05, 1.0, 6.0):
                assert np.all(S.expected_cost_at(d.world, d.Xt, qs + delta, b, h) >= best - 1e-9), (kind, tau, delta)


def test_extra_cost_is_never_negative_and_zero_for_the_oracle():
    d = S.make_dataset("streuung-linear", 3, 50, 1)
    assert S.extra_cost_percent(d.world, d.Xt, S.true_quantile(d.world, d.Xt, 0.8), 0.8) == pytest.approx(0.0, abs=1e-9)
    rng = np.random.default_rng(24)
    for _ in range(20):
        assert S.extra_cost_percent(d.world, d.Xt, rng.uniform(0, 120, len(d.Xt)), 0.8) >= 0.0


def _splitmix_reference(seed, k):
    """Der k-te Ausgabewert (k ab 1) mit ganzen Zahlen von Python, ohne numpy."""
    mask = (1 << 64) - 1
    z = (seed + k * 0x9E3779B97F4A7C15) & mask
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & mask
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & mask
    return z ^ (z >> 31)


def test_splitmix64_matches_an_integer_reference_and_the_published_value():
    assert int(SplitMix64(0).u64(1)[0]) == 0xE220A8397B1DCDAF            # veröffentlichter erster Ausgabewert für Seed 0
    for seed in (0, 1, 12345, 2 ** 40 + 7, 2 ** 63 + 5):
        assert [int(v) for v in SplitMix64(seed).u64(5)] == [_splitmix_reference(seed, k) for k in range(1, 6)]


def test_splitmix64_stream_advances_and_is_reproducible():
    a = SplitMix64(9)
    first, second = a.u64(3), a.u64(3)
    assert [int(v) for v in np.concatenate([first, second])] == [int(v) for v in SplitMix64(9).u64(6)]
    u = SplitMix64(10).uniform(10_000)
    assert 0.0 <= u.min() and u.max() < 1.0 and abs(u.mean() - 0.5) < 0.02
    z = SplitMix64(11).normal(20_000)
    assert abs(z.mean()) < 0.03 and abs(z.std() - 1.0) < 0.03
