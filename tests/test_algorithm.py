"""Kern gegen Handrechnung: die zwei parametrischen Regeln und die Gewichte (DEMO-PLAYBOOK §1b).

Handbeispiel Newsvendor (aus `newsvendor-direkt-demo`, dort von Hand gerechnet): sechs Beobachtungen, x = 0: Nachfragen 4, 5, 6; x = 1: 8, 12, 16; tau = 0,8.
KQ ergibt 5 + 7x (Residuen -1, 0, 1, -4, 0, 4); das Residuum der Rangzahl ceil(0,8 · 6) = 5 ist 1, also KQ + Quantil 6 + 7x; direkt ergibt 6 + 10x, Pinball-Summe 3,0.
Handbeispiel Gewichte (aus `gewichtete-stichprobe-demo`): x = 1 … 5 mit Nachfragen 10, 12, 15, 30, 32, Anfrage x = 3,2, Gauß-Kern mit Bandbreite 1."""

import numpy as np
import pytest

import elw_algorithm as A
import elw_weights as W

X = np.array([[0.0], [0.0], [0.0], [1.0], [1.0], [1.0]])
Y = np.array([4.0, 5.0, 6.0, 8.0, 12.0, 16.0])


def test_ols_and_empirical_quantile_rule_on_the_hand_example():
    assert A.ols_fit(X, Y) == pytest.approx([5.0, 7.0])
    assert A.empirical_quantile(Y - A.predict(A.ols_fit(X, Y), X), 0.8) == pytest.approx(1.0)
    assert A.fit_rule("emp", X, Y, 0.8) == pytest.approx([6.0, 7.0])


def test_quantile_regression_on_the_hand_example():
    coef = A.fit_rule("direkt", X, Y, 0.8)
    q = A.predict(coef, X)
    assert A.pinball_loss(q, Y, 0.8) * 6 == pytest.approx(3.0)                       # Pinball-Summe 3,0 von Hand
    assert [q[0], q[3]] == pytest.approx([6.0, 16.0])                                # Mengen 6 und 16 (x = 0 und x = 1), das 0,8-Quantil je Gruppe


def test_rank_rule_of_the_empirical_quantile():
    v = np.array([5.0, 1.0, 4.0, 2.0, 3.0])
    assert [A.empirical_quantile(v, t) for t in (0.2, 0.4, 0.5, 0.8, 1.0)] == [1.0, 2.0, 3.0, 4.0, 5.0]    # ceil(tau n)-kleinstes Element


def test_newsvendor_cost_is_the_pinball_loss_up_to_the_factor_b_plus_h():
    rng = np.random.default_rng(0)
    q, d = rng.uniform(0, 20, 200), rng.uniform(0, 20, 200)
    for tau in (0.5, 0.8, 0.95):
        b, h = tau / (1 - tau), 1.0
        cost = float(np.mean(b * np.maximum(d - q, 0) + h * np.maximum(q - d, 0)))
        assert cost == pytest.approx((b + h) * A.pinball_loss(q, d, tau))


def test_invalid_tau_and_rule_are_rejected():
    for tau in (0.0, 1.0, -0.2):
        with pytest.raises(ValueError):
            A.quantreg_fit(X, Y, tau)
    with pytest.raises(ValueError):
        A.fit_rule("mittel", X, Y, 0.5)
    with pytest.raises(ValueError):
        A.empirical_quantile(np.array([]), 0.5)


# --- Gewichte ----------------------------------------------------------------------------------------------------------------------------------------

HX = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
HD = np.array([10.0, 12.0, 15.0, 30.0, 32.0])


def test_gaussian_kernel_weights_on_the_hand_example():
    d2 = W.sqdist(np.array([[3.2]]), HX)
    w = W.kernel_weights(d2, 1.0)[0]
    assert w == pytest.approx([0.0359, 0.1963, 0.3953, 0.2928, 0.0798], abs=5e-5)
    assert w.sum() == pytest.approx(1.0)
    assert float(W.weighted_quantile(w[None, :], HD, 0.8)[0]) == 30.0              # kumuliert 0,0359, 0,2322, 0,6275, 0,9203: das 0,8-Quantil ist 30


def test_knn_weights_and_weighted_quantile_on_the_hand_example():
    d2 = W.sqdist(np.array([[3.2]]), HX)
    w = W.knn_weights(W.neighbour_order(d2), 3, 5)[0]
    assert w == pytest.approx([0.0, 1 / 3, 1 / 3, 1 / 3, 0.0])                      # die drei nächsten: x = 2, 3, 4
    assert float(W.weighted_quantile(w[None, :], HD, 0.8)[0]) == 30.0
    assert float(W.weighted_quantile(w[None, :], HD, 0.5)[0]) == 15.0


def test_weight_properties():
    rng = np.random.default_rng(1)
    x, xt = rng.standard_normal((30, 3)), rng.standard_normal((5, 3))
    d2 = W.sqdist(xt, x)
    for h in (0.1, 1.0, 50.0):
        assert W.kernel_weights(d2, h).sum(axis=1) == pytest.approx(np.ones(5))
    assert W.kernel_weights(d2, 1e6) == pytest.approx(np.full((5, 30), 1 / 30), abs=1e-6)       # sehr große Bandbreite: gleiche Gewichte
    y = rng.uniform(10, 50, 30)
    w = W.knn_weights(W.neighbour_order(d2), 10, 30)
    q = [W.weighted_quantile(w, y, t) for t in (0.3, 0.5, 0.8)]
    assert np.all(q[0] <= q[1]) and np.all(q[1] <= q[2])                                        # Monotonie in tau
    assert W.weighted_quantile(w, 3.0 * y, 0.8) == pytest.approx(3.0 * W.weighted_quantile(w, y, 0.8))   # Skalieren der Nachfragen skaliert das Quantil


def test_k_and_bandwidth_grids_respect_the_leave_one_out_limits():
    assert W.k_grid(100) == [5, 10, 20, 40, 80] and W.k_grid(1000) == [5, 10, 20, 40, 80, 160] and W.k_grid(3) == [5]
    assert W.h_grid(3) == pytest.approx(list(W.HGRID)) and W.h_grid(12) == pytest.approx([2.0 * h for h in W.HGRID])
