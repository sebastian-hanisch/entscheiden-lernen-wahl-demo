"""Randfälle: n = 1, n = 2, p >= n, Gleichstände, Extremwerte der Regler und ungültige Eingaben (DEMO-PLAYBOOK §12)."""

import numpy as np
import pytest

import elw_algorithm as A
import elw_constants as C
import elw_diagnose as D
import elw_evaluation as ev
import elw_scenario as S
import elw_weights as W


def test_n_equal_to_one_and_two_do_not_crash_and_choose_kq_plus_quantile():
    for n in (1, 2):
        x = np.arange(n, dtype=float)[:, None]
        y = 50.0 + 3.0 * np.arange(n)
        d = D.diagnose(x, y)
        assert d.rule == D.RULE_EMP and not d.enough_data
        coef = A.fit_rule("emp", x, y, 0.8)
        assert np.all(np.isfinite(coef))


def test_quantile_regression_with_fewer_observations_than_features_still_solves():
    rng = np.random.default_rng(3)
    X, y = rng.standard_normal((4, 6)), rng.uniform(40, 60, 4)
    coef = A.quantreg_fit(X, y, 0.8)
    assert np.all(np.isfinite(coef)) and A.pinball_loss(A.predict(coef, X), y, 0.8) == pytest.approx(0.0, abs=1e-9)     # n <= p: das LP passt exakt


def test_identical_feature_rows_and_identical_demands():
    x = np.zeros((30, 2))
    y = np.full(30, 12.0)
    assert D.diagnose(x, y).rule == D.RULE_EMP
    d2_tr, d2_te = W.distances(x, np.zeros((3, 2)))
    q, _ = W.kernel_rule(x, y, np.zeros((3, 2)), 0.8, (d2_tr, d2_te))
    assert np.all(q == 12.0)                                           # alle Abstände gleich, alle Nachfragen gleich: das Quantil ist die Nachfrage


def test_ties_in_the_demands_do_not_depend_on_the_order():
    rng = np.random.default_rng(4)
    x = rng.standard_normal((40, 2))
    y = rng.integers(0, 4, 40).astype(float)
    perm = rng.permutation(40)
    a, b = D.diagnose(x, y), D.diagnose(x[perm], y[perm])
    assert a.rule == b.rule and a.p_nonlin == pytest.approx(b.p_nonlin, rel=1e-9, abs=1e-15)


def test_the_instance_analysis_runs_at_small_n_with_both_feature_counts_in_every_world():
    for welt in C.WELTEN:
        for n, p in ((C.N_OPTIONS[0], C.P_OPTIONS[-1]), (C.N_OPTIONS[0], C.P_OPTIONS[0])):
            a = ev.analyse(ev.Settings(welt, C.TAU_OPTIONS[0], n, p, 0))
            assert a.rule_extra >= -1e-9 and a.best_extra <= min(a.extras.values()) + 1e-12
            assert a.rule in (D.RULE_GEWICHTE, D.RULE_DIREKT, D.RULE_EMP)


def test_invalid_inputs_are_rejected():
    with pytest.raises(ValueError):
        S.make_dataset("gibt-es-nicht", 3, 10, 0)
    with pytest.raises(ValueError):
        S.make_world("linear", 0, 0)
    with pytest.raises(ValueError):
        S.critical_costs(1.0)
    with pytest.raises(ValueError):
        W.kernel_weights(np.ones((1, 3)), 0.0)
    with pytest.raises(ValueError):
        W.knn_weights(np.zeros((1, 3), dtype=int), 0)


def test_mean_extra_cost_is_never_negative_for_any_rule_on_any_world():
    for welt in C.WELTEN:
        a = ev.analyse(ev.Settings(welt, 0.8, 100, 3, 1))
        assert all(v >= -1e-9 for v in a.extras.values()), welt


def test_the_live_band_and_counts_have_the_shape_the_app_expects():
    b = ev.band("linear", 3, 100, 0.8, seeds=(0, 1, 2))
    assert set(b) == set(C.METHODEN) | {"regel", "beste", "wahl"} and all(len(v) == 3 for v in b.values())
    assert all(r <= e + 1e-12 for r, e in zip(b["beste"], b["regel"]))                       # die beste Wahl je Instanz ist nie schlechter als die Regel
    c = ev.detection_counts("streuung-linear", 3, 100, seeds=(0, 1, 2, 3))
    assert c["runs"] == 4 and 0 <= c["hetero"] <= 4 and sum(c["wahl"].values()) == 4


def test_slope_helper_matches_polyfit():
    rng = np.random.default_rng(5)
    x = rng.standard_normal(50)
    y = 3.0 * x + rng.standard_normal(50)
    assert ev.slope_of(x, y) == pytest.approx(np.polyfit(x, y, 1)[0])
