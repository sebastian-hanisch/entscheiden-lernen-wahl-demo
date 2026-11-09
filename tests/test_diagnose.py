"""Diagnose und Wahlregel an von Hand konstruierten Datensätzen, an der Entscheidungstabelle, an Eigenschaften und Randfällen (DEMO-PLAYBOOK §1b, §12)."""

import numpy as np
import pytest

import elw_diagnose as D
import elw_scenario as S


# --- Handbeispiele -----------------------------------------------------------------------------------------------------------------------------------


def test_quadratic_hand_example_f_is_320_and_the_rule_picks_the_weights(hand_quadratic):
    X, y = hand_quadratic
    xa = np.hstack([np.ones((5, 1)), X])
    p = D.ftest_pvalue(xa, np.hstack([xa, X ** 2]), y, scale=5.0)
    assert p == pytest.approx(1.0 - np.sqrt(320.0 / 322.0), rel=1e-9)     # geschlossene Form für 2 Freiheitsgrade im Nenner
    d = D.diagnose(X, y)
    assert d.p_nonlin == pytest.approx(p) and d.p_nonlin < 0.01
    assert d.rule == D.RULE_GEWICHTE


def test_exactly_linear_data_pick_kq_plus_quantile_and_both_p_values_are_one():
    x = np.arange(12, dtype=float)
    d = D.diagnose(x[:, None], 60.0 + 10.0 * x)
    assert (d.p_nonlin, d.p_hetero) == (1.0, 1.0) and d.rule == D.RULE_EMP


def test_spread_growing_with_x_picks_direct_exactly_at_the_data_threshold(hand_hetero):
    """p = 1: die Schwelle ist 10 (p + 1) = 20 Fälle; 19 Fälle geben KQ plus Quantil, 20 und 21 direkt (inklusive Grenze)."""
    for n, rule in ((19, D.RULE_EMP), (20, D.RULE_DIREKT), (21, D.RULE_DIREKT)):
        d = D.diagnose(*hand_hetero(n))
        assert d.p_hetero < 0.01 and d.p_nonlin > 0.01, n
        assert d.rule == rule and d.enough_data == (n >= 20), n


def test_the_threshold_scales_with_the_number_of_features():
    """Dieselben 30 Fälle, p = 3: die Schwelle ist 40, direkt ist ausgeschlossen, obwohl die Streuung linear wächst."""
    rng = np.random.default_rng(3)
    x = rng.uniform(1, 10, size=(30, 3))
    y = 20 + x[:, 0] + (-1.0) ** np.arange(30) * 0.8 * x[:, 0]
    d = D.diagnose(x, y)
    assert d.p_hetero < 0.01 and not d.enough_data and d.rule != D.RULE_DIREKT


# --- Entscheidungstabelle von Hand --------------------------------------------------------------------------------------------------------------------

DECISION_TABLE = [
    # (p-Wert Krümmung, p-Wert Streuung, n, p, Wahl)
    (0.0001, 0.5, 100, 3, "gewichte"),
    (0.0099, 0.0001, 1000, 3, "gewichte"),       # die Krümmung geht vor der Streuung
    (0.0100, 0.5, 100, 3, "emp"),                # der p-Wert genau 0,01 spricht nicht an
    (0.5, 0.0099, 100, 3, "direkt"),             # 100 >= 40
    (0.5, 0.0100, 100, 3, "emp"),
    (0.5, 0.0001, 40, 3, "direkt"),              # genau 10 (p + 1) Fälle
    (0.5, 0.0001, 39, 3, "emp"),                 # ein Fall zu wenig
    (0.5, 0.0001, 100, 10, "emp"),               # 100 < 110
    (0.5, 0.0001, 110, 10, "direkt"),
    (0.5, 0.5, 1000, 10, "emp"),
]


@pytest.mark.parametrize("pn, ph, n, p, expected", DECISION_TABLE)
def test_choose_rule_follows_the_hand_table(pn, ph, n, p, expected):
    assert D.choose_rule(pn, ph, n, p) == expected


# --- Eigenschaften -----------------------------------------------------------------------------------------------------------------------------------


def _random_dataset(seed, n=60, p=2):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, p))
    y = 50 + 6 * x[:, 0] + (3 + 2 * np.abs(x[:, 0])) * rng.standard_normal(n)
    return x, y


def test_scaling_all_demands_changes_neither_the_p_values_nor_the_choice():
    """Die F-Tests sind skaleninvariant."""
    for seed in range(6):
        x, y = _random_dataset(seed)
        base = D.diagnose(x, y)
        for factor in (1e-3, 7.0, 1e3):
            d = D.diagnose(x, factor * y)
            assert d.p_nonlin == pytest.approx(base.p_nonlin, rel=1e-6, abs=1e-12) and d.p_hetero == pytest.approx(base.p_hetero, rel=1e-6, abs=1e-12)
            assert d.rule == base.rule


def test_shifting_all_demands_changes_nothing():
    for seed in range(6):
        x, y = _random_dataset(seed)
        base = D.diagnose(x, y)
        d = D.diagnose(x, y + 1000.0)
        assert d.p_nonlin == pytest.approx(base.p_nonlin, rel=1e-5, abs=1e-12) and d.p_hetero == pytest.approx(base.p_hetero, rel=1e-5, abs=1e-12)


def test_permuting_the_observations_changes_nothing():
    rng = np.random.default_rng(5)
    for seed in range(6):
        x, y = _random_dataset(seed)
        perm = rng.permutation(len(y))
        a, b = D.diagnose(x, y), D.diagnose(x[perm], y[perm])
        assert a.p_nonlin == pytest.approx(b.p_nonlin, rel=1e-9, abs=1e-15) and a.p_hetero == pytest.approx(b.p_hetero, rel=1e-9, abs=1e-15) and a.rule == b.rule


def test_the_weight_branch_is_chosen_exactly_when_the_curvature_p_value_is_below_threshold():
    seen = set()
    for seed in range(40):
        x, y = _random_dataset(seed, n=40)
        y = y + (8.0 * x[:, 0] ** 2 if seed % 2 else 0.0)
        d = D.diagnose(x, y)
        assert (d.rule == D.RULE_GEWICHTE) == (d.p_nonlin < D.ALPHA)
        assert 0.0 <= d.p_nonlin <= 1.0 and 0.0 <= d.p_hetero <= 1.0
        seen.add(d.rule)
    assert D.RULE_GEWICHTE in seen and len(seen) >= 2            # beide Seiten kommen vor (kein toter Zweig)


def test_every_branch_is_taken_by_some_instance():
    """Zweig-Test: jede der drei Wahlen kommt in den Welten der Demo auf mindestens einer Instanz vor."""
    picks = set()
    for welt in S.WORLD_KINDS:
        for seed in range(6):
            data = S.make_dataset(welt, 3, 1000, seed)
            picks.add(D.diagnose(data.X, data.y).rule)
    assert picks == {D.RULE_EMP, D.RULE_DIREKT, D.RULE_GEWICHTE}


def test_the_choice_depends_only_on_the_training_data():
    a = S.make_dataset("quadratisch", 3, 100, 2)
    b = S.make_dataset("quadratisch", 3, 100, 2)
    assert D.diagnose(a.X, a.y) == D.diagnose(b.X, b.y)


# --- Randfälle ---------------------------------------------------------------------------------------------------------------------------------------


def test_p_greater_or_equal_n_gives_kq_plus_quantile():
    rng = np.random.default_rng(1)
    for n, p in ((3, 3), (4, 5), (2, 2), (1, 1)):
        d = D.diagnose(rng.standard_normal((n, p)), rng.standard_normal(n) * 5 + 50)
        assert d.rule == D.RULE_EMP and d.p_nonlin == 1.0, (n, p)


def test_the_curvature_test_is_undefined_without_enough_degrees_of_freedom():
    """Großes Modell hat 2p + 1 Spalten: bei n = 2p + 1 bleibt kein Freiheitsgrad im Nenner, p-Wert 1."""
    rng = np.random.default_rng(2)
    x = rng.standard_normal((7, 3))
    assert D.diagnose(x, rng.standard_normal(7)).p_nonlin == 1.0
    x = rng.standard_normal((8, 3))
    assert D.diagnose(x, rng.standard_normal(8)).p_nonlin < 1.0


def test_constant_demand_and_all_zero_residuals_do_not_alarm():
    x = np.random.default_rng(4).standard_normal((50, 2))
    d = D.diagnose(x, np.full(50, 17.0))
    assert (d.p_nonlin, d.p_hetero, d.rule) == (1.0, 1.0, D.RULE_EMP)


def test_a_perfect_quadratic_fit_alarms_instead_of_returning_one():
    """Anders als die Vormessung (die bei RSS1 = 0 immer 1 gab): passt nur das große Modell, ist der p-Wert 0."""
    x = np.linspace(-3, 3, 25)[:, None]
    assert D.diagnose(x, 5.0 + x[:, 0] ** 2).p_nonlin == 0.0
