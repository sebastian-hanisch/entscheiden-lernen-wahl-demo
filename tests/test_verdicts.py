"""Verdikte an von Hand gesetzten Werten (DEMO-PLAYBOOK §12): jede Schwelle und jede Zahl im Text steht hier.

Ein Verdikt ist eine reine Funktion (Art, Text). Der Test prüft die Grenzen (höchstens 1 Prozentpunkt oder 5 % über der besten Wahl) und jede genannte Zahl direkt."""

import elw_diagnose as D
import elw_evaluation as ev


def _analysis(rule, rule_extra, best_extra, best="direkt", extras=None):
    s = ev.Settings("streuung-linear", 0.95, 1000, 3, 7)
    d = D.Diagnosis(1000, 3, 0.5, 0.0001, True, rule)
    e = extras or {"emp": 30.0, "direkt": 8.0, "kern": 12.0, "knn": 14.0}
    return ev.Analysis(s, d, e, rule, rule_extra, best, best_extra)


def test_instance_verdict_names_the_choice_its_cost_the_best_choice_and_the_fixed_alternatives():
    kind, text = ev.verdict_instance(_analysis("direkt", 8.0, 8.0))
    assert kind == "success"
    for part in ("p = 3, n = 1000, τ = 0,95, Zufalls-Seed 7", "wählt die Regel **direkt**", "**8,0 %**", "die beste Wahl hätte 8,0 % gehabt (direkt (Quantilregression))",
                 "immer KQ + Quantil 30,0 %", "immer direkt 8,0 %", "immer Gewichte 12,0 %"):
        assert part in text, part


def test_instance_verdict_thresholds_are_one_point_or_five_percent_inclusive():
    assert ev.verdict_instance(_analysis("direkt", 10.0, 9.0))[0] == "success"           # genau 1 Prozentpunkt darüber
    assert ev.verdict_instance(_analysis("direkt", 10.1, 9.0))[0] == "warning"           # 1,1 Punkte und 12 %
    assert ev.verdict_instance(_analysis("direkt", 104.0, 100.0))[0] == "success"        # 4 Punkte, aber 4 % darüber
    assert ev.verdict_instance(_analysis("direkt", 105.1, 100.0))[0] == "warning"
    kind, text = ev.verdict_instance(_analysis("direkt", 20.0, 8.0))
    assert kind == "warning" and "merklich über der besten Wahl" in text


def test_instance_verdict_states_that_the_weight_branch_is_computed_with_the_kernel():
    kind, text = ev.verdict_instance(_analysis("gewichte", 12.0, 8.0))
    assert "Gauß-Kerngewichten gerechnet, nicht mit einem Wald" in text and "wählt die Regel **Gewichte**" in text
    assert "Gauß-Kern" not in ev.verdict_instance(_analysis("direkt", 8.0, 8.0))[1]


def _band(rule_vals, best_vals, wahl, emp, direkt, kern):
    return {"emp": emp, "direkt": direkt, "kern": kern, "knn": kern, "regel": rule_vals, "beste": best_vals, "wahl": wahl}


def test_band_verdict_uses_the_median_and_counts_the_choices():
    b = _band([8.0, 9.0, 40.0, 8.5, 9.5], [7.0, 8.0, 9.0, 7.5, 8.5], ["direkt", "direkt", "emp", "direkt", "gewichte"],
              [30.0, 31.0, 32.0, 33.0, 34.0], [8.0, 9.0, 10.0, 11.0, 12.0], [20.0, 21.0, 22.0, 23.0, 24.0])
    kind, text = ev.verdict_band(b, "streuung-linear", 3, 1000, 0.95)
    assert kind == "success"
    for part in ("Median über 5 feste Instanzen", "Regel 9,0 %", "beste Wahl je Instanz 8,0 %", "immer KQ + Quantil 32,0 %", "immer direkt 10,0 %", "immer Gewichte 22,0 %",
                 "KQ + Quantil in 1, direkt in 3 und die Gewichte in 1 von 5 Instanzen", "τ = 0,95"):
        assert part in text, part


def test_band_verdict_thresholds_compare_the_median_of_the_rule_with_the_best_fixed_median():
    base = dict(best_vals=[5.0] * 3, wahl=["emp"] * 3, emp=[10.0] * 3, direkt=[20.0] * 3, kern=[30.0] * 3)
    assert ev.verdict_band(_band([11.0] * 3, **base), "linear", 3, 100, 0.5)[0] == "success"        # genau 1 Prozentpunkt über dem besten festen Verfahren
    assert ev.verdict_band(_band([11.1] * 3, **base), "linear", 3, 100, 0.5)[0] == "warning"
    kind, text = ev.verdict_band(_band([12.0] * 3, **base), "linear", 3, 100, 0.5)
    assert kind == "warning" and "über dem besten festen Verfahren dieser Welt" in text
    assert ev.verdict_band(_band([10.5] * 3, **base), "linear", 3, 100, 0.5)[1].endswith("bei dem besten der drei festen Verfahren.")


def test_blind_spot_verdict_names_both_counts_and_the_wrong_choice():
    lin = {"runs": 20, "nonlin": 0, "hetero": 20, "wahl": {"emp": 0, "direkt": 20, "gewichte": 0}}
    absv = {"runs": 20, "nonlin": 1, "hetero": 1, "wahl": {"emp": 17, "direkt": 1, "gewichte": 2}}
    kind, text = ev.verdict_blind(lin, absv, 1000, 3)
    assert kind == "info"
    for part in ("n = 1000, p = 3", "in **20 von 20** festen Instanzen", "in **1 von 20**", "in 17 von 20 Instanzen KQ + Quantil"):
        assert part in text, part


def test_rule_vs_fixed_verdict_says_44_percent_and_not_half_and_names_the_sets():
    kind, text = ev.verdict_rule_vs_fixed()
    assert kind == "success"
    for part in ("18,84 %", "33,45 %", "immer Wald-Gewichte", "**44 % weniger Mehrkosten** im Mittel über die vier Welten, nicht die Hälfte", "in den drei übrigen Welten hat ein festes Verfahren kleinere Mehrkosten",
                 "im Median kostet die Regel 9,60 % gegen 9,77 % (immer direkt)", "17,54 %", "18,75 % gegen 33,28 %", "immer KQ + Quantil",
                 "Seeds 100 bis 129", "23,2 % gegen 33,7 %", "31 % weniger", "nicht für echte Daten"):
        assert part in text, part


def test_number_formatting_uses_the_decimal_comma():
    assert ev.de(3.14159, 2) == "3,14" and ev.pct(0.0) == "0,0 %" and ev.tau_text(0.95) == "0,95" and ev.tau_text(0.5) == "0,5"
    assert ev.p_text(0.00001) == "< 0,0001" and ev.p_text(0.0311) == "0,0311"
