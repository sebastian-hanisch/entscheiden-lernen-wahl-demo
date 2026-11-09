"""Zahlen der README: eingefrorene Vormessung (exakt gegen die Rohausgabe und gegen die Zahlen des Plans), vorgerechneter Nachbau (Aggregate exakt aus der Datei,
einzelne Zeilen gegen eine Neurechnung als Band), Zählungen und die Texte, die App und README teilen.

Die Vormessung (Zufallswald, scikit-learn, numpy-Zufallsstrom) wird hier nicht nachgerechnet: der Test prüft, dass die Datei `elw_vormessung.json` die Zahlen der
Rohausgaben und der README enthält. Der Nachbau rechnet mit dem Code dieser Demo; seine Zahlen sind LP und gewichtete Quantile ohne Training, werden aber nur als Bänder
gegen eine Neurechnung geprüft (plattformabhängig in der letzten Stelle)."""

import importlib.util
import re
from pathlib import Path

import pytest

import elw_constants as C
import elw_evaluation as ev
import elw_scenario as S
import elw_vormessung as V

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT.parent / "entscheidungsorientiert-planung" / "vormessung"


def _approx(x, digits=2):
    return pytest.approx(x, abs=0.5 * 10 ** -digits + 1e-9)


# --- Vormessung: Datei gegen Rohausgabe und Plan ----------------------------------------------------------------------------------------------------


def test_frozen_file_equals_the_parsed_raw_outputs():
    if not RAW_DIR.exists():
        pytest.skip("Rohausgaben der Vormessung liegen nicht neben diesem Repo")
    spec = importlib.util.spec_from_file_location("freeze_vormessung", ROOT / "tools" / "freeze_vormessung.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for key, name in mod.FILES.items():
        assert V.frozen()[key] == mod.parse((RAW_DIR / name).read_text(encoding="utf-8")), key


def test_vormessung_totals_seeds_0_to_29():
    g = V.frozen_set("a")["gesamt"]
    assert V.frozen_set("a")["zeilen"] == 1440 and V.frozen_set("a")["seeds"] == 30
    assert {k: g[k]["mittel"] for k in g} == {"emp": 33.68, "direkt": 35.21, "wald": 33.45, "regel": 18.84, "beste": 17.54}
    assert {k: g[k]["median"] for k in g} == {"emp": 13.06, "direkt": 9.77, "wald": 28.50, "regel": 9.60, "beste": 7.87}


def test_vormessung_totals_independent_seeds_100_to_129():
    g = V.frozen_set("b")["gesamt"]
    assert {k: g[k]["mittel"] for k in g} == {"emp": 33.28, "direkt": 35.12, "wald": 34.67, "regel": 18.75, "beste": 17.83}


def test_vormessung_by_world_and_by_n_and_p_seeds_0_to_29():
    w = V.frozen_set("a")["welten"]
    assert (w["linear"]["regel"], w["linear"]["emp"], w["linear"]["beste"]) == (3.5, 3.4, 3.2)
    assert (w["streuung-linear"]["regel"], w["streuung-linear"]["emp"], w["streuung-linear"]["direkt"], w["streuung-linear"]["beste"]) == (11.5, 19.6, 9.0, 8.4)
    assert (w["heteroskedastisch"]["regel"], w["heteroskedastisch"]["emp"], w["heteroskedastisch"]["beste"]) == (13.0, 11.9, 11.3)
    assert (w["nichtlinear"]["regel"], w["nichtlinear"]["emp"], w["nichtlinear"]["direkt"], w["nichtlinear"]["wald"], w["nichtlinear"]["beste"]) == (47.4, 99.8, 108.8, 47.4, 47.3)
    assert w["nichtlinear"]["wahl"] == {"emp": 0, "direkt": 0, "wald": 360} and w["heteroskedastisch"]["wahl"]["emp"] == 306
    npq = V.frozen_set("a")["n_p"]
    assert (npq["n100_p3"]["regel"], npq["n100_p3"]["beste"]) == (23.8, 21.9) and (npq["n100_p10"]["regel"], npq["n100_p10"]["beste"]) == (31.5, 29.4)
    assert (npq["n1000_p3"]["regel"], npq["n1000_p3"]["beste"]) == (9.2, 8.2) and (npq["n1000_p10"]["regel"], npq["n1000_p10"]["beste"]) == (10.8, 10.7)


def test_vormessung_diagnosis_rates():
    a, b = V.frozen_set("a")["diagnose"], V.frozen_set("b")["diagnose"]
    assert (a["nichtlinear_erkannt"], a["fehlalarm"], a["streuung_linear"], a["streuung_betrag"]) == (100, 6, 98, 4)
    assert (b["nichtlinear_erkannt"], b["fehlalarm"], b["streuung_linear"], b["streuung_betrag"]) == (100, 2, 97, 3)


def test_the_rule_saves_44_percent_not_half_of_the_extra_cost_of_the_best_fixed_method():
    """Der Plan sprach von „halbiert“: gemessen ist es das 0,56-fache, also 44 % weniger Mehrkosten als das beste feste Verfahren (in beiden Zufallssätzen)."""
    assert V.best_fixed("a") == ("wald", 33.45) and V.best_fixed("b") == ("emp", 33.28)
    assert V.reduction_vs_best_fixed("a") == _approx(43.68) and V.reduction_vs_best_fixed("b") == _approx(43.66)
    for satz in "ab":
        g = V.frozen_set(satz)["gesamt"]
        assert 0.5 < g["regel"]["mittel"] / V.best_fixed(satz)[1] < 0.6


def test_the_rule_is_below_every_fixed_method_in_all_four_n_p_cells_but_in_one_world_of_four():
    """Zählung je Zufallssatz der Vormessung: In allen 4 Kombinationen aus n und p liegt die Regel unter jedem festen Verfahren; in den Welten nur bei quadratischer
    Wahrheit (dort gleich dem Wald, 47,4 %). In den drei übrigen Welten hat je ein festes Verfahren kleinere Mehrkosten als die Regel: KQ + Quantil bei konstanter Streuung
    (3,5 gegen 3,4) und bei Streuung in |x₁| (13,0 gegen 11,9), direkt bei linear wachsender Streuung (11,5 gegen 9,0)."""
    for satz in "ab":
        w = V.frozen_set(satz)["welten"]
        losing = sorted(name for name, row in w.items() if row["regel"] > min(row["emp"], row["direkt"], row["wald"]))
        assert losing == ["heteroskedastisch", "linear", "streuung-linear"], (satz, losing)
        for key, row in V.frozen_set(satz)["n_p"].items():
            assert row["regel"] < min(row["emp"], row["direkt"], row["wald"]), (satz, key)


# --- Nachbau: Aggregate aus der Datei -----------------------------------------------------------------------------------------------------------------


def test_rebuilt_rows_cover_the_whole_grid_in_both_sets():
    for satz, seeds in (("a", range(30)), ("b", range(100, 130))):
        rs = V.rows(satz)
        assert len(rs) == 1440
        assert {(r["welt"], r["p"], r["n"], r["tau"], r["seed"]) for r in rs} == {(w, p, n, t, s) for w in C.WELTEN for p in C.P_OPTIONS for n in C.N_OPTIONS for t in C.TAU_OPTIONS for s in seeds}


def test_rebuilt_totals_with_kernel_weights_in_place_of_the_forest():
    for satz, expected in (("a", (33.70, 35.68, 38.43, 23.19, 21.52)), ("b", (32.86, 34.81, 38.16, 22.34, 20.63))):
        sm = V.summarize(V.rows(satz))
        got = tuple(sm[k]["mittel"] for k in ("emp", "direkt", "kern", "regel", "beste"))
        assert got == tuple(_approx(x) for x in expected), (satz, got)
        best = min(sm[k]["mittel"] for k in ("emp", "direkt", "kern"))
        assert 0.6 < sm["regel"]["mittel"] / best < 0.75                    # 31 bis 32 % weniger als das beste feste Verfahren, nicht die Hälfte


def test_the_rebuilt_rule_is_below_the_best_fixed_method_in_three_of_four_n_p_cells_and_equal_in_the_fourth():
    for satz in "ab":
        wins, ties = 0, 0
        for n in C.N_OPTIONS:
            for p in C.P_OPTIONS:
                sm = V.by_np(satz, n, p)
                best = min(sm[k]["mittel"] for k in ("emp", "direkt", "kern"))
                if sm["regel"]["mittel"] < best - 1.0:
                    wins += 1
                elif abs(sm["regel"]["mittel"] - best) <= 0.2:
                    ties += 1
        assert (wins, ties) == (3, 1), satz
    sm = V.by_np("a", 100, 10)
    assert abs(sm["regel"]["mittel"] - sm["emp"]["mittel"]) < 0.1 and sm["regel"]["mittel"] == _approx(40.45) and sm["emp"]["mittel"] == _approx(40.36)


def test_rebuilt_rule_always_picks_the_weights_in_the_quadratic_world():
    for satz in "ab":
        rs = [r for r in V.rows(satz) if r["welt"] == "quadratisch"]
        assert len(rs) == 360 and all(r["regel"] == "gewichte" for r in rs)


def test_rebuilt_diagnosis_rates():
    a, b = V.diagnosis_rates("a"), V.diagnosis_rates("b")
    assert a["nichtlinear_erkannt"] == 1.0 and b["nichtlinear_erkannt"] == 1.0
    assert (a["fehlalarm"], b["fehlalarm"]) == (_approx(0.0389, 4), _approx(0.0444, 4))
    assert (a["streuung_linear"], b["streuung_linear"]) == (0.975, 0.975)
    assert (a["streuung_betrag"], b["streuung_betrag"]) == (_approx(0.0083, 4), _approx(0.0583, 4))
    assert (a["streuung_konstant"], b["streuung_konstant"]) == (0.0, _approx(0.0167, 4))


def test_blind_spot_world_costs_in_the_rebuild():
    w = V.by_world("a", "betrag")
    assert (w["regel"]["mittel"], w["beste"]["mittel"], w["emp"]["mittel"]) == (_approx(12.96), _approx(11.27), _approx(12.08))


# --- Nachbau: Zeilen gegen eine Neurechnung (Band) ----------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("welt, p, n, seed", [("quadratisch", 3, 100, 0), ("streuung-linear", 3, 1000, 5), ("betrag", 10, 100, 29), ("linear", 10, 1000, 100)])
def test_stored_rows_match_a_recomputation(welt, p, n, seed):
    satz = "a" if seed < 100 else "b"
    data = S.make_dataset(welt, p, n, seed)
    fresh = ev.extras_for_taus(data, C.TAU_OPTIONS, ("emp", "direkt", "kern"))
    stored = {r["tau"]: r for r in V.rows(satz) if (r["welt"], r["p"], r["n"], r["seed"]) == (welt, p, n, seed)}
    assert len(stored) == 3
    for tau in C.TAU_OPTIONS:
        for m in ("emp", "direkt", "kern"):
            assert fresh[tau][m] == pytest.approx(stored[tau][m], abs=0.01, rel=1e-4), (tau, m)


# --- Texte, die App und README teilen ------------------------------------------------------------------------------------------------------------------


def test_assumptions_table_stands_verbatim_in_the_readme():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert C.ASSUMPTIONS_MARKDOWN.strip() in readme


def test_the_numbers_of_the_assumptions_table_are_the_measured_ones():
    t = C.ASSUMPTIONS_MARKDOWN
    fb = V.frozen_set("b")["gesamt"]
    for number in ("18,75", "33,28", "35,12", "34,67", "17,83"):
        assert number in t
    assert (fb["regel"]["mittel"], fb["emp"]["mittel"], fb["direkt"]["mittel"], fb["wald"]["mittel"], fb["beste"]["mittel"]) == (18.75, 33.28, 35.12, 34.67, 17.83)
    for number in ("0,8 %", "5,8 %", "306 von 360", "13,0 %", "11,3 %", "23,2 %", "33,7 %", "18,84 %", "33,45 %", "40,4 % gegen 40,4 %", "38,3 % gegen 38,5 %"):
        assert number in t
    wa = V.frozen_set("a")
    assert wa["welten"]["heteroskedastisch"]["wahl"]["emp"] == 306 and wa["welten"]["heteroskedastisch"]["regel"] == 13.0 and wa["welten"]["heteroskedastisch"]["beste"] == 11.3
    sm = V.summarize(V.rows("a"))
    assert (sm["regel"]["mittel"], sm["emp"]["mittel"]) == (_approx(23.2, 1), _approx(33.7, 1))
    assert (wa["diagnose"]["streuung_betrag"], V.frozen_set("b")["diagnose"]["streuung_betrag"]) == (4, 3)
    assert V.diagnosis_rates("a")["streuung_betrag"] == _approx(0.008, 3) and V.diagnosis_rates("b")["streuung_betrag"] == _approx(0.058, 3)


def _cell_mean(satz, welt, tau, col, p=3, n=1000):
    from statistics import mean
    return mean(r[col] for r in V.rows(satz) if (r["welt"], r["p"], r["n"], r["tau"]) == (welt, p, n, tau))


def test_tree_asks_the_form_before_the_cost_question_at_tau_half():
    """Prüfbericht: bei τ = 0,5 führte der Baum ohne Frage nach der Form zu „Mittel genügt“. Die Zahlen des Nachbaus, die README und Baum nennen."""
    import elw_tree as T
    for form, leaf in (("ja", "mittel"), ("nein", "gewichte")):
        assert T.walk(T.answers_from(0.5, 1000, 3, {**C.DEFAULT_TREE, "q_form": form})).leaf == leaf
    got = {(satz, welt, col): round(_cell_mean(satz, welt, 0.5, col), 1) for satz in "ab" for welt in ("quadratisch", "linear") for col in ("emp", "kern")}
    assert (got["a", "quadratisch", "emp"], got["a", "quadratisch", "kern"], got["a", "linear", "emp"], got["a", "linear", "kern"]) == (56.6, 15.2, 0.2, 5.4)
    assert (got["b", "quadratisch", "emp"], got["b", "quadratisch", "kern"], got["b", "linear", "emp"], got["b", "linear", "kern"]) == (55.3, 14.7, 0.2, 5.3)
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    for text in ("56,6 %", "15,2 %", "55,3 und 14,7 %", "0,2 % und 5,4 %", "0,2 und 5,3 %"):
        assert text in readme, text


def test_kernel_weights_win_in_one_cell_with_non_monotone_spread():
    """Die README schränkt „Gewichte nur bei gekrümmter Form“ ein: im Nachbau gewinnt der Kern auch bei Streuung in |x₁| (p = 3, n = 1000, τ = 0,95)."""
    got = tuple(round(_cell_mean("a", "betrag", 0.95, col), 1) for col in ("kern", "emp", "direkt"))
    assert got == (14.4, 19.6, 20.2)
    assert "Kerngewichte 14,4 % gegen 19,6 % für KQ + Quantil und 20,2 % für direkt" in (ROOT / "README.md").read_text(encoding="utf-8")


def test_the_rule_equals_the_fixed_forest_in_the_quadratic_world_and_median_edge_is_small():
    for satz, regel in (("a", 47.4), ("b", 48.3)):
        row = V.frozen_set(satz)["welten"]["nichtlinear"]
        assert row["regel"] == regel and min(row["emp"], row["direkt"], row["wald"]) == regel
    g = V.frozen_set("a")["gesamt"]
    assert (g["regel"]["median"], g["direkt"]["median"]) == (9.60, 9.77) and g["direkt"]["median"] == min(g[k]["median"] for k in V.FIXED)


def test_the_readme_names_a_test_count():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert re.search(r"\d+ Tests", readme)


def test_live_detection_counts_of_experiment_4():
    """Zählung über 20 feste Instanzen (Zufalls-Seeds 200000 bis 200019), nur Diagnose: Streuungs-Test bei linear wachsender Streuung und bei Streuung in |x₁|."""
    lin, absv = ev.detection_counts("streuung-linear", 3, 1000), ev.detection_counts("betrag", 3, 1000)
    assert (lin["hetero"], absv["hetero"], lin["runs"]) == (20, 0, 20)
    lin, absv = ev.detection_counts("streuung-linear", 10, 100), ev.detection_counts("betrag", 10, 100)
    assert (lin["hetero"], absv["hetero"]) == (16, 3)
    assert lin["wahl"]["direkt"] == 0 and lin["wahl"]["emp"] == 20            # n = 100 < 110 = 10 (p + 1): direkt ist ausgeschlossen
