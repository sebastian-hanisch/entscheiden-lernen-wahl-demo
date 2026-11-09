"""End-to-end Smoke-Test via Streamlits offiziellem AppTest-Framework: lädt app.py mit den Standardeinstellungen, klickt jeden Button, stellt jeden Regler auf jede Stufe
und prüft, dass kein Python-Fehler auftritt und jeder Abschnitt sein Verdikt zeigt.

Die Regler der App sind Auswahlregler (`select_slider`, `selectbox`) und ein Zahlenfeld; die allgemeinen stehen in der Seitenleiste, die des Baums im Hauptbereich."""

import os

import pytest
from streamlit.testing.v1 import AppTest

import elw_constants as C
import elw_presets as P
import elw_tree as T

APP_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")
TIMEOUT = int(os.environ.get("APPTEST_TIMEOUT", "300"))  # unter Last (parallele Läufe) reichen 180 s nicht immer; Umgebungsvariable zum Anpassen


def _fresh_app() -> AppTest:
    at = AppTest.from_file(APP_PATH, default_timeout=TIMEOUT)
    at.run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    return at


def _verdicts(at) -> list[str]:
    return [e.value for kind in ("success", "info", "warning") for e in getattr(at, kind)]


def test_app_loads_without_exception_and_shows_every_verdict():
    at = _fresh_app()
    verdicts = _verdicts(at)
    for marker in ("**direkt (Quantilregression)** (Stück 2", "Vormessung mit Zufallswald (nicht live), 1440 Zeilen", "wählt die Regel", "Median über 5 feste Instanzen",
                   "**44 % weniger Mehrkosten** im Mittel über die vier Welten", "schlägt die Diagnose auf die Streuung", "Die Streuungs-Diagnose schlägt bei linear wachsender Streuung in 98 %"):
        assert any(marker in v for v in verdicts), marker


def test_every_button_click_does_not_raise():
    labels = [b.label for b in _fresh_app().button]
    assert labels, "keine Buttons gefunden"
    for index, label in enumerate(labels):
        at = _fresh_app()
        at.button[index].click().run(timeout=TIMEOUT)
        assert not at.exception, (label, [str(e) for e in at.exception])


LOESER_REGLER = {"q_linear", "q_modell", "q_loeser", "q_rauschen", "q_ngitter"}      # im Zweig „Entscheidung mit Löser“: sonst ausgegraut (AppTest bedient keine ausgegrauten Regler)
LOESER_PRESET = "Entscheidung mit Löser (SPO+)"


def _control_values(state_key):
    spec = P.SETTING_SPECS[state_key]
    return list(spec.options) if spec.options is not None else [spec.lo, spec.hi]


@pytest.mark.parametrize("state_key", list(P.SETTING_SPECS))
def test_every_control_at_its_levels_does_not_raise(state_key):
    for value in _control_values(state_key):
        at = _fresh_app()
        if state_key in LOESER_REGLER:
            at.button(key=f"preset_{LOESER_PRESET}").click().run(timeout=TIMEOUT)
        for art in ("select_slider", "selectbox", "number_input"):
            try:
                widget = getattr(at, art)(key=state_key)
                break
            except KeyError:
                continue
        else:
            raise AssertionError(state_key)
        widget.set_value(value).run(timeout=TIMEOUT)
        assert not at.exception, (state_key, value, [str(e) for e in at.exception])


def test_the_tree_answer_shows_the_leaf_and_its_piece_link():
    at = _fresh_app()
    leaf = T.LEAVES["direkt"]
    assert any(leaf.verfahren in s.value and leaf.text in s.value for s in at.success)
    at.selectbox(key="q_art").set_value("loeser").run(timeout=TIMEOUT)
    at.selectbox(key="q_loeser").set_value("glaettung").run(timeout=TIMEOUT)
    assert not at.exception
    assert any(T.LEAVES["glaettung"].text in s.value for s in at.success)
    at.selectbox(key="q_garantie").set_value("ja").run(timeout=TIMEOUT)
    assert any(T.LEAVES["schranke"].text in s.value and s.value.startswith("Zusätzlich") for s in at.success)


def test_the_blind_spot_leaf_warns_that_the_rule_would_not_find_it():
    at = _fresh_app()
    at.selectbox(key="q_streuung").set_value("nicht_monoton").run(timeout=TIMEOUT)
    assert any("Die Wahlregel aus Experiment 2 würde es nicht finden" in w.value for w in at.warning)


def test_permalink_sets_the_controls_and_snaps_to_levels():
    at = AppTest.from_file(APP_PATH, default_timeout=TIMEOUT)
    at.query_params["welt"] = "betrag"
    at.query_params["n"] = "250"          # rastet auf 100 ein
    at.query_params["tau"] = "0.8"
    at.query_params["p"] = "10"
    at.query_params["qa"] = "loeser"
    at.query_params["qo"] = "stoerung"
    at.run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    assert at.selectbox(key="welt_select").value == "betrag" and at.select_slider(key="n_select").value == 100
    assert at.select_slider(key="tau_select").value == 0.8 and at.select_slider(key="p_select").value == 10
    assert at.selectbox(key="q_art").value == "loeser" and at.selectbox(key="q_loeser").value == "stoerung"


def test_permalink_ignores_unusable_values():
    at = AppTest.from_file(APP_PATH, default_timeout=TIMEOUT)
    at.query_params["welt"] = "gibt-es-nicht"
    at.query_params["n"] = "abc"
    at.query_params["seed"] = "-4"
    at.query_params["qa"] = "unsinn"
    at.run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    assert at.selectbox(key="welt_select").value == P.SETTING_SPECS["welt_select"].default
    assert at.select_slider(key="n_select").value == C.DEFAULT_N and at.number_input(key="seed_input").value == 0
    assert at.selectbox(key="q_art").value == C.DEFAULT_TREE["q_art"]


def test_headings_name_the_four_experiments():
    at = _fresh_app()
    titles = [s.value for s in at.subheader]
    for n in range(1, 5):
        assert any(f"Experiment {n}" in t for t in titles), n
    assert any("Wo die Annahmen enden" in t for t in titles)
    assert any(e.label == "📐 Mathematische Formulierung" for e in at.expander)
    assert any("Kernfrage" in m.value for m in at.markdown)
