"""Verfahrenswahl: welches Verfahren, gemessen woran? - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Schluss der Linie „Entscheidungsorientiertes Lernen“: ein Entscheidungsbaum über die neun Stücke und eine Wahlregel nach zwei Diagnosen auf den Trainingsdaten
(F-Test auf Krümmung, F-Test auf lineare Streuung). Schnellstart und allgemeine Einstellungen stehen in der Seitenleiste, die Antworten des Baums beim Experiment 1;
Zahlen der Vormessung (Zufallswald, scikit-learn) sind eingefroren und gekennzeichnet, alles andere rechnet live oder steht als vorgerechneter Nachbau da.

Lauffähig mit: streamlit run app.py
"""

import pandas as pd
import streamlit as st

import elw_constants as C
import elw_diagnose as D
import elw_evaluation as ev
import elw_tree as T
import elw_vormessung as V
from elw_evaluation import Settings, de, pct
from elw_layout import dunkles_design, kopf_css
from elw_presets import (
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    options_of,
    randomize_seed,
    sidebar_presets,
    sync_query_params,
)
from elw_visualization import build_abs_residuals, build_band, build_methods, build_rule_vs_fixed

st.set_page_config(page_title="Verfahrenswahl – Sebastian Hanisch", layout="wide")


@st.cache_data(show_spinner=False)
def _analysis(settings):
    return ev.analyse(settings)


@st.cache_data(show_spinner=False)
def _band(welt, p, n, tau):
    return ev.band(welt, p, n, tau)


@st.cache_data(show_spinner=False)
def _detect(welt, p, n):
    return ev.detection_counts(welt, p, n)


@st.cache_data(show_spinner=False)
def _resid(welt, p, n, seed):
    return ev.abs_residuals(welt, p, n, seed)


# Fester Kopf: nur der Titel bleibt beim Scrollen oben (Stylesheet in elw_layout.py).
st.html(kopf_css(dunkles_design()))
with st.container(key="kopf"):
    st.title("🧭 Verfahrenswahl: welches Verfahren, gemessen woran?")
st.markdown(
    """
Die Linie **Entscheidungsorientiertes Lernen** hat neun Verfahren gezeigt, und keines gewinnt überall. Diese Demo ordnet sie: ein **Entscheidungsbaum** sagt, welches Verfahren unter welchen Bedingungen gemessen
gewinnt, und eine **Wahlregel** trifft die Wahl für den Newsvendor aus **zwei Diagnosen auf den Trainingsdaten** (Ist die Form gekrümmt? Hängt die Streuung linear von den Merkmalen ab?).
Gemessen wird in simulierten Welten als **Mehrkosten gegen das Orakel**, das die wahre Verteilung kennt; die Aussagen gelten nur für diese Welten, nicht für echte Daten.

Wer die Verfahrenswahl zuerst ohne Fachbegriffe verstehen will, öffnet **💬 In einfachen Worten**. Weiter unten: vier Experimente (🔬), der Abschnitt **🚧 Wo die Annahmen enden** und der Expander **📐 Mathematische Formulierung**.
"""
)

# erklaerseite:einfach:start (erzeugt von tools/set_demo_explainer.py (Website-Repo) aus tools/content/explainers/welcher-weg-gewinnt-erklaert.json, nicht von Hand ändern)
with st.expander("💬 In einfachen Worten"):
    st.markdown(
        """
Aus früheren Tagen lässt sich eine Bestellmenge auf mehreren Wegen ableiten: den erwarteten Absatz bestellen, eine Prognose mit Sicherheitszuschlag, direkt auf den Kosten lernen oder ähnliche Tage gewichten. In den gemessenen Welten gewinnt kein Weg überall. Die Demo zeigt, wovon es abhängt: von zwei Fragen an die eigenen Daten.

Erste Frage: Ist der Zusammenhang zwischen Umständen und Nachfrage gekrümmt statt gerade? Zweite Frage: Schwankt die Streuung mit den Umständen? Ein Entscheidungsbaum ordnet die Verfahren danach, und eine Wahlregel trifft die Wahl aus den Daten selbst. Sie hat einen blinden Fleck: Eine Streuung, die an beiden Enden groß und in der Mitte klein ist, erkennt sie nicht. Gemessen wird in erfundenen Welten, die Aussagen gelten nicht für echte Daten.

[Ausführlich und verständlich erklärt, mit Fachteil](https://sebastianhanisch.net/welcher-weg-gewinnt-erklaert.html)
        """
    )
# erklaerseite:einfach:end

with st.expander("So funktioniert die Verfahrenswahl", expanded=True):
    st.markdown(
        """
1. **Der Baum:** Fragen nach der Entscheidung (Menge je Fall oder Weg mit Löser), nach der Form der Kosten, der Streuung, der Datenmenge und dem, was man über den Löser weiß, führen zu einem Blatt. Jedes Blatt nennt das Verfahren, das **Stück** der Linie, in dem es gemessen ist, und einen Messwert aus der README dieses Stücks.
2. **Die Regel (Newsvendor):** zeigt ein F-Test, dass Quadrate der Merkmale die lineare Anpassung verbessern (p-Wert unter 0,01), wählt sie **Gewichte**. Sonst: hängt der Betrag der Residuen linear von den Merkmalen ab (p-Wert unter 0,01) und gibt es mindestens 10·(p + 1) Fälle, wählt sie **direkt**. Sonst **KQ plus Quantil**.
3. **Die Prüfung:** die Regel wurde vor der Messung festgelegt und nicht abgestimmt; gemessen wird sie gegen jedes feste Verfahren und gegen die beste Wahl je Instanz, die nur mit Kenntnis der Welt möglich ist.
        """
    )

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

# --- Schnellstart und allgemeine Einstellungen (Seitenleiste); die Antworten des Baums stehen bei Experiment 1 ----------------------------------

with st.sidebar:
    sidebar_presets()  # Schnellstart oben, das aktive Preset ist hervorgehoben
    st.header("⚙️ Einstellungen")
    st.markdown("**Newsvendor-Daten (Experimente 2 bis 4)**")
    welt = st.selectbox("Welt der Daten", options_of("welt_select"), key="welt_select", format_func=lambda k: C.WELT_NAMEN[k],
                        help="Wie die Nachfrage von den Merkmalen abhängt. Gilt für Experiment 2; Experiment 4 vergleicht zwei feste Welten.")
    tau = st.select_slider("Kritisches Verhältnis τ", options_of("tau_select"), key="tau_select", format_func=lambda t: ev.tau_text(t),
                           help="τ = b / (b + h): Anteil der Fehlmengenkosten an der Summe beider Kosten; h = 1 festgelegt. τ = 0,95: eine fehlende Einheit kostet 19-mal so viel wie eine überschüssige. "
                                "Der Baum fragt „Sind die Kosten schief?“ und liest die Antwort hier ab (nein bei τ = 0,5).")
    n = st.select_slider("Beobachtungen n", options_of("n_select"), key="n_select",
                         help="Zahl der Trainingsdaten (Tage mit Merkmalen und beobachteter Nachfrage). Der Baum liest hier ab, ob n mindestens 10·(p + 1) ist.")
    p = st.select_slider("Merkmale p", options_of("p_select"), key="p_select", help="Zahl der Merkmale; nur das erste (x₁) beeinflusst in den Welten die Streuung oder die Wahrheit.")
    st.markdown("**Einzellauf**")
    seed = st.number_input("Zufalls-Seed des Einzellaufs", *bounds("seed_input"), key="seed_input", step=1,
                           help="Wählt den einen Datensatz, den Experiment 2 zeigt und Experiment 4 zeichnet. Das Band in Experiment 2 rechnet über fünf feste Instanzen und hängt davon nicht ab.")
    st.button("🎲 Neuer Zufalls-Seed", width="stretch", on_click=randomize_seed)

sync_query_params({k: st.session_state[k] for k in ("welt_select", "tau_select", "n_select", "p_select", "seed_input",
                                                    "q_art", "q_form", "q_streuung", "q_linear", "q_modell", "q_loeser", "q_rauschen", "q_ngitter", "q_garantie")})

# --- Experiment 1: der Entscheidungsbaum -----------------------------------------------------------------------------------------------------

st.markdown("---")
st.subheader("🔬 Experiment 1: Der Entscheidungsbaum")
st.caption("Beantworten Sie die Fragen für Ihre Lage; nicht benötigte Fragen sind ausgegraut. Zwei Antworten liest der Baum aus der Seitenleiste ab (τ und die Datenmenge). "
           "Jedes Blatt verlinkt auf die Demo des Stücks, in dem das Verfahren gemessen ist; die Zahl im Blatt steht wörtlich in der README dieses Stücks.")

tree_values = {k: st.session_state[k] for k in C.DEFAULT_TREE}
path_now = T.walk(T.answers_from(float(tau), int(n), int(p), tree_values))
on_path = set(path_now.questions)
c1, c2, c3 = st.columns(3)
c1.selectbox("Was wird entschieden?", C.Q_ART, key="q_art", format_func=lambda k: C.Q_ART_NAMEN[k])
c1.selectbox("Newsvendor: Ist der Mittelwert linear in den Merkmalen?", C.Q_FORM, key="q_form", format_func=lambda k: C.Q_FORM_NAMEN[k], disabled="form" not in on_path,
             help="Wird vor der Kostenfrage gestellt: auch bei τ = 0,5 genügt das Mittel bei gekrümmtem Mittelwert nicht (Nachbau in der README, Befunde).")
c1.selectbox("Newsvendor: Hängt die Streuung von x ab?", C.Q_STREUUNG, key="q_streuung", format_func=lambda k: C.Q_STREUUNG_NAMEN[k], disabled="streuung" not in on_path,
             help="Die Streuungs-Diagnose von Experiment 2 erkennt praktisch nur eine Streuung mit linearer Komponente in den Merkmalen; die symmetrisch nicht monotone Streuung in |x₁| bleibt unerkannt (blinder Fleck, Experiment 4).")
c2.selectbox("Löser: Trifft ein lineares Modell die Kosten?", C.Q_LINEAR, key="q_linear", format_func=lambda k: C.Q_LINEAR_NAMEN[k], disabled="linear" not in on_path)
c2.selectbox("Löser: Darf das Modell flexibel sein?", C.Q_MODELL, key="q_modell", format_func=lambda k: C.Q_MODELL_NAMEN[k], disabled="modell" not in on_path)
c2.selectbox("Löser: Was ist über den Löser bekannt?", C.Q_LOESER, key="q_loeser", format_func=lambda k: C.Q_LOESER_NAMEN[k], disabled="loeser" not in on_path)
c3.selectbox("Löser: Sind die Ziele verrauscht?", C.Q_RAUSCHEN, key="q_rauschen", format_func=lambda k: C.Q_RAUSCHEN_NAMEN[k], disabled=not ({"rauschen", "bb_rauschen"} & on_path))
c3.selectbox("Löser mit Ersatzverlust: Trainingsfälle", C.Q_NGITTER, key="q_ngitter", disabled="rauschen" not in on_path,
             help="Die Stücke 3 und 4 messen n = 100 und n = 1000.")
c3.selectbox("Wird eine Garantie für die Kosten verlangt?", C.Q_GARANTIE, key="q_garantie", format_func=lambda k: C.Q_GARANTIE_NAMEN[k])

schief_text = "schief (τ ≠ 0,5)" if abs(float(tau) - 0.5) > 1e-9 else "symmetrisch (τ = 0,5)"
enough_text = f"n = {int(n)} {'≥' if int(n) >= 10 * (int(p) + 1) else '<'} 10·(p + 1) = {10 * (int(p) + 1)}"
st.caption(f"Aus der Seitenleiste abgelesen: Kosten {schief_text}; {enough_text}.")

st.graphviz_chart(T.to_dot(path_now), width="stretch")
st.caption("Hervorgehoben ist der Weg Ihrer Antworten; die Blätter verlinken auf die Demos der Stücke. Über das Symbol oben rechts am Diagramm lässt es sich im Vollbild öffnen.")


def _leaf_box(leaf_id, prefix=""):
    leaf = T.LEAVES[leaf_id]
    st.success(f"{prefix}**{leaf.verfahren}** (Stück {leaf.stueck}: {C.STUECKE[leaf.stueck][1]}). {leaf.text}")
    st.link_button(f"→ Stück {leaf.stueck}: {C.STUECKE[leaf.stueck][1]}", C.stueck_url(leaf.stueck))


_leaf_box(path_now.leaf)
if path_now.extra_leaf:
    _leaf_box(path_now.extra_leaf, "Zusätzlich, für die Garantie: ")
if path_now.leaf == "blind":
    st.warning("Dieses Blatt gibt es nur, wenn Sie die Streuung selbst als nicht monoton einschätzen. Die Wahlregel aus Experiment 2 würde es nicht finden: ihre Diagnose erkennt praktisch nur eine Streuung mit linearer Komponente in den Merkmalen.")

with st.expander("Alle Blätter des Baums mit Messwert und Stück"):
    st.dataframe(
        pd.DataFrame([{"Verfahren": lf.verfahren, "Stück": f"{lf.stueck}: {C.STUECKE[lf.stueck][1]}", "Messwert (aus der README des Stücks)": lf.text, "Demo": C.stueck_url(lf.stueck)}
                      for lf in T.LEAVES.values()]),
        hide_index=True, width="stretch", column_config={"Demo": st.column_config.LinkColumn("Demo", display_text="öffnen")})
    st.caption("Alle Messwerte sind Medianwerte der Mehrkosten gegen das Orakel (Newsvendor) beziehungsweise des Regrets in Prozent (Gitter) über feste Seeds, in den Welten der jeweiligen Demo. "
               "Zahlen aus verschiedenen Stücken sind nicht miteinander vergleichbar (andere Welten, andere Maße).")

# --- Kernfrage -------------------------------------------------------------------------------------------------------------------------------

fz_a, fz_b = V.frozen_set("a")["gesamt"], V.frozen_set("b")["gesamt"]
best_name_a, best_val_a = V.best_fixed("a")
nb_a = V.summarize(V.rows("a"))
st.markdown("---")
st.markdown("## ❓ Kernfrage: Wählt eine einfache Diagnose das Verfahren fast so gut wie die Kenntnis der Welt?")
st.success(
    f"Vormessung mit Zufallswald (nicht live), 1440 Zeilen aus vier Welten, p ∈ {{3, 10}}, n ∈ {{100, 1000}}, τ ∈ {{0,5; 0,8; 0,95}}, Seeds 0 bis 29: mittlere Mehrkosten der **Regel {de(fz_a['regel']['mittel'], 2)} %** "
    f"gegen {de(fz_a['emp']['mittel'], 2)} % (immer KQ + Quantil), {de(fz_a['direkt']['mittel'], 2)} % (immer direkt) und {de(fz_a['wald']['mittel'], 2)} % (immer Wald-Gewichte); die beste Wahl je Zeile mit Kenntnis der Welt "
    f"hätte {de(fz_a['beste']['mittel'], 2)} % gekostet. Gegenüber dem besten festen Verfahren (nach dem Mittel) spart die Regel {de(V.reduction_vs_best_fixed('a'), 0)} % der mittleren Mehrkosten über die vier Welten; "
    f"in drei der vier Welten ist ein festes Verfahren besser, und im Median kostet die Regel {de(fz_a['regel']['median'], 2)} % gegen {de(min(fz_a[k]['median'] for k in V.FIXED), 2)} % für das beste feste Verfahren nach dem Median. "
    f"Mit Gauß-Kerngewichten statt des Waldes (Nachbau mit dem Code dieser Demo, Seeds 0 bis 29) sind es {de(nb_a['regel']['mittel'], 1)} % gegen {de(nb_a['emp']['mittel'], 1)} % (immer KQ + Quantil)."
)

# --- Experiment 2: die Diagnose auf eigenen Daten ---------------------------------------------------------------------------------------------

st.markdown("---")
st.subheader("🔬 Experiment 2: Die Diagnose auf eigenen Daten")
st.caption("Die Welt, τ, n, p und der Zufalls-Seed stammen aus der Seitenleiste. Die App zieht die Trainingsdaten, rechnet die zwei F-Tests, lässt die Regel wählen und bewertet alle Verfahren gegen das Orakel. "
           "Der Zweig „Gewichte“ rechnet hier live mit **Gauß-Kerngewichten** (und zusätzlich kNN-Gewichten), **nicht** mit dem Zufallswald der Vormessung.")
settings = Settings(welt, float(tau), int(n), int(p), int(seed))
with st.spinner("Zieht die Daten, rechnet die Tests und alle Verfahren"):
    a = _analysis(settings)
d = a.diagnosis
m1, m2, m3, m4 = st.columns(4)
m1.metric("p-Wert Krümmung", ev.p_text(d.p_nonlin), delta="unter 0,01: Gewichte" if d.p_nonlin < D.ALPHA else "nicht unter 0,01", delta_color="off")
m2.metric("p-Wert Streuung", ev.p_text(d.p_hetero), delta="unter 0,01" if d.p_hetero < D.ALPHA else "nicht unter 0,01", delta_color="off")
m3.metric("Fälle", f"{d.n}", delta=f"Schwelle 10·(p + 1) = {10 * (d.p + 1)}", delta_color="off")
m4.metric("Wahl der Regel", C.REGEL_NAMEN[a.rule], delta=f"Mehrkosten {pct(a.rule_extra)}", delta_color="off")
st.plotly_chart(build_methods(a.extras, C.REGEL_METHODE[a.rule], a.best_method), width="stretch", key="methods_chart")
st.caption("Balken: Mehrkosten gegen das Orakel auf dieser einen Instanz (exakte erwartete Kosten über 2000 feste Testpunkte). Roter Rand: Wahl der Regel; „beste“: das Verfahren mit den kleinsten Mehrkosten auf dieser Instanz, im Nachhinein und mit Kenntnis der Welt.")
kind, text = ev.verdict_instance(a)
getattr(st, kind)(text)

st.markdown("**Dasselbe über fünf feste Instanzen** (Zufalls-Seeds 100000 bis 100004; Median, Fehlerbalken: Minimum bis Maximum)")
with st.spinner("Rechnet fünf Instanzen (je Instanz Diagnose, ein lineares Programm und die Gewichte)"):
    b = _band(welt, int(p), int(n), float(tau))
st.plotly_chart(build_band(b), width="stretch", key="band_chart")
kind, text = ev.verdict_band(b, welt, int(p), int(n), float(tau))
getattr(st, kind)(text)

# --- Experiment 3: die Regel gegen immer dasselbe Verfahren -----------------------------------------------------------------------------------

st.markdown("---")
st.subheader("🔬 Experiment 3: Die Regel gegen immer dasselbe Verfahren")
st.caption("Mittlere Mehrkosten gegen das Orakel über alle 1440 Zeilen (4 Welten × p ∈ {3, 10} × n ∈ {100, 1000} × τ ∈ {0,5; 0,8; 0,95} × 30 Seeds). Blau: die **Vormessung mit Zufallswald (scikit-learn, nicht live)**, "
           "eingefroren; orange: der **Nachbau mit dem Code dieser Demo**, vorgerechnet (`tools/messung.py`), mit Gauß-Kerngewichten statt des Waldes.")
tab_all, tab_world, tab_np = st.tabs(["Alle Zeilen", "Je Welt", "Je n und p"])


def _fmt(x):
    return de(x, 1)


with tab_all:
    frozen_means = {k: fz_a[k]["mittel"] for k in ("emp", "direkt", "wald", "regel", "beste")}
    rebuilt_means = {k: nb_a[k]["mittel"] for k in ("emp", "direkt", "kern", "regel", "beste")}
    st.plotly_chart(build_rule_vs_fixed(frozen_means, rebuilt_means, "Vormessung mit Zufallswald (nicht live)"), width="stretch", key="rule_chart")
    st.caption("Seeds 0 bis 29. Die Zahlen über den Balken sind die mittleren Mehrkosten in Prozent.")
    rows_all = []
    for satz in ("a", "b"):
        g = V.frozen_set(satz)["gesamt"]
        nb = V.summarize(V.rows(satz))
        rows_all.append({"Satz": f"Vormessung (Wald), {V.SAETZE[satz]}", "immer KQ + Quantil": f"{de(g['emp']['mittel'], 2)} (Median {de(g['emp']['median'], 2)})",
                         "immer direkt": f"{de(g['direkt']['mittel'], 2)} (Median {de(g['direkt']['median'], 2)})", "immer Gewichte": f"{de(g['wald']['mittel'], 2)} (Median {de(g['wald']['median'], 2)})",
                         "Regel": f"{de(g['regel']['mittel'], 2)} (Median {de(g['regel']['median'], 2)})", "beste Wahl je Zeile": f"{de(g['beste']['mittel'], 2)} (Median {de(g['beste']['median'], 2)})"})
        rows_all.append({"Satz": f"Nachbau (Kern), {V.SAETZE[satz]}", "immer KQ + Quantil": f"{de(nb['emp']['mittel'], 2)} (Median {de(nb['emp']['median'], 2)})",
                         "immer direkt": f"{de(nb['direkt']['mittel'], 2)} (Median {de(nb['direkt']['median'], 2)})", "immer Gewichte": f"{de(nb['kern']['mittel'], 2)} (Median {de(nb['kern']['median'], 2)})",
                         "Regel": f"{de(nb['regel']['mittel'], 2)} (Median {de(nb['regel']['median'], 2)})", "beste Wahl je Zeile": f"{de(nb['beste']['mittel'], 2)} (Median {de(nb['beste']['median'], 2)})"})
    st.dataframe(pd.DataFrame(rows_all), hide_index=True, width="stretch")
    kind, text = ev.verdict_rule_vs_fixed()
    getattr(st, kind)(text)

with tab_world:
    rows_w = []
    for welt_name in C.WELTEN:
        for satz in ("a", "b"):
            fz = V.frozen_set(satz)["welten"][C.WELT_IN_VORMESSUNG[welt_name]]
            nb = {k: v["mittel"] for k, v in V.by_world(satz, welt_name).items()}
            rows_w.append({"Welt": C.WELT_NAMEN[welt_name], "Zeilen": f"Vormessung (Wald), {V.SAETZE[satz]}", "immer KQ + Quantil": _fmt(fz["emp"]), "immer direkt": _fmt(fz["direkt"]),
                           "immer Gewichte": _fmt(fz["wald"]), "Regel": _fmt(fz["regel"]), "beste Wahl": _fmt(fz["beste"])})
            rows_w.append({"Welt": C.WELT_NAMEN[welt_name], "Zeilen": f"Nachbau (Kern), {V.SAETZE[satz]}", "immer KQ + Quantil": _fmt(nb["emp"]), "immer direkt": _fmt(nb["direkt"]),
                           "immer Gewichte": _fmt(nb["kern"]), "Regel": _fmt(nb["regel"]), "beste Wahl": _fmt(nb["beste"])})
    st.dataframe(pd.DataFrame(rows_w), hide_index=True, width="stretch")
    st.caption("Mittlere Mehrkosten in Prozent je Welt (360 Zeilen je Satz). In der Welt „Wahrheit quadratisch“ wählt die Regel in allen 360 Zeilen die Gewichte; dort hängt ihr Wert vom Verfahren hinter dem Zweig ab (Wald 47,4 % in Seeds 0 bis 29, 48,3 % in Seeds 100 bis 129; Gauß-Kern im Nachbau höher).")

with tab_np:
    rows_n = []
    for nn in C.N_OPTIONS:
        for pp in C.P_OPTIONS:
            for satz in ("a", "b"):
                fz = V.frozen_set(satz)["n_p"][f"n{nn}_p{pp}"]
                nb = {k: v["mittel"] for k, v in V.by_np(satz, nn, pp).items()}
                rows_n.append({"n, p": f"n = {nn}, p = {pp}", "Zeilen": f"Vormessung (Wald), {V.SAETZE[satz]}", "immer KQ + Quantil": _fmt(fz["emp"]), "immer direkt": _fmt(fz["direkt"]),
                               "immer Gewichte": _fmt(fz["wald"]), "Regel": _fmt(fz["regel"]), "beste Wahl": _fmt(fz["beste"])})
                rows_n.append({"n, p": f"n = {nn}, p = {pp}", "Zeilen": f"Nachbau (Kern), {V.SAETZE[satz]}", "immer KQ + Quantil": _fmt(nb["emp"]), "immer direkt": _fmt(nb["direkt"]),
                               "immer Gewichte": _fmt(nb["kern"]), "Regel": _fmt(nb["regel"]), "beste Wahl": _fmt(nb["beste"])})
    st.dataframe(pd.DataFrame(rows_n), hide_index=True, width="stretch")
    st.caption("Mittlere Mehrkosten in Prozent je Kombination aus Datenmenge und Merkmalszahl (360 Zeilen je Satz).")

# --- Experiment 4: der blinde Fleck -----------------------------------------------------------------------------------------------------------

st.markdown("---")
st.subheader("🔬 Experiment 4: Der blinde Fleck der Diagnose")
st.caption("Die Streuungs-Diagnose prüft, ob der Betrag der Residuen **linear** von den Merkmalen abhängt. Hängt die Streuung von |x₁| ab, wächst sie links und rechts von der Mitte, und eine Gerade sieht nichts. "
           "Gezeigt wird das, behoben wird es nicht. n und p stammen aus der Seitenleiste.")
with st.spinner("Zählt, wie oft die Tests anschlagen (nur Diagnose, kein Lösen)"):
    cnt_lin = _detect("streuung-linear", int(p), int(n))
    cnt_abs = _detect("betrag", int(p), int(n))
x_lin, r_lin = _resid("streuung-linear", int(p), int(n), int(seed))
x_abs, r_abs = _resid("betrag", int(p), int(n), int(seed))
col_l, col_r = st.columns(2)
col_l.plotly_chart(build_abs_residuals(x_lin, r_lin, f"Streuung wächst linear: Steigung {de(ev.slope_of(x_lin, r_lin), 2)}"), width="stretch", key="resid_lin")
col_r.plotly_chart(build_abs_residuals(x_abs, r_abs, f"Streuung in |x₁|: Steigung {de(ev.slope_of(x_abs, r_abs), 2)}"), width="stretch", key="resid_abs")
st.caption("Punkte: Betrag des Residuums der linearen Anpassung über x₁ (eine Instanz, Zufalls-Seed der Seitenleiste). Rote Gerade: die Anpassung, die der Streuungs-Test prüft. Rechts ist sie fast waagerecht, obwohl die Streuung von x₁ abhängt.")
kind, text = ev.verdict_blind(cnt_lin, cnt_abs, int(n), int(p))
getattr(st, kind)(text)
fz_world = V.frozen_set("a")["welten"]
fz_diag = V.frozen_set("a")["diagnose"]
st.info(
    f"Vormessung mit Zufallswald (nicht live), Seeds 0 bis 29, über alle Zeilen der Welt: Die Streuungs-Diagnose schlägt bei linear wachsender Streuung in {fz_diag['streuung_linear']} % der Zeilen an, bei Streuung in |x₁| in {fz_diag['streuung_betrag']} %. "
    f"Die Regel kostet in der Welt mit Streuung in |x₁| im Mittel {de(fz_world['heteroskedastisch']['regel'], 1)} %, die beste Wahl je Zeile {de(fz_world['heteroskedastisch']['beste'], 1)} %."
)

# --- Grenzen ---------------------------------------------------------------------------------------------------------------------------------

st.markdown("---")
st.subheader("🚧 Wo die Annahmen enden")
st.markdown(C.ASSUMPTIONS_MARKDOWN)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    # erklaerseite:lesehilfe:start (erzeugt von tools/set_demo_explainer.py (Website-Repo) aus tools/content/explainers/welcher-weg-gewinnt-erklaert.json, nicht von Hand ändern)
    st.markdown(
        r"""
**Lesehilfe.** Gegeben sind $n$ Beobachtungen $(x_i, y_i)$ mit Merkmalen $x_i \in \mathbb{R}^p$ und der Nachfrage $y_i$; gesucht ist die Wahl eines von drei Verfahren, die eine Bestellregel $q(x)$ liefern. Die Wahl darf nur von den Trainingsdaten abhängen.

| Symbol | Bedeutung |
|---|---|
| $n$, $p$ | Zahl der Beobachtungen, Zahl der Merkmale |
| $\tau = b/(b+h)$ | kritisches Verhältnis: $b$ Kosten je fehlende, $h$ je überschüssige Einheit |
| $X$, $\tilde X$ | Designmatrix $[1, x]$ mit $p + 1$ Spalten, erweiterte Designmatrix $[1, x, x^2]$ (Quadrate elementweise) mit $2p + 1$ Spalten |
| $\mathrm{RSS}_0, \mathrm{RSS}_1$ | Summe der Fehlerquadrate des kleinen und des großen Modells |
| $F$, $p_{\mathrm{Krümmung}}$, $p_{\mathrm{Streuung}}$ | Teststatistik und p-Wert der beiden Tests |
| $\varepsilon_i$ | Betrag des Residuums der linearen Anpassung von $y$ auf $X$ |

[Schritt für Schritt erklärt, mit Herleitung und Literatur](https://sebastianhanisch.net/welcher-weg-gewinnt-erklaert.html#fachleute)
        """
    )
    # erklaerseite:lesehilfe:end
    st.markdown(C.FORMULAS_MARKDOWN)

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zur Reihe: [Entscheidungsorientiert: Regret bis Wahl](https://sebastianhanisch.net/konzepte-entscheidungsorientiert.html)."
)
