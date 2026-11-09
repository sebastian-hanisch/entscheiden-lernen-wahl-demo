"""Konstanten: Regler-Grenzen, feste Annahmen, Namen, Farben, Stücke der Linie und Presets."""

# --- Welten (Newsvendor mit Merkmalen, elw_scenario) -------------------------------------------------------------------------------------------
WELTEN = ("linear", "streuung-linear", "betrag", "quadratisch")
WELT_NAMEN = {
    "linear": "Wahrheit linear, Streuung konstant",
    "streuung-linear": "Streuung wächst linear mit x₁",
    "betrag": "Streuung hängt von |x₁| ab (nicht monoton)",
    "quadratisch": "Wahrheit quadratisch in x₁",
}
# Namen der Welten in der Vormessung (vormessung/out_eo_v11_wahlregel.txt)
WELT_IN_VORMESSUNG = {"linear": "linear", "streuung-linear": "streuung-linear", "betrag": "heteroskedastisch", "quadratisch": "nichtlinear"}

TAU_OPTIONS = (0.5, 0.8, 0.95)
DEFAULT_TAU = 0.95
N_OPTIONS = (100, 1000)
DEFAULT_N = 1000
P_OPTIONS = (3, 10)
DEFAULT_P = 3
SEED_MAX, DEFAULT_SEED = 999999, 7
SWEEP_SEEDS = tuple(range(100000, 100005))   # feste Instanzen des Live-Bandes in Experiment 2: nie ein einzelner Seed als Befund
BAND_SEEDS = tuple(range(20))                # Seeds des vorgerechneten Nachbaus (elw_bands.json)
DIAG_SEEDS = tuple(range(200000, 200020))    # Seeds der Live-Zählung in Experiment 4 (nur F-Tests, kein LP)

# --- Verfahren hinter den Blättern der Wahlregel -----------------------------------------------------------------------------------------------
METHODEN = ("emp", "direkt", "kern", "knn")
METHODE_NAMEN = {
    "emp": "KQ + Quantil der Residuen",
    "direkt": "direkt (Quantilregression)",
    "kern": "Gewichte (Gauß-Kern)",
    "knn": "Gewichte (kNN)",
}
REGEL_METHODE = {"emp": "emp", "direkt": "direkt", "gewichte": "kern"}   # was die Regel in dieser Demo live rechnet (Wald-Zweig: Kerngewichte)
REGEL_NAMEN = {"emp": "KQ + Quantil", "direkt": "direkt", "gewichte": "Gewichte"}
COLOR_EMP = "#4c78a8"
COLOR_DIREKT = "#2F6B65"
COLOR_GEWICHTE = "#e08a1e"
COLOR_REGEL = "#b0413e"
COLOR_BEST = "#555555"
COLOR_KNN = "#9a9a9a"
METHODE_FARBEN = {"emp": COLOR_EMP, "direkt": COLOR_DIREKT, "kern": COLOR_GEWICHTE, "knn": COLOR_KNN}

# --- Die Linie: neun gebaute Stücke, Quelle der Zahlen im Entscheidungsbaum -------------------------------------------------------------------
GITHUB_USER = "sebastian-hanisch"
STUECKE = {
    1: ("regret-statt-fehler-demo", "Regret statt Fehler"),
    2: ("newsvendor-direkt-demo", "Direkt auf den Kosten lernen"),
    3: ("spo-kuerzester-weg-demo", "SPO+ am kürzesten Weg"),
    4: ("zweistufig-oder-integriert-demo", "Zweistufig oder integriert"),
    5: ("gewichtete-stichprobe-demo", "Gewichtete Stichprobe"),
    6: ("nullgradient-glaettung-demo", "Warum der Gradient fehlt: Glättung"),
    7: ("gestoerte-optimierer-demo", "Gestörte Optimierer"),
    8: ("blackbox-loeser-demo", "Der Löser als Black Box"),
    9: ("entscheidungs-schranke-demo", "Eine Schranke für die Entscheidung"),
}


def stueck_url(nr: int) -> str:
    """Adresse der Demo des Stücks (Streamlit-Adresse nach dem Schema `sebastianhanisch-<repo>.streamlit.app`)."""
    return f"https://sebastianhanisch-{STUECKE[nr][0]}.streamlit.app/"


def stueck_github(nr: int) -> str:
    return f"https://github.com/{GITHUB_USER}/{STUECKE[nr][0]}"


# --- Fragen des Entscheidungsbaums (Experiment 1); Antworten als Auswahl, alle Regler stehen immer im Hauptbereich -------------------------------
Q_ART = ("menge", "loeser")
Q_ART_NAMEN = {"menge": "Eine Menge je Fall (Newsvendor, Quantil)", "loeser": "Eine Entscheidung mit Löser (Weg im Netz)"}
Q_FORM = ("ja", "nein")
Q_FORM_NAMEN = {"ja": "ja, der Mittelwert ist linear in den Merkmalen", "nein": "nein, er ist gekrümmt"}
Q_STREUUNG = ("nein", "monoton", "nicht_monoton")
Q_STREUUNG_NAMEN = {
    "nein": "nein, die Streuung ist konstant",
    "monoton": "ja, sie wächst oder fällt linear mit einem Merkmal",
    "nicht_monoton": "ja, aber nicht monoton (zum Beispiel mit |x₁|)",
}
Q_LINEAR = ("ja", "nein")
Q_LINEAR_NAMEN = {"ja": "ja, ein lineares Modell trifft die Kosten", "nein": "nein, die Kosten sind gekrümmt"}
Q_MODELL = ("linear", "flexibel")
Q_MODELL_NAMEN = {"linear": "linear (Vorgabe)", "flexibel": "flexibel (zum Beispiel Gauß-Kern)"}
Q_LOESER = ("spo", "glaettung", "stoerung", "blackbox")
Q_LOESER_NAMEN = {
    "spo": "Löser bekannt, Ersatzverlust möglich (SPO+)",
    "glaettung": "alle Wege aufzählbar, Gibbs-Glättung möglich",
    "stoerung": "nur Aufrufe, gestörte Lösung (Fenchel-Young)",
    "blackbox": "nur Aufrufe, Interpolation (Black Box)",
}
Q_RAUSCHEN = ("ohne", "mit")
Q_RAUSCHEN_NAMEN = {"ohne": "ohne Rauschen", "mit": "mit Rauschen"}
Q_NGITTER = (100, 1000)
Q_GARANTIE = ("nein", "ja")
Q_GARANTIE_NAMEN = {"nein": "nein", "ja": "ja, eine obere Schranke mit Niveau"}

DEFAULT_TREE = {"q_art": "menge", "q_form": "ja", "q_streuung": "monoton", "q_linear": "nein", "q_modell": "linear", "q_loeser": "spo",
                "q_rauschen": "mit", "q_ngitter": 1000, "q_garantie": "nein"}

# --- Presets ---------------------------------------------------------------------------------------------------------------------------------------
PRESETS = {
    "Streuung wächst mit x₁ (direkt)": {"welt": "streuung-linear", "tau": 0.95, "n": 1000, "p": 3, "seed": DEFAULT_SEED, **DEFAULT_TREE},
    "Konstante Streuung, linear (KQ + Quantil)": {"welt": "linear", "tau": 0.95, "n": 1000, "p": 3, "seed": DEFAULT_SEED, **DEFAULT_TREE, "q_streuung": "nein"},
    "Gekrümmte Wahrheit (Gewichte)": {"welt": "quadratisch", "tau": 0.95, "n": 1000, "p": 3, "seed": DEFAULT_SEED, **DEFAULT_TREE, "q_form": "nein"},
    "Blinder Fleck: Streuung in |x₁|": {"welt": "betrag", "tau": 0.95, "n": 1000, "p": 3, "seed": DEFAULT_SEED, **DEFAULT_TREE, "q_streuung": "nicht_monoton"},
    "Entscheidung mit Löser (SPO+)": {"welt": "streuung-linear", "tau": 0.95, "n": 1000, "p": 3, "seed": DEFAULT_SEED, **DEFAULT_TREE, "q_art": "loeser"},
}
PRESET_HELP = {
    "Streuung wächst mit x₁ (direkt)": "Newsvendor, Welt „Streuung wächst linear mit x₁“, p = 3, n = 1000, τ = 0,95: die Regel wählt direkt. Baum: Mittelwert linear, Streuung monoton.",
    "Konstante Streuung, linear (KQ + Quantil)": "Welt „Wahrheit linear, Streuung konstant“, p = 3, n = 1000, τ = 0,95: die Regel wählt KQ plus Quantil.",
    "Gekrümmte Wahrheit (Gewichte)": "Welt „Wahrheit quadratisch in x₁“, p = 3, n = 1000, τ = 0,95: die Regel wählt die Gewichte. Baum: Mittelwert gekrümmt.",
    "Blinder Fleck: Streuung in |x₁|": "Welt „Streuung hängt von |x₁| ab“: die Streuung hängt von x₁ ab, aber nicht monoton; die Diagnose schlägt nicht an, die Regel wählt KQ plus Quantil.",
    "Entscheidung mit Löser (SPO+)": "Der Baum im Zweig „Entscheidung mit Löser“: lineares Modell, Löser bekannt, mit Rauschen, 1000 Trainingsfälle. Die Newsvendor-Einstellungen bleiben wie in der Voreinstellung.",
}

# --- Texte, die App und README wörtlich teilen (ein Test vergleicht) ------------------------------------------------------------------------------
ASSUMPTIONS_MARKDOWN = r"""
| Annahme | Was passiert, wenn sie verletzt ist (Mehrkosten gegen das Orakel, Mittel über die Zeilen) | Ausweg (soweit gemessen) |
|---|---|---|
| **Die Welten der Messung ähneln den eigenen Daten** | Die Regel wurde an derselben Familie von vier Welten gemessen, aus der sie entworfen wurde (Normalrauschen, bei 0 gestutzt, p ∈ {3, 10}, n ∈ {100, 1000}). Die Wiederholung mit den Zufalls-Seeds 100 bis 129 prüft nur die Zufallsziehung: Regel 18,75 % gegen 33,28 / 35,12 / 34,67 % (immer KQ + Quantil / direkt / Wald), beste Wahl je Zeile 17,83 %. Andere Welten und echte Daten sind nicht gemessen. | Nicht gemessen; eigene Daten prüfen. |
| **Die Streuung hängt linear von den Merkmalen ab** | Bei Streuung in \|x₁\| schlägt die Streuungs-Diagnose in der Vormessung in 4 % der Zeilen an (Seeds 100 bis 129: 3 %), im Nachbau in 0,8 % (Seeds 100 bis 129: 5,8 %). Die Regel wählt in der Vormessung in 306 von 360 Zeilen KQ + Quantil und kostet 13,0 % gegen 11,3 % für die beste Wahl je Zeile. | Nicht behoben; ein Test auf nicht monotone Streuung ist nicht gebaut. |
| **Der Zweig „Gewichte“ rechnet mit einem Zufallswald** | Die Zahlen der Regel stammen aus der Vormessung mit scikit-learn und sind nicht live. Im Nachbau dieser Demo mit Gauß-Kerngewichten (Seeds 0 bis 29) kostet die Regel 23,2 % gegen 33,7 % für immer KQ + Quantil, statt 18,84 % gegen 33,45 % (Wald); bei p = 10 und n = 100 gewinnt sie dort nichts gegen immer KQ + Quantil (40,4 % gegen 40,4 %; Seeds 100 bis 129: 38,3 % gegen 38,5 %). | Ein Wald ist nicht Teil der Laufzeit; der Unterschied zwischen Wald und Kern steht in Stück 5. |
| **Die Schwellen 0,01 und 10·(p + 1) sind brauchbar** | Beide Schwellen wurden vor der Messung festgelegt und nicht variiert; wie die Regel auf andere Schwellen reagiert, ist nicht gemessen. | Nicht gemessen. |
| **Die Regel deckt den Löser-Zweig ab** | Die Regel wählt nur zwischen den drei Newsvendor-Verfahren. Der Zweig „Entscheidung mit Löser“ des Baums beruht auf den Bändern der Stücke 3, 4, 6, 7, 8 und 9 in ihren Welten (Gitter 5 × 5, 20 Seeds); eine Diagnose auf den Trainingsdaten gibt es dort nicht. | Keine Regel; die Antworten des Baums sind Annahmen des Nutzers. |
| **Das Orakel ist bekannt** | Mehrkosten gegen das Orakel lassen sich nur in Simulationen messen, in denen die wahre Verteilung bekannt ist. | Auf echten Daten bleibt der Vergleich auf Testdaten. |
"""

FORMULAS_MARKDOWN = r"""
**Lesehilfe.** Gegeben sind n Beobachtungen (xᵢ, yᵢ) mit Merkmalen xᵢ ∈ ℝᵖ und der Nachfrage yᵢ; gesucht ist die Wahl eines von drei Verfahren, die eine Bestellregel q(x) liefern. Die Wahl darf nur von den Trainingsdaten abhängen.

| Symbol | Bedeutung |
|---|---|
| $n$, $p$ | Zahl der Beobachtungen, Zahl der Merkmale |
| $\tau = b/(b+h)$ | kritisches Verhältnis: $b$ Kosten je fehlende, $h$ je überschüssige Einheit |
| $X$, $\tilde X$ | Designmatrix $[1, x]$ mit $p + 1$ Spalten, erweiterte Designmatrix $[1, x, x^2]$ (Quadrate elementweise) mit $2p + 1$ Spalten |
| $\mathrm{RSS}_0, \mathrm{RSS}_1$ | Summe der Fehlerquadrate des kleinen und des großen Modells |
| $F$, $p_{\mathrm{Krümmung}}$, $p_{\mathrm{Streuung}}$ | Teststatistik und p-Wert der beiden Tests |
| $\varepsilon_i$ | Betrag des Residuums der linearen Anpassung, $\lvert y_i - \tilde{x}_i^\top \hat\beta\rvert$ |

**Newsvendor-Kosten.** Mit der Bestellmenge $q$ und der Nachfrage $D$ kostet die Entscheidung $b\,(D-q)_+ + h\,(q-D)_+ = (b+h)\,\rho_\tau(D-q)$, wobei $\rho_\tau$ der Pinball-Verlust ist. Die beste Menge ist das $\tau$-Quantil der Verteilung von $D$ bei Merkmal $x$ (das Orakel).

**Test (a), Krümmung.** Kleines Modell $X$, großes Modell $\tilde X$, beide auf $y$:

$$F_a = \frac{(\mathrm{RSS}_0 - \mathrm{RSS}_1)/p}{\mathrm{RSS}_1/(n - 2p - 1)}, \qquad p_{\mathrm{Krümmung}} = \Pr\bigl(F_{p,\,n-2p-1} \ge F_a\bigr).$$

**Test (b), Streuung.** Die Residuen $\hat r_i$ der linearen Anpassung von $y$ auf $X$ liefern $\varepsilon_i = \lvert \hat r_i \rvert$. Kleines Modell: nur der Achsenabschnitt, großes Modell $X$, beide auf $\varepsilon$:

$$F_b = \frac{(\mathrm{RSS}_0 - \mathrm{RSS}_1)/p}{\mathrm{RSS}_1/(n - p - 1)}, \qquad p_{\mathrm{Streuung}} = \Pr\bigl(F_{p,\,n-p-1} \ge F_b\bigr).$$

**Regel.** Mit der Schwelle $0{,}01$:

$$\text{Wahl} = \begin{cases} \text{Gewichte} & p_{\mathrm{Krümmung}} < 0{,}01, \ \text{direkt} & p_{\mathrm{Krümmung}} \ge 0{,}01,\ p_{\mathrm{Streuung}} < 0{,}01,\ n \ge 10\,(p+1), \ \text{KQ plus Quantil} & \text{sonst.} \end{cases}$$

**Randfälle.** Ist ein Modell nicht definiert ($n - $ Spaltenzahl $\le 0$), gilt der p-Wert 1. Passt schon das kleine Modell bis auf Rundung, gilt ebenfalls 1.

**Mehrkosten.** Mit den erwarteten Kosten $K(\cdot)$ über einen festen Testsatz von Merkmalen: $100\,\bigl(K(\text{Regel}) - K(\text{Orakel})\bigr)/K(\text{Orakel})$ in Prozent.

**Literatur.** Die Tests sind die klassischen F-Tests geschachtelter linearer Modelle; die Regel ist eine Festlegung dieser Demo und keine Aussage aus der Literatur. Die Verfahren hinter den Blättern stehen mit ihren Quellen in den READMEs der neun Stücke.
"""
