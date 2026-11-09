# Verfahrenswahl: welches Verfahren, gemessen woran? – Streamlit-Demo

<!-- TODO nach dem Deploy: direkt hierher den Live-Link setzen (Leerzeile davor und danach):
**[→ Demo live ausprobieren](https://sebastianhanisch-entscheiden-lernen-wahl-demo.streamlit.app/)** -->

Schluss der Linie **Entscheidungsorientiertes Lernen** in der „Konzepte“-Reihe für die Website „Sebastian Hanisch – Operations Research und Machine Learning“. Die Linie hat neun Demos („Stücke“): [regret-statt-fehler-demo](https://github.com/sebastian-hanisch/regret-statt-fehler-demo) (1), [newsvendor-direkt-demo](https://github.com/sebastian-hanisch/newsvendor-direkt-demo) (2), [spo-kuerzester-weg-demo](https://github.com/sebastian-hanisch/spo-kuerzester-weg-demo) (3), [zweistufig-oder-integriert-demo](https://github.com/sebastian-hanisch/zweistufig-oder-integriert-demo) (4), [gewichtete-stichprobe-demo](https://github.com/sebastian-hanisch/gewichtete-stichprobe-demo) (5), [nullgradient-glaettung-demo](https://github.com/sebastian-hanisch/nullgradient-glaettung-demo) (6), [gestoerte-optimierer-demo](https://github.com/sebastian-hanisch/gestoerte-optimierer-demo) (7), [blackbox-loeser-demo](https://github.com/sebastian-hanisch/blackbox-loeser-demo) (8) und [entscheidungs-schranke-demo](https://github.com/sebastian-hanisch/entscheidungs-schranke-demo) (9). Diese Demo misst nichts Neues an den Verfahren der Stücke: Sie **ordnet** sie in einem Entscheidungsbaum, und sie prüft für den Newsvendor eine **Wahlregel nach zwei Diagnosen auf den Trainingsdaten**.

## Warum dieses Problem

Die Linie lehrt nicht „entscheidungsorientiert ist besser“, sondern **wann welcher Weg gewinnt**: SPO+ nur bei gekrümmter Wahrheit und linearem Modell, direktes Lernen der Bestellmenge nur bei von den Merkmalen abhängiger Streuung und genug Daten, Gewichte aus Ähnlichkeit vor allem bei gekrümmter Form (im Nachbau dieser Demo gewinnen sie auch in einer Zelle mit nicht monotoner Streuung, p = 3, n = 1000, τ = 0,95: Kerngewichte 14,4 % gegen 19,6 % für KQ + Quantil und 20,2 % für direkt), Black-Box-Interpolation nur mit sauberen Zielen. **Kernfrage:** Lässt sich das Verfahren aus den Trainingsdaten wählen, statt es aus Kenntnis der Welt zu kennen? Antwort der Messung (Vormessung mit Zufallswald, Welten des Newsvendors): fast, und nicht überall. Die Regel liegt im Mittel bei 18,84 % Mehrkosten gegen 33,45 % für das beste feste Verfahren (nach dem Mittel) und 17,54 % für die beste Wahl mit Kenntnis der Welt; in drei von vier Welten hat aber ein festes Verfahren kleinere Mehrkosten als die Regel, und die Streuungs-Diagnose übersieht eine nicht monotone Streuung (blinder Fleck).

## Modell

**Welten und Kosten** (kopiert aus `newsvendor-direkt-demo`, Herkunft im Modulkopf von `elw_scenario.py`): Nachfrage D = max(0, Y) mit Y ~ Normal(μ(x), σ(x)²), Merkmale x ~ Normal(0, I), h = 1 und b = τ / (1 − τ). Vier Welten: Wahrheit linear mit konstanter Streuung (μ = 60 + 10·(x·β), σ = 8), Streuung wächst linear mit x₁ (σ = max(1, 6 + 4·x₁)), Streuung hängt von |x₁| ab (σ = 4 + 6·|x₁|, nicht monoton) und Wahrheit quadratisch in x₁ (μ + 8·(x₁² − 1), σ = 8). β ist ein zufälliger Einheitsvektor je Zufalls-Seed. Das Orakel ist das wahre τ-Quantil max(0, μ + σ·z_τ). Kosten werden **exakt** als Erwartung über die Nachfrage und über 2000 feste Testpunkte gerechnet; **Mehrkosten** sind der Aufschlag gegen das Orakel in Prozent der Orakelkosten. In der Vormessung (`vormessung/eo_v11_wahlregel.py`) heißen die Welten `linear`, `streuung-linear`, `heteroskedastisch` (hier „|x₁|“) und `nichtlinear` (hier „quadratisch“).

**Die drei Verfahren** (kopiert und gekürzt aus `newsvendor-direkt-demo` und `gewichtete-stichprobe-demo`): *KQ plus Quantil* (Kleinste Quadrate, dazu das Residuum der Rangzahl ⌈τ·n⌉ als gemeinsamer Aufschlag), *direkt* (lineare Quantilregression, ein lineares Programm mit `scipy.optimize.linprog`, HiGHS) und *Gewichte aus Ähnlichkeit* (gewichtetes τ-Quantil der Nachfragen; die Vormessung benutzt einen Zufallswald mit 50 Bäumen, **diese Demo rechnet live Gauß-Kerngewichte** mit per Leave-one-out gewählter Bandbreite und zusätzlich kNN-Gewichte). Ein Zufallswald ist nicht Teil der Laufzeit (kein scikit-learn).

**Die Diagnose** (neu geschrieben nach der Messlogik der Vormessung, `elw_diagnose.py`), nur aus den Trainingsdaten (X, y):

* **Krümmung:** F-Test, ob die Quadrate der Merkmale die lineare Anpassung von y verbessern ([1, X] gegen [1, X, X²]); p-Wert mit `scipy.stats.f.sf`.
* **Streuung:** F-Test, ob der Betrag des Residuums der linearen Anpassung linear von den Merkmalen abhängt ([1] gegen [1, X]). Erkannt wird praktisch nur eine Streuung mit **linearer** Komponente in den Merkmalen; die symmetrisch nicht monotone Streuung in \|x₁\| bleibt unerkannt (blinder Fleck).

**Die Regel** (vor der Messung festgelegt, Schwellen nicht abgestimmt): p-Wert der Krümmung unter 0,01 → **Gewichte**; sonst p-Wert der Streuung unter 0,01 und n ≥ 10·(p + 1) → **direkt**; sonst **KQ plus Quantil**. Randfälle: ist ein Modell nicht definiert (Freiheitsgrade im Nenner höchstens 0), gilt der p-Wert 1; passt schon das kleine Modell bis auf Rundung, ebenfalls 1; passt nur das große Modell exakt, 0 (die Vormessung gab bei Restquadratsumme 0 immer 1; bei Daten mit Rauschen tritt der Fall nicht auf). Formeln und Symboltabelle: Expander „📐 Mathematische Formulierung“ der App.

**Der Entscheidungsbaum** (`elw_tree.py`, Diagramm `entscheidungsbaum.dot`): elf Fragen, sechzehn Blätter. Im Zweig „Menge je Fall“ steht die Frage nach der Form des Mittelwerts vor der Frage nach schiefen Kosten, denn auch bei τ = 0,5 genügt das Mittel bei gekrümmter Wahrheit nicht (Befunde). Die Fragen sind Regler (Experiment 1), zwei Antworten liest der Baum aus der Seitenleiste ab (kritisches Verhältnis τ und die Datenmenge gegen 10·(p + 1)). Jedes Blatt nennt ein Verfahren, das Stück, in dem es gemessen ist, und einen **Messwert, der wörtlich in der README dieses Stücks steht** (ein Test liest die README des Schwesterordners und bricht, wenn sich ein Stück ändert). Jeder Knoten des Diagramms verlinkt auf die Demo des Stücks (`URL` am Knoten; Graphviz `dot` hat die Geometrie erzeugt).

## Methodik

Vier Experimente in der App; Schnellstart und allgemeine Einstellungen (Welt, τ, n, p, Zufalls-Seed) stehen in der Seitenleiste, die Antworten des Baums beim Experiment 1; nicht benötigte Fragen sind ausgegraut:

1. **Der Entscheidungsbaum:** Fragen beantworten, das Diagramm hebt den Weg hervor, das Blatt zeigt Verfahren, Messwert und Link auf das Stück; zusätzlich die Frage nach einer Garantie (Stück 9), die ein zweites Blatt anfügt.
2. **Die Diagnose auf eigenen Daten:** Daten ziehen, beide F-Tests rechnen, die Regel wählen lassen, alle Verfahren live gegen das Orakel bewerten; dazu ein Band über fünf feste Instanzen (Zufalls-Seeds 100000 bis 100004). Der Zweig „Gewichte“ rechnet hier mit Gauß-Kerngewichten, nicht mit dem Wald.
3. **Die Regel gegen immer dasselbe Verfahren:** die Tabelle der Vormessung (Zufallswald, **eingefroren, nicht live**) neben dem **Nachbau mit dem Code dieser Demo** (vorgerechnet mit `tools/messung.py`, Gauß-Kern statt Wald), je Satz Zufalls-Seeds 0 bis 29 und 100 bis 129, über alle Zeilen, je Welt und je Kombination aus n und p.
4. **Der blinde Fleck:** der Betrag der Residuen über x₁ bei linear wachsender Streuung und bei Streuung in |x₁|, dazu die Zählung der Tests über 20 feste Instanzen (Zufalls-Seeds 200000 bis 200019). Gezeigt, nicht behoben.

**Zeilen der Messung:** 4 Welten × p ∈ {3, 10} × n ∈ {100, 1000} × τ ∈ {0,5; 0,8; 0,95} × 30 Zufalls-Seeds = 1440 Zeilen je Satz. „Mittlere Mehrkosten“ ist das Mittel der Mehrkosten über die Zeilen. „Beste Wahl je Zeile“ ist das Verfahren mit den kleinsten Mehrkosten in dieser Zeile, im Nachhinein und mit Kenntnis der Welt; sie ist keine ausführbare Regel, sondern eine Untergrenze.

**Zufall** kommt nur aus SplitMix64 (`elw_rng.py`, Kopie aus `newsvendor-direkt-demo`), nie aus Pythons `hash()`.

## Befunde (gemessen, keine Behauptungen)

Alle Zahlen sind Mehrkosten gegen das Orakel in Prozent, Mittel über die 1440 Zeilen eines Satzes (4 Welten, p ∈ {3, 10}, n ∈ {100, 1000}, τ ∈ {0,5; 0,8; 0,95}, 30 Zufalls-Seeds), wenn nichts anderes dasteht. **Vormessung** heißt: Zufallswald (scikit-learn, 50 Bäume), numpy-Zufallsstrom, Rohausgaben `vormessung/out_eo_v11_wahlregel.txt` (Zufalls-Seeds 0 bis 29) und `vormessung/out_eo_v11_wahlregel_seeds100.txt` (100 bis 129), eingefroren in `elw_vormessung.json` (erzeugt von `tools/freeze_vormessung.py`), nicht live. **Nachbau** heißt: der Code dieser Demo mit Gauß-Kerngewichten statt des Waldes, vorgerechnet in `elw_bands.json` (`tools/messung.py`). Die Welten sind die des Newsvendors; die Aussagen gelten nicht für echte Daten.

| Frage | Befund (Voraussetzung im selben Satz) | Test |
|---|---|---|
| Regel gegen feste Verfahren, Vormessung, Zufalls-Seeds 0 bis 29 | Regel **18,84** (Median 9,60) gegen immer KQ + Quantil 33,68 (13,06), immer direkt 35,21 (9,77), immer Wald-Gewichte 33,45 (28,50); beste Wahl je Zeile 17,54 (7,87). | `test_vormessung_totals_seeds_0_to_29` (exakt, Datei) |
| Dasselbe mit unabhängigen Zufalls-Seeds 100 bis 129, Vormessung | Regel 18,75 gegen 33,28 / 35,12 / 34,67; beste Wahl 17,83. Die Wiederholung prüft nur die Zufallsziehung, nicht die Welten. | `test_vormessung_totals_independent_seeds_100_to_129` |
| „Halbiert“ die Regel die Mehrkosten? | **Nein, 44 % weniger im Mittel über die vier Welten:** 18,84 gegen 33,45 (bestes festes Verfahren nach dem Mittel: immer Wald) beziehungsweise 18,75 gegen 33,28 (immer KQ + Quantil), das 0,56-fache. Die Ersparnis entsteht in der Welt quadratisch (47,4 gegen 99,8 / 108,8 für immer KQ + Quantil / immer direkt); in den drei übrigen Welten hat ein festes Verfahren kleinere Mehrkosten, und im Median liegt die Regel bei 9,60 gegen 9,77 für immer direkt. | `test_the_rule_saves_44_percent_not_half_of_the_extra_cost_of_the_best_fixed_method` |
| Je Welt, Vormessung, Seeds 0 bis 29 (Regel / immer KQ + Quantil / immer direkt / immer Wald / beste Wahl) | konstante Streuung 3,5 / 3,4 / 6,8 / 22,5 / 3,2; Streuung linear 11,5 / 19,6 / 9,0 / 39,5 / 8,4; Streuung in \|x₁\| 13,0 / 11,9 / 16,2 / 24,4 / 11,3; quadratisch 47,4 / 99,8 / 108,8 / 47,4 / 47,3. In der quadratischen Welt wählt die Regel in 360 von 360 Zeilen die Gewichte, bei Streuung in \|x₁\| in 306 von 360 Zeilen KQ + Quantil. | `test_vormessung_by_world_and_by_n_and_p_seeds_0_to_29` |
| Wo liegt die Regel unter jedem festen Verfahren? | In beiden Zufallssätzen der Vormessung in **4 von 4** Kombinationen aus n und p, aber in **keiner der vier** Welten unter jedem festen Verfahren (in der Welt quadratisch ist sie gleich dem Wald: 47,4 gegen 47,4 in Seeds 0 bis 29, 48,3 gegen 48,3 in Seeds 100 bis 129); in den drei übrigen Welten hat je ein festes Verfahren kleinere Mehrkosten als die Regel (KQ + Quantil bei konstanter Streuung und bei Streuung in \|x₁\|, direkt bei linear wachsender Streuung). | `test_the_rule_is_below_every_fixed_method_in_all_four_n_p_cells_but_in_one_world_of_four` |
| Je n und p, Vormessung, Seeds 0 bis 29 (Regel / beste Wahl) | n = 100, p = 3: 23,8 / 21,9; n = 100, p = 10: 31,5 / 29,4; n = 1000, p = 3: 9,2 / 8,2; n = 1000, p = 10: 10,8 / 10,7. | `test_vormessung_by_world_and_by_n_and_p_seeds_0_to_29` |
| Trägt die Diagnose? Vormessung | Krümmung erkannt in 100 % der Zeilen der quadratischen Welt, Fehlalarm in 6 % der übrigen (Seeds 100 bis 129: 2 %); lineare Streuung erkannt in 98 % (97 %); **blinder Fleck:** Streuung in \|x₁\| erkannt in 4 % (3 %). | `test_vormessung_diagnosis_rates` |
| Regel gegen feste Verfahren, Nachbau mit Kerngewichten, Seeds 0 bis 29 | Regel **23,19** gegen immer KQ + Quantil 33,70, immer direkt 35,68, immer Kerngewichte 38,43; beste Wahl 21,52. Das sind 31 % weniger als das beste feste Verfahren (Seeds 100 bis 129: 22,34 gegen 32,86, 32 % weniger; beste Wahl 20,63). | `test_rebuilt_totals_with_kernel_weights_in_place_of_the_forest` |
| Wo gewinnt der Nachbau, wo nicht? | In **3 von 4** Kombinationen aus n und p liegt die Regel mehr als 1 Punkt unter dem besten festen Verfahren; bei n = 100, p = 10 gewinnt sie nichts (Seeds 0 bis 29: 40,45 gegen 40,36 für immer KQ + Quantil; Seeds 100 bis 129: 38,34 gegen 38,47). In der quadratischen Welt wählt sie in 360 von 360 Zeilen die Gewichte. | `test_the_rebuilt_rule_is_below_the_best_fixed_method_in_three_of_four_n_p_cells_and_equal_in_the_fourth`, `test_rebuilt_rule_always_picks_the_weights_in_the_quadratic_world` |
| Diagnose im Nachbau | Krümmung erkannt in 100 %, Fehlalarm 3,9 % (Seeds 100 bis 129: 4,4 %); lineare Streuung erkannt in 97,5 % (97,5 %); Streuung in \|x₁\| in 0,8 % (**5,8 %**); konstante Streuung 0,0 % (1,7 %). Die Quote für den blinden Fleck ist ein seltenes Ereignis und streut zwischen den zwei Sätzen. | `test_rebuilt_diagnosis_rates` |
| Blinder Fleck, live gezählt (20 feste Instanzen, Experiment 4) | n = 1000, p = 3: Streuungs-Test schlägt bei linear wachsender Streuung in 20 von 20 Instanzen an, bei Streuung in \|x₁\| in 0 von 20. n = 100, p = 10: 16 von 20 und 3 von 20; dort wählt die Regel auch bei linear wachsender Streuung in 0 von 20 Instanzen direkt (n = 100 < 110 = 10·(p + 1)). | `test_live_detection_counts_of_experiment_4` |
| Baum bei τ = 0,5: genügt das Mittel auch bei gekrümmter Wahrheit? | **Nein.** Nachbau, p = 3, n = 1000, τ = 0,5, Mittel über 30 Zufalls-Seeds (0 bis 29): Welt quadratisch immer KQ + Quantil 56,6 %, immer Kerngewichte 15,2 %; Welt linear 0,2 % und 5,4 %. Seeds 100 bis 129: 55,3 und 14,7 % (quadratisch), 0,2 und 5,3 % (linear). Das reine Mittel ohne Residuenquantil ist nicht gerechnet. | `test_tree_asks_the_form_before_the_cost_question_at_tau_half` |
| Zeilen des Nachbaus | Gespeicherte Zeilen stimmen mit einer Neurechnung überein (Band: 0,01 Prozentpunkte). | `test_stored_rows_match_a_recomputation` (Band) |
| Zahlen im Baum | Jeder Messwert eines Blattes steht wörtlich in der README seines Stücks. | `tests/test_tree.py` (Verweis-Test, überspringt sich ohne die Schwesterordner) |

Zusätze, **nicht im Test**: Vergleiche von Zahlen zwischen Stücken (die Messwerte der Blätter stammen aus verschiedenen Welten und Maßen und sind nicht untereinander vergleichbar).

### Befunde und Korrekturen gegenüber dem Plan

* **„Die Regel halbiert die Mehrkosten“ trägt nicht.** Der Plan nannte es als Kernbotschaft; gemessen ist es das 0,56-fache, also 44 % weniger mittlere Mehrkosten über die vier Welten als das beste feste Verfahren (Vormessung); im Median ist der Vorsprung klein (9,60 gegen 9,77 für immer direkt). Die Formulierung „halbiert“ steht in App und README nur noch als verneinte Frage, nicht als Behauptung.
* **Logikfehler im Baum behoben: bei τ = 0,5 fehlte die Frage nach der Form.** Die erste Fassung führte „symmetrische Kosten“ ohne weitere Frage zu „Mittel bestellen genügt“. Im Nachbau kostet KQ + Quantil bei τ = 0,5 in der Welt quadratisch aber 56,6 % (p = 3, n = 1000, Mittel über die Seeds 0 bis 29; Kerngewichte 15,2 %), in der Welt linear 0,2 %; die Regel dieser Demo wählt bei Krümmung unabhängig von τ die Gewichte. Der Baum fragt deshalb zuerst die Form und dann die Kosten. Das reine Mittel ohne Residuenquantil ist nicht gerechnet.
* **Der Wald ist nicht live, und mit Kerngewichten fällt der Gewinn kleiner aus.** Der Nachbau mit Gauß-Kern im Zweig „Gewichte“ spart 31 statt 44 %. Ursache in den Zahlen: bei p = 10 sind Kerngewichte viel schlechter als der Wald (Stück 5: bei quadratischer Wahrheit, p = 10, n = 1000, τ = 0,95 kostet der Wald 32,9 %, der Kern 80,8 %); in der quadratischen Welt liegt die Regel im Nachbau bei 64,8 gegen 47,4 % mit Wald. Dass der Wald der Grund ist, ist aus diesen Zahlen abgelesen, nicht durch einen Wald im Nachbau nachgemessen.
* **Die Diagnose-Quoten des Nachbaus weichen von der Vormessung ab** (anderer Zufallsstrom): Fehlalarm 3,9 gegen 6 %, Streuung in \|x₁\| 0,8 gegen 4 %; in den Zufalls-Seeds 100 bis 129 ist die Quote für den blinden Fleck 5,8 gegen 3 %. Die Aussage „der Test schlägt bei nicht monotoner Streuung kaum an“ gilt in allen vier Sätzen (höchstens 5,8 %).
* **Stücke, die vom Plan abweichen, geben ihre Zahlen:** SPO+ bei n = 100 mit Rauschen ist bei Grad 6 in 10 von 20 Zufalls-Seeds besser als KQ (Plan: 2 von 20), in der Stabilitätsprobe mit 100 weiteren Seeds in 29 von 100; Black-Box-Interpolation ist mit sauberen Zielen brauchbar und mit verrauschten in allen 20 Zufalls-Seeds schlechter als KQ; die gestörte Lösung mit σ = 0,1 liegt im Median gleich KQ. Der Baum nennt jeweils die Zahl der gebauten Demo.
* **Flexible Klasse schlägt SPO+:** Stück 4 misst Kern-ETO in 6 von 6 Zellen unter SPO+ (Plan: 5 von 6). Der Baum schickt deshalb „flexibles Modell zugelassen“ zu Stück 4, nicht zu SPO+.
* **Die Brücken des Plans** (inverse Optimierung, Off-Policy-Evaluation aus Logdaten, Wert der Information) stehen nicht im Baum, nur unter „Bewusst nicht umgesetzt“.
* **Quantil der Residuen:** die Vormessung benutzt das linear interpolierte Quantil (`numpy.quantile`), diese Demo die Rangregel ⌈τ·n⌉ aus `newsvendor-direkt-demo`; Kosten exakt statt an gezogenen Nachfragen. Die Ziffern stimmen deshalb nicht mit der Vormessung überein; die Rangfolge der Verfahren in den Mittelwerten ist dieselbe.
* **Name:** der Platzhaltername `entscheiden-lernen-wahl-demo` aus dem Plan bleibt (kein `qa`, `prod`, `hub`, `sso`, `ass`).
* **Footer:** die Linie hat noch keine Website-Spezifikation; der Footer verlinkt vorläufig `demos.html` und ist vor der Website-Einbindung auf die Linie umzustellen (`tools/set_demo_footer.py`).

## Ehrliche Grenzen

* **Simulierte Daten, eine Familie von Welten:** Normalrauschen, bei 0 gestutzt, vier Welten, p ∈ {3, 10}, n ∈ {100, 1000}, τ ∈ {0,5; 0,8; 0,95}. Keine Echtdaten.
* **Die Regel ist an der Familie gemessen, aus der sie entworfen wurde.** Die Wiederholung mit anderen Zufalls-Seeds prüft nur die Zufallsziehung, nicht die Welten. Die Schwellen (0,01 und 10·(p + 1)) wurden nicht variiert.
* **Blinder Fleck:** nicht monotone Streuung wird nicht erkannt; die Regel wählt dort meist KQ + Quantil und liegt bei 13,0 gegen 11,3 % (Vormessung). Er wird gezeigt, nicht behoben.
* **Wald nur eingefroren.** Die Zahlen mit Zufallswald sind eine Vormessung mit scikit-learn (50 Bäume, mindestens 5 je Blatt, nicht abgestimmt) und sagen nichts über Zufallswälder im Allgemeinen. Die Live-Rechnung benutzt Kerngewichte.
* **Der Löser-Zweig des Baums beruht nicht auf einer Diagnose,** sondern auf den Bändern der Stücke 3, 4, 6, 7, 8 und 9 (Gitter 5 × 5, 20 Zufalls-Seeds, eigene Welten). Die Antworten des Baums sind Annahmen des Nutzers.
* **Messwerte der Blätter sind nicht untereinander vergleichbar:** andere Welten, andere Maße (Mehrkosten, Regret), andere Zahl der Zufalls-Seeds. Wo ein Stück „Quellen nur auf Abstract-Ebene gelesen“ vermerkt, gilt das dort; diese Demo zitiert keine Literatur, die F-Tests sind die klassischen Tests geschachtelter linearer Modelle.
* **Der Verweis-Test braucht die Schwesterordner** neben diesem Repo; in der CI und in der Cloud überspringt er sich.
* **Rechenzeit:** Experiment 2 rechnet je Instanz Diagnose, ein lineares Programm und die Gewichte; die Dauer hängt vom Rechner ab und ist nicht festgeschrieben; Ergebnisse werden je Einstellung zwischengespeichert.
* **Nicht gemessen:** andere Schwellen, ein Zufallswald im Nachbau, die Regel auf Echtdaten, eine Diagnose für den Gitter-Zweig.

### Wo die Annahmen enden

(Dieselbe Tabelle wie in der App; ein Test vergleicht beide.)

| Annahme | Was passiert, wenn sie verletzt ist (Mehrkosten gegen das Orakel, Mittel über die Zeilen) | Ausweg (soweit gemessen) |
|---|---|---|
| **Die Welten der Messung ähneln den eigenen Daten** | Die Regel wurde an derselben Familie von vier Welten gemessen, aus der sie entworfen wurde (Normalrauschen, bei 0 gestutzt, p ∈ {3, 10}, n ∈ {100, 1000}). Die Wiederholung mit den Zufalls-Seeds 100 bis 129 prüft nur die Zufallsziehung: Regel 18,75 % gegen 33,28 / 35,12 / 34,67 % (immer KQ + Quantil / direkt / Wald), beste Wahl je Zeile 17,83 %. Andere Welten und echte Daten sind nicht gemessen. | Nicht gemessen; eigene Daten prüfen. |
| **Die Streuung hängt linear von den Merkmalen ab** | Bei Streuung in \|x₁\| schlägt die Streuungs-Diagnose in der Vormessung in 4 % der Zeilen an (Seeds 100 bis 129: 3 %), im Nachbau in 0,8 % (Seeds 100 bis 129: 5,8 %). Die Regel wählt in der Vormessung in 306 von 360 Zeilen KQ + Quantil und kostet 13,0 % gegen 11,3 % für die beste Wahl je Zeile. | Nicht behoben; ein Test auf nicht monotone Streuung ist nicht gebaut. |
| **Der Zweig „Gewichte“ rechnet mit einem Zufallswald** | Die Zahlen der Regel stammen aus der Vormessung mit scikit-learn und sind nicht live. Im Nachbau dieser Demo mit Gauß-Kerngewichten (Seeds 0 bis 29) kostet die Regel 23,2 % gegen 33,7 % für immer KQ + Quantil, statt 18,84 % gegen 33,45 % (Wald); bei p = 10 und n = 100 gewinnt sie dort nichts gegen immer KQ + Quantil (40,4 % gegen 40,4 %; Seeds 100 bis 129: 38,3 % gegen 38,5 %). | Ein Wald ist nicht Teil der Laufzeit; der Unterschied zwischen Wald und Kern steht in Stück 5. |
| **Die Schwellen 0,01 und 10·(p + 1) sind brauchbar** | Beide Schwellen wurden vor der Messung festgelegt und nicht variiert; wie die Regel auf andere Schwellen reagiert, ist nicht gemessen. | Nicht gemessen. |
| **Die Regel deckt den Löser-Zweig ab** | Die Regel wählt nur zwischen den drei Newsvendor-Verfahren. Der Zweig „Entscheidung mit Löser“ des Baums beruht auf den Bändern der Stücke 3, 4, 6, 7, 8 und 9 in ihren Welten (Gitter 5 × 5, 20 Seeds); eine Diagnose auf den Trainingsdaten gibt es dort nicht. | Keine Regel; die Antworten des Baums sind Annahmen des Nutzers. |
| **Das Orakel ist bekannt** | Mehrkosten gegen das Orakel lassen sich nur in Simulationen messen, in denen die wahre Verteilung bekannt ist. | Auf echten Daten bleibt der Vergleich auf Testdaten. |

## Verifikation

190 Tests (`pytest --collect-only -q tests`), alle grün.

* **Kern gegen Handrechnung:** Diagnose am Handbeispiel (n = 5, x = −2 … 2, y = 4, 1, 0, 1, 5: F = 320, p = 1 − √(320/322) in geschlossener Form), exakt lineare Daten, die Schwelle 10·(p + 1) mit 19, 20 und 21 Fällen, Entscheidungstabelle von Hand; Newsvendor-Handbeispiel (KQ 5 + 7x, KQ + Quantil 6 + 7x, direkt 6 + 10x, Pinball-Summe 3,0) und Kerngewichte des Handbeispiels (0,0359 / 0,1963 / 0,3953 / 0,2928 / 0,0798, Quantil 30) (`tests/test_diagnose.py`, `tests/test_algorithm.py`).
* **Orakel-Tests** (`tests/test_oracle_entscheiden_lernen_wahl.py`, anderer Rechenweg): F-Test der Krümmung gegen **exakte Bruchrechnung** mit p-Wert über die t-Verteilung, F-Test der Streuung gegen `scipy.stats.linregress`, beide Tests für p = 3 gegen **Projektionsmatrizen**; Wahlregel gegen die Aufzählung aller Fälle auf einem Raster; Quantilregression gegen **Aufzählung der Ecken** und gegen `sklearn.linear_model.QuantileRegressor` (nur Test); gewichtetes Quantil gegen **exakte Brüche**, Kerngewichte gegen eine `math.fsum`-Schleife; erwartete Kosten gegen **Monte Carlo**, wahres Quantil, Optimalität des Orakels, SplitMix64 gegen eine ganzzahlige Referenz. Im Scratch-Lauf (nicht im Repo) liefen je 300 Instanzen gegen die Brüche, `linregress` und die Projektionen.
* **Eigenschaften:** die Wahl hängt nur von den Trainingsdaten ab; Skalieren und Verschieben der Nachfragen und Vertauschen der Beobachtungen ändern p-Werte und Wahl nicht; der Zweig „Gewichte“ nur bei p-Wert unter 0,01; jede der drei Wahlen kommt in den Welten vor (kein toter Zweig).
* **Randfälle** (`tests/test_edge_cases.py`, `tests/test_diagnose.py`): n = 1 und n = 2, p ≥ n, zu wenige Freiheitsgrade, konstante Nachfrage, alle Residuen null, perfekter quadratischer Fit, Gleichstände, ungültige Eingaben.
* **Baum** (`tests/test_tree.py`): jede Antwort führt zu einer Frage oder einem Blatt, alle 16 Blätter sind erreichbar, Wege nach Handtabelle, Presets landen auf dem genannten Blatt, die DOT-Datei ist die erzeugte, jeder Knoten hat `URL` und `tooltip`, Graphviz `dot` rendert ein Diagramm mit einem Link je Knoten, der Verweis-Test gegen die neun READMEs.
* **Verdikte** (`tests/test_verdicts.py`): jede Schwelle und jede Zahl im Text an handgesetzten Werten.
* **Zahlen der README:** `tests/test_claims.py` (Tabelle der Befunde); die Tabelle „Wo die Annahmen enden“ steht in App und README wortgleich (ein Test vergleicht).
* **Oberfläche und Layout:** AppTest-Rauchtest (jeder Button, jeder Regler auf allen Stufen, Permalink mit Einrasten, ungültige Werte), Presets, aktives Preset hervorgehoben, jeder Regler steht in `SETTING_SPECS` (`tests/test_layout.py`), Footer-Test, Laufzeit-Importe (`tests/test_runtime_imports.py`).
* **Nicht geprüft:** das Aussehen im Browser bei hellem und dunklem Design mit echtem Mausklick auf die Links des Diagramms; die Links führen auf die Adressen der Demos nach dem Schema `sebastianhanisch-<repo>.streamlit.app`, die erst nach dem Deploy der Stücke bestehen.

## Dateistruktur

| Datei | Zweck |
|---|---|
| `.github/workflows/tests.yml` | CI: Push, Pull Request, wöchentlicher Canary |
| `.gitignore` | venv, Caches, Streamlit-Secrets |
| `.streamlit/config.toml` | Streamlit-Einstellungen (keine Nutzungsstatistik) |
| `app.py` | Streamlit-App: Baum, Diagnose, Regel gegen feste Verfahren, blinder Fleck, Grenzen, Formeln |
| `elw_algorithm.py` | Kern: KQ, Rangregel des Quantils, Quantilregression (LP, HiGHS); gekürzte Kopie aus `newsvendor-direkt-demo` |
| `elw_constants.py` | Welten, Verfahren, Stücke der Linie, Fragen des Baums, Presets, Texte für App und README |
| `elw_diagnose.py` | die zwei F-Tests und die Wahlregel |
| `elw_evaluation.py` | Analyse einer Instanz, Live-Band, Zählung der Tests, Verdikte |
| `elw_layout.py` | Fester Kopf (Stylesheet) |
| `elw_presets.py` | Permalink (Einrasten und Begrenzen), Presets (Schnellstart in der Seitenleiste), Zufalls-Seed-Button |
| `elw_rng.py` | SplitMix64 (Kopie aus `newsvendor-direkt-demo`) |
| `elw_scenario.py` | Welten, Daten, wahres Quantil, exakte erwartete Kosten (Kopie aus `newsvendor-direkt-demo`) |
| `elw_tree.py` | Fragen und Blätter des Baums, Weg, DOT-Quelle |
| `elw_visualization.py` | Plotly-Abbildungen (Achsen gesperrt) |
| `elw_vormessung.py` | lädt die eingefrorene Vormessung und den Nachbau, Aggregate |
| `elw_vormessung.json` | eingefrorene Zahlen der Vormessung V11 (Zufallswald, nicht live) |
| `elw_bands.json` | vorgerechneter Nachbau: 2 × 1440 Zeilen |
| `elw_weights.py` | Gewichte aus Ähnlichkeit, gewichtetes Quantil, Leave-one-out (gekürzte Kopie aus `gewichtete-stichprobe-demo`) |
| `entscheidungsbaum.dot` | Entscheidungsbaum als Graphviz-Quelle, jeder Knoten mit `URL` |
| `requirements.txt` | Laufzeit-Bibliotheken (ungepinnt): Streamlit, NumPy, SciPy, pandas, Plotly |
| `requirements-dev.txt` | Laufzeit plus pytest und scikit-learn (nur Gegenprobe) |
| `tools/freeze_vormessung.py` | erzeugt `elw_vormessung.json` aus den Rohausgaben (einmal gelaufen) |
| `tools/messung.py` | rechnet den Nachbau und schreibt `elw_bands.json` |
| `tests/` | Handrechnung, Orakel, Eigenschaften, Randfälle, Baum, Verdikte, Zahlen der README, Presets, Layout, AppTest-Smoke-Test, Footer, Laufzeit-Importe |

## Bewusst nicht umgesetzt

* **Zufallswald live** (kein scikit-learn zur Laufzeit; ein numpy-Wald ist nicht Teil der Demo).
* **Neue Verfahren:** die Demo wählt zwischen den Verfahren der Stücke 1 bis 9, sie führt keines ein.
* **Abstimmen der Schwellen** der Regel und **Beheben des blinden Flecks** (ein Test auf nicht monotone Streuung).
* **Eine Diagnose für den Gitter-Zweig** (Weg im Netz): dort gelten die Bänder der Stücke 3, 4, 6, 7, 8 und 9.
* **Brücken zu verwandten Problemen:** inverse Optimierung („Ziel lernen“ lernt das Gewicht der Ziele aus Entscheidungen, nicht die Kosten), Off-Policy-Evaluation aus Logdaten (wertet fremde Entscheidungsregeln aus Logdaten aus) und [newsvendor-evpi-demo](https://github.com/sebastian-hanisch/newsvendor-evpi-demo) (Wert der Information bei bekannter Verteilung).
* **Echtdaten**, **PDF-Export**, feste Rechenzeiten in App oder README.

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Gebaut mit Streamlit, Plotly, NumPy, SciPy und pandas.

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html).
