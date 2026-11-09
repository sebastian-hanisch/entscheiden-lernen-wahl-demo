"""Der Entscheidungsbaum der Linie: Fragen, Blätter, Weg durch den Baum und die DOT-Quelle des Diagramms (`entscheidungsbaum.dot`).

Jedes Blatt nennt ein Verfahren, das Stück der Linie, in dem es gemessen ist, und einen Messwert. Der Messwert ist wörtlich aus der README des Stücks übernommen:
`marker` sind Zeichenfolgen, die dort stehen müssen (`tests/test_tree.py` liest die README des Schwesterordners und bricht, wenn sich ein Stück ändert),
`weitere` Marker aus der README eines anderen Stücks (Zahl aus einem anderen Stück im Text), `zahlen` die in `text` genannten Zahlen, jede muss in einem Marker vorkommen. Fragen und Antworten stehen als Daten in `QUESTIONS`; die Antworten kommen aus den
Reglern der App (`answers_from`). Nichts im Baum ist neu gemessen: er ordnet die Messungen der neun Stücke.

DOT-Regeln (DEMO-PLAYBOOK, WEBSITE-PLAYBOOK §4): jeder Knoten hat `URL` und `TOOLTIP`, keine `&` in Titeln, Umlaute direkt."""

from dataclasses import dataclass

import elw_constants as C

# Die Frage-Knoten: id -> (Fragetext, {Antwort: nächster Knoten oder Blatt})
QUESTIONS = {
    "art": ("Was wird entschieden?", {"menge": "form", "loeser": "linear"}),
    # Die Form steht vor der Kostenfrage: auch bei τ = 0,5 genügt das Mittel nur bei linearem Mittelwert (Nachbau, Welt quadratisch: README, Befunde).
    "form": ("Ist der Mittelwert linear in den Merkmalen?", {"ja": "schief", "nein": "gewichte"}),
    "schief": ("Sind die Kosten schief (τ ≠ 0,5)?", {"nein": "mittel", "ja": "streuung"}),
    "streuung": ("Hängt die Streuung von x ab?", {"nein": "kq", "monoton": "daten", "nicht_monoton": "blind"}),
    "daten": ("Gibt es mindestens 10·(p + 1) Fälle?", {"ja": "direkt", "nein": "kq_wenig"}),
    "linear": ("Trifft ein lineares Modell die Kosten?", {"ja": "kq_gitter", "nein": "modell"}),
    "modell": ("Darf das Modell flexibel sein?", {"flexibel": "kern", "linear": "loeser"}),
    "loeser": ("Was ist über den Löser bekannt?", {"spo": "rauschen", "glaettung": "glaettung", "stoerung": "stoerung", "blackbox": "bb_rauschen"}),
    "rauschen": ("Rauschen und Zahl der Trainingsfälle?", {"ohne": "spo", "mit_1000": "spo_rausch", "mit_100": "spo_wenig"}),
    "bb_rauschen": ("Sind die Ziele sauber?", {"sauber": "blackbox", "verrauscht": "blackbox_rausch"}),
    "garantie": ("Wird eine Garantie für die Kosten der gewählten Entscheidung verlangt?", {"ja": "schranke"}),
}
START = "art"
EXTRA = "garantie"   # zweiter, vom ersten unabhängiger Zweig: gilt zusätzlich zu jedem Blatt

# Kurzer Name der Antwort an der Kante
EDGE_LABELS = {
    ("art", "menge"): "Menge je Fall", ("art", "loeser"): "Löser",
    ("schief", "nein"): "nein", ("schief", "ja"): "ja",
    ("form", "ja"): "ja", ("form", "nein"): "nein",
    ("streuung", "nein"): "nein", ("streuung", "monoton"): "ja, linear", ("streuung", "nicht_monoton"): "ja, nicht monoton",
    ("daten", "ja"): "ja", ("daten", "nein"): "nein",
    ("linear", "ja"): "ja", ("linear", "nein"): "nein",
    ("modell", "flexibel"): "flexibel", ("modell", "linear"): "linear",
    ("loeser", "spo"): "Ersatzverlust", ("loeser", "glaettung"): "Wege aufzählbar", ("loeser", "stoerung"): "nur Aufrufe, Störung", ("loeser", "blackbox"): "nur Aufrufe, Interpolation",
    ("rauschen", "ohne"): "ohne Rauschen", ("rauschen", "mit_1000"): "Rauschen, n = 1000", ("rauschen", "mit_100"): "Rauschen, n = 100",
    ("bb_rauschen", "sauber"): "sauber", ("bb_rauschen", "verrauscht"): "verrauscht",
    ("garantie", "ja"): "ja",
}


@dataclass(frozen=True)
class Leaf:
    id: str
    stueck: int
    verfahren: str
    text: str            # Messwert mit Bedingung, Zahlen wörtlich aus der README des Stücks
    zahlen: tuple        # die in `text` genannten Zahlen (als Text, mit Dezimalkomma)
    marker: tuple        # Zeichenfolgen, die wörtlich in der README des Stücks stehen müssen
    weitere: tuple = ()  # (Stück, Zeichenfolge) für Zahlen aus der README eines anderen Stücks, die der Text zusätzlich nennt


LEAVES = {
    "mittel": Leaf(
        "mittel", 1, "Mittel bestellen genügt",
        "Bei τ = 0,5 und linearem Mittelwert kostet das Mittel im Median 0,1 % Mehrkosten gegen das Orakel (p = 3, n = 1000); bei τ = 0,8 sind es 42,5 %, bei τ = 0,95 sind es 286,7 %. Das gilt für die Normalverteilung dieser Welt und nicht bei gekrümmtem Mittelwert; dort führt der Baum zu den Gewichten.",
        ("0,1", "42,5", "286,7"), ("0,1 [0,0; 0,5] / **42,5** [36,7; 47,5] / **286,7** [262,7; 306,6] % bei τ = 0,5 / 0,8 / 0,95",)),
    "kq": Leaf(
        "kq", 2, "KQ plus Quantil der Residuen",
        "Bei konstanter Streuung (p = 3, n = 1000, τ = 0,95) kostet KQ plus Quantil im Median 0,3 % Mehrkosten, direkt 0,9 % (direkt ist in 4 von 30 Seeds besser).",
        ("0,3", "0,9", "4 von 30"), ("p = 3, n = 1000: KQ+Q 0,3 % gegen direkt 0,9 % (direkt besser in 4 von 30)",)),
    "direkt": Leaf(
        "direkt", 2, "direkt (Quantilregression)",
        "Bei linear wachsender Streuung (p = 3, n = 1000, τ = 0,95) kostet KQ plus Quantil im Median 34,5 %, direkt 7,8 % (direkt ist in 30 von 30 Seeds besser).",
        ("34,5", "7,8", "30 von 30"), ("34,5 % gegen 7,8 % bei n = 1000 (30 von 30)",)),
    "kq_wenig": Leaf(
        "kq_wenig", 2, "KQ plus Quantil der Residuen (zu wenig Fälle für direkt)",
        "Bei linear wachsender Streuung, aber p = 10 und n = 30 (τ = 0,95) kostet KQ plus Quantil im Median 54,7 %, direkt 185,6 % (direkt ist in 0 von 30 Seeds besser).",
        ("54,7", "185,6", "0 von 30"), ("n = 30: 54,7 % gegen 185,6 % (0 von 30)",)),
    "blind": Leaf(
        "blind", 2, "blinder Fleck: die Diagnose sieht es nicht",
        "Bei Streuung in |x₁| (p = 3, n = 1000, τ = 0,95) kosten KQ plus Quantil im Median 19,5 % und direkt 20,0 % (direkt ist in 7 von 30 Seeds besser): keine der beiden linearen Regeln trifft die Streuung; Gewichte aus Ähnlichkeit sind in dieser Zelle besser (Stück 5: kNN 12,1 gegen KQ + empirisch 19,3 %). Die Streuungs-Diagnose erkennt praktisch nur eine Streuung mit linearer Komponente und schlägt hier kaum an (Experiment 4).",
        ("19,5", "20,0", "7 von 30"), ("19,5 % gegen 20,0 % (7 von 30)",),
        ((5, "KQ + empirisch 19,3 / 19,0, direkt 20,7 / 21,8, **kNN 12,1 / 29,5**"),)),
    "gewichte": Leaf(
        "gewichte", 5, "Gewichte aus Ähnlichkeit (kNN, Kern, Wald)",
        "Bei quadratischer Wahrheit (p = 3, n = 1000, τ = 0,95, Vormessung mit 30 Seeds) kosten KQ plus Quantil 134,9 %, kNN-Gewichte 30,4 %, Kerngewichte 29,6 % und Wald-Gewichte 29,7 %; bei p = 10 liegt nur der Wald noch bei 32,9 %, kNN und Kern bei 81,1 und 80,8 %.",
        ("134,9", "30,4", "29,6", "29,7", "32,9", "81,1", "80,8"),
        ("KQ + empirisch 134,9 / 140,0; direkt 136,5 / 148,1; **kNN 30,4 / 81,1; Kern 29,6 / 80,8; Wald 29,7 / 32,9**",)),
    "kq_gitter": Leaf(
        "kq_gitter", 3, "Kleinste Quadrate (SPO+ gewinnt hier nicht)",
        "Bei linearer Wahrheit (Grad 1, ohne Rauschen) ist SPO+ in 0 von 20 Seeds besser als Kleinste Quadrate; beide liegen bei 0,0 % Regret (n = 100, 1000 und 5000).",
        ("0 von 20", "0,0"), ("Bei Grad 1 ist SPO+ in keiner Zelle ohne Rauschen besser (0 von 20 Seeds)", "| 1 | 100 | 0,0 | 0,0 [0,0; 0,0] | 0,0 [0,0; 0,1] | 0 von 20 |")),
    "kern": Leaf(
        "kern", 4, "flexibles Modell (Gauß-Kern-Ridge, Kern-ETO)",
        "Bei Grad 6 und n = 1000 (ohne Rauschen) liegt Kern-ETO bei 0,05 % Regret, SPO+ bei 0,51 %, KQ bei 2,0 % (Kern-ETO besser als SPO+ in 20 von 20 Seeds); mit Rauschen 0,5 sind es 11,8 %, 12,3 % und 13,1 %.",
        ("0,05", "0,51", "2,0", "20 von 20", "11,8", "12,3", "13,1"),
        ("Bei Grad 6 und n = 1000: Kern-ETO 0,05 %, SPO+ 0,51 %, KQ 2,0 % (Kern-ETO besser als SPO+ in 20 von 20 Seeds, besser als KQ in 20 von 20)",
         "n = 1000: Kern-ETO 11,8 %, SPO+ 12,3 %, KQ 13,1 %, Orakel 10,5 %")),
    "spo": Leaf(
        "spo", 3, "SPO+ (ohne Rauschen)",
        "Bei Grad 6 ohne Rauschen (n = 5000) kostet KQ 2,0 % Regret, SPO+ 0,5 % (SPO+ ist in 20 von 20 Seeds besser).",
        ("2,0", "0,5", "20 von 20"), ("Grad 6, Rauschstufe 0, n = 5000: Regret KQ 2,0 %, SPO+ 0,5 % (SPO+ in 20 von 20 Seeds besser)",)),
    "spo_rausch": Leaf(
        "spo_rausch", 3, "SPO+ (mit Rauschen, genug Fälle)",
        "Bei Grad 6 mit Rauschen 0,5 und n = 1000 liegt das Orakel bei 10,5 %, KQ bei 13,1 % und SPO+ bei 12,3 % (SPO+ ist in 20 von 20 Seeds besser).",
        ("10,5", "13,1", "12,3", "20 von 20"), ("Grad 6, Rauschstufe 0,5, n = 1000: Orakel 10,5 %, KQ 13,1 %, SPO+ 12,3 %", "Mit Rauschen sicher nur bei Grad 6 und n = 1000 oder 5000 (je 20 von 20 Seeds)")),
    "spo_wenig": Leaf(
        "spo_wenig", 3, "Kleinste Quadrate behalten (SPO+ lohnt hier nicht)",
        "Bei n = 100 mit Rauschen 0,5 ist SPO+ bei Grad 1, 2 und 4 in 1, 0 und 0 von 20 Seeds besser als KQ; bei Grad 6 in 10 von 20, in der Stabilitätsprobe mit 100 weiteren Seeds in 29 von 100.",
        ("1, 0 und 0 von 20", "10 von 20", "29 von 100"),
        ("bei Grad 1, 2 und 4 liegt SPO+ im Median über KQ (in 1, 0 und 0 von 20 Seeds besser", "Bei Grad 6 ist es in diesen 20 Seeds ein Münzwurf (10 von 20)", "ergibt SPO+ besser in 29 von 100")),
    "glaettung": Leaf(
        "glaettung", 6, "Gibbs-Glättung des Lösers",
        "Bei Grad 6 und n = 1000 liegt der Regret nach 300 Adam-Schritten für T = 0,03 bis 3 in 19 von 20 oder 20 von 20 Seeds unter dem von KQ; bei T = 0,001 nicht (ohne Rauschen 2,0 gegen 2,0 %).",
        ("19 von 20", "20 von 20", "2,0"),
        ("T = 0,03 bis 3 in 19 von 20 oder 20 von 20 Seeds unter dem von Kleinsten Quadraten (KQ)", "Bei T = 0,001 nicht: ohne Rauschen im Median so hoch wie KQ (2,0 gegen 2,0 %)")),
    "stoerung": Leaf(
        "stoerung", 7, "gestörte Lösung (Fenchel-Young)",
        "Mit Rauschen 0,5 liegt der Median-Regret bei σ = 1 und σ = 3 unter dem von KQ in 20 von 20 Seeds (12,2 und 11,9 gegen 13,1 %); bei σ = 0,1 gleich KQ (13,1 %).",
        ("20 von 20", "12,2", "11,9", "13,1"),
        ("bei σ = 1 und σ = 3 unter dem von KQ in 20 von 20 Seeds (12,2 und 11,9 gegen 13,1 %)", "bei σ = 0,1 in 7 von 20 (13,1 %, gleich KQ)")),
    "blackbox": Leaf(
        "blackbox", 8, "Black-Box-Interpolation (nur mit sauberen Zielen)",
        "Ohne Rauschen (Grad 6, n = 1000) kostet KQ 1,96 % Regret, SPO+ 0,51 % und die Interpolation mit λ = 1 0,57 %.",
        ("1,96", "0,51", "0,57"), ("KQ 1,96 % Regret, SPO+ 0,51 %, Interpolation mit λ = 1 0,57 %",)),
    "blackbox_rausch": Leaf(
        "blackbox_rausch", 8, "Black-Box-Interpolation scheitert bei verrauschten Zielen",
        "Mit Rauschen 0,5 liegt die Interpolation bei λ = 0,3 / 1 / 10 / 100 bei 18,6 / 20,8 / 20,4 / 21,6 %, KQ bei 13,1 %: in allen 20 Seeds ist jede dieser Varianten schlechter als KQ.",
        ("18,6", "20,8", "20,4", "21,6", "13,1", "20"),
        ("KQ 13,1 %, SPO+ 12,3 %, Interpolation bei λ = 0,3 / 1 / 10 / 100 bei 18,6 / 20,8 / 20,4 / 21,6 %", "In **allen 20 Seeds** ist jede dieser vier Varianten schlechter als KQ")),
    "schranke": Leaf(
        "schranke", 9, "konforme Schranke auf den Wert der Entscheidung",
        "Bei Ziel 90 % liegt die Überdeckung der Schranke auf den Wegkosten im Median in allen 9 Zellen zwischen 90,0 und 90,2 %, die der Summe der Kantenschranken zwischen 93,4 und 100,0 %; die Schranke gilt im Mittel über Kalibrierungen, nicht für eine einzelne.",
        ("90", "9", "90,0", "90,2", "93,4", "100,0"),
        ("zwischen 90,0 und 90,2 %; die der Summe der Kantenschranken zwischen 93,4 und 100,0 %", "Die Garantie ist marginal.")),
}

# Zahl der Fragen/Antworten je Regler der App: Schlüssel des Reglers -> Frage, die er beantwortet
QUESTION_OF_WIDGET = {"q_art": "art", "q_form": "form", "q_streuung": "streuung", "q_linear": "linear", "q_modell": "modell", "q_loeser": "loeser",
                      "q_rauschen": "rauschen", "q_ngitter": "rauschen", "q_garantie": "garantie"}


def answers_from(tau: float, n: int, p: int, w: dict) -> dict:
    """Antwort auf jede Frage des Baums aus den Reglern der App: `tau`, `n`, `p` aus der Seitenleiste, `w` die Werte der Baum-Regler (Schlüssel `q_...`)."""
    return {
        "art": w["q_art"],
        "schief": "nein" if abs(tau - 0.5) < 1e-9 else "ja",
        "form": w["q_form"],
        "streuung": w["q_streuung"],
        "daten": "ja" if n >= 10 * (p + 1) else "nein",
        "linear": w["q_linear"],
        "modell": w["q_modell"],
        "loeser": w["q_loeser"],
        "rauschen": "ohne" if w["q_rauschen"] == "ohne" else f"mit_{w['q_ngitter']}",
        "bb_rauschen": "sauber" if w["q_rauschen"] == "ohne" else "verrauscht",
        "garantie": w["q_garantie"],
    }


@dataclass(frozen=True)
class Path:
    questions: tuple      # besuchte Fragen in der Reihenfolge
    answers: tuple        # gegebene Antworten, parallel dazu
    leaf: str             # Blatt des Hauptzweigs
    extra_leaf: str | None  # zusätzliches Blatt (Garantie), sonst None


def walk(answers: dict) -> Path:
    """Weg vom Start zum Blatt; die Garantie-Frage bildet ein zusätzliches Blatt."""
    qs, ans = [], []
    node = START
    while node in QUESTIONS:
        a = answers[node]
        qs.append(node)
        ans.append(a)
        node = QUESTIONS[node][1][a]
    extra = LEAVES["schranke"].id if answers[EXTRA] == "ja" else None
    return Path(tuple(qs), tuple(ans), node, extra)


def all_leaf_ids() -> set:
    """Alle Blätter, die von Fragen aus erreichbar sind (ohne das Garantie-Blatt, das über EXTRA hängt)."""
    out = set()
    for _, (_, kids) in QUESTIONS.items():
        out |= {k for k in kids.values() if k in LEAVES}
    return out


# --- DOT --------------------------------------------------------------------------------------------------------------------------------------------

_ACTIVE_FILL = "#cfe6e2"
_ACTIVE_LEAF = "#f6d9a8"


def _esc(text: str) -> str:
    return text.replace("\\", "\\\\").replace('"', "'").replace("&", "und")


def _wrap(text: str, width: int = 26) -> str:
    text = text.replace(" ≠ ", " ≠ ").replace("(p + 1)", "(p + 1)").replace("n = ", "n = ")
    words, lines, cur = text.split(" "), [], ""
    for w in words:
        if cur and len(cur) + 1 + len(w) > width:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    lines.append(cur)
    return "\\n".join(_esc(x) for x in lines)


def to_dot(path: Path | None = None) -> str:
    """DOT-Quelle des Baums. Mit `path` sind die besuchten Knoten und Kanten hervorgehoben; ohne ist es die statische Datei `entscheidungsbaum.dot`."""
    on_nodes = set(path.questions) | {path.leaf} | ({path.extra_leaf} if path and path.extra_leaf else set()) if path else set()
    on_edges = set(zip(path.questions, path.answers)) if path else set()
    if path and path.extra_leaf:
        on_edges.add((EXTRA, "ja"))
        on_nodes.add(EXTRA)
    lines = [
        "digraph entscheidungsbaum {",
        '  graph [rankdir=LR, fontname="Helvetica", nodesep=0.2, ranksep=0.4, splines=true, pad=0.2, label="Welches Verfahren? Fragen links, Blätter rechts mit dem Stück der Linie", labelloc=t];',
        '  node [fontname="Helvetica", fontsize=14, style=filled, fillcolor="#ffffff", color="#555555", target="_blank"];',
        '  edge [fontname="Helvetica", fontsize=12, color="#777777"];',
    ]
    for qid, (qtext, _) in QUESTIONS.items():
        fill = _ACTIVE_FILL if qid in on_nodes else "#ffffff"
        lines.append(f'  {qid} [shape=box, style="rounded,filled", fillcolor="{fill}", label="{_wrap(qtext, 28)}", URL="{C.stueck_url(first_leaf_piece(qid))}", tooltip="Frage: {_esc(qtext)}"];')
    for lid, leaf in LEAVES.items():
        fill = _ACTIVE_LEAF if lid in on_nodes else "#f3f3f3"
        label = f"{_wrap(leaf.verfahren, 26)}\\nStück {leaf.stueck}"
        lines.append(f'  {lid} [shape=box, fillcolor="{fill}", label="{label}", URL="{C.stueck_url(leaf.stueck)}", tooltip="Stück {leaf.stueck}: {_esc(C.STUECKE[leaf.stueck][1])}"];')
    for qid, (_, kids) in QUESTIONS.items():
        for ans, nxt in kids.items():
            hot = (qid, ans) in on_edges
            attrs = f'label="{_esc(EDGE_LABELS[(qid, ans)])}"' + (', color="#2F6B65", penwidth=2.4, fontcolor="#2F6B65"' if hot else "")
            if qid == EXTRA:
                attrs += ", style=dashed"
            lines.append(f"  {qid} -> {nxt} [{attrs}];")
    lines.append(f'  art -> {EXTRA} [label="zusätzlich", style=dashed, color="#777777"];')
    lines.append("}")
    return "\n".join(lines) + "\n"


def first_leaf_piece(qid: str) -> int:
    """Stück, auf das ein Frage-Knoten verlinkt: das Stück des ersten Blatts unter ihm (jeder Knoten braucht ein URL-Attribut, sonst bleibt das Diagramm für Tastatur und Screenreader unbedienbar)."""
    first = next(iter(QUESTIONS[qid][1].values()))
    return LEAVES[first].stueck if first in LEAVES else first_leaf_piece(first)
