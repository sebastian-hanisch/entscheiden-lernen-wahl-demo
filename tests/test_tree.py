"""Der Entscheidungsbaum: Aufbau, Weg durch den Baum, DOT-Quelle und der Verweis-Test gegen die READMEs der neun Stücke.

Der Verweis-Test liest die README des Schwesterordners (`../<repo>/README.md`) und bricht, wenn sich eine Zahl eines Stücks ändert, ohne dass der Baum nachgezogen wird.
Liegen die Schwesterordner nicht neben diesem Repo (CI, Streamlit Cloud), überspringt er sich."""

import itertools
import re
import shutil
import subprocess
from pathlib import Path

import pytest

import elw_constants as C
import elw_tree as T

ROOT = Path(__file__).resolve().parent.parent
DOT_FILE = ROOT / "entscheidungsbaum.dot"
DOT_EXE = shutil.which("dot") or r"C:\Program Files\Graphviz\bin\dot.exe"

VALUES = {"q_art": C.Q_ART, "q_form": C.Q_FORM, "q_streuung": C.Q_STREUUNG, "q_linear": C.Q_LINEAR, "q_modell": C.Q_MODELL, "q_loeser": C.Q_LOESER,
          "q_rauschen": C.Q_RAUSCHEN, "q_ngitter": C.Q_NGITTER, "q_garantie": C.Q_GARANTIE}


def _leaf_for(tau, n, p, **overrides):
    w = {**C.DEFAULT_TREE, **overrides}
    return T.walk(T.answers_from(tau, n, p, w))


# --- Aufbau und Weg -----------------------------------------------------------------------------------------------------------------------------------


def test_every_answer_leads_to_a_question_or_a_leaf_and_every_leaf_is_reachable():
    ids = set(T.QUESTIONS) | set(T.LEAVES)
    for qid, (_, kids) in T.QUESTIONS.items():
        assert set(kids.values()) <= ids, qid
        for ans in kids:
            assert (qid, ans) in T.EDGE_LABELS, (qid, ans)
    assert set(T.EDGE_LABELS) == {(q, a) for q, (_, kids) in T.QUESTIONS.items() for a in kids}
    reached = set()
    for combo in itertools.product(*VALUES.values()):
        for tau, n, p in ((0.5, 1000, 3), (0.95, 1000, 3), (0.95, 100, 10)):
            reached.add(T.walk(T.answers_from(tau, n, p, dict(zip(VALUES, combo)))).leaf)
    assert reached == set(T.LEAVES) - {"schranke"}                    # das Garantie-Blatt hängt am Zusatzzweig
    assert T.all_leaf_ids() == set(T.LEAVES)


def test_walk_follows_the_hand_table_of_answers():
    table = [
        (dict(tau=0.5, n=1000, p=3), {}, "mittel"),
        (dict(tau=0.5, n=1000, p=3), {"q_form": "nein"}, "gewichte"),                 # auch bei τ = 0,5 genügt das Mittel nicht, wenn die Wahrheit gekrümmt ist
        (dict(tau=0.5, n=1000, p=3), {"q_streuung": "monoton"}, "mittel"),            # bei τ = 0,5 und linearem Mittelwert zählt die Streuung nicht
        (dict(tau=0.8, n=1000, p=3), {"q_form": "nein"}, "gewichte"),
        (dict(tau=0.95, n=1000, p=3), {"q_form": "nein"}, "gewichte"),
        (dict(tau=0.95, n=1000, p=3), {"q_streuung": "nein"}, "kq"),
        (dict(tau=0.95, n=1000, p=3), {"q_streuung": "monoton"}, "direkt"),
        (dict(tau=0.95, n=100, p=10), {"q_streuung": "monoton"}, "kq_wenig"),         # 100 < 10 (10 + 1) = 110
        (dict(tau=0.95, n=100, p=3), {"q_streuung": "monoton"}, "direkt"),            # 100 >= 40
        (dict(tau=0.95, n=1000, p=3), {"q_streuung": "nicht_monoton"}, "blind"),
        (dict(tau=0.95, n=1000, p=3), {"q_art": "loeser", "q_linear": "ja"}, "kq_gitter"),
        (dict(tau=0.95, n=1000, p=3), {"q_art": "loeser", "q_modell": "flexibel"}, "kern"),
        (dict(tau=0.95, n=1000, p=3), {"q_art": "loeser", "q_rauschen": "ohne"}, "spo"),
        (dict(tau=0.95, n=1000, p=3), {"q_art": "loeser", "q_rauschen": "mit", "q_ngitter": 1000}, "spo_rausch"),
        (dict(tau=0.95, n=1000, p=3), {"q_art": "loeser", "q_rauschen": "mit", "q_ngitter": 100}, "spo_wenig"),
        (dict(tau=0.95, n=1000, p=3), {"q_art": "loeser", "q_loeser": "glaettung"}, "glaettung"),
        (dict(tau=0.95, n=1000, p=3), {"q_art": "loeser", "q_loeser": "stoerung"}, "stoerung"),
        (dict(tau=0.95, n=1000, p=3), {"q_art": "loeser", "q_loeser": "blackbox", "q_rauschen": "ohne"}, "blackbox"),
        (dict(tau=0.95, n=1000, p=3), {"q_art": "loeser", "q_loeser": "blackbox", "q_rauschen": "mit"}, "blackbox_rausch"),
    ]
    for args, overrides, leaf in table:
        assert _leaf_for(**args, **overrides).leaf == leaf, (args, overrides)


def test_the_guarantee_question_adds_a_second_leaf_and_never_replaces_the_first():
    for art in C.Q_ART:
        no, yes = _leaf_for(0.95, 1000, 3, q_art=art, q_garantie="nein"), _leaf_for(0.95, 1000, 3, q_art=art, q_garantie="ja")
        assert no.extra_leaf is None and yes.extra_leaf == "schranke" and yes.leaf == no.leaf


def test_presets_land_on_the_leaf_their_help_text_names():
    expected = {
        "Streuung wächst mit x₁ (direkt)": "direkt",
        "Konstante Streuung, linear (KQ + Quantil)": "kq",
        "Gekrümmte Wahrheit (Gewichte)": "gewichte",
        "Blinder Fleck: Streuung in |x₁|": "blind",
        "Entscheidung mit Löser (SPO+)": "spo_rausch",
    }
    assert set(expected) == set(C.PRESETS)
    for name, preset in C.PRESETS.items():
        w = {k: preset[k] for k in C.DEFAULT_TREE}
        assert T.walk(T.answers_from(preset["tau"], preset["n"], preset["p"], w)).leaf == expected[name], name


def test_the_question_for_each_widget_exists():
    assert set(T.QUESTION_OF_WIDGET.values()) <= set(T.QUESTIONS)
    assert set(T.QUESTION_OF_WIDGET) == set(C.DEFAULT_TREE)


# --- DOT ---------------------------------------------------------------------------------------------------------------------------------------------


def test_the_static_dot_file_is_what_the_code_generates():
    assert DOT_FILE.read_text(encoding="utf-8") == T.to_dot()


def test_every_node_has_a_link_to_a_piece_of_the_line_and_a_tooltip():
    dot = T.to_dot()
    node_lines = [ln for ln in dot.splitlines() if re.match(r"\s+\w+ \[shape=box", ln)]
    assert len(node_lines) == len(T.QUESTIONS) + len(T.LEAVES)
    urls = {C.stueck_url(nr) for nr in C.STUECKE}
    for ln in node_lines:
        m = re.search(r'URL="([^"]+)"', ln)
        assert m and m.group(1) in urls, ln
        assert "tooltip=" in ln
    assert "&" not in dot                                    # TITLE und Beschriftungen ohne & (WEBSITE-PLAYBOOK §4)
    for leaf in T.LEAVES.values():
        assert f'URL="{C.stueck_url(leaf.stueck)}"' in next(ln for ln in node_lines if ln.strip().startswith(leaf.id + " "))


def test_question_nodes_link_to_the_first_piece_below_them():
    assert T.first_leaf_piece("art") == 1 and T.first_leaf_piece("form") == 1 and T.first_leaf_piece("streuung") == 2 and T.first_leaf_piece("linear") == 3 and T.first_leaf_piece("garantie") == 9


def test_the_highlighted_path_marks_exactly_the_visited_nodes():
    path = _leaf_for(0.95, 1000, 3, q_garantie="ja")
    dot = T.to_dot(path)
    marked = {m.group(1) for m in re.finditer(r"^\s+(\w+) \[shape=box.*fillcolor=\"(#cfe6e2|#f6d9a8)\"", dot, re.M)}
    assert marked == set(path.questions) | {path.leaf, "schranke", "garantie"}
    assert dot.count("penwidth=2.4") == len(path.questions) + 1                       # jede Kante des Weges plus die Garantie-Kante


@pytest.mark.skipif(not Path(DOT_EXE).exists() and shutil.which("dot") is None, reason="Graphviz dot nicht installiert")
def test_graphviz_renders_the_diagram_with_one_link_per_node():
    out = subprocess.run([DOT_EXE, "-Tsvg"], input=T.to_dot().encode("utf-8"), capture_output=True, timeout=60)
    assert out.returncode == 0, out.stderr.decode("utf-8", "replace")
    svg = out.stdout.decode("utf-8")
    assert svg.count("<a xlink:href=") == len(T.QUESTIONS) + len(T.LEAVES)
    m = re.search(r'viewBox="0\.00 0\.00 ([\d.]+) ([\d.]+)"', svg)
    assert m and float(m.group(1)) > float(m.group(2))                                 # von links nach rechts: breiter als hoch


# --- Verweis-Test gegen die READMEs der Stücke ------------------------------------------------------------------------------------------------------


def _readme(nr):
    path = ROOT.parent / C.STUECKE[nr][0] / "README.md"
    if not path.exists():
        pytest.skip(f"{path.parent.name} liegt nicht neben diesem Repo")
    return path.read_text(encoding="utf-8")


@pytest.mark.parametrize("leaf_id", list(T.LEAVES))
def test_every_marker_of_a_leaf_stands_verbatim_in_the_readme_of_its_piece(leaf_id):
    leaf = T.LEAVES[leaf_id]
    readme = _readme(leaf.stueck)
    for marker in leaf.marker:
        assert marker in readme, f"Stück {leaf.stueck} ({C.STUECKE[leaf.stueck][0]}): {marker!r} steht nicht mehr in der README"
    for nr, marker in leaf.weitere:
        assert marker in _readme(nr), f"Stück {nr} ({C.STUECKE[nr][0]}): {marker!r} steht nicht mehr in der README"


@pytest.mark.parametrize("leaf_id", list(T.LEAVES))
def test_every_number_of_a_leaf_text_comes_from_a_marker_or_a_condition(leaf_id):
    """Die Zahlen des Anzeigetextes sind genau die der Marker; Bedingungen (τ, Rauschstufe) dürfen zusätzlich vorkommen."""
    leaf = T.LEAVES[leaf_id]
    joined = " ".join(leaf.marker + tuple(m for _, m in leaf.weitere))
    for z in leaf.zahlen:
        assert z in joined and z in leaf.text, (leaf_id, z)
    conditions = {"0,5", "0,8", "0,95"}
    for token in re.findall(r"\d+,\d+", leaf.text):
        assert token in joined or token in conditions, (leaf_id, token)
    assert leaf.stueck in C.STUECKE and leaf.text.endswith(".")


def test_leaf_texts_never_use_a_decimal_point():
    for leaf in T.LEAVES.values():
        assert not re.search(r"\d\.\d", leaf.text), leaf.id
