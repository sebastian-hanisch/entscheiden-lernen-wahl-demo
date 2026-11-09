"""Nachbau der Messung V11 (Wahlregel) mit dem Code dieser Demo; schreibt `elw_bands.json`.

Aufruf (Repo-Wurzel):  python tools/messung.py a|b [Ausgabedatei]
  a: Zufalls-Seeds 0 bis 29, b: Zufalls-Seeds 100 bis 129 (unabhängige Wiederholung, wie in der Vormessung).
Je Zeile des Rasters (4 Welten, p in {3, 10}, n in {100, 1000}, tau in {0,5; 0,8; 0,95}, Seed): die beiden p-Werte der Diagnose, die Wahl der Regel und die
Mehrkosten gegen das Orakel (Prozent) von KQ + Quantil, direkt und Gauß-Kerngewichten. Der Wald der Vormessung wird NICHT gerechnet: der Zweig „Gewichte“ der
Regel benutzt hier die Kerngewichte (Leave-one-out-Bandbreite). Die Vormessung selbst steht eingefroren in `elw_vormessung.json` (Wald mit scikit-learn).
Zufall nur über SplitMix64. Die Datei `elw_bands.json` enthält beide Sätze; `merge` fügt die zwei Teildateien zusammen."""

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import elw_constants as C  # noqa: E402
import elw_diagnose as D  # noqa: E402
import elw_evaluation as ev  # noqa: E402
import elw_scenario as S  # noqa: E402

SETS = {"a": tuple(range(30)), "b": tuple(range(100, 130))}
METHODS = ("emp", "direkt", "kern")


def rows_for(seeds):
    rows = []
    t0 = time.time()
    for seed in seeds:
        for welt in C.WELTEN:
            for p in C.P_OPTIONS:
                for n in C.N_OPTIONS:
                    data = S.make_dataset(welt, p, n, seed)
                    diag = D.diagnose(data.X, data.y)
                    ext = ev.extras_for_taus(data, C.TAU_OPTIONS, METHODS)
                    for tau in C.TAU_OPTIONS:
                        e = ext[tau]
                        rows.append([welt, p, n, tau, seed, float(f"{diag.p_nonlin:.6g}"), float(f"{diag.p_hetero:.6g}"), diag.rule,
                                     round(e["emp"], 3), round(e["direkt"], 3), round(e["kern"], 3)])
        print(f"Seed {seed}: {len(rows)} Zeilen, {time.time() - t0:.0f} s", flush=True)
    return rows


def main():
    which = sys.argv[1]
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / f"elw_bands_{which}.json"
    rows = rows_for(SETS[which])
    out.write_text(json.dumps({"satz": which, "seeds": [SETS[which][0], SETS[which][-1]], "spalten": COLUMNS, "zeilen": rows}, ensure_ascii=False) + "\n", encoding="utf-8")


COLUMNS = ["welt", "p", "n", "tau", "seed", "p_nonlin", "p_hetero", "regel", "emp", "direkt", "kern"]

if __name__ == "__main__":
    if sys.argv[1] == "merge":
        a = json.loads((ROOT / "elw_bands_a.json").read_text(encoding="utf-8"))
        b = json.loads((ROOT / "elw_bands_b.json").read_text(encoding="utf-8"))
        data = {"quelle": "tools/messung.py (Nachbau von vormessung/eo_v11_wahlregel.py mit dem Code dieser Demo; Zweig Gewichte = Gauß-Kern, kein Wald)",
                "einheit": "Mehrkosten gegen das Orakel in Prozent je Zeile; p_nonlin und p_hetero sind die p-Werte der beiden F-Tests", "spalten": COLUMNS,
                "satz_a": {"seeds": a["seeds"], "zeilen": a["zeilen"]}, "satz_b": {"seeds": b["seeds"], "zeilen": b["zeilen"]}}
        (ROOT / "elw_bands.json").write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
        (ROOT / "elw_bands_a.json").unlink()
        (ROOT / "elw_bands_b.json").unlink()
    else:
        main()
