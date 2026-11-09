"""Friert die Rohausgaben der Vormessung V11 in `elw_vormessung.json` ein (einmal gelaufen; die Datei steht im Repo).

Quellen (Planungsordner, nicht Teil dieses Repos): `vormessung/out_eo_v11_wahlregel.txt` (Seeds 0 bis 29) und `vormessung/out_eo_v11_wahlregel_seeds100.txt`
(Seeds 100 bis 129). Die Vormessung rechnet mit scikit-learn (Zufallswald, 50 Bäume) und dem numpy-Zufallsstrom; sie ist in der App nicht live.
Aufruf (Repo-Wurzel):  python tools/freeze_vormessung.py [Ordner der Rohausgaben]"""

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DIR = ROOT.parent / "entscheidungsorientiert-planung" / "vormessung"
FILES = {"satz_a": "out_eo_v11_wahlregel.txt", "satz_b": "out_eo_v11_wahlregel_seeds100.txt"}
NUM = r"(-?\d+(?:\.\d+)?)"


def parse(text: str) -> dict:
    out = {"gesamt": {}, "welten": {}, "n_p": {}, "diagnose": {}}
    m = re.search(r"(\d+) Zeilen \((\d+) Seeds\)", text)
    out["zeilen"], out["seeds"] = int(m.group(1)), int(m.group(2))
    for name, key in (("immer     emp", "emp"), ("immer  direkt", "direkt"), ("immer    wald", "wald"), ("Regel         ", "regel"),
                      ("beste Wahl je Zeile (Orakel)", "beste")):
        m = re.search(re.escape(name) + r":\s+" + NUM + r"\s+Median\s+" + NUM, text)
        out["gesamt"][key] = {"mittel": float(m.group(1)), "median": float(m.group(2))}
    for m in re.finditer(r"^\s+(linear|streuung-linear|heteroskedastisch|nichtlinear): immer emp\s+" + NUM + r" \| immer direkt\s+" + NUM + r" \| immer wald\s+" + NUM
                         + r" \| Regel\s+" + NUM + r" \| Orakel\s+" + NUM + r" \| Wahl \{'emp': (\d+), 'direkt': (\d+), 'wald': (\d+)\}", text, re.M):
        g = m.groups()
        out["welten"][g[0]] = {"emp": float(g[1]), "direkt": float(g[2]), "wald": float(g[3]), "regel": float(g[4]), "beste": float(g[5]),
                               "wahl": {"emp": int(g[6]), "direkt": int(g[7]), "wald": int(g[8])}}
    for m in re.finditer(r"^\s+n =\s*(\d+), p =\s*(\d+): immer emp\s+" + NUM + r" \| immer direkt\s+" + NUM + r" \| immer wald\s+" + NUM + r" \| Regel\s+" + NUM
                         + r" \| Orakel\s+" + NUM, text, re.M):
        g = m.groups()
        out["n_p"][f"n{g[0]}_p{g[1]}"] = {"emp": float(g[2]), "direkt": float(g[3]), "wald": float(g[4]), "regel": float(g[5]), "beste": float(g[6])}
    m = re.search(r"Diagnose Nichtlinearität: erkannt in (\d+) % der nichtlinearen Zeilen, Fehlalarm in (\d+) % der übrigen", text)
    out["diagnose"]["nichtlinear_erkannt"], out["diagnose"]["fehlalarm"] = int(m.group(1)), int(m.group(2))
    for name, key in (("streuung-linear", "streuung_linear"), ("heteroskedastisch", "streuung_betrag"), ("linear", "streuung_linear_welt")):
        m = re.search(r"Diagnose Streuung \(" + re.escape(name) + r"\): p < 0,01 in (\d+) % der Zeilen", text)
        out["diagnose"][key] = int(m.group(1))
    assert len(out["welten"]) == 4 and len(out["n_p"]) == 4, "Auswertung unvollständig"
    return out


def main():
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_DIR
    data = {
        "quelle": "vormessung/out_eo_v11_wahlregel.txt (Seeds 0 bis 29) und vormessung/out_eo_v11_wahlregel_seeds100.txt (Seeds 100 bis 129); Zufallswald mit scikit-learn, nicht live",
        "einheit": "Mehrkosten gegen das Orakel in Prozent (Mittel über die Zeilen; bei gesamt auch der Median)",
        "welt_namen": {"linear": "linear", "streuung-linear": "streuung-linear", "heteroskedastisch": "betrag", "nichtlinear": "quadratisch"},
    }
    for key, name in FILES.items():
        data[key] = parse((src / name).read_text(encoding="utf-8"))
    (ROOT / "elw_vormessung.json").write_text(json.dumps(data, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
