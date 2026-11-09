"""Die eingefrorene Vormessung V11 (Wahlregel mit Zufallswald, scikit-learn, nicht live) und der vorgerechnete Nachbau mit dem Code dieser Demo.

`elw_vormessung.json` entsteht aus den Rohausgaben der Vormessung (`tools/freeze_vormessung.py`). `elw_bands.json` entsteht aus `tools/messung.py`: der Nachbau
rechnet mit SplitMix64, exakten Kosten und Gauß-Kerngewichten statt des Waldes (zweite Spalte der Tabellen in der App)."""

import json
from functools import lru_cache
from pathlib import Path
from statistics import mean, median

import elw_constants as C

HERE = Path(__file__).resolve().parent
SAETZE = {"a": "Zufalls-Seeds 0 bis 29", "b": "Zufalls-Seeds 100 bis 129"}
FIXED = ("emp", "direkt", "wald")
FIXED_NAMEN = {"emp": "immer KQ + Quantil", "direkt": "immer direkt", "wald": "immer Wald-Gewichte"}


@lru_cache(maxsize=1)
def frozen() -> dict:
    return json.loads((HERE / "elw_vormessung.json").read_text(encoding="utf-8"))


def frozen_set(satz: str) -> dict:
    return frozen()[f"satz_{satz}"]


def best_fixed(satz: str) -> tuple[str, float]:
    """Das feste Verfahren mit den kleinsten mittleren Mehrkosten über alle Zeilen der Vormessung (Name, Mittel)."""
    g = frozen_set(satz)["gesamt"]
    name = min(FIXED, key=lambda m: g[m]["mittel"])
    return name, g[name]["mittel"]


def reduction_vs_best_fixed(satz: str) -> float:
    """Wie viel Prozent der mittleren Mehrkosten des besten festen Verfahrens die Regel spart (Mittel über alle Zeilen)."""
    g = frozen_set(satz)["gesamt"]
    best = best_fixed(satz)[1]
    return 100.0 * (best - g["regel"]["mittel"]) / best


# --- Nachbau (elw_bands.json) ------------------------------------------------------------------------------------------------------------------------


@lru_cache(maxsize=1)
def bands() -> dict:
    return json.loads((HERE / "elw_bands.json").read_text(encoding="utf-8"))


def rows(satz: str) -> list[dict]:
    """Zeilen des Nachbaus als Dictionaries; `regel_extra` ist die Mehrkosten des von der Regel gewählten Verfahrens, `beste_extra` die der besten Wahl je Zeile."""
    b = bands()
    cols = b["spalten"]
    out = []
    for r in b[f"satz_{satz}"]["zeilen"]:
        d = dict(zip(cols, r))
        d["regel_extra"] = d[C.REGEL_METHODE[d["regel"]]]
        d["beste_extra"] = min(d["emp"], d["direkt"], d["kern"])
        out.append(d)
    return out


def summarize(rs: list[dict]) -> dict:
    """Mittel (und Median) der Mehrkosten über die Zeilen: immer-X, Regel, beste Wahl je Zeile."""
    out = {}
    for key, col in (("emp", "emp"), ("direkt", "direkt"), ("kern", "kern"), ("regel", "regel_extra"), ("beste", "beste_extra")):
        vals = [r[col] for r in rs]
        out[key] = {"mittel": mean(vals), "median": median(vals)}
    return out


def by_world(satz: str, welt: str) -> dict:
    return summarize([r for r in rows(satz) if r["welt"] == welt])


def by_np(satz: str, n: int, p: int) -> dict:
    return summarize([r for r in rows(satz) if r["n"] == n and r["p"] == p])


def diagnosis_rates(satz: str, alpha: float = 0.01) -> dict:
    """Trefferquoten der beiden F-Tests im Nachbau (Anteil der Zeilen mit p-Wert unter alpha), je Welt."""
    rs = rows(satz)
    nl = [r for r in rs if r["welt"] == "quadratisch"]
    other = [r for r in rs if r["welt"] != "quadratisch"]
    rate = lambda sel, col: sum(r[col] < alpha for r in sel) / len(sel)  # noqa: E731
    return {
        "nichtlinear_erkannt": rate(nl, "p_nonlin"),
        "fehlalarm": rate(other, "p_nonlin"),
        "streuung_linear": rate([r for r in rs if r["welt"] == "streuung-linear"], "p_hetero"),
        "streuung_betrag": rate([r for r in rs if r["welt"] == "betrag"], "p_hetero"),
        "streuung_konstant": rate([r for r in rs if r["welt"] == "linear"], "p_hetero"),
    }
