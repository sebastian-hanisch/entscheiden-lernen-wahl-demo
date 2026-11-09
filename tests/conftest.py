"""Gemeinsame Fixtures: die von Hand konstruierten Datensätze, an denen Diagnose und Regel geprüft werden (DEMO-PLAYBOOK §1b)."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def hand_quadratic():
    """n = 5, ein Merkmal x = -2, -1, 0, 1, 2 und y = x² plus 1 am letzten Punkt: y = 4, 1, 0, 1, 5.

    Von Hand (orthogonale Basis 1, x, x² - 2 mit den Normen 5, 10, 14): Mittel 2,2, Steigung 0,2, Koeffizient von x² - 2 gleich 16/14 = 8/7.
    RSS0 = 18,4 (nur [1, x]), RSS1 = 43 - (24,2 + 0,4 + 128/7) = 4/35. F = (18,4 - 4/35) / ((4/35) / 2) = 320 mit 1 und 2 Freiheitsgraden,
    und für 2 Freiheitsgrade im Nenner gilt in geschlossener Form p = 1 - sqrt(F / (F + 2)) = 0,003110."""
    return np.array([[-2.0], [-1.0], [0.0], [1.0], [2.0]]), np.array([4.0, 1.0, 0.0, 1.0, 5.0])


def alternating_spread(n: int):
    """x = 1 .. n, y = 10 + (-1)^i · 0,5 · x: die Streuung um die Gerade wächst linear mit x (|Residuum| ≈ 0,5 x), die Anpassung ist nicht gekrümmt."""
    x = np.arange(1, n + 1, dtype=float)
    return x[:, None], 10.0 + (-1.0) ** np.arange(n) * 0.5 * x


@pytest.fixture
def hand_hetero():
    return alternating_spread
