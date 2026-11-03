"""
1-cosine discrete gust model (CS 25 / FAR 25 style).

Vertical gust velocity:
    w_g(s) = (U_ds / 2) * (1 - cos(π s / H))   for 0 ≤ s ≤ 2H
           = 0 otherwise

where s = V * t  (distance travelled), H = gust gradient distance.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass


@dataclass
class GustProfile:
    """Parameters of a single 1-cos discrete gust."""
    U_ds: float          # design gust velocity [m/s]
    H: float             # gust gradient length [m]
    V: float             # true airspeed [m/s]

    def duration(self) -> float:
        """Time to traverse the full gust (2H / V)."""
        return 2.0 * self.H / self.V

    def velocity(self, t: float | np.ndarray) -> float | np.ndarray:
        """
        Vertical gust velocity w_g(t).
        Accepts scalar or array of times.
        """
        s = self.V * np.asarray(t)
        w = np.zeros_like(s, dtype=float)
        mask = (s >= 0.0) & (s <= 2.0 * self.H)
        w[mask] = (self.U_ds / 2.0) * (1.0 - np.cos(np.pi * s[mask] / self.H))
        return w if isinstance(t, np.ndarray) else float(w)

    def alpha(self, t: float | np.ndarray) -> float | np.ndarray:
        """Effective angle-of-attack increment Δα = w_g / V."""
        return self.velocity(t) / self.V


def standard_gust_lengths() -> np.ndarray:
    """Typical discrete-gust gradient lengths used in certification [m]."""
    return np.array([30.0, 50.0, 75.0, 100.0, 125.0, 150.0, 200.0, 250.0, 300.0])
