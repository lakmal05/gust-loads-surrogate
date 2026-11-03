"""
Simple strip-theory aerodynamics for a high-aspect-ratio wing.

Quasi-steady lift on each strip:
    L' = ½ ρ V² c Clα (α + Δα_gust [+ θ])
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class AeroProperties:
    """Aerodynamic parameters (constant along span for baseline)."""
    rho: float = 1.225          # air density [kg/m³]
    V: float = 50.0             # true airspeed [m/s]
    chord: float = 1.5          # chord length c [m]
    Cl_alpha: float = 2 * np.pi # lift-curve slope [1/rad]
    alpha0: float = 0.0         # rigid geometric AoA [rad]


class StripTheory:
    """
    Quasi-steady strip theory.

    The wing is discretised into the same elements as the structural mesh.
    Lift is applied as nodal forces on the free DOFs.
    """

    def __init__(self, aero: AeroProperties, n_elements: int, span: float):
        self.aero = aero
        self.n_el = n_elements
        self.span = span
        self.L_el = span / n_elements
        self.strip_centres = (np.arange(n_elements) + 0.5) * self.L_el

    def dynamic_pressure(self) -> float:
        return 0.5 * self.aero.rho * self.aero.V ** 2

    def lift_per_length(self, alpha_eff: np.ndarray) -> np.ndarray:
        q = self.dynamic_pressure()
        return q * self.aero.chord * self.aero.Cl_alpha * alpha_eff

    def nodal_forces_from_strips(self, lift_per_length: np.ndarray) -> np.ndarray:
        """
        Distribute strip lift equally to the two nodes of each element.
        Returns force vector of length n_free (root DOFs already removed).
        """
        n_nodes = self.n_el + 1
        n_dof = 2 * n_nodes
        F = np.zeros(n_dof)

        for e in range(self.n_el):
            Fe = lift_per_length[e] * self.L_el
            F[2 * e]     += Fe / 2.0
            F[2 * e + 2] += Fe / 2.0

        return F[2:]  # drop root DOFs

    def aero_force(
        self,
        structural_rotation: np.ndarray,
        gust_alpha: Optional[np.ndarray] = None,
        include_structural_alpha: bool = False,
    ) -> np.ndarray:
        """
        Compute aerodynamic force vector on free DOFs.

        include_structural_alpha=False → pure forced response (recommended baseline).
        include_structural_alpha=True  → full aeroelastic coupling (can be unstable).
        """
        alpha_eff = np.full(self.n_el, float(self.aero.alpha0))

        if include_structural_alpha:
            theta_nodes = structural_rotation[1::2]
            theta_strip = np.zeros(self.n_el)
            for e in range(self.n_el):
                if e == 0:
                    theta_strip[e] = theta_nodes[0]
                else:
                    theta_strip[e] = 0.5 * (theta_nodes[e - 1] + theta_nodes[e])
            alpha_eff = alpha_eff + theta_strip

        if gust_alpha is not None:
            alpha_eff = alpha_eff + np.asarray(gust_alpha)

        L_prime = self.lift_per_length(alpha_eff)
        return self.nodal_forces_from_strips(L_prime)
