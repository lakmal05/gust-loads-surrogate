"""
Euler-Bernoulli beam finite element model for a cantilever wing.

Assumptions
-----------
- Uniform or piecewise-constant properties along span
- Only bending (no torsion for baseline)
- Consistent mass matrix
- Root fixed (cantilever boundary conditions)
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Tuple


@dataclass
class BeamProperties:
    """Physical properties of a uniform beam (wing spar idealisation)."""
    length: float          # total span L [m]
    EI: float              # flexural rigidity [N.m²]
    mass_per_length: float # linear density μ [kg/m]
    n_elements: int = 10   # number of finite elements


def element_stiffness(EI: float, L: float) -> np.ndarray:
    """
    4x4 local stiffness matrix for Euler-Bernoulli beam element
    DOFs: [v1, θ1, v2, θ2]
    """
    k = EI / L**3
    return k * np.array([
        [ 12,   6*L,  -12,   6*L],
        [6*L, 4*L**2, -6*L, 2*L**2],
        [-12,  -6*L,   12,  -6*L],
        [6*L, 2*L**2, -6*L, 4*L**2],
    ], dtype=float)


def element_mass(mu: float, L: float) -> np.ndarray:
    """
    4x4 consistent mass matrix for Euler-Bernoulli beam element.
    """
    m = mu * L / 420.0
    return m * np.array([
        [156,   22*L,   54,  -13*L],
        [22*L,  4*L**2, 13*L, -3*L**2],
        [54,    13*L,  156,  -22*L],
        [-13*L, -3*L**2, -22*L, 4*L**2],
    ], dtype=float)


class CantileverBeam:
    """
    Assembled cantilever beam (wing) model.
    Nodes 0 … n_elements, node 0 is the fixed root.
    """

    def __init__(self, props: BeamProperties):
        self.props = props
        self.n_el = props.n_elements
        self.n_nodes = self.n_el + 1
        self.n_dof = 2 * self.n_nodes          # v, θ at each node
        self.L_el = props.length / self.n_el

        self.K = np.zeros((self.n_dof, self.n_dof))
        self.M = np.zeros((self.n_dof, self.n_dof))
        self._assemble()
        self._apply_bc()

    def _assemble(self) -> None:
        K_e = element_stiffness(self.props.EI, self.L_el)
        M_e = element_mass(self.props.mass_per_length, self.L_el)

        for e in range(self.n_el):
            # global DOF indices for this element
            i = 2 * e
            dofs = [i, i + 1, i + 2, i + 3]
            for a in range(4):
                for b in range(4):
                    self.K[dofs[a], dofs[b]] += K_e[a, b]
                    self.M[dofs[a], dofs[b]] += M_e[a, b]

    def _apply_bc(self) -> None:
        """Fix root node (v0 = θ0 = 0) by removing first two DOFs."""
        free = np.arange(2, self.n_dof)
        self.K_free = self.K[np.ix_(free, free)].copy()
        self.M_free = self.M[np.ix_(free, free)].copy()
        self.free_dofs = free
        self.n_free = len(free)

    def static_deflection(self, loads: np.ndarray) -> np.ndarray:
        """
        Solve K u = F for a static load vector defined on free DOFs.
        loads : array of length n_free (vertical forces and moments)
        Returns full displacement vector (including zeros at root).
        """
        u_free = np.linalg.solve(self.K_free, loads)
        u = np.zeros(self.n_dof)
        u[self.free_dofs] = u_free
        return u

    def tip_load_deflection(self, P: float) -> float:
        """
        Analytic check: tip vertical force P → tip deflection.
        Returns tip vertical displacement.
        """
        F = np.zeros(self.n_free)
        # tip vertical force is the last-but-one free DOF (v at tip)
        # free DOFs: v1,θ1,v2,θ2,…,vN,θN
        F[-2] = P
        u = self.static_deflection(F)
        return u[-2]  # tip vertical displacement

    def natural_frequencies(self, n_modes: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solve generalised eigenproblem K φ = ω² M φ.
        Returns frequencies [Hz] and mode-shape matrix (free DOFs).
        """
        # Use scipy for generalised eigenvalue problem
        from scipy.linalg import eigh
        w2, modes = eigh(self.K_free, self.M_free, subset_by_index=[0, n_modes - 1])
        freqs = np.sqrt(np.maximum(w2, 0.0)) / (2 * np.pi)  # Hz
        return freqs, modes

    def node_coordinates(self) -> np.ndarray:
        """Span-wise coordinates of nodes [0 … L]."""
        return np.linspace(0.0, self.props.length, self.n_nodes)


# Analytic references for validation
def analytic_tip_deflection(P: float, L: float, EI: float) -> float:
    """Cantilever tip deflection under tip load: δ = P L³ / (3 EI)"""
    return P * L**3 / (3.0 * EI)


def analytic_first_frequency(L: float, EI: float, mu: float) -> float:
    """
    First natural frequency of a uniform cantilever beam (Hz).
    ω1 = (1.875)^2 * sqrt(EI / (μ L^4))
    """
    beta = 1.87510407  # first root of cos β cosh β + 1 = 0
    omega = (beta ** 2) * np.sqrt(EI / (mu * L**4))
    return omega / (2 * np.pi)
