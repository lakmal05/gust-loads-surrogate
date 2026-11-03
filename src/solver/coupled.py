"""
Time-domain coupled aeroelastic solver for gust response.

Equation of motion (free DOFs):
    M ü + C ú + K u = F_aero(t, u, gust)

Integration: Newmark-β average acceleration (γ=½, β=¼).
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional

from src.structure.beam import CantileverBeam, BeamProperties, element_stiffness
from src.aero.strip import StripTheory, AeroProperties
from src.gust.discrete import GustProfile


@dataclass
class SimulationResult:
    time: np.ndarray
    tip_deflection: np.ndarray
    root_bending_moment: np.ndarray
    gust_velocity: np.ndarray
    peak_rbm: float
    time_of_peak: float


class CoupledGustSimulator:
    """
    Full-order aeroelastic simulation of a cantilever wing
    subjected to a discrete 1-cos gust.
    """

    def __init__(
        self,
        beam: CantileverBeam,
        aero: StripTheory,
        damping_ratio: float = 0.02,
    ):
        self.beam = beam
        self.aero = aero
        self.n_free = beam.n_free

        # Rayleigh damping tuned to first mode
        freqs, _ = beam.natural_frequencies(1)
        omega1 = 2 * np.pi * max(freqs[0], 1e-3)
        alpha = 2 * damping_ratio * omega1
        self.C = alpha * beam.M_free

        self.M = beam.M_free.copy()
        self.K = beam.K_free.copy()

    def _root_bending_moment(self, u_full: np.ndarray) -> float:
        """Recover root bending moment from first-element internal forces."""
        u = np.zeros(self.beam.n_dof)
        u[self.beam.free_dofs] = u_full
        L = self.beam.L_el
        EI = self.beam.props.EI
        Ke = element_stiffness(EI, L)
        u_e = np.array([0.0, 0.0, u[2], u[3]])
        f_e = Ke @ u_e
        return float(f_e[1])  # moment reaction at root

    def simulate(
        self,
        gust: GustProfile,
        t_end: Optional[float] = None,
        dt: float = 0.002,
        n_steps: Optional[int] = None,
    ) -> SimulationResult:
        """
        Newmark-β average-acceleration scheme.
        """
        if t_end is None:
            t_end = max(gust.duration() * 3.0, 2.0)
        if n_steps is None:
            n_steps = int(np.ceil(t_end / dt))
        t = np.linspace(0.0, t_end, n_steps + 1)
        dt = float(t[1] - t[0])

        gamma = 0.5
        beta = 0.25

        # Effective mass matrix (constant)
        M_eff = self.M + gamma * dt * self.C + beta * (dt ** 2) * self.K
        M_eff_inv = np.linalg.inv(M_eff)

        u = np.zeros(self.n_free)
        v = np.zeros(self.n_free)
        a = np.zeros(self.n_free)

        tip_hist = np.zeros(n_steps + 1)
        rbm_hist = np.zeros(n_steps + 1)
        gust_hist = np.zeros(n_steps + 1)

        for i in range(n_steps):
            ti = t[i + 1]

            wg = gust.velocity(ti)
            alpha_g = np.full(self.aero.n_el, wg / max(gust.V, 1e-6))

            # Explicit aero force evaluated with displacement at start of step
            F = self.aero.aero_force(u, gust_alpha=alpha_g)

            # Predictors
            u_pred = u + dt * v + (0.5 - beta) * (dt ** 2) * a
            v_pred = v + (1.0 - gamma) * dt * a

            # Solve for new acceleration
            rhs = F - self.C @ v_pred - self.K @ u_pred
            a_new = M_eff_inv @ rhs

            # Correctors
            u_new = u_pred + beta * (dt ** 2) * a_new
            v_new = v_pred + gamma * dt * a_new

            u, v, a = u_new, v_new, a_new

            tip_hist[i + 1] = u[-2]
            rbm_hist[i + 1] = self._root_bending_moment(u)
            gust_hist[i + 1] = wg

        peak_idx = int(np.argmax(np.abs(rbm_hist)))
        return SimulationResult(
            time=t,
            tip_deflection=tip_hist,
            root_bending_moment=rbm_hist,
            gust_velocity=gust_hist,
            peak_rbm=float(rbm_hist[peak_idx]),
            time_of_peak=float(t[peak_idx]),
        )


def default_wing() -> Tuple[CantileverBeam, StripTheory]:
    """Representative high-aspect-ratio wing."""
    props = BeamProperties(
        length=15.0,
        EI=2.5e7,
        mass_per_length=40.0,
        n_elements=12,
    )
    beam = CantileverBeam(props)
    aero_props = AeroProperties(rho=1.225, V=80.0, chord=2.0, Cl_alpha=2 * np.pi)
    aero = StripTheory(aero_props, n_elements=props.n_elements, span=props.length)
    return beam, aero
