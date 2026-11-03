"""
Worst-case gust search on the surrogate using a simple Bayesian-style loop
(Gaussian-process free: we use the neural ensemble mean + UCB acquisition).
"""

from __future__ import annotations

from typing import Callable, Dict, Tuple

import numpy as np

from src.surrogate.models import DeepEnsemble, FEATURE_NAMES
from src.solver.dataset import ParameterBounds


def expected_improvement(mean: np.ndarray, std: np.ndarray, best: float, xi: float = 0.01) -> np.ndarray:
    """EI for maximisation of |load| (we work with absolute peak)."""
    from scipy.stats import norm
    # we maximise the absolute value of the predicted load
    # treat "best" as the best (largest) |load| seen so far
    improvement = mean - best - xi
    Z = improvement / (std + 1e-12)
    return improvement * norm.cdf(Z) + std * norm.pdf(Z)


def bayesian_search(
    surrogate: DeepEnsemble,
    n_iter: int = 40,
    n_candidates: int = 500,
    seed: int = 0,
) -> Dict:
    """
    Sequential search that maximises the predicted absolute peak RBM.
    At each step we draw random candidates, pick the one with highest UCB,
    evaluate the surrogate, and update the "observed" best.
    """
    bounds = ParameterBounds()
    rng = np.random.default_rng(seed)
    names = FEATURE_NAMES
    low = np.array([getattr(bounds, n)[0] for n in names])
    high = np.array([getattr(bounds, n)[1] for n in names])

    # initial random observations
    X_obs = low + (high - low) * rng.random((8, len(names)))
    mean, std = surrogate.predict(X_obs)
    y_obs = np.abs(mean)  # we care about magnitude

    history = []
    for it in range(n_iter):
        # random candidate pool
        X_cand = low + (high - low) * rng.random((n_candidates, len(names)))
        m, s = surrogate.predict(X_cand)
        # Upper Confidence Bound (exploratory)
        ucb = np.abs(m) + 1.5 * s
        best_idx = int(np.argmax(ucb))
        x_next = X_cand[best_idx]
        m_next, s_next = m[best_idx], s[best_idx]

        X_obs = np.vstack([X_obs, x_next])
        y_obs = np.append(y_obs, np.abs(m_next))
        history.append({
            "iter": it,
            **dict(zip(names, x_next)),
            "pred_abs_rbm": float(np.abs(m_next)),
            "pred_std": float(s_next),
        })

    best_i = int(np.argmax(y_obs))
    best = {
        **dict(zip(names, X_obs[best_i])),
        "pred_abs_rbm": float(y_obs[best_i]),
    }
    return {"best": best, "history": history}


def verify_with_solver(params: Dict) -> float:
    """Re-evaluate the found point with the high-fidelity solver."""
    from src.structure.beam import BeamProperties, CantileverBeam
    from src.aero.strip import AeroProperties, StripTheory
    from src.gust.discrete import GustProfile
    from src.solver.coupled import CoupledGustSimulator

    props = BeamProperties(
        length=15.0,
        EI=2.5e7,
        mass_per_length=40.0 * params["mass_factor"],
        n_elements=12,
    )
    beam = CantileverBeam(props)
    aero = StripTheory(
        AeroProperties(rho=params["rho"], V=params["V"], chord=2.0, Cl_alpha=2 * np.pi),
        n_elements=props.n_elements,
        span=props.length,
    )
    sim = CoupledGustSimulator(beam, aero, damping_ratio=0.04)
    gust = GustProfile(U_ds=params["U_ds"], H=params["H"], V=params["V"])
    res = sim.simulate(gust, dt=0.002)
    return float(res.peak_rbm)
