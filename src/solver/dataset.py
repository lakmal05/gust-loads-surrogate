"""
Generate a design-of-experiments dataset of peak gust loads
using Latin Hypercube / Sobol sampling + multiprocessing.
"""

from __future__ import annotations

import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import qmc

from src.structure.beam import BeamProperties, CantileverBeam
from src.aero.strip import AeroProperties, StripTheory
from src.gust.discrete import GustProfile
from src.solver.coupled import CoupledGustSimulator


@dataclass
class ParameterBounds:
    """Ranges for the design variables."""
    H: Tuple[float, float] = (30.0, 300.0)       # gust length [m]
    U_ds: Tuple[float, float] = (5.0, 25.0)      # gust amplitude [m/s]
    V: Tuple[float, float] = (40.0, 120.0)       # airspeed [m/s]
    rho: Tuple[float, float] = (0.4, 1.225)      # density (altitude proxy)
    mass_factor: Tuple[float, float] = (0.7, 1.3) # mass scaling


def _run_single(params: Dict) -> Dict:
    """Worker function – must be picklable (top-level)."""
    H = params["H"]
    U_ds = params["U_ds"]
    V = params["V"]
    rho = params["rho"]
    mass_factor = params["mass_factor"]

    props = BeamProperties(
        length=15.0,
        EI=2.5e7,
        mass_per_length=40.0 * mass_factor,
        n_elements=10,  # slightly coarser for speed
    )
    beam = CantileverBeam(props)
    aero_props = AeroProperties(rho=rho, V=V, chord=2.0, Cl_alpha=2 * np.pi)
    aero = StripTheory(aero_props, n_elements=props.n_elements, span=props.length)
    sim = CoupledGustSimulator(beam, aero, damping_ratio=0.04)

    gust = GustProfile(U_ds=U_ds, H=H, V=V)
    # adaptive time step roughly proportional to shortest period
    dt = min(0.003, gust.duration() / 80.0)
    result = sim.simulate(gust, dt=dt)

    return {
        **params,
        "peak_rbm": result.peak_rbm,
        "time_of_peak": result.time_of_peak,
        "max_tip": float(np.max(np.abs(result.tip_deflection))),
    }


def generate_dataset(
    n_samples: int = 500,
    n_workers: int = 4,
    seed: int = 42,
    method: str = "lhs",
    out_path: str = "data/gust_loads_dataset.csv",
) -> pd.DataFrame:
    """
    Create a Latin-Hypercube (or Sobol) sample and evaluate the solver in parallel.
    """
    bounds = ParameterBounds()
    names = ["H", "U_ds", "V", "rho", "mass_factor"]
    low = np.array([getattr(bounds, n)[0] for n in names])
    high = np.array([getattr(bounds, n)[1] for n in names])

    if method == "sobol":
        sampler = qmc.Sobol(d=len(names), scramble=True, seed=seed)
        # Sobol prefers power-of-2; we take the next power and trim
        m = int(np.ceil(np.log2(max(n_samples, 2))))
        sample = sampler.random_base2(m=m)[:n_samples]
    else:
        sampler = qmc.LatinHypercube(d=len(names), seed=seed)
        sample = sampler.random(n=n_samples)

    sample = qmc.scale(sample, low, high)
    param_list = [dict(zip(names, row)) for row in sample]

    results: List[Dict] = []
    t0 = time.time()
    print(f"Running {n_samples} cases with {n_workers} workers …")

    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_run_single, p): i for i, p in enumerate(param_list)}
        for fut in as_completed(futures):
            idx = futures[fut]
            try:
                res = fut.result()
                results.append(res)
                if len(results) % 50 == 0:
                    print(f"  {len(results)}/{n_samples} done")
            except Exception as exc:
                print(f"  case {idx} failed: {exc}")

    elapsed = time.time() - t0
    print(f"Finished in {elapsed:.1f} s  ({elapsed/max(len(results),1):.2f} s/case)")

    df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Saved → {out_path}  ({len(df)} rows)")
    return df


def split_dataset(
    df: pd.DataFrame,
    test_frac: float = 0.15,
    edge_frac: float = 0.05,
    seed: int = 0,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split into train / val / test, keeping a separate edge-case test set
    (points near the boundary of the parameter space).
    """
    rng = np.random.default_rng(seed)
    # normalise features to [0,1] to detect edges
    feat_cols = ["H", "U_ds", "V", "rho", "mass_factor"]
    X = df[feat_cols].values
    Xn = (X - X.min(0)) / (X.max(0) - X.min(0) + 1e-12)
    # edge = any coordinate < edge_frac or > 1-edge_frac
    is_edge = np.any((Xn < edge_frac) | (Xn > 1 - edge_frac), axis=1)

    edge_df = df[is_edge].copy()
    core_df = df[~is_edge].copy()

    # further split core into train/val/test
    n = len(core_df)
    idx = rng.permutation(n)
    n_test = int(test_frac * n)
    n_val = int(0.15 * n)
    test_idx = idx[:n_test]
    val_idx = idx[n_test:n_test + n_val]
    train_idx = idx[n_test + n_val:]

    train = core_df.iloc[train_idx].reset_index(drop=True)
    val = core_df.iloc[val_idx].reset_index(drop=True)
    # final test = random core test + all edge cases
    test = pd.concat([core_df.iloc[test_idx], edge_df], ignore_index=True)

    return train, val, test


if __name__ == "__main__":
    # small demo set
    df = generate_dataset(n_samples=40, n_workers=2, out_path="data/demo_dataset.csv")
    print(df.describe())
