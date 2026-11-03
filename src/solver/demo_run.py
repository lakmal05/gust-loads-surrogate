"""
Quick demonstration: run one gust case and save plots.
"""

from __future__ import annotations

import os
import numpy as np
import matplotlib.pyplot as plt

from src.solver.coupled import CoupledGustSimulator, default_wing
from src.gust.discrete import GustProfile


def main():
    beam, aero = default_wing()
    sim = CoupledGustSimulator(beam, aero, damping_ratio=0.03)

    gust = GustProfile(U_ds=12.0, H=80.0, V=80.0)
    result = sim.simulate(gust, dt=0.002)

    out_dir = os.path.join(os.path.dirname(__file__), "..", "..", "data")
    os.makedirs(out_dir, exist_ok=True)

    fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)

    axes[0].plot(result.time, result.gust_velocity, "b-")
    axes[0].set_ylabel("Gust vel. [m/s]")
    axes[0].set_title(f"1-cos gust  H={gust.H} m, Uds={gust.U_ds} m/s")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(result.time, result.tip_deflection, "g-")
    axes[1].set_ylabel("Tip defl. [m]")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(result.time, result.root_bending_moment / 1e3, "r-")
    axes[2].axvline(result.time_of_peak, color="k", ls="--", alpha=0.5,
                    label=f"peak @ {result.time_of_peak:.2f}s")
    axes[2].set_ylabel("Root BM [kN·m]")
    axes[2].set_xlabel("Time [s]")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout()
    plot_path = os.path.join(out_dir, "demo_gust_response.png")
    fig.savefig(plot_path, dpi=120)
    print(f"Saved plot → {plot_path}")
    print(f"Peak root bending moment = {result.peak_rbm/1e3:.2f} kN·m "
          f"at t = {result.time_of_peak:.3f} s")
    plt.close(fig)


if __name__ == "__main__":
    main()
