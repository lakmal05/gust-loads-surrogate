"""
Basic smoke + sanity tests for the coupled gust simulation.
"""

import numpy as np
import pytest

from src.solver.coupled import CoupledGustSimulator, default_wing
from src.gust.discrete import GustProfile


@pytest.fixture
def simulator():
    beam, aero = default_wing()
    return CoupledGustSimulator(beam, aero, damping_ratio=0.03)


def test_gust_runs_without_crash(simulator):
    gust = GustProfile(U_ds=10.0, H=50.0, V=80.0)
    result = simulator.simulate(gust, dt=0.005)
    assert result.time is not None
    assert len(result.tip_deflection) == len(result.time)
    assert np.isfinite(result.peak_rbm)


def test_zero_gust_gives_near_zero_response(simulator):
    """With U_ds = 0 the response should stay essentially zero."""
    gust = GustProfile(U_ds=0.0, H=50.0, V=80.0)
    result = simulator.simulate(gust, t_end=1.0, dt=0.01)
    assert np.max(np.abs(result.tip_deflection)) < 1e-6
    assert np.max(np.abs(result.root_bending_moment)) < 1e-3


def test_larger_gust_gives_larger_load(simulator):
    g1 = GustProfile(U_ds=5.0, H=50.0, V=80.0)
    g2 = GustProfile(U_ds=15.0, H=50.0, V=80.0)
    r1 = simulator.simulate(g1, dt=0.005)
    r2 = simulator.simulate(g2, dt=0.005)
    assert abs(r2.peak_rbm) > abs(r1.peak_rbm)
