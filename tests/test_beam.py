"""
Validation tests for the Euler-Bernoulli cantilever beam FEM.
"""

import numpy as np
import pytest

from src.structure.beam import (
    BeamProperties,
    CantileverBeam,
    analytic_tip_deflection,
    analytic_first_frequency,
)


@pytest.fixture
def uniform_beam():
    """A simple uniform aluminium-like wing spar."""
    props = BeamProperties(
        length=10.0,          # 10 m half-span
        EI=1.0e7,             # N.m²
        mass_per_length=50.0, # kg/m
        n_elements=20,
    )
    return CantileverBeam(props)


def test_tip_deflection_matches_analytic(uniform_beam):
    """Static tip load must recover the classic PL³/3EI formula."""
    P = 1000.0  # N
    numerical = uniform_beam.tip_load_deflection(P)
    analytic = analytic_tip_deflection(P, uniform_beam.props.length, uniform_beam.props.EI)
    rel_err = abs(numerical - analytic) / abs(analytic)
    assert rel_err < 1e-3, f"Tip deflection error {rel_err:.2e} > 0.1 %"


def test_first_natural_frequency(uniform_beam):
    """First bending frequency should be close to analytic value."""
    freqs, _ = uniform_beam.natural_frequencies(n_modes=3)
    f1_num = freqs[0]
    f1_ana = analytic_first_frequency(
        uniform_beam.props.length,
        uniform_beam.props.EI,
        uniform_beam.props.mass_per_length,
    )
    rel_err = abs(f1_num - f1_ana) / f1_ana
    # With 20 elements the error is typically < 0.5 %
    assert rel_err < 0.01, f"Frequency error {rel_err:.2e} > 1 %"


def test_stiffness_positive_definite(uniform_beam):
    """Free stiffness matrix must be positive definite."""
    eigvals = np.linalg.eigvalsh(uniform_beam.K_free)
    assert np.all(eigvals > 0), "K_free is not positive definite"


def test_mass_positive_definite(uniform_beam):
    """Consistent mass matrix must be positive definite."""
    eigvals = np.linalg.eigvalsh(uniform_beam.M_free)
    assert np.all(eigvals > 0), "M_free is not positive definite"


def test_mesh_refinement_improves_frequency():
    """Higher element count should reduce frequency error."""
    props_coarse = BeamProperties(length=10.0, EI=1e7, mass_per_length=50.0, n_elements=5)
    props_fine = BeamProperties(length=10.0, EI=1e7, mass_per_length=50.0, n_elements=40)
    beam_c = CantileverBeam(props_coarse)
    beam_f = CantileverBeam(props_fine)
    f_ana = analytic_first_frequency(10.0, 1e7, 50.0)
    err_c = abs(beam_c.natural_frequencies(1)[0][0] - f_ana) / f_ana
    err_f = abs(beam_f.natural_frequencies(1)[0][0] - f_ana) / f_ana
    assert err_f < err_c
