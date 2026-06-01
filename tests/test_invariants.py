"""Structural invariant tests for the diagnostic framework.

These verify the analytic identities and the qualitative signatures that the
manuscript relies on: the FLRW baseline, the Bianchi-I and Kasner Weyl values,
the LTB closed form and its homogeneous limit, the Buchert redundancy identity,
the magnetic-axis activation, and the observer-tilt stability.
"""
import numpy as np
import pytest

from phase_space_efe import compute_diagnostics_EB, AXES, EPS
from phase_space_efe.benchmarks import (
    bianchi_weyl_C2, simulate_flrw, simulate_bianchi, simulate_kasner,
    simulate_ltb, simulate_gw_flrw, ltb_tilt,
)


def test_flrw_all_axes_zero():
    df = simulate_flrw(np.linspace(0.7, 5.0, 20))
    for c in AXES:
        assert df[c].abs().max() < 1e-10


def test_bianchi_isotropic_limit_zero():
    # Equal exponents are isotropic, so the Weyl invariant vanishes.
    assert abs(bianchi_weyl_C2(1.0, (2 / 3, 2 / 3, 2 / 3))) < 1e-12


def test_kasner_weyl_value():
    # Vacuum Kasner has the known value C^2 t^4 = 64/27 for (2/3, 2/3, -1/3).
    val = bianchi_weyl_C2(1.0, (2 / 3, 2 / 3, -1 / 3))
    assert val == pytest.approx(64.0 / 27.0, rel=1e-9)


def test_bianchi_magnetic_axis_zero():
    df = simulate_bianchi(np.linspace(0.7, 5.0, 20))
    assert df["I_B"].abs().max() < 1e-12


def test_kasner_shear_dominates_and_electric_present():
    row = simulate_kasner(np.linspace(0.7, 5.0, 20)).iloc[-1]
    assert max(AXES, key=lambda a: row[a]) == "I_sigma"
    assert row["I_E"] > 0 and row["I_B"] < 1e-12


def test_ltb_closed_form_positive_and_matter_dominant():
    row = simulate_ltb(np.linspace(0.7, 5.0, 20)).iloc[-1]
    assert row["I_rho"] > 0 and row["I_E"] > 0
    assert row["I_B"] < 1e-12
    assert max(AXES, key=lambda a: row[a]) == "I_rho"


def test_ltb_homogeneous_limit_vanishes():
    # With a vanishing bang-time profile the model is homogeneous: every axis -> 0.
    row = simulate_ltb(np.array([1.0]), tb_amp=1e-9).iloc[0]
    for c in AXES:
        assert row[c] < 1e-6


def test_buchert_redundancy_identity():
    # |Q_D|/<theta>^2 == |(2/3) I_theta - 2 I_sigma| to round-off.
    rng = np.random.default_rng(1)
    theta = 1.5 + 0.2 * rng.standard_normal(500)
    sigma2 = np.abs(0.01 * rng.standard_normal(500))
    d = compute_diagnostics_EB(np.ones(500), theta, sigma2, np.zeros(500), np.zeros(500))
    recon = abs((2 / 3) * d["I_theta"] - 2 * d["I_sigma"])
    assert abs(d["I_Q_derived"] - recon) < 1e-12


def test_unified_scale_C2_relation():
    # The reported Weyl scalar equals 8(<E^2> - <B^2>).
    E2 = np.full(10, 0.7); B2 = np.full(10, 0.3)
    d = compute_diagnostics_EB(np.ones(10), np.ones(10), np.zeros(10), E2, B2)
    assert d["C2_weyl"] == pytest.approx(8.0 * (0.7 - 0.3), rel=1e-12)


def test_gw_magnetic_axis_active():
    df = simulate_gw_flrw(np.linspace(1.5, 3.5, 6))
    assert df["I_B"].min() > 0
    # electric and magnetic parts are comparable (not exactly equal) on the background
    rel = (df["I_E"] - df["I_B"]).abs().max() / max(df["I_E"].max(), df["I_B"].max())
    assert 0.0 < rel < 0.5


def test_observer_tilt_keeps_dominant_axis_and_zero_magnetic():
    for v0 in (0.0, 0.01, 0.05, 0.1):
        d = ltb_tilt(v0)
        assert max(AXES, key=lambda a: d[a]) == "I_rho"
        assert d["I_B"] < 1e-12
