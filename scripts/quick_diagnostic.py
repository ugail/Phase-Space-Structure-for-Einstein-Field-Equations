#!/usr/bin/env python3
"""A minimal worked example.

Shows two things: how to compute the diagnostic vector for a built-in benchmark,
and how to drop in your own fields. Run from the repository root after installing
the package with `pip install -e .`.

    python scripts/quick_diagnostic.py
"""
import numpy as np

from phase_space_efe import compute_diagnostics_EB, AXES
from phase_space_efe.benchmarks import simulate_ltb, bianchi_weyl_C2


def example_builtin():
    """Diagnostic vector for the LTB benchmark at its final sampled time."""
    times = np.linspace(0.7, 5.0, 80)
    row = simulate_ltb(times).iloc[-1]
    print("LTB benchmark, final time t = %.3f" % row["t"])
    for c in AXES:
        print(f"  {c:8s} = {row[c]:.3e}")
    print(f"  dominant axis: {max(AXES, key=lambda a: row[a])}")


def example_custom():
    """Audit your own domain fields.

    Supply the density, expansion, shear scalar sigma^2 = (1/2) sigma_ab sigma^ab,
    and the Weyl scalars E_ab E^ab and B_ab B^ab over the averaging domain. Pass
    proper-volume weights if the domain is not uniform. With curvature_scale=None
    the unified normalisation K_D = <theta^4>_D is used.
    """
    n = 400
    rng = np.random.default_rng(0)
    rho = 1.0 + 0.05 * rng.standard_normal(n)   # mild matter inhomogeneity
    theta = np.full(n, 1.5)                      # uniform expansion
    sigma2 = np.zeros(n)                         # no shear
    E2 = np.full(n, 1e-3); B2 = np.zeros(n)      # purely electric Weyl
    d = compute_diagnostics_EB(rho, theta, sigma2, E2, B2)
    print("\nCustom field audit:")
    for c in AXES:
        print(f"  {c:8s} = {d[c]:.3e}")
    print(f"  Weyl scalar C^2 = 8(E^2 - B^2) = {d['C2_weyl']:.3e}")


if __name__ == "__main__":
    example_builtin()
    example_custom()
