"""phase_space_efe: a geometric diagnostic phase space for inhomogeneous cosmological dynamics.

The toolkit organises standard kinematic, curvature, and constraint quantities into
a compact, non-redundant set of dimensionless diagnostics with an explicit electric
and magnetic Weyl split, and compares analytic and perturbative cosmological
solution families as points and trajectories in one diagnostic space.
"""
from .core import (
    EPS, AXES, RELIABILITY_TOL, DEPARTURE_FLOOR, WEAK_TOL,
    domain_average, domain_variance, buchert_terms, compute_diagnostics_EB,
    background_kretschmann, background_kretschmann_dust,
    add_departure_score, add_driver_decomposition, classify_regime, loo_nearest_centroid,
)
from . import benchmarks
from . import figures
from . import pipeline

__version__ = "1.0.0"

__all__ = [
    "EPS", "AXES", "RELIABILITY_TOL", "DEPARTURE_FLOOR", "WEAK_TOL",
    "domain_average", "domain_variance", "buchert_terms", "compute_diagnostics_EB",
    "background_kretschmann", "background_kretschmann_dust",
    "add_departure_score", "add_driver_decomposition", "classify_regime",
    "loo_nearest_centroid", "benchmarks", "figures", "pipeline", "__version__",
]
