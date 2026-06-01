"""Core diagnostic primitives for the geometric phase space.

This module defines the dimensionless diagnostic axes, the electric and magnetic
Weyl normalisation, the derived Buchert backreaction, and the descriptive
classification. The axes form the vector

    X_cosmo = (I_rho, I_theta, I_sigma, I_E, I_B, I_H, I_M),

which is non-redundant in the algebraic sense that no Buchert-type derived
coordinate is carried as a separate axis. All Weyl axes use a single unified
curvature normalisation K_D = <theta^4>_D + eps.
"""
import numpy as np

EPS = 1e-12

# The five structure axes used for separability and classification. The two
# reliability indicators I_H and I_M are reported separately as constraint flags.
AXES = ["I_rho", "I_theta", "I_sigma", "I_E", "I_B"]

# Documented classification thresholds, stated explicitly in the manuscript.
# A time slice is flagged unreliable when either constraint residual exceeds
# RELIABILITY_TOL. Departures below DEPARTURE_FLOOR are FLRW-like, and departures
# below WEAK_TOL are labelled "weak". These are operational labels, not universal
# physical thresholds. The continuous diagnostic vector is the primary output.
RELIABILITY_TOL = 1e-3
DEPARTURE_FLOOR = 1e-4
WEAK_TOL = 1e-2


def domain_average(x, weights=None):
    """Domain average of a field, optionally weighted by the proper volume element."""
    x = np.asarray(x, float)
    if weights is None:
        return float(np.mean(x))
    w = np.asarray(weights, float)
    return float(np.sum(w * x) / (np.sum(w) + EPS))


def domain_variance(x, weights=None):
    """Domain variance of a field, optionally weighted."""
    x = np.asarray(x, float)
    mu = domain_average(x, weights)
    if weights is None:
        return float(np.mean((x - mu) ** 2))
    w = np.asarray(weights, float)
    return float(np.sum(w * (x - mu) ** 2) / (np.sum(w) + EPS))


def buchert_terms(theta, sigma2, weights=None):
    """Buchert kinematical backreaction Q_D and its variance and shear parts.

    Q_D = (2/3)(<theta^2> - <theta>^2) - 2<sigma^2>. The normalised magnitude
    |Q_D|/<theta>^2 is returned as I_Q_derived, and is algebraically fixed by
    I_theta and I_sigma, which is why Q_D is reported as derived rather than as a
    separate axis.
    """
    theta = np.asarray(theta, float)
    sigma2 = np.asarray(sigma2, float)
    mth = domain_average(theta, weights)
    mth2 = domain_average(theta ** 2, weights)
    var = mth2 - mth ** 2
    msig = domain_average(sigma2, weights)
    Qvar = (2.0 / 3.0) * var
    Qshear = -2.0 * msig
    Q = Qvar + Qshear
    den = mth ** 2 + EPS
    return {
        "Q_D": float(Q), "Q_var": float(Qvar), "Q_shear": float(Qshear),
        "I_Q_derived": float(abs(Q) / den),
        "I_Q_var": float(abs(Qvar) / den),
        "I_Q_shear": float(abs(Qshear) / den),
    }


def compute_diagnostics_EB(rho, theta, sigma2, E2, B2, H_constraint=0.0,
                           M_constraint=0.0, weights=None, curvature_scale=None):
    """Non-redundant diagnostic coordinates with the electric/magnetic Weyl split.

    Parameters
    ----------
    rho, theta, sigma2 : array_like
        Rest-mass density, expansion, and shear scalar sigma^2 = (1/2) sigma_ab sigma^ab.
    E2, B2 : array_like
        The scalars E_ab E^ab and B_ab B^ab over the domain.
    H_constraint, M_constraint : float
        Hamiltonian and momentum constraint residuals, reported as reliability flags.
    weights : array_like, optional
        Proper volume weights for the domain average.
    curvature_scale : float, optional
        Curvature normalisation. When None, the unified scale <theta^4>_D + eps is
        used for every benchmark, which makes I_E and I_B comparable across families.
    """
    rho = np.asarray(rho, float); theta = np.asarray(theta, float)
    sigma2 = np.asarray(sigma2, float); E2 = np.asarray(E2, float); B2 = np.asarray(B2, float)
    rm = domain_average(rho, weights); tm = domain_average(theta, weights)
    I_rho = domain_variance(rho, weights) / (rm ** 2 + EPS)
    I_theta = domain_variance(theta, weights) / (tm ** 2 + EPS)
    I_sigma = domain_average(sigma2, weights) / (tm ** 2 + EPS)
    if curvature_scale is None:
        curvature_scale = domain_average(theta ** 4, weights) + EPS
    mE = domain_average(E2, weights); mB = domain_average(B2, weights)
    I_E = mE / (curvature_scale + EPS); I_B = mB / (curvature_scale + EPS)
    C2 = 8.0 * (mE - mB)  # Weyl scalar C_abcd C^abcd
    out = {
        "I_rho": float(I_rho), "I_theta": float(I_theta), "I_sigma": float(I_sigma),
        "I_E": float(I_E), "I_B": float(I_B),
        "I_H": float(abs(H_constraint)), "I_M": float(abs(M_constraint)),
        "C2_weyl": float(C2), "rho_mean": float(rm), "theta_mean": float(tm),
    }
    out.update(buchert_terms(theta, sigma2, weights))
    return out


def background_kretschmann(H, Hdot):
    """Reference FLRW Kretschmann scale, retained only for the normalisation
    sensitivity check. The default normalisation is the unified scale <theta^4>_D."""
    return 12.0 * ((Hdot + H ** 2) ** 2 + H ** 4)


def background_kretschmann_dust(t):
    return background_kretschmann(2.0 / (3.0 * t), -2.0 / (3.0 * t ** 2))


def add_departure_score(df, score_name="D_FLRW"):
    """Add a min-max normalised mean of the five axes as a single departure score."""
    out = df.copy()
    for c in AXES:
        v = out[c].values
        rng = np.nanmax(v) - np.nanmin(v)
        out[c + "_norm"] = (v - np.nanmin(v)) / rng if rng > 1e-15 else np.zeros_like(v)
    out[score_name] = out[[c + "_norm" for c in AXES]].mean(axis=1)
    return out


def add_driver_decomposition(df, small_tol=DEPARTURE_FLOOR):
    """Add per-axis driver fractions and the dominant driver label."""
    out = df.copy()
    tot = out[AXES].sum(axis=1) + EPS
    for c in AXES:
        out[c + "_driver_fraction"] = out[c] / tot
    dom = (out[[c + "_driver_fraction" for c in AXES]].idxmax(axis=1)
           .str.replace("_driver_fraction", "", regex=False))
    out["dominant_driver"] = np.where(out[AXES].max(axis=1) < small_tol, "none", dom)
    return out


def classify_regime(row, rel=RELIABILITY_TOL, small=DEPARTURE_FLOOR, weak=WEAK_TOL):
    """Descriptive label from the dominant axis, with reliability and FLRW floors."""
    if row["I_H"] > rel or row["I_M"] > rel:
        return "numerically unreliable"
    vals = {
        "matter-inhomogeneous": row["I_rho"], "expansion-inhomogeneous": row["I_theta"],
        "anisotropy-dominated": row["I_sigma"], "electric-Weyl-dominated": row["I_E"],
        "magnetic-Weyl-dominated": row["I_B"],
    }
    lab = max(vals, key=vals.get); v = vals[lab]
    if v < small:
        return "FLRW-like"
    if v < weak:
        return "weak " + lab
    return lab


def loo_nearest_centroid(X, lab):
    """Leave-one-out nearest-centroid accuracy in the diagnostic space."""
    X = np.asarray(X, float); lab = np.asarray(lab)
    models = np.unique(lab); correct = 0
    for i in range(len(X)):
        cc = {}
        for m in models:
            mask = lab == m
            if m == lab[i]:
                idx = np.where(mask)[0]; idx = idx[idx != i]; cc[m] = X[idx].mean(0)
            else:
                cc[m] = X[mask].mean(0)
        pred = min(cc, key=lambda m: np.sum((X[i] - cc[m]) ** 2))
        correct += (pred == lab[i])
    return correct / len(X)
