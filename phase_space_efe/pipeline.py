"""Full reproduction pipeline.

`run(out_dir)` evaluates the six benchmarks over their dynamical histories,
assembles the diagnostic frame, runs the classification, the robustness sweeps,
the averaging-domain study, the constraint-contamination experiment, the
observer-tilt test, and the principal component projection with the label-free
separability measures. It writes every table, the run metadata, and every figure
to `out_dir`, and copies the six manuscript figures into a `figures_for_paper`
folder. Output is local, with no dependence on Google Drive or Colab.
"""
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.linalg import svd

from . import benchmarks as bm
from . import figures as fg
from .core import (AXES, EPS, RELIABILITY_TOL, DEPARTURE_FLOOR, WEAK_TOL,
                   add_departure_score, add_driver_decomposition, classify_regime,
                   loo_nearest_centroid)

# The six figures referenced by the manuscript, under their paper filenames.
PAPER_FIGURES = [
    "Fig_heatmap_final_time", "D1b_magnetic_weyl_validation", "C1_pca_phase_space",
    "B1_bianchi_anisotropy_sweep", "B2_ltb_amplitude_sweep", "B3_ltb_resolution",
]


def build_diagnostics(times=None, gw_times=None, seed=7):
    """Run the six benchmarks and return the assembled, classified diagnostic frame."""
    np.random.seed(seed)
    if times is None:
        times = np.linspace(0.7, 5.0, 80)
    if gw_times is None:
        gw_times = np.linspace(1.5, 3.5, 40)
    df_flrw = bm.simulate_flrw(times)
    df_bianchi = bm.simulate_bianchi(times)
    df_ltb = bm.simulate_ltb(times)
    df_pflrw = bm.simulate_perturbed_flrw_consistent(times)
    df_kasner = bm.simulate_kasner(times)
    df_gw = bm.simulate_gw_flrw(gw_times)
    df = pd.concat([df_flrw, df_bianchi, df_ltb, df_pflrw, df_kasner, df_gw],
                   ignore_index=True)
    df = add_departure_score(df)
    df = add_driver_decomposition(df)
    df["regime"] = df.apply(classify_regime, axis=1)
    df["I_Q_recon"] = np.abs((2 / 3) * df["I_theta"] - 2 * df["I_sigma"])
    return df, df_bianchi


def add_pca(df):
    """Standardised two-component PCA projection (SVD) appended to the frame."""
    Xmat = df[AXES].values.astype(float)
    Xs = (Xmat - Xmat.mean(0)) / (Xmat.std(0) + 1e-12)
    U, S, Vt = svd(Xs, full_matrices=False)
    PC = Xs @ Vt[:2].T
    df["PC1"] = PC[:, 0]; df["PC2"] = PC[:, 1]
    evr = (S[:2] ** 2 / (S ** 2).sum())
    return df, Xs, evr


def separability(df, Xs):
    """Label-free separability metrics, full history and developed late-time regime."""
    try:
        from sklearn.metrics import silhouette_score
    except Exception:
        silhouette_score = None
    labels = df["model"].values
    late = (df["t"] >= df["t"].median()).values
    res = {}
    if silhouette_score is not None:
        res["silhouette_all"] = float(silhouette_score(Xs, labels))
        res["silhouette_late"] = float(silhouette_score(Xs[late], labels[late]))
    res["loo_nc_all"] = float(loo_nearest_centroid(Xs, labels))
    res["loo_nc_late"] = float(loo_nearest_centroid(Xs[late], labels[late]))
    return res


def run(out_dir=None, verbose=True):
    """Run the full pipeline and write all tables, figures, and metadata to out_dir."""
    out_dir = Path(out_dir) if out_dir is not None else Path("Results")
    fig_dir = out_dir / "figures"; tab_dir = out_dir / "tables"; met_dir = out_dir / "metadata"
    for d in (out_dir, fig_dir, tab_dir, met_dir):
        d.mkdir(parents=True, exist_ok=True)
    fg.set_pub_style()

    def log(*a):
        if verbose:
            print(*a)

    log("Running benchmarks ...")
    df, df_bianchi = build_diagnostics()
    df, Xs, evr = add_pca(df)
    sep = separability(df, Xs)

    log("Running robustness studies ...")
    df_bsweep = bm.bianchi_anisotropy_sweep()
    df_lsweep = bm.ltb_amplitude_sweep()
    df_res = bm.ltb_resolution_study()
    df_dom = bm.domain_dependence_study()
    df_contam = bm.constraint_contamination(df_bianchi.iloc[-1].to_dict())
    df_tilt = bm.observer_tilt_study()

    # ---- tables ----
    df.to_csv(tab_dir / "all_diagnostics.csv", index=False)
    for nm, d in [("bianchi_sweep", df_bsweep), ("ltb_amplitude_sweep", df_lsweep),
                  ("ltb_resolution", df_res), ("domain_dependence", df_dom),
                  ("constraint_contamination", df_contam), ("observer_tilt", df_tilt),
                  ("pca_projection", df[["model", "t", "PC1", "PC2"]])]:
        d.to_csv(tab_dir / f"{nm}.csv", index=False)
    pd.crosstab(df["model"], df["regime"]).to_csv(tab_dir / "regime_counts.csv")
    pd.crosstab(df["model"], df["dominant_driver"]).to_csv(tab_dir / "dominant_driver_counts.csv")

    # ---- figures ----
    log("Generating figures ...")
    fg.fig_heatmap(df, fig_dir)
    fg.fig_pca(df, fig_dir)
    fg.fig_magnetic_validation(df, fig_dir)
    fg.fig_components_over_time(df, fig_dir)
    fg.fig_bianchi_sweep(df_bsweep, fig_dir)
    fg.fig_ltb_amplitude_sweep(df_lsweep, fig_dir)
    fg.fig_ltb_resolution(df_res, fig_dir)

    # ---- manuscript figure bundle ----
    paper_dir = fig_dir / "figures_for_paper"; paper_dir.mkdir(exist_ok=True)
    for name in PAPER_FIGURES:
        src = fig_dir / f"{name}.png"
        if src.exists():
            (paper_dir / src.name).write_bytes(src.read_bytes())

    # ---- metadata ----
    iq_err = float((df["I_Q_derived"] - df["I_Q_recon"]).abs().max())
    gw = df[df["model"] == "GW-FLRW"]
    gw_rel = float((gw["I_E"] - gw["I_B"]).abs().max() / max(gw["I_E"].max(), gw["I_B"].max()))
    meta = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "diagnostic_axes": ["I_rho", "I_theta", "I_sigma", "I_E", "I_B", "I_H", "I_M"],
        "thresholds": {"reliability_tol": RELIABILITY_TOL,
                       "departure_floor": DEPARTURE_FLOOR, "weak_tol": WEAK_TOL, "eps": EPS},
        "pca_explained_variance_ratio_first2": [float(x) for x in evr],
        "separability": sep,
        "backreaction_identity_max_abs_error": iq_err,
        "gw_electric_magnetic_comparability": gw_rel,
        "notes": [
            "Weyl split into electric/magnetic; C^2=8(E^2-B^2).",
            "Unified curvature normalisation K_D=<theta^4>_D+eps used for all benchmarks.",
            "Axes are non-redundant (no derived Buchert axis), not dynamically independent.",
            "I_B nonzero only for the tensor (GW) benchmark; zero (round-off) for all others.",
            "GW electric/magnetic comparability: E^2~B^2, C^2 suppressed; I_B>0.",
            "Observer-tilt (LTB): dominant axis stays I_rho, I_B stays 0 up to v0=0.1.",
        ],
    }
    (met_dir / "run_metadata.json").write_text(json.dumps(meta, indent=2))

    # ---- archive ----
    zip_path = out_dir / "results_archive.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(out_dir.rglob("*")):
            if p.is_file() and p != zip_path:
                zf.write(p, p.relative_to(out_dir))

    log(f"explained variance (first two PCs): {evr.round(3).tolist()}")
    log(f"separability: {sep}")
    log(f"backreaction identity max abs error: {iq_err:.2e}")
    log(f"GW electric/magnetic comparability: {gw_rel:.3f}")
    log(f"Done. Wrote tables, figures, and metadata to {out_dir}")
    return df
