"""Publication-quality figure generation.

Every figure is written as a 300 dpi PNG using Computer-Modern math text. The
functions take the assembled diagnostic frame (and the robustness tables where
needed) plus an output directory, and return the path written.
"""
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.ticker import AutoMinorLocator
from matplotlib.lines import Line2D

from .core import AXES

PALETTE = {
    "FLRW": "#444444", "Bianchi-I": "#1b7837", "LTB": "#1f4e9e",
    "Perturbed-FLRW": "#c8430f", "Kasner": "#7b3294", "GW-FLRW": "#d6a000",
}
MODEL_ORDER = ["FLRW", "Bianchi-I", "Kasner", "LTB", "Perturbed-FLRW", "GW-FLRW"]


def set_pub_style():
    """Apply the publication plotting style (serif, Computer-Modern math, 300 dpi)."""
    plt.rcParams.update({
        "figure.dpi": 120, "savefig.dpi": 300,
        "font.family": "serif", "mathtext.fontset": "cm", "font.size": 12,
        "axes.titlesize": 13, "axes.labelsize": 15, "legend.fontsize": 12,
        "xtick.labelsize": 12, "ytick.labelsize": 12,
        "axes.linewidth": 0.9, "lines.linewidth": 2.0,
        "xtick.direction": "in", "ytick.direction": "in",
        "xtick.top": True, "ytick.right": True,
        "xtick.major.size": 5, "ytick.major.size": 5,
        "xtick.minor.size": 2.8, "ytick.minor.size": 2.8,
        "legend.frameon": True, "legend.framealpha": 0.96,
        "legend.edgecolor": "0.7", "axes.axisbelow": True,
    })


def style_axes(ax):
    if ax.get_xscale() == "linear":
        ax.xaxis.set_minor_locator(AutoMinorLocator())
    if ax.get_yscale() == "linear":
        ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.grid(True, which="major", color="0.85", lw=0.6)


def _save(fig, out_dir, name, **kw):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.png"
    fig.savefig(path, dpi=300, bbox_inches="tight", **kw)
    plt.close(fig)
    return path


def fig_heatmap(df, out_dir):
    """Manuscript Fig. 1: final-time diagnostic profile across benchmarks."""
    labels = [r"$I_\rho$", r"$I_\theta$", r"$I_\sigma$", r"$I_E$", r"$I_B$"]
    fin = df.sort_values("t").groupby("model").tail(1).set_index("model").reindex(MODEL_ORDER)
    H = np.array([[max(float(fin.loc[m, c]), 1e-12) for c in AXES] for m in MODEL_ORDER])
    Hlog = np.clip(np.log10(H), -9, 0)
    fig, ax = plt.subplots(figsize=(7.2, 4.6), constrained_layout=True)
    im = ax.imshow(Hlog, aspect="auto", cmap="viridis", vmin=-9, vmax=0)
    ax.set_xticks(range(len(AXES))); ax.set_xticklabels(labels, fontsize=15)
    ax.set_yticks(range(len(MODEL_ORDER))); ax.set_yticklabels(MODEL_ORDER, fontsize=12)
    ax.set_xticks(np.arange(-0.5, len(AXES), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(MODEL_ORDER), 1), minor=True)
    ax.grid(which="minor", color="white", lw=1.3); ax.tick_params(which="minor", length=0)
    ax.tick_params(which="major", length=0)
    for i, m in enumerate(MODEL_ORDER):
        for j, c in enumerate(AXES):
            v = float(fin.loc[m, c])
            txt = "0" if v < 1e-11 else (f"{v:.1e}" if v < 0.1 else f"{v:.2f}")
            ax.text(j, i, txt, ha="center", va="center", fontsize=9.5,
                    color="white" if Hlog[i, j] < -3.2 else "black")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cb.set_label(r"$\log_{10}$ (diagnostic amplitude)", fontsize=12)
    return _save(fig, out_dir, "Fig_heatmap_final_time")


def fig_pca(df, out_dir):
    """Two-panel PCA trajectory plot (classifier-free visual separability check)."""
    def fade(color, n, light=0.80):
        base = np.array(mcolors.to_rgb(color))
        f = np.linspace(light, 0.0, max(n, 1))[:, None]
        return (1.0 - f) * base[None, :] + f * np.ones(3)[None, :]

    order = MODEL_ORDER
    mark = {"FLRW": "o", "Bianchi-I": "s", "Kasner": "D",
            "LTB": "o", "Perturbed-FLRW": "^", "GW-FLRW": "v"}
    ls = {"Perturbed-FLRW": (0, (5, 2))}
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(11.0, 5.0))
    for ax in (axA, axB):
        for m in order:
            gp = df[df.model == m].sort_values("t")
            if gp.empty:
                continue
            x = gp["PC1"].to_numpy(); y = gp["PC2"].to_numpy(); col = PALETTE[m]
            ax.plot(x, y, color=col, lw=0.9, alpha=0.55, zorder=1, ls=ls.get(m, "-"))
            ax.scatter(x, y, s=14, marker=mark[m], c=fade(col, len(x)),
                       edgecolor=col, linewidths=0.3, zorder=2)
            if len(x) >= 2 and (abs(x[-1] - x[-2]) + abs(y[-1] - y[-2])) > 1e-9:
                ax.annotate("", xy=(x[-1], y[-1]), xytext=(x[-2], y[-2]), zorder=3,
                            arrowprops=dict(arrowstyle="-|>", color=col, lw=1.0))
        ax.set_xlabel(r"$\mathrm{PC}_1$"); ax.set_ylabel(r"$\mathrm{PC}_2$"); style_axes(ax)
    axA.set_title("(a) full projection", fontsize=12)
    axB.set_title("(b) near-FLRW region", fontsize=12)
    axB.set_xlim(0.2, 2.2); axB.set_ylim(-3.2, 3.0)
    handles = [Line2D([0], [0], color=PALETTE[m], marker=mark[m], lw=1.4, ms=6.5,
                      ls=ls.get(m, "-"), markeredgecolor=PALETTE[m], label=m) for m in order]
    fig.legend(handles=handles, loc="lower center", ncol=6, frameon=True,
               bbox_to_anchor=(0.5, -0.08), handletextpad=0.4, columnspacing=1.3,
               borderaxespad=0.0)
    fig.subplots_adjust(left=0.065, right=0.99, top=0.93, bottom=0.20, wspace=0.22)
    return _save(fig, out_dir, "C1_pca_phase_space", pad_inches=0.25)


def fig_magnetic_validation(df, out_dir):
    """Magnetic Weyl axis I_B over time; nonzero only for the tensor benchmark."""
    fig, ax = plt.subplots(figsize=(6.8, 4.6), constrained_layout=True)
    gw = df[df["model"] == "GW-FLRW"]
    ax.axhspan(1e-19, 1e-15, color="0.88", zorder=0)
    ax.axhline(1e-16, color="0.55", ls="--", lw=1.1, zorder=1)
    ax.plot(gw["t"], gw["I_B"], "o", ms=6, color=PALETTE["GW-FLRW"], mec="k", mew=0.4,
            label=r"GW$-$FLRW (tensor mode)", zorder=3)
    ax.set_yscale("log"); ax.set_ylim(1e-18, 1e-2); ax.set_xlim(0.7, 5.0)
    ax.set_xlabel(r"time $t$"); ax.set_ylabel(r"$I_B$  (magnetic Weyl)")
    style_axes(ax); ax.legend(loc="center left", bbox_to_anchor=(0.02, 0.55))
    ax.text(0.5, 0.07, "FLRW, Bianchi-I, LTB, Kasner, perturbed-FLRW:  " + r"$I_B=0$ (round-off floor)",
            transform=ax.transAxes, ha="center", va="bottom", fontsize=10.5,
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.8", lw=0.7))
    return _save(fig, out_dir, "D1b_magnetic_weyl_validation")


def fig_components_over_time(df, out_dir):
    """Five per-axis component plots over time on a log scale."""
    floor = 1e-9; paths = []
    specs = [("I_rho", r"$I_\rho$"), ("I_theta", r"$I_\theta$"), ("I_sigma", r"$I_\sigma$"),
             ("I_E", r"$I_E$"), ("I_B", r"$I_B$")]
    for comp, lab in specs:
        fig, ax = plt.subplots(figsize=(6.6, 4.5), constrained_layout=True)
        drawn = []
        for m, gp in df.groupby("model"):
            y = gp[comp].values
            if np.nanmax(y) <= floor:
                continue
            ax.plot(gp["t"], np.clip(y, floor, None), label=m, color=PALETTE.get(m)); drawn.append(m)
        ax.set_yscale("log"); ax.set_ylim(floor, None)
        ax.set_xlabel(r"time $t$"); ax.set_ylabel(lab + r"$(t)$"); style_axes(ax)
        ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0.0)
        zero = [m for m in PALETTE if m not in drawn]
        if zero:
            ax.text(0.015, 0.04, (", ".join(zero)) + r"$:\ $" + lab + r"$=0$",
                    transform=ax.transAxes, fontsize=10.5, va="bottom",
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.85", lw=0.6))
        paths.append(_save(fig, out_dir, f"component_{comp}_over_time"))
    return paths


def fig_bianchi_sweep(df_bsweep, out_dir):
    fig, ax = plt.subplots(figsize=(6.4, 4.5), constrained_layout=True)
    ax.plot(df_bsweep["epsilon"], df_bsweep["I_sigma"], color="#1f4e9e", label=r"$I_\sigma$  (shear)")
    ax.plot(df_bsweep["epsilon"], df_bsweep["I_E"], "--", color="#c8430f", label=r"$I_E$  (electric Weyl)")
    ax.set_xlabel(r"anisotropy parameter $\varepsilon$"); ax.set_ylabel(r"diagnostic amplitude")
    ax.set_xlim(0, 0.2); ax.set_ylim(bottom=0); style_axes(ax); ax.legend(loc="upper left")
    return _save(fig, out_dir, "B1_bianchi_anisotropy_sweep")


def fig_ltb_amplitude_sweep(df_lsweep, out_dir):
    fig, ax = plt.subplots(figsize=(6.4, 4.5), constrained_layout=True)
    for c, lab, sty in [("I_rho", r"$I_\rho$", "-"), ("I_sigma", r"$I_\sigma$", "--"),
                        ("I_E", r"$I_E$", ":")]:
        ax.plot(df_lsweep["amplitude"], df_lsweep[c], sty, label=lab)
    ax.set_xlabel(r"bang-time amplitude $A$"); ax.set_ylabel(r"diagnostic amplitude")
    ax.set_xlim(0, 0.6); ax.set_ylim(bottom=0); style_axes(ax); ax.legend(loc="upper left")
    return _save(fig, out_dir, "B2_ltb_amplitude_sweep")


def fig_ltb_resolution(df_res, out_dir):
    fig, ax = plt.subplots(figsize=(6.4, 4.5), constrained_layout=True)
    for c, lab in [("I_rho", r"$I_\rho$"), ("I_sigma", r"$I_\sigma$"), ("I_E", r"$I_E$")]:
        ax.plot(df_res["n_r"], df_res[c], "o-", label=lab)
    ax.set_xlabel(r"radial resolution $N_r$"); ax.set_ylabel(r"diagnostic amplitude")
    ax.set_ylim(bottom=0); style_axes(ax); ax.legend(loc="center right")
    return _save(fig, out_dir, "B3_ltb_resolution")
