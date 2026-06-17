# Geometric Phase-Space Structure in Cosmological Solutions of Einstein's Field Equations

*Geometric Phase-Space Structure in Cosmological Solutions of Einstein's Field Equations*

Cite: Ugail, H. (2026). Geometric Phase-Space Structure in Cosmological Solutions of Einstein's Field Equations. ArXiv. https://arxiv.org/abs/2606.1707

Relativistic cosmological models can depart from the Friedmann-Lemaitre-Robertson-Walker (FLRW) idealisation in more than one way. Matter can become inhomogeneous, the local expansion rate can vary from place to place, the expansion can turn anisotropic, and the free gravitational field can develop electric or magnetic Weyl curvature. A single departure-from-FLRW number cannot say which of these is at work. **This is an open-source toolkit that organises standard kinematic, curvature, and constraint quantities into a compact, non-redundant set of dimensionless diagnostics with an explicit electric and magnetic Weyl split.** For a chosen observer field and averaging domain it reports the diagnostic vector

    X_cosmo = (I_rho, I_theta, I_sigma, I_E, I_B, I_H, I_M),

namely matter inhomogeneity, expansion inhomogeneity, shear, the electric and magnetic Weyl parts, and two reliability indicators built from the Hamiltonian and momentum constraints. Solution families are then compared as points and trajectories in one diagnostic space. The framework is observer-explicit and domain-explicit. It is a transparent diagnostic, not a new invariant classification of spacetime.

<img width="2015" height="1384" alt="component_I_E_over_time" src="https://github.com/user-attachments/assets/8c45ecb1-85d8-45f9-8148-385952236ca4" />


## What does this measure?

We reproduce, across the six benchmark families, the following.

1. **Each benchmark occupies a distinct region of the diagnostic space, matching its physical mechanism.** FLRW sits at the origin. Bianchi-I and Kasner are dominated by shear and electric Weyl curvature, with Kasner an order of magnitude stronger. LTB spreads a moderate signal across matter inhomogeneity, expansion inhomogeneity, shear, and electric Weyl curvature. The scalar-perturbed model is dominated by matter inhomogeneity, and the gravitational-wave benchmark is the only model with a nonzero magnetic Weyl axis.

2. **The electric and magnetic Weyl split recovers information a scalar Weyl measure hides.** A solution can carry a large magnetic Weyl field while the Weyl scalar `C^2 = 8(E^2 - B^2)` stays small or changes sign. The tensor benchmark activates the magnetic axis while the five non-radiative benchmarks return it at the round-off floor.

3. **The Buchert backreaction is algebraically determined, not an independent axis.** The identity `|Q_D|/<theta>^2 = |(2/3) I_theta - 2 I_sigma|` holds to round-off, so the backreaction is reported as a derived explanatory quantity rather than a separate coordinate.

4. **The classification is robust.** It is stable under changes of perturbation amplitude, spatial resolution, averaging domain, constraint reliability, curvature normalisation, and a leading-order observer tilt. Under the radial LTB tilt the dominant axis stays `I_rho` and the magnetic axis stays zero.

5. **Separability is reported honestly as a classifier-free check.** The first two principal components carry about 71% of the variance, the silhouette score rises from about 0.47 over the full history to about 0.78 in the developed late-time regime, and a leave-one-out nearest-centroid assignment recovers the correct regime about 0.80 of the time over the full history and about 0.96 at late times.

## Contents

- **`run_pipeline_full.ipynb`** — the full reproduction notebook. Runs the benchmark suite, the symbolic curvature verification, the robustness sweeps, the averaging-domain and observer-tilt studies, the constraint-contamination experiment, the PCA projection with separability metrics, and the manuscript figures.

- **`run_experiments.py`** — the same full pipeline as a single command-line script. Runs at production fidelity and writes every CSV, figure, and the run metadata to `./Results`. Pure NumPy, SciPy, SymPy, and matplotlib, CPU only.

- **`verify_results.py`** — a quick verification script for reviewers. Loads the precomputed tables in `Results/` and confirms that every headline number is reproducible from those tables. Runs in a couple of seconds, needs no GPU, and prints a clean PASS/FAIL summary.

- **`phase_space_efe/`** — the Python package. Contains the diagnostic primitives, the electric and magnetic Weyl split, the derived Buchert backreaction, and the classification (`core.py`), the six benchmark simulators and the robustness, domain, contamination, and tilt studies (`benchmarks.py`), the publication figure generators (`figures.py`), and the end-to-end pipeline (`pipeline.py`).

- **`scripts/quick_diagnostic.py`** — a minimal worked example showing how to compute the diagnostic vector for a built-in benchmark and how to drop in your own fields.

- **`tests/`** — a pytest suite verifying the FLRW baseline, the Bianchi-I and Kasner Weyl values, the LTB closed form and its homogeneous limit, the Buchert redundancy identity, the magnetic-axis activation, and the observer-tilt stability.

- **`Results/`** — the precomputed result tables, figures, and run metadata for the paper, sufficient to verify every reported number without re-running the pipeline.

- **`requirements.txt`** — full dependencies (NumPy, SciPy, pandas, matplotlib, SymPy, scikit-learn). **`requirements-verify.txt`** — minimal dependencies for `verify_results.py` only (NumPy, pandas).

## Quick start

The fastest way to check the headline numbers is:

```
git clone https://github.com/ugail/Phase-Space-Structure-for-Einstein-Field-Equations.git
cd Phase-Space-Structure-for-Einstein-Field-Equations
pip install -r requirements-verify.txt
python verify_results.py
```

The script loads the precomputed CSVs in `Results/`, recomputes every headline quantity, and prints a PASS/FAIL summary. A full pass takes a couple of seconds on any laptop.

To run the worked example:

```
pip install -e .
python scripts/quick_diagnostic.py
```

## Reproducing the full pipeline

Re-running the full pipeline requires Python 3.10 or later. No GPU is needed. Install the full requirements first:

```
pip install -r requirements.txt
pip install -e .
```

To reproduce all reported results from a clean run, use the script:

```
python run_experiments.py
```

It writes every CSV, figure, and the run metadata to `./Results` by default. Set the environment variable `PSEFE_OUT`, or pass `--out PATH`, to change the output directory. A full run takes a few minutes on a modern CPU, dominated by the symbolic gravitational-wave curvature computation.

The same pipeline is available as a notebook:

```
jupyter notebook run_pipeline_full.ipynb
```

## Tests

To run the structural invariant tests:

```
pip install pytest
pytest tests
```

The tests verify that FLRW returns every axis at the round-off floor, that the Bianchi-I isotropic limit and Kasner vacuum value of the Weyl invariant are correct, that the LTB closed form is positive and reduces to zero in the homogeneous limit, that the Buchert redundancy identity holds to round-off, that the magnetic axis is activated only by the tensor benchmark, and that the leading-order observer tilt leaves the dominant axis and the zero magnetic axis unchanged.

## Diagnostic axes

For a chosen observer field `u^a`, an averaging domain `D`, and a time slice, the toolkit reports:

- **`I_rho`** — matter inhomogeneity, `Var_D(rho)/<rho>^2`.
- **`I_theta`** — expansion inhomogeneity, `Var_D(theta)/<theta>^2`.
- **`I_sigma`** — shear, `<sigma^2>/<theta>^2` with `sigma^2 = (1/2) sigma_ab sigma^ab`.
- **`I_E`** — electric Weyl, `<E_ab E^ab>/K_D`.
- **`I_B`** — magnetic Weyl, `<B_ab B^ab>/K_D`.
- **`I_H`, `I_M`** — Hamiltonian and momentum constraint residuals, reported as reliability flags rather than physical structure.

All Weyl axes use a single unified curvature normalisation `K_D = <theta^4>_D + eps`, applied to every benchmark so that `I_E` and `I_B` are comparable across matter, vacuum, and radiative families. The Buchert backreaction `Q_D` is reported as a derived quantity through `|Q_D|/<theta>^2 = |(2/3) I_theta - 2 I_sigma|`, so it is not carried as a separate axis. The word non-redundant is used in this algebraic sense; the axes are not claimed to be dynamically independent, since the Einstein equations couple them, and the PCA quantifies the resulting correlations.

## Who is this for?

- **Researchers in relativistic cosmology** who want a compact, reproducible way to compare exact, perturbative, and numerical solution families on a common geometric footing.
- **Researchers in numerical-relativity cosmology** who want a diagnostic that separates physical structure from numerical reliability through the constraint-residual flags, with a worked contamination experiment.
- **Reviewers** who want to verify every headline number from precomputed tables in a couple of seconds, without re-running the pipeline.

## Citation

If you use this toolkit or the precomputed result tables, please cite the paper:

> Ugail, H. (2026). Geometric Phase-Space Structure in Cosmological Solutions of Einstein's Field Equations. ArXiv. https://arxiv.org/abs/2606.1707

A related methodological idea, measuring the preservation or loss of theoretically meaningful structure rather than relying on a single generic score, appears in:

> H. Ugail and N. Howard. *Symmetry-Organised Complexity in Quantum Neural Networks*. Symmetry, 2026, 18(6), 912. doi:10.3390/sym18060912.

## License

Released under the MIT License. See `LICENSE` for the full text.
