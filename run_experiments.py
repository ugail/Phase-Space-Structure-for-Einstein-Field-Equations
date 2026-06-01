#!/usr/bin/env python3
"""Run the full reproduction pipeline at production fidelity.

Evaluates the six benchmarks, the robustness sweeps, the averaging-domain study,
the constraint-contamination experiment, the observer-tilt test, and the PCA
projection, then writes every table, figure, and the run metadata.

Output goes to ./Results by default. Set the environment variable PSEFE_OUT to
choose a different directory, or pass --out PATH. Pure NumPy, SciPy, SymPy, and
matplotlib, CPU only.

    python run_experiments.py
    python run_experiments.py --out /tmp/my_run
"""
import argparse
import os

from phase_space_efe.pipeline import run


def main():
    ap = argparse.ArgumentParser(description="Run the full CosmoDiag pipeline.")
    ap.add_argument("--out", default=os.environ.get("PSEFE_OUT", "Results"),
                    help="output directory (default: ./Results or $PSEFE_OUT)")
    ap.add_argument("--quiet", action="store_true", help="suppress progress output")
    args = ap.parse_args()
    run(out_dir=args.out, verbose=not args.quiet)


if __name__ == "__main__":
    main()
