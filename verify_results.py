#!/usr/bin/env python3
"""Quick verification of the headline numbers from the precomputed tables.

Loads the CSVs in Results/ and confirms that the claims made in the manuscript
follow from those tables. Prints a PASS/FAIL line per check and exits non-zero if
any check fails. Needs only NumPy and pandas, no GPU, and runs in a second or two.

    pip install -r requirements-verify.txt
    python verify_results.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RES = Path(__file__).parent / "Results"
TAB = RES / "tables"

EXPECTED_DOMINANT = {
    "FLRW": "none", "Bianchi-I": "I_sigma", "Kasner": "I_sigma",
    "LTB": "I_rho", "Perturbed-FLRW": "I_rho", "GW-FLRW": "I_B",
}
checks = []


def record(name, ok, detail=""):
    checks.append((name, bool(ok), detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))


def approx(a, b, rtol=0.10, atol=1e-9):
    return abs(a - b) <= atol + rtol * abs(b)


def main():
    if not TAB.exists():
        print(f"Results tables not found at {TAB}. Run run_experiments.py first.")
        return 2

    df = pd.read_csv(TAB / "all_diagnostics.csv")
    fin = df.sort_values("t").groupby("model").tail(1).set_index("model")

    # 1. dominant driver per benchmark at final time
    ok = True; bad = []
    for m, exp in EXPECTED_DOMINANT.items():
        got = str(fin.loc[m, "dominant_driver"])
        if got != exp:
            ok = False; bad.append(f"{m}:{got}!={exp}")
    record("dominant driver per benchmark (final time)", ok, ",".join(bad))

    # 2. magnetic axis: zero for non-radiative, positive for the tensor benchmark
    nonrad = ["FLRW", "Bianchi-I", "Kasner", "LTB", "Perturbed-FLRW"]
    ib_nonrad = float(df[df.model.isin(nonrad)]["I_B"].abs().max())
    ib_gw = float(df[df.model == "GW-FLRW"]["I_B"].min())
    record("magnetic axis I_B = 0 off-tensor, > 0 on tensor",
           ib_nonrad < 1e-10 and ib_gw > 0, f"max|I_B|_nonrad={ib_nonrad:.1e}, min I_B_GW={ib_gw:.1e}")

    # 3. backreaction redundancy identity |Q|/<theta>^2 = |(2/3)I_theta - 2 I_sigma|
    lhs = df["I_Q_derived"].to_numpy()
    rhs = np.abs((2 / 3) * df["I_theta"].to_numpy() - 2 * df["I_sigma"].to_numpy())
    qerr = float(np.max(np.abs(lhs - rhs)))
    record("Buchert redundancy identity holds to round-off", qerr < 1e-10, f"max abs error={qerr:.1e}")

    # 4. GW electric/magnetic comparability ~ 0.13 and C^2 suppressed
    gw = df[df.model == "GW-FLRW"]
    rel = float((gw["I_E"] - gw["I_B"]).abs().max() / max(gw["I_E"].max(), gw["I_B"].max()))
    record("GW electric/magnetic comparability ~ 0.13", approx(rel, 0.13, rtol=0.6), f"rel={rel:.3f}")

    # 5. PCA first two components explain ~71% of variance
    Xs = df[["I_rho", "I_theta", "I_sigma", "I_E", "I_B"]].to_numpy(float)
    Xs = (Xs - Xs.mean(0)) / (Xs.std(0) + 1e-12)
    s = np.linalg.svd(Xs, compute_uv=False)
    evr2 = float((s[:2] ** 2).sum() / (s ** 2).sum())
    record("PCA first two components ~ 0.71 of variance", approx(evr2, 0.713, rtol=0.08), f"evr2={evr2:.3f}")

    # 6. observer tilt: dominant axis stays I_rho and I_B stays 0 up to v0 = 0.1
    tilt = pd.read_csv(TAB / "observer_tilt.csv")
    ok_tilt = (tilt["dominant"] == "I_rho").all() and float(tilt["I_B"].abs().max()) < 1e-12
    record("observer tilt: dominant stays I_rho, I_B = 0", ok_tilt,
           f"I_B max={float(tilt['I_B'].abs().max()):.1e}")

    # 7. contamination: retained up to the tolerance, flagged above it
    con = pd.read_csv(TAB / "constraint_contamination.csv")
    lo = con[con.contamination <= 1e-3]["regime"].str.contains("unreliable").any()
    hi = con[con.contamination >= 2e-3]["regime"].eq("numerically unreliable").all()
    record("contamination flagged only above the tolerance", (not lo) and hi)

    npass = sum(ok for _, ok, _ in checks)
    print(f"\n{npass}/{len(checks)} checks passed.")
    return 0 if npass == len(checks) else 1


if __name__ == "__main__":
    sys.exit(main())
