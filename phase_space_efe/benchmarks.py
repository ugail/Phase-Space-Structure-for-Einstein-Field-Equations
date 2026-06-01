"""Benchmark cosmologies and robustness studies.

Six benchmarks exercise the diagnostic framework, namely FLRW, Bianchi-I, Kasner,
Lemaitre-Tolman-Bondi dust, scalar-perturbed FLRW, and tensor-perturbed FLRW.
The module also provides the controlled sweeps used in the robustness section,
the averaging-domain study, the constraint-contamination experiment, and the
leading-order observer-tilt test.
"""
import numpy as np
import pandas as pd
import sympy as sp

from .core import EPS, AXES, compute_diagnostics_EB, classify_regime, RELIABILITY_TOL


# ---------------------------------------------------------------------------
# Exact homogeneous and vacuum models
# ---------------------------------------------------------------------------
def bianchi_weyl_C2(t, p):
    """Closed-form Weyl scalar C_abcd C^abcd for a Bianchi-I metric a_i = t^{p_i}."""
    p = np.asarray(p, float); H = p / t; dH = -p / t ** 2; A = dH + H ** 2; th = H.sum()
    pair = sum((H[i] * H[j]) ** 2 for i in range(3) for j in range(i + 1, 3))
    K = 4 * ((A ** 2).sum() + pair); R00 = -A.sum(); Rii = dH + H * th
    Ric2 = R00 ** 2 + (Rii ** 2).sum(); Rsc = 2 * dH.sum() + (H ** 2).sum() + th ** 2
    return float(K - 2 * Ric2 + Rsc ** 2 / 3)


def simulate_flrw(times, n=400):
    rows = []
    for t in times:
        a = t ** (2 / 3); H = 2 / (3 * t); th = 3 * H; rho0 = 1 / a ** 3
        d = compute_diagnostics_EB(np.full(n, rho0), np.full(n, th),
                                   np.zeros(n), np.zeros(n), np.zeros(n))
        d.update(model="FLRW", t=t); rows.append(d)
    return pd.DataFrame(rows)


def simulate_bianchi(times, p=(0.52, 0.67, 0.81), n=400):
    p = np.asarray(p, float); rows = []
    for t in times:
        Hi = p / t; th = Hi.sum(); Hb = th / 3; s2 = 0.5 * np.sum((Hi - Hb) ** 2)
        rho0 = 1 / (t ** p.sum() + EPS); C2 = bianchi_weyl_C2(t, p)
        d = compute_diagnostics_EB(np.full(n, rho0), np.full(n, th), np.full(n, s2),
                                   np.full(n, C2 / 8.0), np.zeros(n))  # B = 0
        d.update(model="Bianchi-I", t=t); rows.append(d)
    return pd.DataFrame(rows)


def simulate_kasner(times, p=(2 / 3, 2 / 3, -1 / 3), n=400):
    p = np.asarray(p, float); rows = []
    for t in times:
        Hi = p / t; th = Hi.sum(); Hb = th / 3; s2 = 0.5 * np.sum((Hi - Hb) ** 2)
        C2 = bianchi_weyl_C2(t, p)
        d = compute_diagnostics_EB(np.full(n, 1.0), np.full(n, th if abs(th) > 1e-9 else 1e-9),
                                   np.full(n, s2), np.full(n, C2 / 8.0), np.zeros(n))  # B = 0
        d.update(model="Kasner", t=t); rows.append(d)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Exact inhomogeneous dust (Lemaitre-Tolman-Bondi)
# ---------------------------------------------------------------------------
def simulate_ltb(times, n_r=700, M0=1.0, tb_amp=0.60, r0=0.30, r_min=0.03, r_max=1.0):
    r = np.linspace(r_min, r_max, n_r); M = M0 * r ** 3; Mp = 3 * M0 * r ** 2
    tB = tb_amp * np.exp(-(r / r0) ** 2); A = (9 * M / 2) ** (1 / 3); rows = []
    for t in times:
        tau = t - tB
        if np.any(tau <= 0):
            raise ValueError("times too early for chosen tb_amp")
        R = A * tau ** (2 / 3); Rdot = (2 / 3) * R / tau
        Rp = np.gradient(R, r); Rdotp = np.gradient(Rdot, r)
        Rps = np.where(np.abs(Rp) < 1e-10, 1e-10, Rp)
        rho = np.maximum(Mp / (4 * np.pi * R ** 2 * Rps), 1e-12)
        Hperp = Rdot / R; Hpar = Rdotp / Rps; th = Hpar + 2 * Hperp
        s2 = (1 / 3) * (Hpar - Hperp) ** 2
        C2 = 48.0 * (M / R ** 3 - Mp / (3 * R ** 2 * Rps)) ** 2
        w = np.abs(R ** 2 * Rps)
        d = compute_diagnostics_EB(rho, th, s2, C2 / 8.0, np.zeros_like(C2), weights=w)  # B = 0
        d.update(model="LTB", t=t); rows.append(d)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Constraint-consistent perturbations
# ---------------------------------------------------------------------------
def simulate_perturbed_flrw_consistent(times, n=32, eps0=0.02):
    L = 2 * np.pi; dx = L / n; xv = np.linspace(0, L, n, endpoint=False)
    X, Y, Z = np.meshgrid(xv, xv, xv, indexing="ij")
    pat = np.sin(X) + 0.6 * np.sin(2 * Y + 0.3) + 0.4 * np.cos(Z + X); pat -= pat.mean()
    k = np.fft.fftfreq(n, d=dx) * 2 * np.pi; KX, KY, KZ = np.meshgrid(k, k, k, indexing="ij")
    k2 = KX ** 2 + KY ** 2 + KZ ** 2; k2[0, 0, 0] = 1.0; G4pi = 1.0

    def hess(fk, a, b):
        Kv = [KX, KY, KZ]; return np.real(np.fft.ifftn(-(Kv[a] * Kv[b]) * fk))

    rows = []
    for t in times:
        eta = t; aN = eta ** 2; Hc = 2 / eta  # matter-dominated (conformal): a ~ eta^2
        Phi = eps0 * pat; Phik = np.fft.fftn(Phi)
        lapPhi = np.real(np.fft.ifftn(-k2 * Phik))
        delta_rho = (lapPhi - 3 * Hc ** 2 * Phi) / (G4pi * aN ** 2)
        ham_res = lapPhi - 3 * Hc ** 2 * Phi - G4pi * aN ** 2 * delta_rho
        rhobar = 3 * Hc ** 2 / (2 * G4pi * aN ** 2); rho = rhobar + delta_rho
        gP = [np.real(np.fft.ifftn(1j * Kv * Phik)) for Kv in (KX, KY, KZ)]
        v = [-(Hc * g) / (G4pi * aN ** 3 * rhobar) for g in gP]
        mom_res = Hc * gP[0] + G4pi * aN ** 3 * rhobar * v[0]
        divv = sum(np.real(np.fft.ifftn(1j * Kv * np.fft.fftn(vi)))
                   for Kv, vi in zip((KX, KY, KZ), v))
        theta = 3 * (2 / eta ** 3) + divv / aN
        Hxx = hess(Phik, 0, 0) / aN ** 2; Hyy = hess(Phik, 1, 1) / aN ** 2; Hzz = hess(Phik, 2, 2) / aN ** 2
        Hxy = hess(Phik, 0, 1) / aN ** 2; Hxz = hess(Phik, 0, 2) / aN ** 2; Hyz = hess(Phik, 1, 2) / aN ** 2
        lap = Hxx + Hyy + Hzz; Exx = Hxx - lap / 3; Eyy = Hyy - lap / 3; Ezz = Hzz - lap / 3
        E2 = Exx ** 2 + Eyy ** 2 + Ezz ** 2 + 2 * (Hxy ** 2 + Hxz ** 2 + Hyz ** 2)
        d = compute_diagnostics_EB(rho.ravel(), theta.ravel(), np.zeros(rho.size),
                                   E2.ravel(), np.zeros(E2.size),
                                   H_constraint=np.max(np.abs(ham_res)),
                                   M_constraint=np.max(np.abs(mom_res)))
        d.update(model="Perturbed-FLRW", t=t); rows.append(d)
    return pd.DataFrame(rows)


def _gw_EB_lambdas(k=2.0, eps=1e-3):
    """Symbolic electric and magnetic Weyl scalars for a transverse-traceless mode."""
    eta, z = sp.symbols('eta z', real=True); n = 4
    cc = [eta, sp.symbols('x'), sp.symbols('y'), z]
    a = eta; Hm = sp.sin(k * eta) / (k * eta); hp = eps * Hm * sp.cos(k * z)  # on-shell TT (a = eta)
    g = sp.diag(-a ** 2, a ** 2 * (1 + hp), a ** 2 * (1 - hp), a ** 2); gi = g.inv()
    D = lambda f, i: sp.diff(f, cc[i])
    G = [[[sum(gi[A, d] * (D(g[d, b], c) + D(g[d, c], b) - D(g[b, c], d)) for d in range(n)) / 2
           for c in range(n)] for b in range(n)] for A in range(n)]
    Rm = [[[[D(G[A][b][d], c) - D(G[A][b][c], d)
             + sum(G[A][c][e] * G[e][b][d] - G[A][d][e] * G[e][b][c] for e in range(n))
             for d in range(n)] for c in range(n)] for b in range(n)] for A in range(n)]
    Rl = [[[[sum(g[A, e] * Rm[e][b][c][d] for e in range(n))
             for d in range(n)] for c in range(n)] for b in range(n)] for A in range(n)]
    Ric = sp.Matrix([[sum(Rm[A][b][A][d] for A in range(n)) for d in range(n)] for b in range(n)])
    Rsc = sum(gi[b, d] * Ric[b, d] for b in range(n) for d in range(n))
    C = [[[[Rl[A][b][c][d]
            - sp.Rational(1, 2) * (g[A, c] * Ric[b, d] - g[A, d] * Ric[b, c]
                                   - g[b, c] * Ric[A, d] + g[b, d] * Ric[A, c])
            + sp.Rational(1, 6) * Rsc * (g[A, c] * g[b, d] - g[A, d] * g[b, c])
            for d in range(n)] for c in range(n)] for b in range(n)] for A in range(n)]
    u = [1 / a, 0, 0, 0]; sg = sp.sqrt(-g.det())
    E = sp.Matrix([[sum(C[A][c][b][d] * u[c] * u[d] for c in range(n) for d in range(n))
                    for b in range(n)] for A in range(n)])

    def lcf(perm):
        arr = list(perm); s = 1
        if len(set(perm)) < 4:
            return 0
        for i in range(4):
            for j in range(i + 1, 4):
                if arr[i] > arr[j]:
                    s = -s
        return s

    B = sp.zeros(n)
    for A in range(n):
        for c in range(n):
            tot = 0
            for b in range(n):
                for d in range(n):
                    for e in range(n):
                        ea = sg * lcf((A, b, d, e))
                        if ea == 0:
                            continue
                        for dd in range(n):
                            for ee in range(n):
                                for f in range(n):
                                    tot += (sp.Rational(1, 2) * ea * gi[d, dd] * gi[e, ee]
                                            * C[dd][ee][c][f] * u[b] * u[f])
            B[A, c] = tot
    invf = lambda M: sum(M[A, b] * sum(gi[A, c] * gi[b, d] * M[c, d]
                                       for c in range(n) for d in range(n))
                         for A in range(n) for b in range(n))
    return sp.lambdify((eta, z), invf(E), 'numpy'), sp.lambdify((eta, z), invf(B), 'numpy')


def simulate_gw_flrw(times, k=16.0, eps=1.5e-2, nz=640, nperiod=24):
    """Sub-horizon TT mode. The mean-square Weyl amplitude is averaged over one
    wave period so the oscillating mode gives smooth, positive I_E and I_B."""
    fE, fB = _gw_EB_lambdas(k, eps); zs = np.linspace(0, 2 * np.pi, nz, endpoint=False)
    T = 2 * np.pi / k; rows = []
    for eta in times:
        ee = np.linspace(eta - T / 2, eta + T / 2, nperiod)
        E2 = float(np.mean([np.mean(fE(e, zs)) for e in ee]))
        B2 = float(np.mean([np.mean(fB(e, zs)) for e in ee]))
        th = 3.0 / eta ** 2  # radiation-background expansion proxy
        d = compute_diagnostics_EB(np.array([1.0]), np.full(1, th), np.zeros(1),
                                   np.array([E2]), np.array([B2]))
        d.update(model="GW-FLRW", t=eta); rows.append(d)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Robustness studies
# ---------------------------------------------------------------------------
def bianchi_anisotropy_sweep(eps_values=None, t=2.0):
    if eps_values is None:
        eps_values = np.linspace(0, 0.2, 41)
    rows = []
    for e in eps_values:
        p = (2 / 3 - e, 2 / 3, 2 / 3 + e); Hi = np.array(p) / t; th = Hi.sum()
        s2 = 0.5 * np.sum((Hi - th / 3) ** 2)
        C2 = bianchi_weyl_C2(t, p); Cs = th ** 4 + EPS  # unified scale <theta^4>
        rows.append(dict(epsilon=e, I_sigma=s2 / th ** 2, I_E=(C2 / 8) / Cs, Q_shear=-2 * s2 / th ** 2))
    return pd.DataFrame(rows)


def ltb_amplitude_sweep(amps=None):
    if amps is None:
        amps = np.linspace(0.0, 0.6, 13)
    rows = []
    for A0 in amps:
        if A0 < 1e-9:
            rows.append(dict(amplitude=0.0, I_rho=0, I_theta=0, I_sigma=0, I_E=0)); continue
        d = simulate_ltb(np.array([1.0]), tb_amp=A0).iloc[0]
        rows.append(dict(amplitude=A0, I_rho=d.I_rho, I_theta=d.I_theta, I_sigma=d.I_sigma, I_E=d.I_E))
    return pd.DataFrame(rows)


def ltb_resolution_study(resolutions=(200, 400, 700, 1000, 1400)):
    rows = []
    for Nr in resolutions:
        d = simulate_ltb(np.array([1.0]), n_r=Nr).iloc[0]
        rows.append(dict(n_r=Nr, I_rho=d.I_rho, I_theta=d.I_theta, I_sigma=d.I_sigma, I_E=d.I_E))
    return pd.DataFrame(rows)


def ltb_on_domain(rmax, t=1.0, n_r=700, tb_amp=0.60, r0=0.30):
    r = np.linspace(0.03, rmax, n_r); M0 = 1.0; M = M0 * r ** 3; Mp = 3 * M0 * r ** 2
    tB = tb_amp * np.exp(-(r / r0) ** 2); A = (9 * M / 2) ** (1 / 3); tau = t - tB
    R = A * tau ** (2 / 3); Rdot = (2 / 3) * R / tau
    Rp = np.gradient(R, r); Rps = np.where(np.abs(Rp) < 1e-10, 1e-10, Rp)
    rho = np.maximum(Mp / (4 * np.pi * R ** 2 * Rps), 1e-12)
    Hpar = np.gradient(Rdot, r) / Rps; Hperp = Rdot / R
    th = Hpar + 2 * Hperp; s2 = (1 / 3) * (Hpar - Hperp) ** 2
    C2 = 48.0 * (M / R ** 3 - Mp / (3 * R ** 2 * Rps)) ** 2; w = np.abs(R ** 2 * Rps)
    return compute_diagnostics_EB(rho, th, s2, C2 / 8.0, np.zeros_like(C2), weights=w)


def domain_dependence_study(rmaxes=(0.4, 0.6, 0.8, 1.0)):
    rows = []
    for rmax in rmaxes:
        d = ltb_on_domain(rmax)
        rows.append(dict(r_max=rmax, I_rho=d["I_rho"], I_sigma=d["I_sigma"], I_E=d["I_E"]))
    return pd.DataFrame(rows)


def constraint_contamination(base_row, levels=(0.0, 1e-4, 1e-3, 2e-3, 1e-2)):
    rows = []
    for lv in levels:
        row = dict(base_row); row["I_H"] = lv; row["I_M"] = lv
        rows.append(dict(contamination=lv, exceeds_tol=lv > RELIABILITY_TOL,
                         regime=classify_regime(row)))
    return pd.DataFrame(rows)


def ltb_tilt(v0, t=1.0, n_r=700, tb_amp=0.60, r0=0.30, r_min=0.03, r_max=1.0):
    """Leading-order radial observer tilt v(r) = v0 r(1-r) applied to the LTB congruence.

    The magnetic axis stays zero because a radial boost of the spherically symmetric
    (Petrov-D) electric Weyl field is along a principal direction.
    """
    r = np.linspace(r_min, r_max, n_r); M = r ** 3; Mp = 3 * r ** 2
    tB = tb_amp * np.exp(-(r / r0) ** 2); A = (9 * M / 2) ** (1 / 3); tau = t - tB
    R = A * tau ** (2 / 3); Rdot = (2 / 3) * R / tau
    Rp = np.gradient(R, r); Rps = np.where(np.abs(Rp) < 1e-10, 1e-10, Rp)
    rho = np.maximum(Mp / (4 * np.pi * R ** 2 * Rps), 1e-12)
    Hpar = np.gradient(Rdot, r) / Rps; Hperp = Rdot / R; th = Hpar + 2 * Hperp
    w = np.abs(R ** 2 * Rps)
    dl = lambda f: np.gradient(f, r) / Rps  # proper radial derivative (1/R') d/dr
    v = v0 * r * (1.0 - r)
    th_t = th + (dl(v) + 2 * v / R)                       # tilted expansion
    s2_t = (1.0 / 3.0) * ((Hpar - Hperp) + (dl(v) - v / R)) ** 2  # aligned radial shear
    C2 = 48.0 * (M / R ** 3 - Mp / (3 * R ** 2 * Rps)) ** 2
    return compute_diagnostics_EB(rho, th_t, s2_t, C2 / 8.0, np.zeros_like(C2), weights=w)


def observer_tilt_study(v0_values=(0.0, 0.01, 0.05, 0.1)):
    rows = []
    for v0 in v0_values:
        d = ltb_tilt(v0); drv = max(AXES, key=lambda c: d[c])
        rows.append(dict(tilt_v0=v0, I_rho=d["I_rho"], I_theta=d["I_theta"],
                         I_sigma=d["I_sigma"], I_E=d["I_E"], I_B=d["I_B"], dominant=drv))
    return pd.DataFrame(rows)
