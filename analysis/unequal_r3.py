"""Calibrate the modal sensitivity ratio r = (dln f_v/deps)/(dln f_u/deps) for the partial modulation masks from
static runs with eps offset +-0.05 in the mask region; predict the first/second-order sideband polarization with
(i) the fitted two-mode model (unequal q_u, q_v) and (ii) static FDTD central differences; compare with dynamic runs."""
import sys, os; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tmcp_paths import *  # ROOT, DATA, R1, R2, R2C, R3, COMM, OUT, FIG, TAB
import sys, os, json, numpy as np
sys.path.insert(0, f"{R2}")
os.environ.setdefault("FC", "5.754e9"); os.environ.setdefault("DATA", f"{R1}")
import fdtd_core as fc, adiabatic as ad
from scipy.optimize import least_squares
import twomode_r3 as tm   # runs its analyses on import (fast); provides fit/model_deriv/pol/base/q

FC = 5.754e9; R3 = f"{R3}"


def Hc(path):
    d = np.load(path); nb = int(d["nbase"]); f = d["freqs"][:nb]; S = d["S"][:nb]
    prop = np.exp(1j * 2 * np.pi * f / fc.c0)
    return f, np.array([d["Ex"][:nb] / S * prop, d["Ey"][:nb] / S * prop])


def at(fH, fv):
    f, H = fH
    return np.array([np.interp(fv, f, H[c].real) + 1j * np.interp(fv, f, H[c].imag) for c in range(2)])


base = tm.base; f1, f2, Q1, Q2 = base["p"]; C = base["C"]
f0, H0 = ad.H_curve(2.2); sc = None
out = {}
for mask in ("uquad", "patch"):
    pp, pm = f"{R3}/{mask}_p.npz", f"{R3}/{mask}_m.npz"
    if not (os.path.exists(pp) and os.path.exists(pm)):
        continue
    res = {}
    # fit shifted resonances with Q, modal vectors fixed (scale of H fixed by the baseline fit normalization)
    for tag, path in (("p", pp), ("m", pm)):
        f, H = Hc(path); band = (f > 5.45e9) & (f < 6.15e9); fb = f[band]
        Hb = H[:, band] / np.abs(H0[:, (f0 > 5.45e9) & (f0 < 6.15e9)]).max()
        def r_(p):
            a, b = p
            Mod = C[0][:, None] / (1 + 1j * Q1 * (fb / a - a / fb)) + C[1][:, None] / (1 + 1j * Q2 * (fb / b - b / fb)) + C[2][:, None]
            d = (Mod - Hb).ravel(); return np.concatenate([d.real, d.imag])
        s = least_squares(r_, [f1, f2], x_scale=[1e7, 1e7])
        res[tag] = s.x
    dfu = (res["p"][0] - res["m"][0]) / 0.1 / f1; dfv = (res["p"][1] - res["m"][1]) / 0.1 / f2   # dln f/deps
    q_u, q_v = -2 * dfu * 2.2, -2 * dfv * 2.2          # f_i ~ eps^(-q_i/2)
    r = dfv / dfu
    # predictions
    p1 = tm.pol(tm.model_deriv(base, FC, d=1, qu=q_u, qv=q_v)); p2 = tm.pol(tm.model_deriv(base, FC, d=2, qu=q_u, qv=q_v))
    Hp, Hm, Hz = at(Hc(pp), FC), at(Hc(pm), FC), at(ad.H_curve(2.2), FC)
    fd1 = tm.pol((Hp - Hm) / 0.1); fd2 = tm.pol((Hp - 2 * Hz + Hm) / 0.05 ** 2)
    rel1 = float(20 * np.log10(np.linalg.norm((Hp - Hm) / 0.1 * 0.05 / 2) / np.linalg.norm(Hz)))
    out[mask] = dict(dlnfu=float(dfu), dlnfv=float(dfv), r=float(r), q_u=float(q_u), q_v=float(q_v),
                     model_d1=p1, model_d2=p2, fd_d1=fd1, fd_d2=fd2, fd_E1_dBc=rel1)
    print(mask, "dlnf/deps u %.4f v %.4f  r = %.2f | model d1 AR %.1f LR %.1f d2 LR %.1f | FD d1 AR %.1f LR %.1f d2 AR %.1f LR %.1f | E1 %.1f dBc"
          % (dfu, dfv, r, p1["AR"], p1["LR"], p2["LR"], fd1["AR"], fd1["LR"], fd2["AR"], fd2["LR"], rel1))
# uniform reference
dl = -0.86 / 2 / 2.2
print("uniform: dlnf/deps = %.4f" % dl)
json.dump(out, open(f"{R2}/unequal_r3.json", "w"), indent=1)
