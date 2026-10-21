"""
Round-3 two-mode analyses:
 (a) sensitivity of the predicted sideband polarization to the fitted parameters (refits over different
     frequency windows + one-at-a-time perturbations)                         [reviewer 1, comment 6]
 (b) unequal modulation of the two modes: first-/second-order sideband polarization vs the ratio
     r = (d ln f_v/d eps)/(d ln f_u/d eps)                                    [reviewer 1, comment 5]
 (c) non-perturbative two-mode harmonics at delta_eps = 0.2: modal Fourier coefficients per order
                                                                              [reviewer 2, comment 3]
"""
import sys, os; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tmcp_paths import *  # ROOT, DATA, R1, R2, R2C, R3, COMM, OUT, FIG, TAB
import sys, os, json, numpy as np
os.environ.setdefault("FC", "5.754e9"); os.environ.setdefault("DATA", f"{R1}")
import fdtd_core as fc, adiabatic as ad
from scipy.optimize import least_squares

FC = 5.754e9; EPS0 = 2.2
f_all, H_all = ad.H_curve(2.2)
q, _ = ad.fit_q()


def fit(fmin, fmax):
    band = (f_all > fmin) & (f_all < fmax); fb = f_all[band]; Hb = H_all[:, band]
    sc = np.abs(Hb).max(); Hb = Hb / sc
    def basis(p):
        f1, f2, Q1, Q2 = p
        return np.stack([1/(1+1j*Q1*(fb/f1-f1/fb)), 1/(1+1j*Q2*(fb/f2-f2/fb)), np.ones_like(fb)], 1)
    def lin(p):
        A = basis(p); return A, np.linalg.lstsq(A, Hb.T, rcond=None)[0]
    def res(p):
        A, C = lin(p); r = (A @ C - Hb.T).ravel(); return np.concatenate([r.real, r.imag])
    best = None
    for f1 in np.linspace(5.5e9, 5.75e9, 4):
        for f2 in np.linspace(5.8e9, 6.0e9, 4):
            s = least_squares(res, [f1, f2, 25, 25], x_scale=[1e8, 1e8, 10, 10],
                              bounds=([5.2e9, 5.2e9, 3, 3], [6.4e9, 6.4e9, 300, 300]))
            if best is None or s.cost < best.cost: best = s
    A, C = lin(best.x)
    err = np.sqrt(np.sum(np.abs(A @ C - Hb.T)**2) / np.sum(np.abs(Hb)**2))
    return dict(p=best.x, C=C, err=err)


def model_deriv(F, fv, eps=EPS0, d=0, qu=None, qv=None, eps_t=None):
    """H (d=0) or eps-derivatives; qu, qv: scaling exponents of the two modes (default q)."""
    f1, f2, Q1, Q2 = F["p"]; C = F["C"]
    qs = (qu if qu is not None else q, qv if qv is not None else q)
    e = eps if eps_t is None else eps_t
    out = 0
    for i, (fi0, Qi) in enumerate(((f1, Q1), (f2, Q2))):
        qi = qs[i]; fi = fi0 * (e/EPS0)**(-qi/2)
        x = Qi*(fv/fi - fi/fv); M = 1/(1+1j*x)
        if d == 0:
            val = M
        else:
            dfi = -(qi/2)*fi/e; d2fi = (qi/2)*(qi/2+1)*fi/e**2
            dxf = -Qi*(fv/fi**2 + 1/fv); d2xf = 2*Qi*fv/fi**3
            dx = dxf*dfi; d2x = d2xf*dfi**2 + dxf*d2fi
            val = -1j*M**2*dx if d == 1 else (-2*M**3*dx**2 - 1j*M**2*d2x)
        out = out + np.multiply.outer(C[i], val)
    if d == 0: out = out + (C[2][:, None] if np.ndim(val) else C[2])
    return out


def pol(E):
    Ex, Ey = E[0], E[1]; R, L = fc.cp_components(Ex, Ey)
    return dict(AR=float(fc.axial_ratio(Ex, Ey)), LR=float(20*np.log10(abs(L)/abs(R))),
                EyEx=float(20*np.log10(abs(Ey)/abs(Ex))))


out = {}
# ---------------- (a) sensitivity ----------------
base = fit(5.45e9, 6.15e9)
windows = [(5.45e9, 6.15e9), (5.50e9, 6.05e9), (5.55e9, 6.00e9), (5.40e9, 6.20e9), (5.60e9, 5.95e9)]
refits = []
for w in windows:
    F = fit(*w)
    refits.append(dict(window=[w[0]/1e9, w[1]/1e9], f1=F["p"][0]/1e9, f2=F["p"][1]/1e9, Q1=F["p"][2], Q2=F["p"][3],
                       err=F["err"], d1=pol(model_deriv(F, FC, d=1)), d2=pol(model_deriv(F, FC, d=2)),
                       carrier=pol(model_deriv(F, FC))))
out["refits"] = refits
pert = []
names = ["f1", "f2", "Q1", "Q2"]
steps = [5e6, 5e6, 0.10, 0.10]          # +-5 MHz, +-10 %
for i, (nm, st) in enumerate(zip(names, steps)):
    for sg in (-1, 1):
        F = dict(p=base["p"].copy(), C=base["C"].copy())
        F["p"][i] = F["p"][i] + sg*st if i < 2 else F["p"][i]*(1 + sg*st)
        pert.append(dict(param=nm, sign=sg, d1=pol(model_deriv(F, FC, d=1)), d2=pol(model_deriv(F, FC, d=2)),
                         carrier=pol(model_deriv(F, FC))))
for sg in (-1, 1):            # modal amplitude ratio +-1 dB
    F = dict(p=base["p"].copy(), C=base["C"].copy()); F["C"][1] = F["C"][1]*10**(sg/20)
    pert.append(dict(param="amp2", sign=sg, d1=pol(model_deriv(F, FC, d=1)), d2=pol(model_deriv(F, FC, d=2)),
                     carrier=pol(model_deriv(F, FC))))
for sg in (-1, 1):            # q +-10 %
    F = base
    pert.append(dict(param="q", sign=sg, d1=pol(model_deriv(F, FC, d=1, qu=q*(1+0.1*sg), qv=q*(1+0.1*sg))),
                     d2=pol(model_deriv(F, FC, d=2, qu=q*(1+0.1*sg), qv=q*(1+0.1*sg))), carrier=pol(model_deriv(F, FC))))
out["perturb"] = pert
# ---------------- (b) unequal modulation ----------------
rs = np.linspace(0.0, 2.0, 41); ub = []
for r in rs:
    # keep the mean sensitivity, change the ratio: q_u = 2q/(1+r), q_v = 2qr/(1+r)
    qu = 2*q/(1+r); qv = 2*q*r/(1+r)
    ub.append(dict(r=float(r), d1=pol(model_deriv(base, FC, d=1, qu=qu, qv=qv)), d2=pol(model_deriv(base, FC, d=2, qu=qu, qv=qv))))
out["unequal"] = ub
# ---------------- (c) non-perturbative harmonics at de = 0.2 ----------------
Mn = 4096; th = 2*np.pi*np.arange(Mn)/Mn
harm = {}
for de in (0.05, 0.2):
    e_t = EPS0 + de*np.sin(th)
    f1, f2, Q1, Q2 = base["p"]; C = base["C"]
    modal = []
    for (fi0, Qi) in ((f1, Q1), (f2, Q2)):
        fi = fi0*(e_t/EPS0)**(-q/2); modal.append(1/(1+1j*Qi*(FC/fi - fi/FC)))
    nn = np.arange(-4, 5)
    cm = np.array([[np.mean(M*np.exp(-1j*k*th)) for k in nn] for M in modal])      # modal Fourier coeffs
    E = np.einsum("ic,in->cn", C[:2], cm); E[:, 4] += C[2]
    harm[str(de)] = [dict(n=int(k), mod1=float(abs(cm[0, i])), mod2=float(abs(cm[1, i])),
                          dphase_deg=float(np.degrees(np.angle(cm[1, i]/cm[0, i]))), **pol(E[:, i]),
                          P_dBc=float(10*np.log10(np.sum(abs(E[:, i])**2)/np.sum(abs(E[:, 4])**2)))) for i, k in enumerate(nn)]
out["harm2mode"] = harm
out["base"] = dict(f1=base["p"][0]/1e9, f2=base["p"][1]/1e9, Q1=base["p"][2], Q2=base["p"][3], err=base["err"])
json.dump(out, open(f"{R2}/twomode_r3.json", "w"), indent=1)
print("refits:")
for r in refits: print("  %s f %.3f %.3f Q %.1f %.1f err %.4f | d1 AR %.1f LR %.1f | d2 AR %.1f LR %.1f | carrier AR %.2f" %
                       (r["window"], r["f1"], r["f2"], r["Q1"], r["Q2"], r["err"], r["d1"]["AR"], r["d1"]["LR"], r["d2"]["AR"], r["d2"]["LR"], r["carrier"]["AR"]))
print("perturbations:")
for p in pert: print("  %s %+d | d1 AR %.1f LR %.1f | d2 AR %.1f LR %.1f | carrier AR %.2f" % (p["param"], p["sign"], p["d1"]["AR"], p["d1"]["LR"], p["d2"]["AR"], p["d2"]["LR"], p["carrier"]["AR"]))
print("unequal (r, d1 AR, d1 LR, d2 LR):")
for u in ub[::4]: print("  %.2f %.1f %.1f %.1f" % (u["r"], u["d1"]["AR"], u["d1"]["LR"], u["d2"]["LR"]))
for de, hh in harm.items():
    print("two-mode nonperturbative de", de)
    for h in hh: print("  n=%+d |c1| %.3f |c2| %.3f dphi %7.1f AR %5.1f LR %6.1f P %6.1f" % (h["n"], h["mod1"], h["mod2"], h["dphase_deg"], h["AR"], h["LR"], h["P_dBc"]))
