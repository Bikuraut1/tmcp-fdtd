"""
Two-mode (cavity) model fitted to the static FDTD broadside transfer function, used to
predict sideband polarization quantitatively (reviewer 1, comment 3).

  H(f; eps) = sum_i  p_i * M_i(f; eps) + b,   M_i = 1/(1 + j Q_i (f/f_i - f_i/f)),
  p_i (complex 2-vectors: Ex, Ey modal radiation vectors), f_i(eps) = f_i0 (eps/eps0)^(-q/2).
Fitted on eps = 2.2 only; q taken from adiabatic.fit_q.  Derivatives with respect to eps are then
analytic, and the quasi-static sideband phasors are
  E_{+-1} = +-(de/2j) dH/deps,   E_{+-2} = -(de^2/8) d2H/deps2.
"""
import sys, os; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tmcp_paths import *  # ROOT, DATA, R1, R2, R2C, R3, COMM, OUT, FIG, TAB
import sys, os, json, numpy as np
os.environ.setdefault("FC", "5.754e9"); os.environ.setdefault("DATA", f"{R1}")
import fdtd_core as fc, adiabatic as ad
from scipy.optimize import least_squares

FC = float(os.environ["FC"]); EPS0 = 2.2
f, H = ad.H_curve(2.2)
band = (f > 5.45e9) & (f < 6.15e9)
fb, Hb = f[band], H[:, band]
scale = np.abs(Hb).max(); Hb = Hb / scale


def basis(fv, p):
    f1, f2, Q1, Q2 = p
    M1 = 1 / (1 + 1j*Q1*(fv/f1 - f1/fv)); M2 = 1 / (1 + 1j*Q2*(fv/f2 - f2/fv))
    return np.stack([M1, M2, np.ones_like(M1)], 1)


def lin(p):
    A = basis(fb, p)
    C = np.linalg.lstsq(A, Hb.T, rcond=None)[0]     # (3, 2): rows = M1, M2, background
    return A, C


def resid(p):
    A, C = lin(p); r = (A @ C - Hb.T).ravel()
    return np.concatenate([r.real, r.imag])


best = None
for f1 in np.linspace(5.6e9, 5.8e9, 5):
    for f2 in np.linspace(5.8e9, 6.0e9, 5):
        s = least_squares(resid, [f1, f2, 30, 30], x_scale=[1e8, 1e8, 10, 10],
                          bounds=([5.3e9, 5.3e9, 3, 3], [6.3e9, 6.3e9, 300, 300]))
        if best is None or s.cost < best.cost: best = s
p = best.x; A, C = lin(p)
fit_err = np.sqrt(np.sum(np.abs(A @ C - Hb.T)**2) / np.sum(np.abs(Hb)**2))
q, _ = ad.fit_q()


def model(fv, eps, d=0):
    """H and its eps-derivatives (d = 0, 1, 2) at frequency fv for permittivity eps."""
    f1, f2, Q1, Q2 = p
    out = np.zeros((2,) + np.shape(fv), complex)
    for i, (fi0, Qi) in enumerate(((f1, Q1), (f2, Q2))):
        fi = fi0 * (eps/EPS0)**(-q/2)
        x = Qi*(fv/fi - fi/fv)
        dfi = -(q/2)*fi/eps; d2fi = (q/2)*(q/2 + 1)*fi/eps**2
        dx_dfi = -Qi*(fv/fi**2 + 1/fv); d2x_dfi2 = 2*Qi*fv/fi**3
        dx = dx_dfi*dfi; d2x = d2x_dfi2*dfi**2 + dx_dfi*d2fi
        M = 1/(1 + 1j*x); M1 = -1j*M**2; M2 = -2*M**3
        val = M if d == 0 else (M1*dx if d == 1 else M2*dx**2 + M1*d2x)
        out += C[i][:, None] * val if np.ndim(fv) else C[i] * val
    if d == 0: out = out + (C[2][:, None] if np.ndim(fv) else C[2])
    return out


def pol(E):
    Ex, Ey = E
    R, L = fc.cp_components(Ex, Ey)
    return dict(EyEx_dB=float(20*np.log10(abs(Ey)/abs(Ex))), phase_deg=float(np.degrees(np.angle(Ey/Ex))),
                AR_dB=float(fc.axial_ratio(Ex, Ey)), LR_dB=float(20*np.log10(abs(L)/abs(R))))


res = dict(fit=dict(f1_GHz=p[0]/1e9, f2_GHz=p[1]/1e9, Q1=p[2], Q2=p[3], rel_rms=float(fit_err), q=q,
                    p1=[complex(c).__repr__() for c in C[0]], p2=[complex(c).__repr__() for c in C[1]],
                    b=[complex(c).__repr__() for c in C[2]]))
# modal polarization vectors
for i in (0, 1):
    res["fit"][f"mode{i+1}"] = pol(C[i])
# predictions at the carrier
res["carrier"] = pol(model(FC, EPS0))
for de in (0.05, 0.2):
    E1p = (de/2j)*model(FC, EPS0, 1); E1m = -E1p
    E2 = -(de**2/8)*model(FC, EPS0, 2)
    res[f"de{de}"] = {"+1": pol(E1p), "-1": pol(E1m), "+-2": pol(E2),
                      "E1_dBc": float(20*np.log10(np.linalg.norm(E1p)/np.linalg.norm(model(FC, EPS0)))),
                      "E2_dBc": float(20*np.log10(np.linalg.norm(E2)/np.linalg.norm(model(FC, EPS0))))}
# finite-difference (direct static FDTD) derivatives at FC, for comparison
Hc = {e: ad.H_curve(e) for e in (2.15, 2.2, 2.25)}
interp = lambda e: np.array([np.interp(FC, Hc[e][0], Hc[e][1][c].real) + 1j*np.interp(FC, Hc[e][0], Hc[e][1][c].imag)
                             for c in range(2)]) / scale
d1 = (interp(2.25) - interp(2.15)) / 0.1
d2 = (interp(2.25) - 2*interp(2.2) + interp(2.15)) / 0.05**2
res["fd_static"] = {"dH": pol(d1), "d2H": pol(d2)}
res["model_static_deriv"] = {"dH": pol(model(FC, EPS0, 1)), "d2H": pol(model(FC, EPS0, 2))}
# sweep of first/second-order sideband polarization vs carrier frequency (model and FD)
fs = np.linspace(5.65e9, 5.90e9, 26)
sweep = []
for fv in fs:
    I = lambda e: np.array([np.interp(fv, Hc[e][0], Hc[e][1][c].real) + 1j*np.interp(fv, Hc[e][0], Hc[e][1][c].imag)
                            for c in range(2)]) / scale
    fd1 = (I(2.25) - I(2.15))/0.1; fd2 = (I(2.25) - 2*I(2.2) + I(2.15))/0.05**2
    sweep.append(dict(f_GHz=fv/1e9, carrier=pol(model(fv, EPS0)), carrier_fdtd=pol(I(2.2)),
                      m1=pol(model(fv, EPS0, 1)), fd1=pol(fd1), m2=pol(model(fv, EPS0, 2)), fd2=pol(fd2)))
res["sweep"] = sweep
json.dump(res, open(f"{R2}/twomode.json", "w"), indent=1)
print(json.dumps({k: v for k, v in res.items() if k != "sweep"}, indent=1))
for s in sweep[::3]:
    print(round(s["f_GHz"], 3), "carrier AR %.1f/%.1f" % (s["carrier"]["AR_dB"], s["carrier_fdtd"]["AR_dB"]),
          "| dH AR %.1f/%.1f LR %.1f/%.1f" % (s["m1"]["AR_dB"], s["fd1"]["AR_dB"], s["m1"]["LR_dB"], s["fd1"]["LR_dB"]),
          "| d2H AR %.1f/%.1f LR %.1f/%.1f" % (s["m2"]["AR_dB"], s["fd2"]["AR_dB"], s["m2"]["LR_dB"], s["fd2"]["LR_dB"]))
