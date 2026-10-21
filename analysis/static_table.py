"""Table 1 of the letter: static performance (coarse grid, refined grid, HFSS).
AR from cubic-spline interpolation of the complex broadside field; band edges are interpolated crossings."""
import sys, os; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tmcp_paths import *  # ROOT, DATA, R1, R2, R2C, R3, COMM, OUT, FIG, TAB
import csv, json, numpy as np
from scipy.interpolate import CubicSpline
from scipy.signal import argrelmin
import fdtd_core as fc


def ar_band(f, Ex=None, Ey=None, ar=None, lo=5.6e9, hi=6.0e9):
    ff = np.linspace(max(lo, np.min(f)), min(hi, np.max(f)), 40001)
    if ar is None:
        o = np.argsort(f); a = fc.axial_ratio(CubicSpline(f[o], Ex[o])(ff), CubicSpline(f[o], Ey[o])(ff))
    else:
        a = CubicSpline(f, ar)(ff)
    k = np.argmin(a)
    return a[k], ff[k], ff[:k][a[:k] > 3][-1], ff[k:][a[k:] > 3][0]


def s11_band(f, s, lo=5.4e9, hi=6.4e9):
    m = (f > lo) & (f < hi); f, s = f[m], s[m]
    ff = np.linspace(f.min(), f.max(), 100001); ss = np.interp(ff, f, s)
    b = ff[ss < -10]
    mins = [(ff[i], ss[i]) for i in argrelmin(ss, order=2000)[0] if ss[i] < -5]
    return b[0], b[-1], mins


def m(x, d):
    s = f"{x:.{d}f}"; return s.replace("-", "$-$")


rows = []
d = np.load(f"{R3}/dense_base.npz"); nb = int(d["nbase"])          # coarse grid, 2-MHz DFT grid available
fd = d["freqs"][:nb]
nd = len(d["freqs"]) - 1                                         # coarse row: AR on the 2-MHz DFT grid
rows.append((r"In-house, $\Delta x=\SI{0.50}{mm}$", ar_band(d["freqs"][nb:nd], ar=fc.axial_ratio(d["Ex"][nb:nd], d["Ey"][nb:nd])), s11_band(d["f"], d["S11_dB"]), True))
r = np.load(f"{R1}/refined.npz")
rows.append((r"In-house, $\Delta x=\SI{0.25}{mm}$", ar_band(r["freqs"], r["Ex"], r["Ey"], lo=5.65e9), s11_band(r["f"], r["S11_dB"]), False))
def load_csv(p):
    rr = [x for x in csv.reader(open(p)) if x]
    return np.array([[float(v) for v in x[:2]] for x in rr[1:]])
ha = load_csv(f"{COMM}/hfss/Axial_Ratio_Plot1.csv"); hs = load_csv(f"{COMM}/hfss/S_Parameter_Plot1.csv")
rows.append(("HFSS", ar_band(ha[:, 0] * 1e9, ar=ha[:, 1]), s11_band(hs[:, 0] * 1e9, hs[:, 1]), False))
# check: spline interpolation of the 21.7-MHz grid vs the dense grid (coarse model)
a_s, f_s, lo_s, hi_s = ar_band(fd, d["Ex"][:nb], d["Ey"][:nb])
check = dict(spline_ARmin=float(a_s), spline_f=float(f_s / 1e9), spline_band=[float(lo_s / 1e9), float(hi_s / 1e9)])

L = [r"\begin{table}[t]", r"\centering",
     r"\caption{Static performance of the antenna ($\delta\eps=0$). Frequencies in GHz. Band edges are interpolated crossings of the \SI{3}{dB} AR and \SI{-10}{dB} $|S_{11}|$ levels; the AR of the in-house results is evaluated from a cubic-spline interpolation of the complex broadside field (checked against a \SI{2}{MHz} DFT grid on the coarse grid).}",
     r"\label{tab:static}", r"\footnotesize", r"\setlength{\tabcolsep}{3.5pt}", r"\begin{tabular}{@{}lccccc@{}}", r"\toprule",
     r"Solver & AR$_{\min}$ (dB) & $f(\mathrm{AR}_{\min})$ & \SI{3}{dB} AR band & \SI{-10}{dB} $S_{11}$ band & $S_{11}$ minima (GHz/dB) \\", r"\midrule"]
for lab, (a, fa, lo, hi), (b0, b1, mins), foot in rows:
    L.append(f"{lab} & {a:.2f} & {fa/1e9:.3f} & {lo/1e9:.3f}--{hi/1e9:.3f} & {b0/1e9:.3f}--{b1/1e9:.3f}" + (r"$^{a}$" if foot else "")
             + " & " + "; ".join(f"{x/1e9:.3f}/{m(y, 1)}" for x, y in mins[:2]) + r" \\")
L += [r"\bottomrule", r"\end{tabular}", "", r"\smallskip", r"\footnotesize $^{a}$The lower resonance reaches only \SI{-9.6}{dB} on the coarse grid.", r"\end{table}"]
open(f"{TAB}/tab_static.tex", "w").write("\n".join(L) + "\n")
json.dump(check, open(f"{TAB}/tab_static_dense_check.json", "w"), indent=1)
print("\n".join(L)); print("dense-grid check (coarse):", check)
