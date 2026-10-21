import sys, os; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tmcp_paths import *  # ROOT, DATA, R1, R2, R2C, R3, COMM, OUT, FIG, TAB
import os, sys, csv
import numpy as np
from scipy.interpolate import CubicSpline
from fig_style import plt, COL_W, C
import fdtd_core as fc

WITH_CST = os.environ.get("CST", "0") == "1"
def load_csv(p):
    rows = [r for r in csv.reader(open(p)) if r]
    return np.array([[float(x) for x in r[:2]] for r in rows[1:]])
hs = load_csv(f"{COMM}/hfss/S_Parameter_Plot1.csv"); ha = load_csv(f"{COMM}/hfss/Axial_Ratio_Plot1.csv")
cs = np.loadtxt(f"{COMM}/cst/Truncated_Patch_S11.txt", comments="#")
ca = np.loadtxt(f"{COMM}/cst/Truncated_Patch_AR.txt", comments="#")
ih = np.load(f"{R1}/pulse_eps2.200.npz")
ref = np.load(f"{R1}/refined.npz")
dense = f"{R3}/dense_base.npz"


def ar_curve(freqs, Ex, Ey, ff):
    """AR from cubic-spline interpolation of the complex broadside field (smooth between DFT samples)."""
    o = np.argsort(freqs); f = freqs[o]
    sx = CubicSpline(f, Ex[o]); sy = CubicSpline(f, Ey[o])
    return fc.axial_ratio(sx(ff), sy(ff))


ff = np.linspace(5.4e9, 6.35e9, 1901)
fig, (a, b) = plt.subplots(2, 1, figsize=(COL_W, 3.3), sharex=True)
fr = (5.4, 6.4)
m = (ih["f"] > fr[0]*1e9) & (ih["f"] < fr[1]*1e9)
a.plot(ih["f"][m]/1e9, ih["S11_dB"][m], color=C["blue"], label=r"In-house FDTD, $\Delta x$ = 0.50 mm")
m2 = (ref["f"] > fr[0]*1e9) & (ref["f"] < fr[1]*1e9)
a.plot(ref["f"][m2]/1e9, ref["S11_dB"][m2], color=C["green"], lw=0.9, label=r"In-house FDTD, $\Delta x$ = 0.25 mm")
a.plot(hs[:, 0], hs[:, 1], "--", color=C["red"], label="HFSS")
if WITH_CST:
    a.plot(cs[:, 0], cs[:, 1], ":", color=C["orange"], label="CST ($W$ = 16.0 mm)")
a.axhline(-10, color=C["grey"], lw=0.5, ls="-.")
a.set_ylabel(r"$|S_{11}|$ (dB)"); a.set_ylim(-25, 0)
a.legend(loc="lower left", handlelength=2.0, fontsize=6)
a.text(0.98, 0.06, "(a)", transform=a.transAxes, ha="right")
if os.path.exists(dense):
    d = np.load(dense); fd = d["freqs"][:-1]
    b.plot(ff/1e9, ar_curve(fd, d["Ex"][:-1], d["Ey"][:-1], ff), color=C["blue"])
else:
    b.plot(ff/1e9, ar_curve(ih["freqs"][:-1], ih["Ex"][:-1], ih["Ey"][:-1], ff), color=C["blue"])
b.plot(ff/1e9, ar_curve(ref["freqs"], ref["Ex"], ref["Ey"], ff), color=C["green"], lw=0.9)
b.plot(ha[:, 0], ha[:, 1], "--", color=C["red"])
if WITH_CST:
    b.plot(ca[:, 0], ca[:, 1], ":", marker="o", ms=2, color=C["orange"])
b.axhline(3, color=C["grey"], lw=0.5, ls="-.")
b.set_ylim(0, 12); b.set_xlim(*fr)
b.set_xlabel("Frequency (GHz)"); b.set_ylabel("Broadside AR (dB)")
b.text(0.98, 0.88, "(b)", transform=b.transAxes, ha="right")
fig.tight_layout(h_pad=0.4)
name = "figS2_cst" if WITH_CST else "fig2_validation"
fig.savefig(f"{FIG}/{name}.pdf"); fig.savefig(f"{FIG}/{name}.png", dpi=200)
