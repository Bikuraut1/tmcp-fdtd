"""Fig. 7: sideband polarization vs the modal tuning ratio r = (dln f_v/deps)/(dln f_u/deps):
two-mode model (lines), static FDTD derivatives and time-varying FDTD for partial modulation profiles (markers)."""
import sys, os; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tmcp_paths import *  # ROOT, DATA, R1, R2, R2C, R3, COMM, OUT, FIG, TAB
import json, numpy as np, sys
from fig_style import plt, COL_W, C
import fdtd_core as fc
t = json.load(open(f"{R2}/twomode_r3.json"))["unequal"]
U = json.load(open(f"{R2}/unequal_r3.json"))
runs = {json.loads(l)["tag"]: json.loads(l) for l in open(f"{R3}/harm.jsonl")}
r = np.array([x["r"] for x in t])
fig, (a, b) = plt.subplots(2, 1, figsize=(COL_W, 3.2), sharex=True)
a.plot(r, [x["d1"]["AR"] for x in t], color=C["blue"], label="two-mode model")
b.plot(r, [x["d2"]["LR"] for x in t], color=C["blue"])
mk = {"uquad": "s", "patch": "D", "uniform": "o"}
names = {"uquad": "two quadrants", "patch": "under patch only", "uniform": "uniform"}
pts = {"uniform": (1.0, runs.get("E_200_0.05"))}
for k in ("uquad", "patch"):
    if k in U: pts[k] = (U[k]["r"], runs.get(f"{k}_200_0.05"))
for k, (rv, o) in pts.items():
    if o is None: continue
    w = o["windows"][-1]; n = np.array(o["harmonics"])
    Ex = np.array(w["Ex_re"]) + 1j*np.array(w["Ex_im"]); Ey = np.array(w["Ey_re"]) + 1j*np.array(w["Ey_im"])
    R, L = fc.cp_components(Ex, Ey)
    for nn, mfc in ((-1, "w"), (1, C["red"])):
        i = list(n).index(nn); a.plot(rv, w["AR_dB"][i], mk[k], color=C["red"], mfc=mfc, ms=4)
    for nn, mfc in ((-2, "w"), (2, C["red"])):
        i = list(n).index(nn); b.plot(rv, 20*np.log10(abs(L[i])/abs(R[i])), mk[k], color=C["red"], mfc=mfc, ms=4)
    a.plot([], [], mk[k], color=C["red"], ms=4, label=f"FDTD, {names[k]}")
for k in ("uquad", "patch"):
    if k in U:
        a.plot(U[k]["r"], U[k]["fd_d1"]["AR"], "x", color="k", ms=5, mew=1.0)
        b.plot(U[k]["r"], U[k]["fd_d2"]["LR"], "x", color="k", ms=5, mew=1.0)
T0 = json.load(open(f"{R2}/twomode.json"))["fd_static"]
a.plot(1.0, T0["dH"]["AR_dB"], "x", color="k", ms=5, mew=1.0); b.plot(1.0, T0["d2H"]["LR_dB"], "x", color="k", ms=5, mew=1.0)
a.plot([], [], "x", color="k", ms=5, mew=1.0, label="static FDTD derivative")
a.set_ylabel("AR of $n=\\pm1$ (dB)"); a.set_ylim(0, 40); a.axhline(3, color=C["grey"], lw=0.5, ls="-.")
a.legend(fontsize=5.5, loc="upper right", ncol=2, handlelength=1.5)
b.set_ylabel("$E_L/E_R$ of $n=\\pm2$ (dB)"); b.set_xlabel(r"Modal tuning ratio $r$")
b.axhline(0, color=C["grey"], lw=0.5); b.set_ylim(-20, 5); b.set_xlim(0, 2)
a.text(0.02, 0.88, "(a)", transform=a.transAxes); b.text(0.02, 0.88, "(b)", transform=b.transAxes)
fig.tight_layout(h_pad=0.3)
fig.savefig(f"{FIG}/fig7_unequal.pdf"); fig.savefig(f"{FIG}/fig7_unequal.png", dpi=200)
