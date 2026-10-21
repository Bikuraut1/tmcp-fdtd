"""Fig. 5 (revised): LHCP/RHCP patterns of the carrier and of the n = +1, +2 sidebands,
xz and yz planes, delta_eps = 0.05, f_m = f_c/200 (coarse grid, last DFT window)."""
import sys, os; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tmcp_paths import *  # ROOT, DATA, R1, R2, R2C, R3, COMM, OUT, FIG, TAB
import os, sys, numpy as np
from fig_style import plt, COL_W, C
TAG = os.environ.get("TAG", "200_0.05")
d = np.load(f"{R2C}/harm_{TAG}.npz")
th, ph = d["theta"], d["phi"]

def circ(Et, Ep):
    return (Et + 1j*Ep)/np.sqrt(2), (Et - 1j*Ep)/np.sqrt(2)

Ref = np.max(np.abs(d["Et_0"])**2 + np.abs(d["Ep_0"])**2)

def cut(A, phi_deg):
    i1 = np.argmin(np.abs(ph - np.deg2rad(phi_deg))); i2 = np.argmin(np.abs(ph - np.deg2rad(phi_deg + 180)))
    ang = np.concatenate([-np.rad2deg(th[::-1][:-1]), np.rad2deg(th)])
    return ang, np.concatenate([A[i2, ::-1][:-1], A[i1, :]])

fig, axs = plt.subplots(2, 3, figsize=(COL_W*1.0, 2.9), sharex=True, sharey=True)
for j, n in enumerate((0, 1, 2)):
    R, L = circ(d[f"Et_{n}"], d[f"Ep_{n}"])
    for i, phd in enumerate((0, 90)):
        ax = axs[i, j]
        a, v = cut(10*np.log10(np.abs(L)**2/Ref + 1e-12), phd); ax.plot(a, v, color=C["blue"], lw=0.9, label="LHCP")
        a, v = cut(10*np.log10(np.abs(R)**2/Ref + 1e-12), phd); ax.plot(a, v, "--", color=C["orange"], lw=0.9, label="RHCP")
        ax.set_xlim(-90, 90); ax.set_xticks([-90, 0, 90]); ax.set_ylim(-60, 2)
        if i == 0: ax.set_title({0: "carrier $n=0$", 1: "$n=+1$", 2: "$n=+2$"}[n], pad=2, fontsize=7)
        if j == 0: ax.set_ylabel(("$xz$-plane" if phd == 0 else "$yz$-plane") + "\n(dB)", fontsize=7)
        if i == 1: ax.set_xlabel(r"$\theta$ (deg)", fontsize=7)
axs[0, 0].legend(loc="lower center", fontsize=5.5, handlelength=1.6)
fig.tight_layout(w_pad=0.2, h_pad=0.3)
fig.savefig(f"{FIG}/fig5_sb_patterns.pdf"); fig.savefig(f"{FIG}/fig5_sb_patterns.png", dpi=200)
for n in (0, 1, 2, -1, -2):
    R, L = circ(d[f"Et_{n}"], d[f"Ep_{n}"]); U = np.abs(d[f"Et_{n}"])**2 + np.abs(d[f"Ep_{n}"])**2
    i, k = np.unravel_index(np.argmax(U), U.shape)
    # hpbw (total) in xz and yz
    print(n, "max at theta %.0f phi %.0f" % (np.rad2deg(th[k]), np.rad2deg(ph[i])),
          "broadside L,R dB rel carrier max: %.1f %.1f" % (10*np.log10(abs(L[0, 0])**2/Ref), 10*np.log10(abs(R[0, 0])**2/Ref)))
