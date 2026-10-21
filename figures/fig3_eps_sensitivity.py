import sys, os; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tmcp_paths import *  # ROOT, DATA, R1, R2, R2C, R3, COMM, OUT, FIG, TAB
import sys
import numpy as np
from fig_style import plt, COL_W, C

import fdtd_core as fc

import os
FC = float(os.environ.get("FC", "5.754e9")) / 1e9
fig, (a, b) = plt.subplots(2, 1, figsize=(COL_W, 3.3), sharex=True)
EPS = (2.0, 2.1, 2.2, 2.3, 2.4)
cols = {2.0: C["purple"], 2.1: C["orange"], 2.2: C["blue"], 2.3: C["green"], 2.4: C["red"]}
lss = {2.0: (0, (1, 1)), 2.1: "--", 2.2: "-", 2.3: "-.", 2.4: (0, (5, 1, 1, 1, 1, 1))}
for eps in EPS:
    p = f"{R1}/pulse_eps{eps:.3f}.npz"
    if not os.path.exists(p): p = f"{R2C}/pulse_eps{eps:.3f}.npz"
    d = np.load(p)
    f = d["freqs"][:-1] / 1e9; Ex, Ey = d["Ex"][:-1], d["Ey"][:-1]
    R, L = fc.cp_components(Ex, Ey)
    a.plot(f, fc.axial_ratio(Ex, Ey), ls=lss[eps], color=cols[eps], label=rf"$\varepsilon_r$ = {eps:.2f}")
    m = (d["f"] / 1e9 > 5.3) & (d["f"] / 1e9 < 6.4)
    b.plot(d["f"][m] / 1e9, d["S11_dB"][m], ls=lss[eps], color=cols[eps])
for ax in (a, b):
    ax.axvline(FC, color=C["grey"], lw=0.6, ls=":")
a.axhline(3, color=C["grey"], lw=0.5, ls="-.")
a.set_ylim(0, 15); a.set_yticks([0, 5, 10, 15]); a.set_ylabel("Broadside AR (dB)")
a.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=5, fontsize=5.6, handlelength=2.0, columnspacing=0.8, handletextpad=0.4)
a.text(FC + 0.008, 0.5, r"$f_c$", fontsize=7)
a.text(0.02, 0.88, "(a)", transform=a.transAxes)
b.set_ylim(-25, 0); b.set_ylabel(r"$|S_{11}|$ (dB)"); b.set_xlabel("Frequency (GHz)")
b.set_xlim(FC - 0.35, FC + 0.45)
b.text(0.02, 0.08, "(b)", transform=b.transAxes)
fig.tight_layout(h_pad=0.4)
fig.savefig(f"{FIG}/fig3_eps.pdf"); fig.savefig(f"{FIG}/fig3_eps.png", dpi=200)
