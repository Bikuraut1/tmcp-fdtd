"""Fig. S1: modulated-cavity benchmark (delta_eps = 0.2, f_m = f0/20): FDTD (D-based and naive) vs Hill-equation reference."""
import sys, os; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tmcp_paths import *  # ROOT, DATA, R1, R2, R2C, R3, COMM, OUT, FIG, TAB
import numpy as np
from fig_style import plt, COL_W, C
d = np.load(f"{R2}/bench_de0.2_N20.npz")
ref, eD, eN, dt, f0 = d["ref"], d["eD"], d["eN"], float(d["dt"]), float(d["f0"])
t = (np.arange(len(ref)) + 1) * dt * 1e9
sc = np.abs(ref).max()
fig, (a, b) = plt.subplots(2, 1, figsize=(COL_W, 3.0))
z = t < 3.0
a.plot(t[z], ref[z] / sc, color=C["grey"], lw=2.2, alpha=0.45, label="Hill-equation reference")
a.plot(t[z], eD[z] / sc, color=C["blue"], lw=0.7, label="FDTD, D-based update")
a.plot(t[z], eN[z] / sc, "--", color=C["red"], lw=0.7, label="FDTD, naive update")
a.set_xlim(0, 3); a.set_ylim(-1.15, 1.6); a.set_ylabel("$E_y$ (norm.)"); a.set_xlabel("Time (ns)")
a.legend(loc="upper center", fontsize=5.5, ncol=3, handlelength=1.5, columnspacing=0.6)
a.text(0.94, 0.06, "(a)", transform=a.transAxes)
# max error per carrier period
P = int(round(1 / (f0 * dt))); nb = len(t) // P
eb = lambda e: np.abs(e[:nb * P] - ref[:nb * P]).reshape(nb, P).max(1) / sc
tb = t[:nb * P].reshape(nb, P).mean(1)
b.semilogy(tb, eb(eD), color=C["blue"], lw=0.8, label="D-based update (this work)")
b.semilogy(tb, eb(eN), color=C["red"], lw=0.8, label="naive update")
b.set_ylim(1e-6, 1e-1); b.set_ylabel("Max. error / max $|E_y|$"); b.set_xlabel("Time (ns)")
b.legend(loc="lower right", fontsize=6); b.text(0.01, 0.06, "(b)", transform=b.transAxes)
fig.tight_layout(h_pad=0.4)
fig.savefig(f"{FIG}/figS1_benchmark.pdf"); fig.savefig(f"{FIG}/figS1_benchmark.png", dpi=200)
