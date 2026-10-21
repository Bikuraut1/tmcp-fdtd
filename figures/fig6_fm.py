import sys, os; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tmcp_paths import *  # ROOT, DATA, R1, R2, R2C, R3, COMM, OUT, FIG, TAB
import sys, os
import numpy as np
from fig_style import plt, COL_W, C

import fdtd_core as fc
from adiabatic import adiabatic_harmonics, fit_q
from analyze_mod import load_runs

Q_LW = float(os.environ.get("QLW", "239.8"))   # modal linewidth f_c/Q (MHz)
q, _ = fit_q()
runs = [r for r in load_runs() if r["delta_eps"] > 0]
fig, (a, b) = plt.subplots(2, 1, figsize=(COL_W, 3.5), sharex=True)
mk = {0.05: "o", 0.2: "s"}
for de, col in ((0.05, C["blue"]), (0.2, C["red"])):
    rr = sorted([r for r in runs if abs(r["delta_eps"] - de) < 1e-9], key=lambda r: r["f_mod"])
    if not rr:
        continue
    x = np.array([r["f_mod"] / 1e6 / Q_LW for r in rr])
    def lev(r, n):
        i0 = list(r["n"]).index(0); i = list(r["n"]).index(n)
        return 10 * np.log10(r["Prad"][i] / r["Prad"][i0])
    a.plot(x, [lev(r, -1) for r in rr], mk[de] + "--", color=col, ms=3.5, lw=0.8, mfc="w", label=rf"$n=-1$, $\delta\varepsilon$={de}")
    a.plot(x, [lev(r, +1) for r in rr], mk[de] + "-", color=col, ms=3.5, lw=0.8, label=rf"$n=+1$, $\delta\varepsilon$={de}")
    n, En = adiabatic_harmonics(de, q); i0 = list(n).index(0); i1 = list(n).index(1)
    lvl = 20 * np.log10(np.linalg.norm(En[:, i1]) / np.linalg.norm(En[:, i0]))
    a.axhline(lvl, color=col, lw=0.6, ls=":")
    # polarization: XPD (LHCP/RHCP) of n = +-2 and carrier AR
    xpd = lambda r, nn: 20 * np.log10(abs(r["EL"][list(r["n"]).index(nn)]) / abs(r["ER"][list(r["n"]).index(nn)]))
    b.plot(x, [xpd(r, 0) for r in rr], mk[de] + "-", color=col, ms=3.5, lw=0.8, label=rf"$n=0$, $\delta\varepsilon$={de}")
    b.plot(x, [xpd(r, 2) for r in rr], mk[de] + "--", color=col, ms=3.5, lw=0.8, mfc="w",
           label=rf"$n=+2$, $\delta\varepsilon$={de}")
    b.plot(x, [xpd(r, -2) for r in rr], mk[de] + ":", color=col, ms=2.5, lw=0.8, mfc="w", mew=0.6,
           label=rf"$n=-2$, $\delta\varepsilon$={de}")
a.set_xscale("log"); a.set_ylabel("First-sideband power (dBc)")
a.legend(fontsize=5.5, ncol=2, loc="lower left", handlelength=2.0)
a.set_ylim(-30, 0)
a.text(0.98, 0.9, "(a)", transform=a.transAxes, ha="right")
b.axhline(0, color=C["grey"], lw=0.5)
b.set_ylabel("LHCP/RHCP ratio (dB)"); b.set_xlabel(r"$f_m\,Q/f_c$")
b.legend(fontsize=5.2, ncol=3, loc="lower right", handlelength=1.8, columnspacing=0.6)
b.set_ylim(-30, 33)
b.text(0.98, 0.9, "(b)", transform=b.transAxes, ha="right")
from matplotlib.ticker import NullFormatter, FixedLocator
b.xaxis.set_major_locator(FixedLocator([0.1, 0.3, 1, 3])); b.set_xticklabels(["0.1", "0.3", "1", "3"])
b.xaxis.set_minor_formatter(NullFormatter()); a.xaxis.set_minor_formatter(NullFormatter())
b.set_xlim(0.1, 4)
fig.tight_layout(h_pad=0.4)
fig.savefig(f"{FIG}/fig6_fm.pdf", pad_inches=0.06); fig.savefig(f"{FIG}/fig6_fm.png", dpi=200)
for r in sorted(runs, key=lambda r: (r["delta_eps"], r["f_mod"])):
    i0 = list(r["n"]).index(0)
    print(r["tag"], "f_mQ/f_c=%.2f" % (r["f_mod"]/1e6/Q_LW), "Prad dBc", np.round(10*np.log10(r["Prad"]/r["Prad"][i0]), 2),
          "AR", np.round(r["AR_dB"], 2), "XPD", np.round(20*np.log10(np.abs(r["EL"])/np.abs(r["ER"])), 1))
