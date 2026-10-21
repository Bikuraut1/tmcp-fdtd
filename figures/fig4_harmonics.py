import sys, os; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tmcp_paths import *  # ROOT, DATA, R1, R2, R2C, R3, COMM, OUT, FIG, TAB
import sys, json
import numpy as np
from fig_style import plt, COL_W, C
import fdtd_core as fc
from adiabatic import adiabatic_harmonics, fit_q
NHP = 4
q, _ = fit_q()
runs = {}
for l in open(f"{R2C}/harm.jsonl"):
    o = json.loads(l); w = o["windows"][-1]
    Ex = np.array(w["Ex_re"]) + 1j*np.array(w["Ex_im"]); Ey = np.array(w["Ey_re"]) + 1j*np.array(w["Ey_im"])
    n = np.array(o["harmonics"]); sel = np.abs(n) <= NHP
    ER, EL = fc.cp_components(Ex[sel], Ey[sel])
    runs[o["tag"]] = dict(n=n[sel], EL=EL, ER=ER, AR_dB=np.array(w["AR_dB"])[sel])
cases = [("200_0.05", 0.05), ("200_0.2", 0.2)]
fig, axs = plt.subplots(2, 2, figsize=(COL_W * 1.0, 3.6), sharex=True,
                        gridspec_kw=dict(height_ratios=[1.35, 1]))
w = 0.36
for j, (tag, de) in enumerate(cases):
    ax, axp = axs[0, j], axs[1, j]
    n, En = adiabatic_harmonics(de, q, nh=NHP)
    Ra, La = fc.cp_components(En[0], En[1]); i0 = list(n).index(0)
    ref_a = abs(La[i0])
    if tag in runs:
        r = runs[tag]; i0f = list(r["n"]).index(0); ref = abs(r["EL"][i0f])
        ax.bar(r["n"] - w/2, 20*np.log10(np.abs(r["EL"])/ref) + 70, bottom=-70, width=w, color=C["blue"], label="FDTD, LHCP")
        ax.bar(r["n"] + w/2, 20*np.log10(np.abs(r["ER"])/ref) + 70, bottom=-70, width=w, color=C["orange"], label="FDTD, RHCP")
        axp.plot(r["n"], r["AR_dB"], "o", color=C["blue"], ms=3.5, label="FDTD")
    ax.plot(n - w/2, 20*np.log10(np.abs(La)/ref_a), "k_", ms=7, mew=1.0, label="adiabatic model")
    ax.plot(n + w/2, 20*np.log10(np.abs(Ra)/ref_a), "k_", ms=7, mew=1.0)
    axp.plot(n, fc.axial_ratio(En[0], En[1]), "x", color="k", ms=4, mew=0.8, label="adiabatic model")
    ax.set_ylim(-70, 5); axp.set_ylim(0, 52); axp.set_yticks([0, 10, 20, 30, 40, 50])
    ax.set_title(rf"$\delta\varepsilon$ = {de}", pad=2)
    axp.set_xlabel("Harmonic order $n$"); axp.set_xticks([-4, -2, 0, 2, 4])
    axp.axhline(3, color=C["grey"], lw=0.5, ls="-.")
    if j == 0:
        ax.set_ylabel("Broadside level (dB)\nre carrier LHCP")
        axp.set_ylabel("AR (dB)")
    ax.text(0.03, 0.92, "(%s)" % "ab"[j], transform=ax.transAxes, fontsize=7)
    axp.text(0.03, 0.85, "(%s)" % "cd"[j], transform=axp.transAxes, fontsize=7)
h, l = axs[0, 1].get_legend_handles_labels()
fig.legend(h, l, loc="upper center", ncol=3, fontsize=6.5, bbox_to_anchor=(0.55, 1.035), handlelength=1.4)
axs[1, 1].legend(loc="upper right", fontsize=6, handlelength=1.0)
fig.tight_layout(h_pad=0.3, w_pad=0.6, rect=(0, 0, 1, 0.96))
fig.savefig(f"{FIG}/fig4_harmonics.pdf"); fig.savefig(f"{FIG}/fig4_harmonics.png", dpi=200)
