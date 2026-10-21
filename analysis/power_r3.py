import sys, os; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tmcp_paths import *  # ROOT, DATA, R1, R2, R2C, R3, COMM, OUT, FIG, TAB
"""Power balance and Manley-Rowe check of the energy-accounting runs (round 3).
    python analysis/power_r3.py [TAG ...]       (default: every run with energy=1)"""
import json, numpy as np, sys
runs = {json.loads(l)["tag"]: json.loads(l) for l in open(f"{R3}/harm.jsonl")}
for tag in sys.argv[1:] or [t for t, o in runs.items() if o["opts"].get("energy")]:
    o = runs[tag]; w = o["windows"][-1]; e = o["energy_acc"]
    n = np.array(o["harmonics"]); f = np.array(o["freqs"]); i0 = list(n).index(0)
    F = np.array(w["flux"]); Fs = np.array(w["flux_sub"]); P = np.array(w["Prad"]); Pa = np.array(w["Pacc"])
    Ps = e.get("src", np.nan)
    print("==", tag, o["opts"])
    print(" Psrc %.5e  Pacc(port,all) %.5e  W_mod(box) %.3e  P_diel(box) %.3e  sum flux %.5e  sum NTFF %.5e" % (Ps, Pa.sum(), e["mod_box"], e["diel_box"], F.sum(), P.sum()))
    print(" closure (sumflux + diel - Psrc - mod)/Psrc = %.4f" % ((F.sum() + e["diel_box"] - Ps - e["mod_box"]) / Ps))
    print(" W_mod/Psrc = %.2e ; Manley-Rowe f_c*sum(flux_n/f_n)/(Psrc - ?): MR = %.5f vs sumflux/Psrc %.5f" % (e["mod_box"] / Ps, FC_ := f[i0] * np.sum(F / f) / Ps, F.sum() / Ps))
    print(" sideband fraction: flux %.4f  NTFF %.4f ; surface/substrate share of flux %.4f" % (F[n != 0].sum() / F.sum(), P[n != 0].sum() / P.sum(), Fs.sum() / F.sum()))
    for k in range(len(n)):
        if abs(n[k]) <= 4:
            print("  n=%+d flux/Psrc %.4f (sub %.4f) NTFF/Psrc %.4f | flux dBc %6.2f NTFF dBc %6.2f" % (n[k], F[k] / Ps, Fs[k] / Ps, P[k] / Ps, 10 * np.log10(F[k] / F[i0]), 10 * np.log10(P[k] / P[i0])))
