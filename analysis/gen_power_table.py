"""Harmonic-resolved power balance table (Table 5) from the energy-accounting runs."""
import sys, os; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tmcp_paths import *  # ROOT, DATA, R1, R2, R2C, R3, COMM, OUT, FIG, TAB
import json, numpy as np
runs = {json.loads(l)["tag"]: json.loads(l) for l in open(f"{R3}/harm.jsonl")}
M = f"{TAB}"
cols = [("E_lossless_200_0.2", r"$\tan\delta=0$"), ("E_200_0.2", r"$\tan\delta=0.0009$")]
def pct(x): return f"{100*x:.2f}".replace("-", "$-$")
rows = {}
for tag, _ in cols:
    o = runs[tag]; w = o["windows"][-1]; e = o["energy_acc"]; n = np.array(o["harmonics"])
    F = np.array(w["flux"]); Fs = np.array(w["flux_sub"]); P = np.array(w["Prad"]); Ps = e["src"]
    r = {}
    for k in (0, 1, 2, 3):
        sel = np.abs(n) == k
        r[k] = (F[sel].sum() / Ps, P[sel].sum() / P.sum() * F.sum() / Ps)
    sel = np.abs(n) >= 4; r[4] = (F[sel].sum() / Ps, P[sel].sum() / P.sum() * F.sum() / Ps)
    r["tot"] = F.sum() / Ps; r["sub"] = Fs.sum() / Ps; r["diel"] = e["diel_box"] / Ps; r["mod"] = e["mod_box"] / Ps
    r["res"] = (F.sum() + e["diel_box"] - Ps - e["mod_box"]) / Ps
    f = np.array(o["freqs"]); i0 = list(n).index(0)
    r["MR"] = (f[i0] * np.sum(F / f) + e["diel_box"] - Ps) / Ps        # Manley-Rowe residual (loss counted at f_c)
    r["sb"] = F[n != 0].sum() / Ps
    r["port_sb"] = float(np.sum(np.array(w["Pacc"])[n != 0]) / Ps)
    r["Pacc"] = float(np.sum(w["Pacc"]) / Ps)
    rows[tag] = r
L = [r"\begin{table}[t]", r"\centering",
     r"\caption{Harmonic-resolved power balance for $\delta\eps=0.2$, $f_m=f_c/200$, in percent of the power $P_s$ delivered by the impressed source. Harmonic powers are the exact Poynting flux through the Huygens box (radiation plus the part guided in the substrate), with the box closed down to the ground.}",
     r"\label{tab:power}", r"\small", r"\begin{tabular}{@{}lrr@{}}", r"\toprule",
     " & " + " & ".join(c[1] for c in cols) + r" \\", r"\midrule"]
labels = {0: r"Carrier, $n=0$", 1: r"$n=\pm1$", 2: r"$n=\pm2$", 3: r"$n=\pm3$", 4: r"$|n|\ge4$"}
for k in (0, 1, 2, 3, 4):
    L.append(labels[k] + " & " + " & ".join(pct(rows[t][k][0]) for t, _ in cols) + r" \\")
L.append(r"\midrule")
L.append(r"Total flux, $\sum_nP_n$ & " + " & ".join(pct(rows[t]["tot"]) for t, _ in cols) + r" \\")
L.append(r"\quad of which sidebands, $\sum_{n\neq0}P_n$ & " + " & ".join(pct(rows[t]["sb"]) for t, _ in cols) + r" \\")
L.append(r"Dielectric loss in the box & " + " & ".join(pct(rows[t]["diel"]) for t, _ in cols) + r" \\")
L.append(r"Power from the modulation, $W_m$ & " + " & ".join(pct(rows[t]["mod"]) for t, _ in cols) + r" \\")
L.append(r"Residual, $\sum P_n+P_d-P_s-W_m$ & " + " & ".join(pct(rows[t]["res"]) for t, _ in cols) + r" \\")
L.append(r"Port power at $n\neq0$ (current source) & " + " & ".join(pct(rows[t]["port_sb"]) for t, _ in cols) + r" \\")
L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
open(f"{M}/tab_power.tex", "w").write("\n".join(L) + "\n")
print("\n".join(L)); print(json.dumps(rows, indent=0, default=float))
