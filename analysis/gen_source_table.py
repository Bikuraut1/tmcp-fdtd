"""Table: soft (ideal current) source vs 50-ohm resistive source, f_m = f_c/200."""
import sys, os; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tmcp_paths import *  # ROOT, DATA, R1, R2, R2C, R3, COMM, OUT, FIG, TAB
import json, numpy as np, sys
import fdtd_core as fc
old = {json.loads(l)["tag"]: json.loads(l) for l in open(f"{R2C}/harm.jsonl")}
new = {json.loads(l)["tag"]: json.loads(l) for l in open(f"{R3}/harm.jsonl")}
cases = [("soft, 0.05", new["E_200_0.05"]), (r"\SI{50}{\ohm}, 0.05", new["R50_200_0.05"]),
         ("soft, 0.2", new["E_200_0.2"]), (r"\SI{50}{\ohm}, 0.2", old["R50_200_0.2"])]
def m(x, d=1):
    s = f"{x:.{d}f}";  s = s.lstrip("-") if float(s) == 0 else s
    return s.replace("-", "$-$")
R = []
for lab, o in cases:
    w = o["windows"][-1]; n = np.array(o["harmonics"]); i0 = list(n).index(0)
    P = np.array(w["Prad"]); Pa = np.array(w["Pacc"])
    Ex = np.array(w["Ex_re"]) + 1j*np.array(w["Ex_im"]); Ey = np.array(w["Ey_re"]) + 1j*np.array(w["Ey_im"])
    Rr, L = fc.cp_components(Ex, Ey)
    ret = -np.sum(Pa[(n != 0)][Pa[n != 0] < 0])            # sideband power absorbed by the source resistance
    out = P.sum() + ret
    d = dict(lab=lab)
    for k in (-2, -1, 1, 2):
        i = list(n).index(k); d[f"P{k}"] = 10*np.log10(P[i]/P[i0]); d[f"AR{k}"] = w["AR_dB"][i]; d[f"LR{k}"] = 20*np.log10(abs(L[i])/abs(Rr[i]))
    d["AR0"] = w["AR_dB"][i0]
    d["frac_rad"] = P[n != 0].sum()/P.sum(); d["ret"] = ret/out; d["conv"] = P[n != 0].sum()/out
    R.append(d)
rows = [(r"$P_{-1}/P_0$, $P_{+1}/P_0$ (dB)", lambda d: m(d["P-1"]) + ", " + m(d["P1"])),
        (r"$P_{-2}/P_0$, $P_{+2}/P_0$ (dB)", lambda d: m(d["P-2"]) + ", " + m(d["P2"])),
        (r"Carrier AR (dB)", lambda d: m(d["AR0"])),
        (r"AR of $n=-1$, $+1$ (dB)", lambda d: m(d["AR-1"]) + ", " + m(d["AR1"])),
        (r"$E_L/E_R$ of $n=-1$, $+1$ (dB)", lambda d: m(d["LR-1"]) + ", " + m(d["LR1"])),
        (r"AR of $n=-2$, $+2$ (dB)", lambda d: m(d["AR-2"]) + ", " + m(d["AR2"])),
        (r"$E_L/E_R$ of $n=-2$, $+2$ (dB)", lambda d: m(d["LR-2"]) + ", " + m(d["LR2"])),
        (r"Radiated sideband fraction, $\sum_{n\neq0}P_n/\sum_nP_n$ (\%)", lambda d: m(100*d["frac_rad"])),
        (r"Sideband power returned to the source (\%)$^{a}$", lambda d: m(100*d["ret"])),
        (r"Radiated sideband power, share of output (\%)$^{a}$", lambda d: m(100*d["conv"]))]
L = [r"\begin{table}[t]", r"\centering",
     r"\caption{Effect of the source impedance for $f_m=f_c/200$: ideal current (soft) source versus a \SI{50}{\ohm} resistive source, for $\delta\eps=0.05$ and $0.2$.}",
     r"\label{tab:source}", r"\footnotesize", r"\setlength{\tabcolsep}{3pt}", r"\begin{tabular}{@{}lcccc@{}}", r"\toprule",
     r"Source, $\delta\eps$ & " + " & ".join(d["lab"] for d in R) + r" \\", r"\midrule"]
for lab, fn in rows:
    L.append(lab + " & " + " & ".join(fn(d) for d in R) + r" \\")
L += [r"\bottomrule", r"\end{tabular}", "", r"\smallskip",
      r"\footnotesize $^{a}$Relative to the total output power, i.e., the radiated power at all harmonics plus the sideband power absorbed by the source resistance.",
      r"\end{table}"]
open(f"{TAB}/tab_source.tex", "w").write("\n".join(L) + "\n")
print("\n".join(L))
