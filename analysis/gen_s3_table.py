"""Table S3: coarse vs refined grid at f_m = f_c/200."""
import sys, os; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tmcp_paths import *  # ROOT, DATA, R1, R2, R2C, R3, COMM, OUT, FIG, TAB
import json, numpy as np, sys
import fdtd_core as fc
new = {json.loads(l)["tag"]: json.loads(l) for l in open(f"{R3}/harm.jsonl")}
def m(x, d=1):
    s = f"{x:.{d}f}"; s = s.lstrip("-") if float(s) == 0 else s
    return s.replace("-", "$-$")
def data(o):
    w = o["windows"][-1]; n = np.array(o["harmonics"]); P = np.array(w["Prad"]); i0 = list(n).index(0)
    Ex = np.array(w["Ex_re"]) + 1j*np.array(w["Ex_im"]); Ey = np.array(w["Ey_re"]) + 1j*np.array(w["Ey_im"]); R, L = fc.cp_components(Ex, Ey)
    return {int(k): (10*np.log10(P[i]/P[i0]), w["AR_dB"][i], 20*np.log10(abs(L[i])/abs(R[i]))) for i, k in enumerate(n)}, P[n != 0].sum()/P.sum()
L = [r"\begin{tabular}{@{}rrrrrrr@{}}\toprule",
     r" & \multicolumn{3}{c}{Coarse, $f_c=\SI{5.754}{GHz}$} & \multicolumn{3}{c}{Refined, $f_c=\SI{5.833}{GHz}$} \\",
     r"$n$ & $P_n/P_0$ (dB) & AR (dB) & $E_L/E_R$ (dB) & $P_n/P_0$ (dB) & AR (dB) & $E_L/E_R$ (dB) \\\midrule"]
summary = {}
for de, ct, rt in ((0.05, "E_200_0.05", "ref_200_0.05"), (0.2, "E_200_0.2", "ref_200_0.2")):
    if rt not in new: continue
    C, fcz = data(new[ct]); Rf, frz = data(new[rt])
    L.append(rf"\multicolumn{{7}}{{@{{}}l}}{{\emph{{$\delta\eps={de}$}} (sideband fraction {100*fcz:.1f}\% and {100*frz:.1f}\%)}} \\")
    for k in (-3, -2, -1, 0, 1, 2, 3):
        a, b = C[k], Rf[k]
        L.append(("$-$%d" % -k if k < 0 else ("+%d" % k if k > 0 else "0")) + " & " + " & ".join(m(v) for v in a) + " & " + " & ".join(m(v) for v in b) + r" \\")
    dl = {k: Rf[k][0] - C[k][0] for k in (-3, -2, -1, 1, 2, 3)}
    summary[de] = dict(dlev12=max(abs(dl[k]) for k in (-2, -1, 1, 2)), dlev3=max(abs(dl[k]) for k in (-3, 3)),
                       dLR12=max(abs(Rf[k][2] - C[k][2]) for k in (-2, -1, 1, 2)), dAR0=Rf[0][1] - C[0][1],
                       dAR_cp=max(abs(Rf[k][1] - C[k][1]) for k in (-3, -2, -1, 0, 1, 2, 3) if C[k][1] < 10),
                       fr=(fcz, frz), asym1=(C[-1][0] - C[1][0], Rf[-1][0] - Rf[1][0]))
    L.append(r"\addlinespace")
L[-1] = r"\bottomrule\end{tabular}"
open(f"{TAB}/tab_s3_rows.tex", "w").write("\n".join(L) + "\n")
print("\n".join(L)); print(json.dumps(summary, indent=1, default=float))
