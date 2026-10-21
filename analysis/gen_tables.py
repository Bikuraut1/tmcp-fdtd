"""Generate LaTeX tables for the revision from analysis_rev.json / twomode.json (no hand transcription)."""
import sys, os; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tmcp_paths import *  # ROOT, DATA, R1, R2, R2C, R3, COMM, OUT, FIG, TAB
import json, sys, os, numpy as np
os.environ.setdefault("FC", "5.754e9"); os.environ.setdefault("DATA", f"{R1}")
import fdtd_core as fc, adiabatic as ad
R = json.load(open(f"{R2}/analysis_rev.json"))
T = json.load(open(f"{R2}/twomode.json"))
runs = {json.loads(l)["tag"]: json.loads(l) for l in open(f"{R2C}/harm.jsonl")}
M = f"{TAB}"

def m(x, d=1):
    s = f"{x:.{d}f}"
    if float(s) == 0: s = s.lstrip("-")
    return s.replace("-", "$-$")

# ---------------- harmonic table ----------------
lines = [r"\begin{table}[t]", r"\centering",
         r"\caption{Harmonics radiated by the modulated antenna for $f_m=f_c/200=\SI{28.8}{MHz}$ (time-varying FDTD, coarse grid). $P_n$: hemisphere-integrated radiated power relative to the carrier; AR and $E_L/E_R$ (LHCP/RHCP amplitude ratio) at broadside; $\Delta$: FDTD minus the adiabatic model (scaling law, as in Fig.~\ref{fig:harm}). Note that $P_n$ is an integrated power, whereas Fig.~\ref{fig:harm} shows broadside field components.}",
         r"\label{tab:harm}", r"\small", r"\setlength{\tabcolsep}{4pt}",
         r"\begin{tabular}{@{}rcrrrlrr@{}}", r"\toprule",
         r"$n$ & $f_c+nf_m$ (GHz) & $P_n$ (dBc) & AR (dB) & $E_L/E_R$ (dB) & State & $\Delta P_n$ (dB) & $\Delta$AR (dB) \\", r"\midrule"]
for tag, de in (("200_0.05", 0.05), ("200_0.2", 0.2)):
    r = R[tag]; tab = {row["n"]: row for row in r["table"]}
    lines.append(rf"\multicolumn{{8}}{{@{{}}l}}{{\emph{{$\delta\eps={de}$}}}} \\")
    for n in (-3, -2, -1, 0, 1, 2, 3):
        row = tab[n]; a = r["adiab_scaled"][str(n)] if str(n) in r["adiab_scaled"] else r["adiab_scaled"][n]
        dP = row["P_dBc"] - a["P_dBc"]; dA = row["AR_dB"] - a["AR_dB"]
        show_dev = True
        lines.append(("$-$" + str(-n) if n < 0 else ("+" + str(n) if n > 0 else "0")) + f" & {row['f_GHz']:.4f} & {m(row['P_dBc'])} & {m(row['AR_dB'])} & {m(row['LR_dB'])} & {row['state']} & "
                     + (f"{m(dP)} & {m(dA)}" if (show_dev and n != 0) else ("-- & " + m(dA) if n == 0 else "-- & --")) + r" \\")
    lines.append(r"\addlinespace")
lines[-1] = r"\bottomrule"
lines += [r"\end{tabular}", "", r"\smallskip",
          r"\footnotesize State: LHCP/RHCP if AR $<\SI{3}{dB}$; LHCP- or RHCP-dominant if $\SI{3}{dB}\le$ AR $\le\SI{10}{dB}$; linear if AR $>\SI{10}{dB}$. For nearly linear harmonics (AR $>\SI{15}{dB}$) the AR is ill-conditioned, and $\Delta$AR is not a meaningful measure of agreement. Harmonics up to $|n|=6$ were computed; for $|n|=4$ they are at $-61$ to $-66$~dBc ($\delta\eps=0.05$) and $-22$ to $-24$~dBc ($\delta\eps=0.2$) (Supporting Information).",
          r"\end{table}"]
open(f"{M}/tab_harm.tex", "w").write("\n".join(lines) + "\n")

# ---------------- polarization comparison table ----------------
def pol(Ex, Ey):
    Rr, L = fc.cp_components(Ex, Ey)
    return dict(EyEx_dB=20*np.log10(abs(Ey)/abs(Ex)), phase_deg=np.degrees(np.angle(Ey/Ex)), AR_dB=fc.axial_ratio(Ex, Ey),
                LR_dB=20*np.log10(abs(L)/abs(Rr)))
w = runs["200_0.05"]["windows"][-1]
Ex = np.array(w["Ex_re"]) + 1j*np.array(w["Ex_im"]); Ey = np.array(w["Ey_re"]) + 1j*np.array(w["Ey_im"])
q, _ = ad.fit_q(); n, En = ad.adiabatic_harmonics(0.05, q, nh=3)
adi = {int(k): pol(En[0, i], En[1, i]) for i, k in enumerate(n)}
rows = []
def fmt(p): return f"{m(p['EyEx_dB'])} & {m(p['phase_deg'], 0)} & {m(p['AR_dB'])} & {m(p['LR_dB'])}"
for order, d1, d2 in ((1, "+1", "dH"), (2, "+-2", "d2H")):
    lab = r"$\pm1$" if order == 1 else r"$\pm2$"
    ideal = r"$-\infty$ & -- & $\infty$ & 0" if order == 1 else r"0 & $-90$ & 0 & $-\infty$"
    rows.append(rf"\multicolumn{{5}}{{@{{}}l}}{{\emph{{Sidebands $n={lab[1:-1]}$}}}} \\")
    rows.append(rf"Ideal two-mode ($x_{{u,v}}=\pm1$) & {ideal} \\")
    rows.append(rf"Fitted two-mode model, Eq.~(\ref{{eq:qs}}) & {fmt(T['de0.05'][d1])} \\")
    rows.append(rf"Static FDTD derivative, Eq.~(\ref{{eq:qs}}) & {fmt(T['fd_static'][d2])} \\")
    rows.append(rf"Adiabatic model & {fmt(adi[order])} \\")
    rows.append(rf"Time-varying FDTD, $n=-{order}$ & {fmt(pol(Ex[6-order], Ey[6-order]))} \\")
    rows.append(rf"Time-varying FDTD, $n=+{order}$ & {fmt(pol(Ex[6+order], Ey[6+order]))} \\")
    rows.append(r"\addlinespace")
rows[-1] = r"\bottomrule"
tab = [r"\begin{table}[t]", r"\centering",
       r"\caption{Polarization of the first- and second-order sidebands at $f_c$ for $\delta\eps=0.05$, $f_m=f_c/200$: ratio and phase of $E_y/E_x$, AR and LHCP/RHCP ratio at broadside. The carrier is LHCP ($E_L/E_R=\SI{22.5}{dB}$ static).}",
       r"\label{tab:pol}", r"\small", r"\setlength{\tabcolsep}{4pt}",
       r"\begin{tabular}{@{}lrrrr@{}}", r"\toprule",
       r" & $|E_y/E_x|$ (dB) & $\angle E_y/E_x$ ($^\circ$) & AR (dB) & $E_L/E_R$ (dB) \\", r"\midrule"] + rows + \
      [r"\end{tabular}", "", r"\smallskip",
       r"\footnotesize For the first-order sidebands, $|E_y|\ll|E_x|$, so the phase of $E_y/E_x$ is poorly conditioned and has little effect on the polarization.",
       r"\end{table}"]
open(f"{M}/tab_pol.tex", "w").write("\n".join(tab) + "\n")
print(open(f"{M}/tab_harm.tex").read()); print(open(f"{M}/tab_pol.tex").read())
