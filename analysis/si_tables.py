"""Supporting-Information tables: Table S2 (Huygens-surface sensitivity), Table S3 (two-mode sensitivity rows),
and the scaling-law table of S6."""
import sys, os; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tmcp_paths import *  # ROOT, DATA, R1, R2, R2C, R3, COMM, OUT, FIG, TAB
import json, numpy as np
from scipy.interpolate import CubicSpline
import fdtd_core as fc


def jl(path):
    return {json.loads(l)["tag"]: json.loads(l) for l in open(path)}


def m(x, d=2):
    s = f"{x:.{d}f}"; s = s.lstrip("-") if float(s) == 0 else s
    return s.replace("-", "$-$")


# ---- Table S2: harmonic levels vs Huygens surface (delta_eps = 0.2, f_m = f_c/200) ----
new, old = jl(f"{R3}/harm.jsonl"), jl(f"{R2C}/harm.jsonl")
rows = [("Original box (bottom 2 cells above ground)", old["200_0.2"], "Prad"),
        ("Box closed down to the ground", new["E_200_0.2"], "Prad"),
        ("Exact Poynting flux, closed box", new["E_200_0.2"], "flux"),
        ("Larger box (10/30-cell clearance)", new["bigbox_200_0.2"], "Prad"),
        ("Finite board, closed six-faced box", new["finite_200_0.2"], "Prad")]
L = [r"\begin{tabular}{@{}lrrrrr@{}}\toprule", r" & $n=-2$ & $n=-1$ & $n=+1$ & $n=+2$ & Sideband fraction \\\midrule"]
for lab, o, key in rows:
    w = o["windows"][-1]; P = np.array(w[key]); n = np.array(o["harmonics"]); i0 = list(n).index(0)
    vals = [10 * np.log10(P[list(n).index(k)] / P[i0]) for k in (-2, -1, 1, 2)]
    L.append(lab + " & " + " & ".join(m(v) for v in vals) + " & %.1f\\%% \\\\" % (100 * P[n != 0].sum() / P.sum()))
L.append(r"\bottomrule\end{tabular}")
open(f"{TAB}/tab_s4_rows.tex", "w").write("\n".join(L) + "\n")

# ---- Table S3: two-mode sensitivity rows ----
t = json.load(open(f"{R2}/twomode_r3.json"))
lab = {"f1": r"$f_u$", "f2": r"$f_v$", "Q1": r"$Q_u$", "Q2": r"$Q_v$", "amp2": r"$|\mathbf p_v|$", "q": r"$q$"}
step = {"f1": r"\SI{5}{MHz}", "f2": r"\SI{5}{MHz}", "Q1": "10\\%", "Q2": "10\\%", "amp2": r"\SI{1}{dB}", "q": "10\\%"}
r0 = t["refits"][0]
L = [r"Fitted values & " + " & ".join(m(v, 1) for v in (r0["d1"]["AR"], r0["d1"]["LR"], r0["d2"]["AR"], r0["d2"]["LR"], r0["carrier"]["AR"])) + r" \\"]
for p in t["perturb"]:
    L.append(("$+$" if p["sign"] > 0 else "$-$") + step[p["param"]] + " in " + lab[p["param"]] + " & "
             + " & ".join(m(v, 1) for v in (p["d1"]["AR"], p["d1"]["LR"], p["d2"]["AR"], p["d2"]["LR"], p["carrier"]["AR"])) + r" \\")
open(f"{TAB}/tab_s5_rows.tex", "w").write("\n".join(L) + "\n")

# ---- S6: scaling-law table ----
a = json.load(open(f"{R2}/analysis_rev.json"))["scaling_check"]
q = 0.86


def armin(eps):
    for p in (f"{R1}/pulse_eps{eps:.3f}.npz", f"{R2C}/pulse_eps{eps:.3f}.npz"):
        if os.path.exists(p):
            d = np.load(p); f = d["freqs"][:-1]
            ff = np.linspace(5.45e9, 6.15e9, 70001)
            return ff[np.argmin(fc.axial_ratio(CubicSpline(f, d["Ex"][:-1])(ff), CubicSpline(f, d["Ey"][:-1])(ff)))]


f0 = armin(2.2); E = (2.0, 2.1, 2.15, 2.25, 2.3, 2.4)
L = [r"$\eps_r$ & " + " & ".join(str(e) for e in E) + r" \\\midrule",
     r"rms error of the scaling law (\%) & " + " & ".join(f"{100 * a[str(e)]:.1f}" for e in E) + r" \\",
     r"AR-minimum frequency, direct (GHz) & " + " & ".join(f"{armin(e) / 1e9:.3f}" for e in E) + r" \\",
     r"AR-minimum frequency, scaling law (GHz) & " + " & ".join(f"{f0 / (e / 2.2) ** (q / 2) / 1e9:.3f}" for e in E) + r" \\\bottomrule"]
open(f"{TAB}/tab_s6_scaling_rows.tex", "w").write("\n".join(L) + "\n")
print(open(f"{TAB}/tab_s4_rows.tex").read()); print(open(f"{TAB}/tab_s5_rows.tex").read()); print("\n".join(L))
