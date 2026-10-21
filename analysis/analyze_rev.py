"""Post-processing for the revision: harmonic tables, steady-state check, power balance,
adiabatic-model checks (scaling law vs direct eps-interpolation), source/loss/grid comparisons."""
import sys, os; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tmcp_paths import *  # ROOT, DATA, R1, R2, R2C, R3, COMM, OUT, FIG, TAB
import sys, os, json, glob, numpy as np
os.environ.setdefault("FC", "5.754e9"); os.environ.setdefault("DATA", f"{R1}")
import fdtd_core as fc, adiabatic as ad
from scipy.interpolate import CubicSpline

REV = f"{R2}"; CO = f"{REV}/coarse"
FC = float(os.environ["FC"]); EPS0 = 2.2
out = {}


def load_harm():
    runs = {}
    for l in open(f"{CO}/harm.jsonl"):
        o = json.loads(l); runs[o["tag"]] = o
    return runs


def win_arrays(w):
    Ex = np.array(w["Ex_re"]) + 1j*np.array(w["Ex_im"]); Ey = np.array(w["Ey_re"]) + 1j*np.array(w["Ey_im"])
    return Ex, Ey


def harm_table(o, w=-1):
    W = o["windows"][w]; Ex, Ey = win_arrays(W)
    n = np.array(o["harmonics"]); i0 = list(n).index(0)
    P = np.array(W["Prad"]); R, L = fc.cp_components(Ex, Ey)
    rows = []
    for k, nn in enumerate(n):
        ar = float(fc.axial_ratio(Ex[k], Ey[k])); lr = float(20*np.log10(abs(L[k])/abs(R[k])))
        state = "linear" if ar > 10 else ("LHCP-dom." if lr > 0 else "RHCP-dom.")
        if ar < 3: state = "LHCP" if lr > 0 else "RHCP"
        rows.append(dict(n=int(nn), f_GHz=o["freqs"][k]/1e9, P_dBc=float(10*np.log10(P[k]/P[i0])), AR_dB=ar, LR_dB=lr,
                         state=state, EyEx_dB=float(20*np.log10(abs(Ey[k])/abs(Ex[k])))))
    return rows


def adiabatic_table(de, q, nh=6, H=None):
    n, En = ad.adiabatic_harmonics(de, q, nh=nh) if H is None else H(de, nh)
    i0 = list(n).index(0); R, L = fc.cp_components(En[0], En[1])
    P = np.abs(En[0])**2 + np.abs(En[1])**2
    return {int(nn): dict(P_dBc=float(10*np.log10(P[k]/P[i0])), AR_dB=float(fc.axial_ratio(En[0, k], En[1, k])),
                          LR_dB=float(20*np.log10(abs(L[k])/abs(R[k])))) for k, nn in enumerate(n)}


# --- direct (non-scaled) adiabatic model from static runs at many eps ---
def static_H(eps):
    for path in (f"{CO}/pulse_eps{eps:.3f}.npz", f"{R1}/pulse_eps{eps:.3f}.npz"):
        if os.path.exists(path):
            d = np.load(path); f = d["freqs"][:-1]; S = d["S"][:-1]
            prop = np.exp(1j*2*np.pi*f/fc.c0)
            return f, np.array([d["Ex"][:-1]/S*prop, d["Ey"][:-1]/S*prop])
    return None


def direct_adiabatic(fcar=FC):
    eps_list = [e for e in (2.0, 2.1, 2.15, 2.2, 2.25, 2.3, 2.4) if static_H(e) is not None]
    Hc = []
    for e in eps_list:
        f, H = static_H(e)
        Hc.append([np.interp(fcar, f, H[c].real) + 1j*np.interp(fcar, f, H[c].imag) for c in range(2)])
    Hc = np.array(Hc)
    spl = [CubicSpline(eps_list, Hc[:, c]) for c in range(2)]
    def model(de, nh, M=4096):
        th = 2*np.pi*np.arange(M)/M; e = EPS0 + de*np.sin(th)
        E = np.array([spl[c](e) for c in range(2)])
        n = np.arange(-nh, nh+1)
        return n, np.array([[np.mean(E[c]*np.exp(-1j*k*th)) for k in n] for c in range(2)])
    return eps_list, model, Hc


def scaling_check(q):
    """rms relative error of H(f;eps) ~ H0(f (eps/eps0)^(q/2)) within f_c +- 150 MHz, per eps."""
    f0, H0 = static_H(2.2); res = {}
    for e in (2.0, 2.1, 2.15, 2.25, 2.3, 2.4):
        if static_H(e) is None: continue
        f, H = static_H(e); s = (e/EPS0)**(q/2)
        band = (f*s > f0[0]) & (f*s < f0[-1]) & (abs(f - FC) < 0.15e9)
        Hs = np.array([np.interp(f[band]*s, f0, H0[c].real) + 1j*np.interp(f[band]*s, f0, H0[c].imag) for c in range(2)])
        res[e] = float(np.sqrt(np.sum(np.abs(Hs - H[:, band])**2)/np.sum(np.abs(H[:, band])**2)))
        # AR-minimum frequency, direct vs scaled
        ar = fc.axial_ratio(H[0], H[1]); res[f"ARmin_f_{e}"] = float(f[np.argmin(np.where(abs(f-FC) < 0.4e9, ar, 99))]/1e9)
        res[f"ARmin_f_scaled_{e}"] = float(f0[np.argmin(np.where(abs(f0-FC) < 0.4e9, fc.axial_ratio(H0[0], H0[1]), 99))]/s/1e9)
    return res


if __name__ == "__main__":
    q, qerr = ad.fit_q(); out["q"] = q
    runs = load_harm()
    out["scaling_check"] = scaling_check(q)
    eps_list, dmodel, Hc = direct_adiabatic()
    out["direct_eps_list"] = eps_list
    for tag, o in runs.items():
        r = dict(opts=o["opts"], N=o["N_ratio"], de=o["delta_eps"], fm_MHz=o["f_mod"]/1e6, runtime=o["runtime"])
        r["table"] = harm_table(o)
        n = np.array(o["harmonics"]); i0 = list(n).index(0)
        W = o["windows"][-1]; P = np.array(W["Prad"]); Pa = np.array(W["Pacc"]); f = np.array(o["freqs"])
        r["sumPrad_over_Pacc0"] = float(P.sum()/Pa[i0])
        r["MR_ratio"] = float(np.sum((P - np.where(n == 0, 0, Pa))/f)/(Pa[i0]/f[i0]))   # Manley-Rowe (lossless only)
        r["frac_beyond3"] = float(P[abs(n) > 3].sum()/P.sum())
        r["frac_sidebands"] = float(P[n != 0].sum()/P.sum())
        r["frac_sidebands_7"] = float(P[(n != 0) & (abs(n) <= 3)].sum()/P[abs(n) <= 3].sum())
        r["Pacc_sb_over_Pacc0"] = (Pa/Pa[i0]).tolist()
        r["mod_work_over_Pacc0"] = float((P.sum() - Pa.sum())/Pa[i0])     # radiated - accepted (excl. dielectric loss)
        if len(o["windows"]) > 1:
            Ex0, Ey0 = win_arrays(o["windows"][0]); Ex1, Ey1 = win_arrays(o["windows"][1])
            d = np.sqrt(np.abs(Ex1-Ex0)**2 + np.abs(Ey1-Ey0)**2)/np.sqrt(np.abs(Ex1)**2 + np.abs(Ey1)**2)
            r["ss_rel_change"] = d.tolist()
            P0 = np.array(o["windows"][0]["Prad"]); r["ss_dB_change"] = (10*np.log10(P/P0)).tolist()
            r["ss_AR_change"] = (np.array(W["AR_dB"]) - np.array(o["windows"][0]["AR_dB"])).tolist()
        if o["f0"] == FC and o["opts"].get("tand", 0.0009) == 0.0009:
            a = adiabatic_table(o["delta_eps"], q)
            r["adiab_scaled"] = a
            r["adiab_direct"] = adiabatic_table(o["delta_eps"], q, H=dmodel)
        out[tag] = r
    json.dump(out, open(f"{REV}/analysis_rev.json", "w"), indent=1)
    for tag in runs:
        r = out[tag]; print("\n==", tag, r["opts"], "runtime %.0f" % r["runtime"])
        print(" sumPrad/Pacc0 %.4f  beyond3 %.4f  sb %.3f  sb7 %.3f MR %.4f" % (r["sumPrad_over_Pacc0"], r["frac_beyond3"],
              r["frac_sidebands"], r["frac_sidebands_7"], r["MR_ratio"]))
        for row in r["table"]:
            s = " n=%+d %.4f GHz %6.1f dBc AR %5.1f LR %6.1f %s" % (row["n"], row["f_GHz"], row["P_dBc"], row["AR_dB"], row["LR_dB"], row["state"])
            if "adiab_scaled" in r:
                a = r["adiab_scaled"][row["n"]]; b = r["adiab_direct"][row["n"]]
                s += " | ad %6.1f dBc AR %5.1f | dir %6.1f AR %5.1f" % (a["P_dBc"], a["AR_dB"], b["P_dBc"], b["AR_dB"])
            if "ss_dB_change" in r: s += " | ss %.1e %.3f dB" % (r["ss_rel_change"][row["n"]+6], r["ss_dB_change"][row["n"]+6])
            print(s)
    print(json.dumps(out["scaling_check"], indent=0))
