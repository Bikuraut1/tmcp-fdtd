"""Analysis of the time-modulated runs + quasi-static (QS) model.
Writes mod/numbers.json (every number quoted in the paper comes from here)."""
import json, os, numpy as np, fdtd_core as fc

import tmcp_paths; FC = float(os.environ.get("FC", "5.754e9")); DATA = os.environ.get("DATA", str(tmcp_paths.R1))
def load_runs():
    runs = [json.loads(l) for l in open(f"{DATA}/results.jsonl")]
    for r in runs:
        r["Ex"] = np.array(r["Ex_re"]) + 1j * np.array(r["Ex_im"])
        r["Ey"] = np.array(r["Ey_re"]) + 1j * np.array(r["Ey_im"])
        r["ER"], r["EL"] = fc.cp_components(r["Ex"], r["Ey"])   # corrected convention
        r["n"] = np.array(r["harmonics"]); r["Prad"] = np.array(r["Prad"])
    return runs

def pulse(eps):
    d = np.load(f"{DATA}/pulse_eps{eps:.3f}.npz")
    k = len(d["freqs"]) - 1           # last DFT frequency = FC
    H = np.array([d["Ex"][k], d["Ey"][k]]) / d["S"][k]
    return d, H

def qs_model():
    out = {}
    if not all(os.path.exists(f"{DATA}/pulse_eps{e:.3f}.npz") for e in (2.15, 2.2, 2.25)):
        return None
    d0, H0 = pulse(2.2); _, Hm = pulse(2.15); _, Hp = pulse(2.25)
    dH = (Hp - Hm) / 0.10
    R0, L0 = fc.cp_components(H0[0], H0[1]); Rd, Ld = fc.cp_components(dH[0], dH[1])
    out["AR_carrier_static_dB"] = float(fc.axial_ratio(H0[0], H0[1]))
    out["AR_dHde_dB"] = float(fc.axial_ratio(dH[0], dH[1]))
    out["XPD_carrier_static_dB"] = float(20 * np.log10(abs(L0) / abs(R0)))
    out["XPD_dHde_dB"] = float(20 * np.log10(abs(Ld) / abs(Rd)))          # >0: LHCP-dominant
    out["rel_sens_total"] = float(np.linalg.norm(dH) / np.linalg.norm(H0))   # |dH/de| / |H|
    out["rel_sens_L"] = float(abs(Ld) / abs(L0)); out["rel_sens_R_over_L0"] = float(abs(Rd) / abs(L0))
    out["H0"] = H0; out["dH"] = dH
    # static numbers at FC from the eps=2.2 pulse run
    out["eta_fc"] = float(d0["Prad_fc"] / d0["Pacc_fc"])
    U = (np.abs(d0["Et_fc"]) ** 2 + np.abs(d0["Ep_fc"]) ** 2) / (2 * fc.eta0)
    out["D_fc_dBi"] = float(10 * np.log10(4 * np.pi * U.max() / d0["Prad_fc"]))
    out["G_fc_dBi"] = float(10 * np.log10(4 * np.pi * U.max() / d0["Pacc_fc"]))
    out["Greal_fc_dBi"] = float(10 * np.log10(4 * np.pi * U.max() / d0["Pinc_fc"]))
    out["S11_fc_dB"] = float(np.interp(FC, d0["f"], d0["S11_dB"]))
    return out

def summarize(runs, qs):
    rows = []
    for r in runs:
        i0 = list(r["n"]).index(0)
        row = dict(tag=r["tag"], N=r["N_ratio"], f_m_MHz=r["f_mod"] / 1e6, de=r["delta_eps"])
        row["Prad_dBc"] = (10 * np.log10(r["Prad"] / r["Prad"][i0])).tolist()
        row["AR_dB"] = list(r["AR_dB"])
        row["XPD_dB"] = (20 * np.log10(np.abs(r["EL"]) / np.abs(r["ER"]))).tolist()   # LHCP/RHCP per harmonic
        row["EL_dBc"] = (20 * np.log10(np.abs(r["EL"]) / abs(r["EL"][i0]))).tolist()
        row["ER_dBc"] = (20 * np.log10(np.abs(r["ER"]) / abs(r["EL"][i0]))).tolist()
        row["sideband_power_fraction"] = float(1 - r["Prad"][i0] / r["Prad"].sum())
        if qs is not None and r["delta_eps"] > 0:
            de = r["delta_eps"]
            # QS: E_{+-1}/E_0 = +-(de/2j) dH/H0  (per component); carrier at modulation ~ H0
            E1 = (de / 2) * qs["dH"]
            R1, L1 = fc.cp_components(E1[0], E1[1]); R0, L0 = fc.cp_components(*qs["H0"])
            row["QS_EL1_dBc"] = float(20 * np.log10(abs(L1) / abs(L0)))
            row["QS_ER1_dBc"] = float(20 * np.log10(abs(R1) / abs(L0)))
            row["QS_AR1_dB"] = qs["AR_dHde_dB"]
            row["QS_XPD1_dB"] = qs["XPD_dHde_dB"]
            row["QS_E1_total_dBc"] = float(20 * np.log10(np.linalg.norm(E1) / np.linalg.norm(qs["H0"])))
            Etot = np.sqrt(np.abs(r["Ex"]) ** 2 + np.abs(r["Ey"]) ** 2)
            row["FDTD_E1_total_dBc"] = (20 * np.log10(Etot / Etot[i0])).tolist()
        rows.append(row)
    return rows

if __name__ == "__main__":
    runs = load_runs() if os.path.exists(f"{DATA}/results.jsonl") else []
    qs = qs_model()
    rows = summarize(runs, qs)
    out = dict(qs={k: v for k, v in (qs or {}).items() if k not in ("H0", "dH")}, runs=rows)
    json.dump(out, open(f"{DATA}/numbers.json", "w"), indent=1)
    np.set_printoptions(precision=2, suppress=True)
    print(json.dumps(out["qs"], indent=1))
    for r in rows:
        print(r["tag"], "f_m=%.1f MHz de=%.2f" % (r["f_m_MHz"], r["de"]))
        for k in ("Prad_dBc", "AR_dB", "XPD_dB", "EL_dBc", "ER_dBc", "FDTD_E1_total_dBc"):
            if k in r:
                print("   %-18s" % k, np.round(r[k], 2))
        for k in ("QS_EL1_dBc", "QS_ER1_dBc", "QS_AR1_dB", "QS_XPD1_dB", "QS_E1_total_dBc", "sideband_power_fraction"):
            if k in r:
                print("   %-18s %.2f" % (k, r[k]))
