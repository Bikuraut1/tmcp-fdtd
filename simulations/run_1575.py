import sys, os; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tmcp_paths import *  # ROOT, DATA, R1, R2, R2C, R3, COMM, OUT, FIG, TAB
import json, sys, numpy as np, fdtd_core as fc
from mod_study import harmonic_run
H = 1.575e-3; DX = 15.9e-3 / 32
GEOM = dict(L_cells=32, W_cells=32, notch=4, feed_off_cells=8, dx=DX, dy=DX, dz=H/5, h=H, pcb_margin=24)
import os; FC = float(os.environ.get("FC", "5.754e9"))   # carrier = static AR minimum of the coarse grid
OUT = f"{R1}/results.jsonl"

def pulse_static(eps):
    m = fc.build_model(eps_base=eps, **GEOM)
    Nt = int(os.environ.get("NT_PULSE", "16000"))
    N_fft = 2 ** int(np.ceil(np.log2(Nt * 4)))
    fa = np.arange(N_fft // 2 + 1) / (m["dt"] * N_fft)
    freqs = np.concatenate([fa[(fa >= 5.3e9) & (fa <= 6.2e9)], [FC]])
    res = fc.run(m, Nt=Nt, dft_freqs=freqs, verbose=False)
    ps = fc.port_spectrum(res)
    E = np.array([fc.broadside(res, k) for k in range(len(freqs))])
    # source spectrum at each DFT frequency (same window) for transfer functions
    w = res["dft_window"]; t = res["time"]
    S = np.array([np.sum(res["src"] * w * np.exp(-1j * 2 * np.pi * f * t)) for f in freqs])
    th = np.linspace(0, np.pi/2, 46); ph = np.linspace(0, 2*np.pi, 91)
    kc = len(freqs) - 1
    Et, Ep = fc.far_field(res, kc, th, ph); P, U = fc.radiated_power(Et, Ep, th, ph)
    Pacc_fc = float(np.interp(FC, ps["f"], ps["Pacc"])); Pinc_fc = float(np.interp(FC, ps["f"], ps["Pinc"]))
    np.savez(f"{R1}/pulse_eps{eps:.3f}.npz", freqs=freqs, Ex=E[:, 0], Ey=E[:, 1], S=S, f=ps["f"], S11_dB=ps["S11_dB"],
             Zin=ps["Zin"], Pacc=ps["Pacc"], Pinc=ps["Pinc"], energy=res["energy"], dt=m["dt"], Nt=Nt,
             Et_fc=Et, Ep_fc=Ep, theta=th, phi=ph, Prad_fc=P, Pacc_fc=Pacc_fc, Pinc_fc=Pinc_fc)
    print("pulse", eps, "done", round(res["runtime"]), "s", flush=True)

jobs = sys.argv[1:]
for j in jobs:
    kind, *a = j.split(":")
    if kind == "pulse":
        pulse_static(float(a[0]))
    else:
        N, de = int(a[0]), float(a[1])
        o = harmonic_run(GEOM, FC, N, de, settle=float(os.environ.get("SETTLE", "12e-9")), tag=j)
        with open(OUT, "a") as fh:
            fh.write(json.dumps(o) + "\n")
        print("harm", j, "done", round(o["runtime"]), "s | AR", np.round(o["AR_dB"], 2).tolist(),
              "| Prad dBc", np.round(10*np.log10(np.array(o["Prad"])/o["Prad"][3]), 1).tolist(), flush=True)
