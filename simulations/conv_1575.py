import sys, os; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tmcp_paths import *  # ROOT, DATA, R1, R2, R2C, R3, COMM, OUT, FIG, TAB
# In-house refined run of the 1.575 mm design: dx, dy, dz all halved (64 cells across patch, 10 through substrate)
import json, numpy as np, fdtd_core as fc
H = 1.575e-3; DX = 15.9e-3 / 64
m = fc.build_model(L_cells=64, W_cells=64, notch=8, feed_off_cells=16, dx=DX, dy=DX, dz=H/10, h=H, pcb_margin=48, npml=10)
Nt = int(round(16000 * 7.0e-13 / m["dt"]))
N_fft = 2 ** int(np.ceil(np.log2(Nt * 4)))
fa = np.arange(N_fft // 2 + 1) / (m["dt"] * N_fft)
freqs = fa[(fa >= 5.3e9) & (fa <= 6.4e9)]
print("Nx,Ny,Nz", m["Nx"], m["Ny"], m["Nz"], "Nt", Nt, "dt", m["dt"], flush=True)
res = fc.run(m, Nt=Nt, dft_freqs=freqs, verbose=True, progress_every=2000)
ps = fc.port_spectrum(res)
E = np.array([fc.broadside(res, k) for k in range(len(freqs))])
AR = fc.axial_ratio(E[:, 0], E[:, 1])
np.savez(f"{R1}/refined.npz", freqs=freqs, AR=AR, Ex=E[:, 0], Ey=E[:, 1], f=ps["f"], S11_dB=ps["S11_dB"], Zin=ps["Zin"], energy=res["energy"], dt=m["dt"])
k = np.argmin(AR); print(json.dumps(dict(ARmin=float(AR[k]), f_ARmin=float(freqs[k]), runtime=res["runtime"])), flush=True)
