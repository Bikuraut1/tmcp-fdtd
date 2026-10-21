import sys, os; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tmcp_paths import *  # ROOT, DATA, R1, R2, R2C, R3, COMM, OUT, FIG, TAB
import sys, json, os, numpy as np
from mod_study import harmonic_run
H = 1.575e-3; DX = 15.9e-3 / 64
GEOM = dict(L_cells=64, W_cells=64, notch=8, feed_off_cells=16, dx=DX, dy=DX, dz=H/10, h=H, pcb_margin=48, npml=10)
FC = 5.833e9   # refined-grid static AR minimum (d1575/refined.npz)
for spec in sys.argv[1:]:
    N, de = spec.split(",")
    o = harmonic_run(GEOM, FC, int(N), float(de), settle=12e-9, tag=f"ref:{N}:{de}")
    with open(f"{R2}/refined_dyn.jsonl", "a") as fh:
        fh.write(json.dumps(o) + "\n")
    print("DONE", spec, round(o["runtime"]), "s AR", np.round(o["AR_dB"], 2).tolist(),
          "Prad dBc", np.round(10*np.log10(np.array(o["Prad"])/o["Prad"][3]), 1).tolist(), flush=True)
