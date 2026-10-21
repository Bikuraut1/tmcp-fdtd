"""
Second-revision runs (round 3). Same physics as run_rev.py, plus:
  grid=ref           refined grid (dx = 15.9 mm/64, dz = h/10); default carrier 5.833 GHz
  mask=patch|uquad   modulate (or offset statically) only part of the substrate:
                       patch  - the substrate under the patch footprint (both modes ~ equally)
                       uquad  - the two patch quadrants on the untruncated (u) diagonal (u mode >> v mode)
  sdeps=X            static permittivity offset in the mask region (pulse runs, calibration)
  dense=1            pulse runs: add a 2-MHz DFT grid over 5.60-5.95 GHz (AR bandwidth)
  nh=K               harmonics |n| <= K (default 6)
  energy=1           accumulate modulation work and dielectric loss over the last DFT window
Every harmonic run also stores the exact near-field Poynting flux through the Huygens surface per harmonic.
Jobs: pulse:EPS[:k=v...]   harm:N:DE[:k=v...]        Output: rev/r3/
"""
import sys, os; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tmcp_paths import *  # ROOT, DATA, R1, R2, R2C, R3, COMM, OUT, FIG, TAB
import sys, os, json, numpy as np
import fdtd_core as fc

H = 1.575e-3
GRIDS = {"coarse": (dict(L_cells=32, W_cells=32, notch=4, feed_off_cells=8, dx=15.9e-3/32, dy=15.9e-3/32, dz=H/5, h=H,
                         pcb_margin=24), 5.754e9),
         "ref": (dict(L_cells=64, W_cells=64, notch=8, feed_off_cells=16, dx=15.9e-3/64, dy=15.9e-3/64, dz=H/10, h=H,
                      pcb_margin=48, npml=10), 5.833e9)}
OUT = f"{R3}"; os.makedirs(OUT, exist_ok=True)
MODEL_KEYS = {"tand": float, "air_gap": int, "npml": int, "pcb_margin": int, "below": int, "cfl": float}
RUN_KEYS = {"src_R": float, "hb_lat": int, "hb_top": int, "hb_bot": int, "closed": lambda v: bool(int(v)),
            "sdeps": float}
OTHER = {"settle": float, "nwin": int, "nh": int, "energy": int, "dense": int, "grid": str, "mask": str, "tag": str}


def parse(items):
    mk, rk, ok = {}, {}, {}
    for it in items:
        k, v = it.split("=")
        if k in MODEL_KEYS: mk[k] = MODEL_KEYS[k](v)
        elif k in RUN_KEYS: rk[k] = RUN_KEYS[k](v)
        elif k in OTHER: ok[k] = OTHER[k](v)
        else: raise ValueError(k)
    if "sdeps" in rk: rk["static_deps"] = rk.pop("sdeps")
    return mk, rk, ok


def make_mask(m, kind):
    nsx = m["sub_x2"] - m["sub_x1"] + 1; nsy = m["sub_y2"] - m["sub_y1"] + 1
    M = np.zeros((nsx, nsy), bool)
    i0 = m["px1"] - m["sub_x1"]; j0 = m["py1"] - m["sub_y1"]; W = m["W_cells"]; L = m["L_cells"]
    if kind == "patch":
        M[i0:i0 + W, j0:j0 + L] = True
    elif kind == "uquad":       # untruncated diagonal: (x low, y low) and (x high, y high) quadrants
        M[i0:i0 + W // 2, j0:j0 + L // 2] = True; M[i0 + W // 2:i0 + W, j0 + L // 2:j0 + L] = True
    elif kind == "vquad":
        M[i0 + W // 2:i0 + W, j0:j0 + L // 2] = True; M[i0:i0 + W // 2, j0 + L // 2:j0 + L] = True
    else:
        raise ValueError(kind)
    return M


def setup(mk, ok):
    G, fdef = GRIDS[ok.get("grid", "coarse")]
    g = dict(G); g.update(mk); cfl = g.pop("cfl", 0.9)
    m = fc.build_model(**g); m["dt"] *= cfl / 0.9
    FC = float(os.environ.get("FC", fdef))
    return m, FC


def pulse_static(eps, mk, rk, ok, tag):
    mk = dict(mk); mk["eps_base"] = eps
    G, fdef = GRIDS[ok.get("grid", "coarse")]
    g = dict(G); g.update(mk); m = fc.build_model(**g); FC = float(os.environ.get("FC", fdef))
    if "mask" in ok: rk = dict(rk, mod_mask=make_mask(m, ok["mask"]))
    Nt = int(os.environ.get("NT_PULSE", "16000"))
    N_fft = 2 ** int(np.ceil(np.log2(Nt * 4)))
    fa = np.arange(N_fft // 2 + 1) / (m["dt"] * N_fft)
    base = fa[(fa >= 5.3e9) & (fa <= 6.4e9)]
    dense = np.arange(5.60e9, 5.95e9 + 1, 2e6) if ok.get("dense") else np.array([])
    freqs = np.concatenate([base, dense, [FC]])
    res = fc.run(m, Nt=Nt, dft_freqs=freqs, verbose=False, **rk)
    ps = fc.port_spectrum(res)
    E = np.array([fc.broadside(res, k) for k in range(len(freqs))])
    w = res["dft_window"]; t = res["time"]
    S = np.array([np.sum(res["src"] * w * np.exp(-1j * 2 * np.pi * f * t)) for f in freqs])
    th = np.linspace(0, np.pi / 2, 46); ph = np.linspace(0, 2 * np.pi, 91)
    Et, Ep = fc.far_field(res, len(freqs) - 1, th, ph); P, U = fc.radiated_power(Et, Ep, th, ph)
    np.savez(f"{OUT}/{tag}.npz", freqs=freqs, nbase=len(base), Ex=E[:, 0], Ey=E[:, 1], S=S, f=ps["f"],
             S11_dB=ps["S11_dB"], Zin=ps["Zin"], Pacc=ps["Pacc"], energy=res["energy"], dt=m["dt"], Nt=Nt,
             Et_fc=Et, Ep_fc=Ep, theta=th, phi=ph, Prad_fc=P, Pacc_fc=float(np.interp(FC, ps["f"], ps["Pacc"])),
             opts=json.dumps(dict({k: v for k, v in rk.items() if k != "mod_mask"}, **mk, **ok)))
    print("pulse", tag, "done", round(res["runtime"]), "s", flush=True)


def harmonic_run(N_ratio, de, mk, rk, ok, tag):
    m, FC = setup(mk, ok)
    if "mask" in ok: rk = dict(rk, mod_mask=make_mask(m, ok["mask"]))
    NH = ok.get("nh", 6); n_win = ok.get("nwin", 1); settle = ok.get("settle", 8.0) * 1e-9
    f_mod = FC / N_ratio; T_mod = 1 / f_mod
    spT = int(np.ceil(T_mod / m["dt"])); m["dt"] = T_mod / spT; m["f0"] = FC
    Ns = int(np.ceil(settle / m["dt"])); Nt = Ns + n_win * spT
    W = np.zeros((n_win, Nt))
    for w in range(n_win): W[w, Ns + w * spT: Ns + (w + 1) * spT] = 1.0
    ew = W[-1] > 0 if ok.get("energy") else None
    freqs = FC + f_mod * np.arange(-NH, NH + 1)
    res = fc.run(m, Nt=Nt, dft_freqs=freqs, source="cw", delta_eps=de, f_mod=f_mod, dft_window=W,
                 verbose=True, progress_every=20000, energy_window=ew, **rk)
    res["accJ"] = res["accJ"] * (2.0 / spT); res["accM"] = res["accM"] * (2.0 / spT)
    closed = rk.get("closed", False)
    th = np.linspace(0, np.pi if closed else np.pi / 2, 91 if closed else 46); ph = np.linspace(0, 2 * np.pi, 91)
    out = dict(tag=tag, f0=FC, f_mod=f_mod, N_ratio=N_ratio, delta_eps=de, Nt=Nt, dt=m["dt"], settle=settle,
               opts=dict({k: v for k, v in rk.items() if k != "mod_mask"}, **mk, **ok), freqs=freqs.tolist(),
               harmonics=list(range(-NH, NH + 1)), windows=[], energy_acc=res["energy_acc"])
    pats = {}
    for w in range(n_win):
        r = dict(res)
        if n_win > 1: r["accJ"] = res["accJ"][w]; r["accM"] = res["accM"][w]
        sl = slice(Ns + w * spT, Ns + (w + 1) * spT); t = res["time"][sl]
        coef = lambda x: 2.0 / spT * np.array([np.sum(x[sl] * np.exp(-1j * 2 * np.pi * f * t)) for f in freqs])
        V = coef(res["v_feed"]); I = coef(res["i_feed"]) * np.exp(1j * np.pi * freqs * m["dt"])
        Exb, Eyb, Prad, Dmax, Fl, Fls = [], [], [], [], [], []
        for k in range(len(freqs)):
            Ex, Ey = fc.broadside(r, k)
            Et, Ep = fc.far_field(r, k, th, ph)
            P, U = fc.radiated_power(Et, Ep, th, ph)
            ft, fs = fc.box_flux(r, k)
            Exb.append(Ex); Eyb.append(Ey); Prad.append(P); Dmax.append(4 * np.pi * U.max() / max(P, 1e-300))
            Fl.append(ft); Fls.append(fs)
            if w == n_win - 1 and abs(k - NH) <= 2:
                pats[f"Et_{k - NH}"] = Et; pats[f"Ep_{k - NH}"] = Ep
        Exb, Eyb = np.array(Exb), np.array(Eyb)
        out["windows"].append(dict(Ex_re=Exb.real.tolist(), Ex_im=Exb.imag.tolist(), Ey_re=Eyb.real.tolist(),
                                   Ey_im=Eyb.imag.tolist(), AR_dB=fc.axial_ratio(Exb, Eyb).tolist(),
                                   Prad=list(map(float, Prad)), Dmax=list(map(float, Dmax)),
                                   flux=Fl, flux_sub=Fls, V_re=V.real.tolist(), V_im=V.imag.tolist(),
                                   I_re=I.real.tolist(), I_im=I.imag.tolist(),
                                   Pacc=(0.5 * np.real(V * np.conj(I))).tolist(),
                                   Pacc_td=float(np.mean(res["v_feed"][sl] * 0.5 * (res["i_feed"][sl] + np.roll(res["i_feed"], 1)[sl])))))
    out["runtime"] = res["runtime"]
    np.savez(f"{OUT}/harm_{tag}.npz", theta=th, phi=ph, **pats)
    with open(f"{OUT}/harm.jsonl", "a") as fh: fh.write(json.dumps(out) + "\n")
    w = out["windows"][-1]; P = np.array(w["Prad"])
    print("harm", tag, "done", round(res["runtime"]), "s | AR", np.round(w["AR_dB"], 1).tolist(),
          "| dBc", np.round(10 * np.log10(P / P[NH]), 1).tolist(), "| energy", res["energy_acc"], flush=True)


if __name__ == "__main__":
    for j in sys.argv[1:]:
        parts = j.split(":")
        if parts[0] == "pulse":
            mk, rk, ok = parse(parts[2:]); pulse_static(float(parts[1]), mk, rk, ok, ok.get("tag", f"pulse_{parts[1]}"))
        else:
            mk, rk, ok = parse(parts[3:])
            harmonic_run(int(parts[1]), float(parts[2]), mk, rk, ok, ok.get("tag", f"{parts[1]}_{parts[2]}"))
