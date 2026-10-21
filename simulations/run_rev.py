"""
Supplementary coarse-grid runs for the revision.

Jobs (argv), each  kind:spec[:key=val,...]
  pulse:EPS[:k=v...]          static pulse run, eps_base = EPS
  harm:N:DE[:k=v...]          CW + modulation, f_m = FC/N, NH = 6, two consecutive one-period DFT windows
Options k=v: tand, air_gap, npml, pcb_margin (model);  src_R, hb_lat, hb_top, hb_bot (run);  tag
Output: rev/coarse/*.npz (pulse), rev/coarse/harm.jsonl + rev/coarse/harm_<tag>.npz (patterns)
"""
import sys, os; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tmcp_paths import *  # ROOT, DATA, R1, R2, R2C, R3, COMM, OUT, FIG, TAB
import sys, os, json, numpy as np
import fdtd_core as fc

H = 1.575e-3; DX = 15.9e-3 / 32
GEOM = dict(L_cells=32, W_cells=32, notch=4, feed_off_cells=8, dx=DX, dy=DX, dz=H/5, h=H, pcb_margin=24)
FC = float(os.environ.get("FC", "5.754e9"))
OUT = f"{R2C}"; os.makedirs(OUT, exist_ok=True)
MODEL_KEYS = {"tand", "air_gap", "npml", "pcb_margin", "eps_base", "below", "cfl"}
RUN_KEYS = {"src_R", "hb_lat", "hb_top", "hb_bot", "closed"}
HARM_KEYS = {"settle", "nwin"}
NH = 6
HK = {}


def parse_opts(items):
    mk, rk, tag = {}, {}, None
    for it in items:
        k, v = it.split("=")
        if k in HARM_KEYS: HK[k] = float(v); continue
        if k in MODEL_KEYS: mk[k] = float(v) if k in ("tand", "eps_base", "cfl") else int(v)
        elif k in RUN_KEYS: rk[k] = float(v) if k == "src_R" else (bool(int(v)) if k == "closed" else int(v))
        elif k == "tag": tag = v
        else: raise ValueError(k)
    return mk, rk, tag


def window_view(res, w):
    r = dict(res)
    if res["n_windows"] > 1:
        r["accJ"] = res["accJ"][w]; r["accM"] = res["accM"][w]
    return r


def pulse_static(eps, mk, rk, tag):
    g = dict(GEOM); g.update(mk); g["eps_base"] = eps
    m = fc.build_model(**g)
    Nt = int(os.environ.get("NT_PULSE", "16000"))
    if mk.get("tand", 0.0009) > 0.005: Nt = int(Nt)          # lossier -> decays faster; same length is ample
    N_fft = 2 ** int(np.ceil(np.log2(Nt * 4)))
    fa = np.arange(N_fft // 2 + 1) / (m["dt"] * N_fft)
    freqs = np.concatenate([fa[(fa >= 5.3e9) & (fa <= 6.4e9)], [FC]])
    res = fc.run(m, Nt=Nt, dft_freqs=freqs, verbose=False, **rk)
    ps = fc.port_spectrum(res)
    E = np.array([fc.broadside(res, k) for k in range(len(freqs))])
    w = res["dft_window"]; t = res["time"]
    S = np.array([np.sum(res["src"] * w * np.exp(-1j * 2 * np.pi * f * t)) for f in freqs])
    thmax = np.pi if rk.get("closed", False) else np.pi/2      # closed box: full sphere
    th = np.linspace(0, thmax, 46 if thmax < 3 else 91); ph = np.linspace(0, 2*np.pi, 91)
    kc = len(freqs) - 1
    Et, Ep = fc.far_field(res, kc, th, ph); P, U = fc.radiated_power(Et, Ep, th, ph)
    # radiated power on a coarse frequency subset (efficiency vs f)
    sub = np.arange(0, len(freqs) - 1, max(1, (len(freqs) - 1) // 24))
    th2 = np.linspace(0, thmax, 31 if thmax < 3 else 61); ph2 = np.linspace(0, 2*np.pi, 61)
    Psub = np.array([fc.radiated_power(*fc.far_field(res, k, th2, ph2), th2, ph2)[0] for k in sub])
    name = tag or f"pulse_eps{eps:.3f}"
    np.savez(f"{OUT}/{name}.npz", freqs=freqs, Ex=E[:, 0], Ey=E[:, 1], S=S, f=ps["f"], S11_dB=ps["S11_dB"],
             Zin=ps["Zin"], Pacc=ps["Pacc"], Pinc=ps["Pinc"], energy=res["energy"], dt=m["dt"], Nt=Nt,
             Et_fc=Et, Ep_fc=Ep, theta=th, phi=ph, Prad_fc=P, Umax_fc=U.max(),
             Pacc_fc=float(np.interp(FC, ps["f"], ps["Pacc"])), Pinc_fc=float(np.interp(FC, ps["f"], ps["Pinc"])),
             f_sub=freqs[sub], Prad_sub=Psub, opts=json.dumps(dict(mk, **rk)))
    print("pulse", name, "done", round(res["runtime"]), "s", flush=True)


def harmonic_run(N_ratio, de, mk, rk, tag, settle=12e-9, n_win=2):
    g = dict(GEOM); g.update(mk); cfl = g.pop("cfl", 0.9)
    m = fc.build_model(**g); m["dt"] *= cfl / 0.9
    f_mod = FC / N_ratio; T_mod = 1 / f_mod
    spT = int(np.ceil(T_mod / m["dt"])); m["dt"] = T_mod / spT; m["f0"] = FC
    Ns = int(np.ceil(settle / m["dt"])); Nt = Ns + n_win * spT
    W = np.zeros((n_win, Nt))
    for w in range(n_win): W[w, Ns + w*spT: Ns + (w+1)*spT] = 1.0
    freqs = FC + f_mod * np.arange(-NH, NH + 1)
    res = fc.run(m, Nt=Nt, dft_freqs=freqs, source="cw", delta_eps=de, f_mod=f_mod,
                 dft_window=W, verbose=True, progress_every=20000, **rk)
    res["accJ"] = res["accJ"] * (2.0 / spT); res["accM"] = res["accM"] * (2.0 / spT)
    th = np.linspace(0, np.pi/2, 46); ph = np.linspace(0, 2*np.pi, 91)
    out = dict(tag=tag, f0=FC, f_mod=f_mod, N_ratio=N_ratio, delta_eps=de, Nt=Nt, dt=m["dt"], settle=settle,
               opts=dict(mk, **rk), freqs=freqs.tolist(), harmonics=list(range(-NH, NH + 1)), windows=[])
    pats = {}
    for w in range(n_win):
        r = window_view(res, w)
        sl = slice(Ns + w*spT, Ns + (w+1)*spT); t = res["time"][sl]
        coef = lambda x: 2.0 / spT * np.array([np.sum(x[sl] * np.exp(-1j*2*np.pi*f*t)) for f in freqs])
        V = coef(res["v_feed"]); I = coef(res["i_feed"]) * np.exp(1j*np.pi*freqs*m["dt"])
        Exb, Eyb, Prad, Dmax = [], [], [], []
        for k in range(len(freqs)):
            Ex, Ey = fc.broadside(r, k)
            Et, Ep = fc.far_field(r, k, th, ph)
            P, U = fc.radiated_power(Et, Ep, th, ph)
            Exb.append(Ex); Eyb.append(Ey); Prad.append(P); Dmax.append(4*np.pi*U.max()/max(P, 1e-300))
            if w == n_win - 1 and k - NH in (0, 1, 2, -1, -2):
                pats[f"Et_{k-NH}"] = Et; pats[f"Ep_{k-NH}"] = Ep
        Exb, Eyb = np.array(Exb), np.array(Eyb)
        R, L = fc.cp_components(Exb, Eyb)
        out["windows"].append(dict(Ex_re=Exb.real.tolist(), Ex_im=Exb.imag.tolist(), Ey_re=Eyb.real.tolist(),
                                   Ey_im=Eyb.imag.tolist(), AR_dB=fc.axial_ratio(Exb, Eyb).tolist(),
                                   RHCP=np.abs(R).tolist(), LHCP=np.abs(L).tolist(), Prad=list(map(float, Prad)),
                                   Dmax=list(map(float, Dmax)), V_re=V.real.tolist(), V_im=V.imag.tolist(),
                                   I_re=I.real.tolist(), I_im=I.imag.tolist(),
                                   Pacc=(0.5*np.real(V*np.conj(I))).tolist()))
    out["runtime"] = res["runtime"]
    np.savez(f"{OUT}/harm_{tag}.npz", theta=th, phi=ph, **pats)
    with open(f"{OUT}/harm.jsonl", "a") as fh: fh.write(json.dumps(out) + "\n")
    w = out["windows"][-1]; P = np.array(w["Prad"])
    print("harm", tag, "done", round(res["runtime"]), "s | AR", np.round(w["AR_dB"], 1).tolist(),
          "| dBc", np.round(10*np.log10(P/P[NH]), 1).tolist(), flush=True)


if __name__ == "__main__":
    for j in sys.argv[1:]:
        parts = j.split(":")
        if parts[0] == "pulse":
            mk, rk, tag = parse_opts(parts[2:]); pulse_static(float(parts[1]), mk, rk, tag)
        else:
            HK.clear(); mk, rk, tag = parse_opts(parts[3:])
            harmonic_run(int(parts[1]), float(parts[2]), mk, rk, tag or f"{parts[1]}_{parts[2]}",
                         settle=HK.get("settle", 12) * 1e-9, n_win=int(HK.get("nwin", 2)))
