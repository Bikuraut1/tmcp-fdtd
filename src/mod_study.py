"""
Time-modulated-substrate study (CW carrier, sinusoidal eps modulation).

For f_mod = f0 / N_ratio the steady state is periodic with T_mod, so a
rectangular DFT over exactly one modulation period gives the exact Fourier
(harmonic) coefficients at f0 + n f_mod (no leakage).  dt is reduced slightly
so that T_mod is an integer number of time steps.
"""
import sys, json, numpy as np, fdtd_core as fc

NH = 3                         # harmonics n = -NH..NH


def harmonic_run(geom, f0, N_ratio, delta_eps, settle=12e-9, tag=None, n_periods=1):
    m = fc.build_model(**geom)
    f_mod = f0 / N_ratio
    T_mod = 1.0 / f_mod
    steps_per_T = int(np.ceil(T_mod / m["dt"]))
    m["dt"] = T_mod / steps_per_T          # exact integer steps per period (<= 0.9 CFL)
    m["f0"] = f0
    N_settle = int(np.ceil(settle / m["dt"]))
    M = steps_per_T * n_periods
    Nt = N_settle + M
    win = np.zeros(Nt); win[N_settle:] = 1.0
    freqs = f0 + f_mod * np.arange(-NH, NH + 1)
    res = fc.run(m, Nt=Nt, dft_freqs=freqs, source="cw", delta_eps=delta_eps, f_mod=f_mod,
                 dft_window=win, verbose=True, progress_every=10000)
    # Fourier coefficients: X = 2/M * sum x[n] e^{-j w n dt}
    res["accJ"] *= 2.0 / M; res["accM"] *= 2.0 / M
    t = res["time"][N_settle:]
    def coef(x):
        return 2.0 / M * np.array([np.sum(x[N_settle:] * np.exp(-1j * 2 * np.pi * f * t)) for f in freqs])
    V = coef(res["v_feed"])
    I = coef(res["i_feed"]) * np.exp(1j * np.pi * freqs * m["dt"])
    theta = np.linspace(0, np.pi / 2, 46); phi = np.linspace(0, 2 * np.pi, 91)
    out = dict(tag=tag, f0=f0, f_mod=f_mod, N_ratio=N_ratio, delta_eps=delta_eps, Nt=Nt, dt=m["dt"],
               freqs=freqs.tolist(), harmonics=list(range(-NH, NH + 1)))
    Exb, Eyb, Prad, Dmax = [], [], [], []
    for k in range(len(freqs)):
        Ex, Ey = fc.broadside(res, k)
        Et, Ep = fc.far_field(res, k, theta, phi)
        P, U = fc.radiated_power(Et, Ep, theta, phi)
        Exb.append(Ex); Eyb.append(Ey); Prad.append(P); Dmax.append(4 * np.pi * U.max() / max(P, 1e-300))
    Exb, Eyb = np.array(Exb), np.array(Eyb)
    R, Lh = fc.cp_components(Exb, Eyb)
    out.update(Ex_re=Exb.real.tolist(), Ex_im=Exb.imag.tolist(), Ey_re=Eyb.real.tolist(), Ey_im=Eyb.imag.tolist(),
               AR_dB=fc.axial_ratio(Exb, Eyb).tolist(),
               RHCP=np.abs(R).tolist(), LHCP=np.abs(Lh).tolist(),
               Prad=list(map(float, Prad)), Dmax=list(map(float, Dmax)),
               V_re=V.real.tolist(), V_im=V.imag.tolist(), I_re=I.real.tolist(), I_im=I.imag.tolist(),
               Pacc=(0.5 * np.real(V * np.conj(I))).tolist(), runtime=res["runtime"])
    return out


if __name__ == "__main__":
    geom = json.loads(sys.argv[1])
    f0 = float(sys.argv[2])
    outfile = sys.argv[3]
    for spec in sys.argv[4:]:
        N_ratio, de = spec.split(",")
        o = harmonic_run(geom, f0, int(N_ratio), float(de), tag=spec)
        with open(outfile, "a") as fh:
            fh.write(json.dumps(o) + "\n")
        print("DONE", spec, "runtime %.0f s" % o["runtime"],
              "AR(dB) per harmonic:", np.round(o["AR_dB"], 2).tolist(),
              "Prad rel (dB):", np.round(10 * np.log10(np.array(o["Prad"]) / o["Prad"][NH]), 1).tolist(), flush=True)
