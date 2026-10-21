"""Non-perturbative adiabatic model.
E(t) = H(f_c; eps(t)) with H(f; eps) ~= H0(f * (eps/eps0)^(q/2)) (frequency scaling of
the static transfer function), q calibrated from static runs at eps0 +- 0.05.
Harmonic n of E over one modulation cycle gives the adiabatic sideband prediction."""
import numpy as np, fdtd_core as fc

import os; import tmcp_paths; FC = float(os.environ.get("FC", "5.754e9")); EPS0 = 2.2; DATA = os.environ.get("DATA", str(tmcp_paths.R1))

def H_curve(eps):
    d = np.load(f"{DATA}/pulse_eps{eps:.3f}.npz")
    f = d["freqs"][:-1]; S = d["S"][:-1]
    # strip the free-space propagation phase e^{-jk r} (r = 1 m) used by the NTFF: it does not
    # scale with the antenna and must be excluded from the frequency-scaling map
    prop = np.exp(+1j * 2 * np.pi * f / fc.c0 * 1.0)
    return f, np.array([d["Ex"][:-1] / S * prop, d["Ey"][:-1] / S * prop])

def fit_q():
    """Best q such that H(f; eps) ~= H0(f * (eps/eps0)^(q/2)) for eps = 2.15, 2.25."""
    f0, H0 = H_curve(2.2)
    qs = np.linspace(0.6, 1.2, 241); err = []
    for q in qs:
        e = 0
        for eps in (2.15, 2.25):
            f, H = H_curve(eps)
            s = (eps / EPS0) ** (q / 2)
            band = (f * s > f0[0]) & (f * s < f0[-1]) & (abs(f - FC) < 0.15e9)
            Hs = np.array([np.interp(f[band] * s, f0, H0[c].real) + 1j * np.interp(f[band] * s, f0, H0[c].imag) for c in range(2)])
            e += np.sum(np.abs(Hs - H[:, band]) ** 2) / np.sum(np.abs(H[:, band]) ** 2)
        err.append(e)
    k = int(np.argmin(err))
    return float(qs[k]), float(np.sqrt(err[k] / 2))   # rms relative error of the scaling fit

def adiabatic_harmonics(de, q, nh=3, M=4096):
    f0, H0 = H_curve(2.2)
    th = 2 * np.pi * np.arange(M) / M
    eps_t = EPS0 + de * np.sin(th)
    ff = FC * (eps_t / EPS0) ** (q / 2)
    E = np.array([np.interp(ff, f0, H0[c].real) + 1j * np.interp(ff, f0, H0[c].imag) for c in range(2)])
    # harmonic n: (1/M) sum E(th) e^{-j n th}
    n = np.arange(-nh, nh + 1)
    En = np.array([[np.mean(E[c] * np.exp(-1j * k * th)) for k in n] for c in range(2)])
    return n, En

if __name__ == "__main__":
    q, rel = fit_q()
    print("q =", q, "scaling-fit rms rel. error =", rel)
    for de in (0.05, 0.2):
        n, En = adiabatic_harmonics(de, q)
        R, L = fc.cp_components(En[0], En[1])
        i0 = list(n).index(0)
        print("de", de)
        print("  AR  ", np.round(fc.axial_ratio(En[0], En[1]), 2))
        print("  |E| dBc", np.round(20*np.log10(np.sqrt(abs(En[0])**2+abs(En[1])**2)/np.sqrt(abs(En[0,i0])**2+abs(En[1,i0])**2)), 2))
        print("  XPD L/R", np.round(20*np.log10(abs(L)/abs(R)), 1))
