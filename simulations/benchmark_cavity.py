"""
Independent benchmark of the time-varying update (reviewer 1, comment 4).

A PEC cavity (a x b x d) uniformly filled with eps_r(t) = eps_r0 + de*sin(2 pi f_m t)
supports the TE101 mode Ey ~ sin(pi x/a) sin(pi z/d).  Because the medium is uniform,
the mode shape is preserved and its displacement amplitude D(t) obeys exactly
        D'' + w0^2 * eps_r0/eps_r(t) * D = 0,      E = D/(eps0*eps_r(t))
(a Hill equation; w0 = mode frequency at eps_r0).  The FDTD engine's own update
kernels (update_E / update_H of fdtd_core) are run on this cavity with
  (a) the paper's D-based update  E^{n+1} = (eps^n E^n + dt curlH)/eps^{n+1}
  (b) a naive E-based update       E^{n+1} = E^n + dt curlH / eps^{n+1}
and compared with a high-accuracy ODE solution (scipy, rtol 1e-11).
"""
import sys, os; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tmcp_paths import *  # ROOT, DATA, R1, R2, R2C, R3, COMM, OUT, FIG, TAB
import sys, json, numpy as np
import fdtd_core as fc
from scipy.integrate import solve_ivp

eps0, mu0, c0 = fc.eps0, fc.mu0, fc.c0
NXc, NZc, NY = 40, 40, 4          # cells across a and d; Ny nodes (field uniform in y)
dx = dz = 0.5e-3; dy = 0.5e-3
Nx, Nz = NXc + 1, NZc + 1
er0 = 2.2
dt = 0.9 / (c0 * np.sqrt(1/dx**2 + 1/dy**2 + 1/dz**2))
# discrete (Yee) TE101 frequency at er0, including time discretisation
kx, kz = np.pi / (NXc * dx), np.pi / (NZc * dz)
v = c0 / np.sqrt(er0)
s = v * dt * np.sqrt((np.sin(kx*dx/2)/dx)**2 + (np.sin(kz*dz/2)/dz)**2)
w_d = 2 / dt * np.arcsin(s)
f0 = w_d / (2*np.pi)

def fdtd(de, fm, Nt, scheme):
    shp = (Nx, NY, Nz)
    Ex, Ey, Ez, Hx, Hy, Hz = (np.zeros(shp) for _ in range(6))
    ii = np.arange(Nx)[:, None, None]; kk = np.arange(Nz)[None, None, :]
    Ey[:, :NY-1, :] = (np.sin(np.pi*ii/NXc) * np.sin(np.pi*kk/NZc) * np.ones((1, NY-1, 1)))
    psi = [np.zeros(shp) for _ in range(12)]
    oneX, oneY, oneZ = np.ones(Nx), np.ones(NY), np.ones(Nz)
    zX, zY, zZ = np.zeros(Nx), np.zeros(NY), np.zeros(Nz)
    fX, fY, fZ = np.zeros(Nx, bool), np.zeros(NY, bool), np.zeros(Nz, bool)
    C1 = np.ones(shp); C2 = np.ones(shp)
    ic, kc = Nx//2, Nz//2
    out = np.zeros(Nt)
    for n in range(Nt):
        fc.update_H(Ex, Ey, Ez, Hx, Hy, Hz, *psi[:6], zX, zX, oneX, fX, zY, zY, oneY, fY, zZ, zZ, oneZ, fZ,
                    dx, dy, dz, dt/mu0)
        e_new = eps0*(er0 + de*np.sin(2*np.pi*fm*(n+1)*dt)); e_old = eps0*(er0 + de*np.sin(2*np.pi*fm*n*dt))
        if scheme == "D":
            C1[:] = 1/(e_new/dt); C2[:] = e_old/dt
        else:  # naive: eps(t) dE/dt = curl H
            C1[:] = 1/(e_new/dt); C2[:] = e_new/dt
        fc.update_E(Ex, Ey, Ez, Hx, Hy, Hz, *psi[6:], zX, zX, oneX, fX, zY, zY, oneY, fY, zZ, zZ, oneZ, fZ,
                    C1, C2, C1, C2, C1, C2, dx, dy, dz)
        out[n] = Ey[ic, 1, kc]
    return out   # E at t = (n+1) dt

def ode(de, fm, Nt):
    er = lambda t: er0 + de*np.sin(2*np.pi*fm*t)
    E0 = np.sin(np.pi*(Nx//2)/NXc) * np.sin(np.pi*(Nz//2)/NZc)
    D0 = eps0*er0*E0
    Dp0 = -(dt/2) * w_d**2 * D0          # matches the leapfrog start H^{-1/2} = 0
    rhs = lambda t, y: [y[1], -w_d**2 * er0/er(t) * y[0]]
    t = (np.arange(Nt) + 1) * dt
    sol = solve_ivp(rhs, (0, t[-1]), [D0, Dp0], t_eval=t, rtol=1e-11, atol=1e-30, method="DOP853")
    return sol.y[0] / (eps0*er(t))

res = {"f0_GHz": f0/1e9, "dt_ps": dt*1e12}
cases = [(0.05, f0/5), (0.2, f0/5), (0.2, f0/20)]
for de, fm in cases:
    Tm = 1/fm; Nt = int(round(20*Tm/dt))
    ref = ode(de, fm, Nt)
    for sch in ("D", "naive"):
        e = fdtd(de, fm, Nt, sch)
        err = np.max(np.abs(e - ref)) / np.max(np.abs(ref))
        # sideband content: Hann-windowed spectrum over the record, peak heights vs reference
        w = np.hanning(Nt); F = np.abs(np.fft.rfft(e*w, 8*Nt)); Fr = np.abs(np.fft.rfft(ref*w, 8*Nt))
        pk = np.argmax(Fr); lev_err = 20*np.log10(F[pk]/Fr[pk])
        res[f"de={de},fm=f0/{round(f0/fm)},{sch}"] = dict(max_rel_err=float(err), carrier_level_err_dB=float(lev_err), Nt=Nt)
        print(de, round(f0/fm), sch, "max rel err %.2e" % err, "peak level err %.3f dB" % lev_err, flush=True)
    np.savez(f"{R2}/bench_de{de}_N{round(f0/fm)}.npz", ref=ref, eD=fdtd(de, fm, Nt, "D"),
             eN=fdtd(de, fm, Nt, "naive"), dt=dt, f0=f0, fm=fm)
json.dump(res, open(f"{R2}/benchmark.json", "w"), indent=1)
