"""
Fast 3-D FDTD engine for the corner-truncated CP patch on a (optionally)
time-modulated substrate.

Physics, grid, CPML, feed, PEC handling and Huygens-surface sampling are
identical to CP_Patch_FDTD_Final2.py (bug-fixed version).  Differences are
purely computational:
  * the field / CPML updates run in Numba-compiled loops (multithreaded);
  * the Huygens-surface currents are Fourier-transformed ON THE FLY at a
    user-supplied list of frequencies (running DFT), so no time series has
    to be stored -> long CW runs fit in memory;
  * geometry (patch size, notch, feed offset), modulation and excitation are
    parameters of run().
All grid indices keep the MATLAB 1-based convention of the original code.
"""
import time as _time
import numpy as np
import numba as nb

c0 = 299792458.0
mu0 = 4 * np.pi * 1e-7
eps0 = 8.8541878128e-12
eta0 = np.sqrt(mu0 / eps0)


def mround(x):
    return int(np.sign(x) * np.floor(abs(x) + 0.5))


def rng(a, b):
    return slice(a - 1, b)


# --------------------------------------------------------------------------
# Numba kernels
# --------------------------------------------------------------------------
@nb.njit(parallel=True, fastmath=False, cache=True)
def update_H(Ex, Ey, Ez, Hx, Hy, Hz,
             psi_Ezy, psi_Eyz, psi_Exz, psi_Ezx, psi_Eyx, psi_Exy,
             bH_x, cH_x, kH_x, mH_x, bH_y, cH_y, kH_y, mH_y, bH_z, cH_z, kH_z, mH_z,
             dx, dy, dz, cH):
    Nx, Ny, Nz = Ex.shape
    # Hx : (0..Nx-1, 0..Ny-2, 0..Nz-2)
    for i in nb.prange(Nx):
        for j in range(Ny - 1):
            for k in range(Nz - 1):
                dEz_dy = (Ez[i, j + 1, k] - Ez[i, j, k]) / dy
                dEy_dz = (Ey[i, j, k + 1] - Ey[i, j, k]) / dz
                if mH_y[j]:
                    psi_Ezy[i, j, k] = bH_y[j] * psi_Ezy[i, j, k] + cH_y[j] * dEz_dy
                if mH_z[k]:
                    psi_Eyz[i, j, k] = bH_z[k] * psi_Eyz[i, j, k] + cH_z[k] * dEy_dz
                Hx[i, j, k] -= cH * ((dEz_dy / kH_y[j] + psi_Ezy[i, j, k])
                                     - (dEy_dz / kH_z[k] + psi_Eyz[i, j, k]))
    # Hy : (0..Nx-2, 0..Ny-1, 0..Nz-2)
    for i in nb.prange(Nx - 1):
        for j in range(Ny):
            for k in range(Nz - 1):
                dEx_dz = (Ex[i, j, k + 1] - Ex[i, j, k]) / dz
                dEz_dx = (Ez[i + 1, j, k] - Ez[i, j, k]) / dx
                if mH_z[k]:
                    psi_Exz[i, j, k] = bH_z[k] * psi_Exz[i, j, k] + cH_z[k] * dEx_dz
                if mH_x[i]:
                    psi_Ezx[i, j, k] = bH_x[i] * psi_Ezx[i, j, k] + cH_x[i] * dEz_dx
                Hy[i, j, k] -= cH * ((dEx_dz / kH_z[k] + psi_Exz[i, j, k])
                                     - (dEz_dx / kH_x[i] + psi_Ezx[i, j, k]))
    # Hz : (0..Nx-2, 0..Ny-2, 0..Nz-1)
    for i in nb.prange(Nx - 1):
        for j in range(Ny - 1):
            for k in range(Nz):
                dEy_dx = (Ey[i + 1, j, k] - Ey[i, j, k]) / dx
                dEx_dy = (Ex[i, j + 1, k] - Ex[i, j, k]) / dy
                if mH_x[i]:
                    psi_Eyx[i, j, k] = bH_x[i] * psi_Eyx[i, j, k] + cH_x[i] * dEy_dx
                if mH_y[j]:
                    psi_Exy[i, j, k] = bH_y[j] * psi_Exy[i, j, k] + cH_y[j] * dEx_dy
                Hz[i, j, k] -= cH * ((dEy_dx / kH_x[i] + psi_Eyx[i, j, k])
                                     - (dEx_dy / kH_y[j] + psi_Exy[i, j, k]))


@nb.njit(parallel=True, fastmath=False, cache=True)
def update_E(Ex, Ey, Ez, Hx, Hy, Hz,
             psi_Hzy, psi_Hyz, psi_Hxz, psi_Hzx, psi_Hyx, psi_Hxy,
             bE_x, cE_x, kE_x, mE_x, bE_y, cE_y, kE_y, mE_y, bE_z, cE_z, kE_z, mE_z,
             C1x, C2x, C1y, C2y, C1z, C2z, dx, dy, dz):
    Nx, Ny, Nz = Ex.shape
    # Ex : (0..Nx-2, 1..Ny-2, 1..Nz-2)
    for i in nb.prange(Nx - 1):
        for j in range(1, Ny - 1):
            for k in range(1, Nz - 1):
                dHz_dy = (Hz[i, j, k] - Hz[i, j - 1, k]) / dy
                dHy_dz = (Hy[i, j, k] - Hy[i, j, k - 1]) / dz
                if mE_y[j]:
                    psi_Hzy[i, j, k] = bE_y[j] * psi_Hzy[i, j, k] + cE_y[j] * dHz_dy
                if mE_z[k]:
                    psi_Hyz[i, j, k] = bE_z[k] * psi_Hyz[i, j, k] + cE_z[k] * dHy_dz
                curl = (dHz_dy / kE_y[j] + psi_Hzy[i, j, k]) - (dHy_dz / kE_z[k] + psi_Hyz[i, j, k])
                Ex[i, j, k] = C1x[i, j, k] * (C2x[i, j, k] * Ex[i, j, k] + curl)
    # Ey : (1..Nx-2, 0..Ny-2, 1..Nz-2)
    for i in nb.prange(1, Nx - 1):
        for j in range(Ny - 1):
            for k in range(1, Nz - 1):
                dHx_dz = (Hx[i, j, k] - Hx[i, j, k - 1]) / dz
                dHz_dx = (Hz[i, j, k] - Hz[i - 1, j, k]) / dx
                if mE_z[k]:
                    psi_Hxz[i, j, k] = bE_z[k] * psi_Hxz[i, j, k] + cE_z[k] * dHx_dz
                if mE_x[i]:
                    psi_Hzx[i, j, k] = bE_x[i] * psi_Hzx[i, j, k] + cE_x[i] * dHz_dx
                curl = (dHx_dz / kE_z[k] + psi_Hxz[i, j, k]) - (dHz_dx / kE_x[i] + psi_Hzx[i, j, k])
                Ey[i, j, k] = C1y[i, j, k] * (C2y[i, j, k] * Ey[i, j, k] + curl)
    # Ez : (1..Nx-2, 1..Ny-2, 0..Nz-2)
    for i in nb.prange(1, Nx - 1):
        for j in range(1, Ny - 1):
            for k in range(Nz - 1):
                dHy_dx = (Hy[i, j, k] - Hy[i - 1, j, k]) / dx
                dHx_dy = (Hx[i, j, k] - Hx[i, j - 1, k]) / dy
                if mE_x[i]:
                    psi_Hyx[i, j, k] = bE_x[i] * psi_Hyx[i, j, k] + cE_x[i] * dHy_dx
                if mE_y[j]:
                    psi_Hxy[i, j, k] = bE_y[j] * psi_Hxy[i, j, k] + cE_y[j] * dHx_dy
                curl = (dHy_dx / kE_x[i] + psi_Hyx[i, j, k]) - (dHx_dy / kE_y[j] + psi_Hxy[i, j, k])
                Ez[i, j, k] = C1z[i, j, k] * (C2z[i, j, k] * Ez[i, j, k] + curl)


@nb.njit(parallel=True, cache=True)
def dft_accumulate(accJ, accM, J, M, ph, w):
    # accJ/accM: (Nf, 3, P) complex; J/M: (3, P) float; ph: (Nf,) complex
    Nf, _, P = accJ.shape
    for f in nb.prange(Nf):
        p_ = ph[f] * w
        for c in range(3):
            for p in range(P):
                accJ[f, c, p] += J[c, p] * p_
                accM[f, c, p] += M[c, p] * p_


# --------------------------------------------------------------------------
# Geometry / model set-up
# --------------------------------------------------------------------------
def build_model(L_cells=32, W_cells=32, notch=4, feed_off_cells=8,
                eps_base=2.2, tand=0.0009, f0=5.8e9, npml=10, air_gap=0,
                pcb_margin=24, dx=0.5e-3, dy=0.5e-3, dz=0.315e-3, h=1.575e-3, below=0):
    """feed_off_cells: feed offset from patch centre along y (cells).
    Original design: L=W=32, notch=4, offset = 8 (4 mm)."""
    m = dict()
    lam = c0 / f0
    h_cells = mround(h / dz)
    Nx = W_cells + 2 * (pcb_margin + air_gap + npml)
    Ny = L_cells + 2 * (pcb_margin + air_gap + npml)
    pad = npml + air_gap + pcb_margin
    px1, px2 = pad + 1, pad + W_cells
    py1, py2 = pad + 1, pad + L_cells
    sub_x1, sub_x2 = px1 - pcb_margin, px2 + pcb_margin
    sub_y1, sub_y2 = py1 - pcb_margin, py2 + pcb_margin
    fs_above = mround((lam / 4) / dz) + 5
    Nz = npml + below + h_cells + fs_above + npml   # below: air cells under the ground (finite-board runs)
    z_gnd = npml + 1 + below
    z_patch = z_gnd + h_cells
    sub_z1, sub_z2 = z_gnd, z_patch - 1
    feed_x = px1 + mround(W_cells / 2)
    feed_y = py1 + mround(L_cells / 2) - feed_off_cells
    dt = 0.90 / (c0 * np.sqrt((1 / dx) ** 2 + (1 / dy) ** 2 + (1 / dz) ** 2))

    Npi, Npj = px2 - px1 + 1, py2 - py1 + 1
    patch_mask = np.ones((Npi, Npj), dtype=bool)
    if notch > 0:
        patch_mask[0:notch, Npj - notch:Npj] = False
        patch_mask[Npi - notch:Npi, 0:notch] = False

    m.update(locals())
    m.pop("m")
    return m


def cpml_coeffs(N, sig_max, npml, dt, m_=4, kappa_max=7, alpha_max=0.15):
    def calc_b(sig, k, a):
        return np.exp(-(sig / k + a) * (dt / eps0))

    def calc_c(sig, k, a, b):
        return sig / (k * (sig + k * a) + 1e-20) * (b - 1)

    sigE = np.zeros(N); kE = np.ones(N); aE = np.zeros(N)
    sigH = np.zeros(N); kH = np.ones(N); aH = np.zeros(N)
    for i in range(1, npml + 1):
        xE = (npml - i + 1) / npml; xH = (npml - i + 0.5) / npml
        sigE[i-1] = sig_max * xE ** m_; kE[i-1] = 1 + (kappa_max - 1) * xE ** m_; aE[i-1] = alpha_max * (1 - xE)
        sigH[i-1] = sig_max * xH ** m_; kH[i-1] = 1 + (kappa_max - 1) * xH ** m_; aH[i-1] = alpha_max * (1 - xH)
        xE_R = i / npml; xH_R = (i - 0.5) / npml
        idxE = N - npml + i
        sigE[idxE-1] = sig_max * xE_R ** m_; kE[idxE-1] = 1 + (kappa_max - 1) * xE_R ** m_; aE[idxE-1] = alpha_max * (1 - xE_R)
        idxH = N - npml + i - 1
        if idxH > 0:
            sigH[idxH-1] = sig_max * xH_R ** m_; kH[idxH-1] = 1 + (kappa_max - 1) * xH_R ** m_; aH[idxH-1] = alpha_max * (1 - xH_R)
    bE = calc_b(sigE, kE, aE); cE = calc_c(sigE, kE, aE, bE)
    bH = calc_b(sigH, kH, aH); cH = calc_c(sigH, kH, aH, bH)
    mE = np.zeros(N, dtype=np.bool_); mE[:npml] = True; mE[N - npml:] = True
    mH = np.zeros(N, dtype=np.bool_); mH[:npml] = True; mH[N - npml - 1:N - 1] = True
    kE[~mE] = 1; kH[~mH] = 1
    return bE, cE, kE, mE, bH, cH, kH, mH


# --------------------------------------------------------------------------
# Main simulation
# --------------------------------------------------------------------------
def run(model, Nt=12000, dft_freqs=None, source="pulse", delta_eps=0.0, f_mod=0.0,
        cw_ramp=2e-9, dft_window=None, verbose=True, progress_every=2000,
        src_R=None, hb_lat=5, hb_top=20, hb_bot=2, closed=False,
        mod_mask=None, static_deps=0.0, energy_window=None):
    """(revision options)
    src_R  : None -> soft gap source (original); float -> resistive voltage source with
             internal resistance src_R (ohm), distributed over the gap cells in parallel.
    dft_window may be 2-D (n_windows, Nt): one independent DFT accumulation per window.
    hb_lat, hb_top, hb_bot : Huygens-box clearances (cells) from the patch edges / ground.
    mod_mask : 2-D bool array over the substrate (len(SX), len(SY)); only these columns are
               modulated (None: whole substrate, original behaviour). static_deps adds a static
               permittivity offset in the same region (for sensitivity calibration).
    energy_window : bool array (Nt); over these steps the code accumulates the time-averaged
               power delivered by the modulation to the field, -1/2 d(eps)/dt |E|^2, and the
               dielectric loss sigma|E|^2, inside the Huygens box and in the whole substrate.
    closed : six-faced Huygens box with its bottom face hb_bot cells BELOW the ground and no
             ground image in the NTFF (for finite boards: build_model(air_gap>0, below>0)).
    """
    """
    source: 'pulse' (Gaussian-modulated sinusoid of the original code) or
            'cw' (sinusoid at f0 with a raised-cosine ramp of length cw_ramp).
    dft_freqs: frequencies (Hz) at which the Huygens currents (and the port
            V, I) are accumulated with a running DFT.
    dft_window: array of length Nt multiplying the samples in the running DFT
            (default: the original 5 % end taper).
    Returns a dict with time series v_feed, i_feed, energy and the DFT data.
    """
    g = model
    Nx, Ny, Nz = g["Nx"], g["Ny"], g["Nz"]
    dx, dy, dz, dt = g["dx"], g["dy"], g["dz"], g["dt"]
    npml, f0, eps_base = g["npml"], g["f0"], g["eps_base"]
    z_gnd, z_patch = g["z_gnd"], g["z_patch"]
    px1, px2, py1, py2 = g["px1"], g["px2"], g["py1"], g["py2"]
    feed_x, feed_y = g["feed_x"], g["feed_y"]
    SX = rng(g["sub_x1"], g["sub_x2"]); SY = rng(g["sub_y1"], g["sub_y2"]); SZ = rng(g["sub_z1"], g["sub_z2"])
    sub_z = np.arange(g["sub_z1"], g["sub_z2"] + 1)
    patch_mask = g["patch_mask"]

    time = np.arange(Nt) * dt
    if source == "pulse":
        tau_p = 1.2 / 8e9; t0_p = 4.5 * tau_p
        src = np.exp(-((time - t0_p) / tau_p) ** 2) * np.sin(2 * np.pi * f0 * time)
    elif source == "cw":
        ramp = np.where(time < cw_ramp, 0.5 * (1 - np.cos(np.pi * time / cw_ramp)), 1.0)
        src = ramp * np.sin(2 * np.pi * f0 * time)
    else:
        raise ValueError(source)

    # ---- CPML ----
    m_ = 4
    sx = (m_ + 1) / (150 * np.pi * dx); sy = (m_ + 1) / (150 * np.pi * dy); sz = (m_ + 1) / (150 * np.pi * dz)
    bE_x, cE_x, kE_x, mE_x, bH_x, cH_x, kH_x, mH_x = cpml_coeffs(Nx, sx, npml, dt)
    bE_y, cE_y, kE_y, mE_y, bH_y, cH_y, kH_y, mH_y = cpml_coeffs(Ny, sy, npml, dt)
    bE_z, cE_z, kE_z, mE_z, bH_z, cH_z, kH_z, mH_z = cpml_coeffs(Nz, sz, npml, dt)

    # ---- materials (identical to the fixed script) ----
    eps_r_dp = g["tand"] * eps_base
    sigma_sub = 2 * np.pi * f0 * eps0 * eps_r_dp
    Sigma_3D = np.zeros((Nx, Ny, Nz)); Sigma_3D[SX, SY, SZ] = sigma_sub
    Eps_static = np.ones((Nx, Ny, Nz)) * eps0
    r_inner, r_outer = 1, 4
    eps_coax = 2.1 * eps0
    r_loop = max(r_inner + 1, mround((r_inner + r_outer) / 2))
    X_grid, Y_grid = np.meshgrid(np.arange(1, Nx + 1), np.arange(1, Ny + 1), indexing="ij")
    dist_sq = (X_grid - feed_x) ** 2 + (Y_grid - feed_y) ** 2
    inner = dist_sq <= r_inner ** 2
    coax = (dist_sq > r_inner ** 2) & (dist_sq < r_outer ** 2)
    outer = (dist_sq >= r_outer ** 2) & (dist_sq <= (r_outer + 1) ** 2)
    for z in range(z_gnd, z_patch + 1):
        Eps_static[:, :, z - 1][coax] = eps0 * eps_base
    for z in sub_z:
        lay = Sigma_3D[:, :, z - 1]; lay[coax] = 0; lay[inner] = 0
    n_sub = len(sub_z)
    inner_sub = inner[SX, SY]; coax_sub = coax[SX, SY]
    outer_sub = ~coax_sub & ~inner_sub
    outer_sub3 = np.repeat(outer_sub[:, :, None], n_sub, axis=2)
    Eps_sub_base = np.ones(outer_sub3.shape) * (eps0 * eps_base)
    Sigma_sub = np.zeros(outer_sub3.shape)
    for kk, z in enumerate(sub_z):
        lay = Eps_sub_base[:, :, kk]; lay[coax_sub] = eps_coax; lay[inner_sub] = eps0
        Sigma_sub[:, :, kk] = Sigma_3D[SX, SY, z - 1]
    if mod_mask is None:
        M2 = np.ones(outer_sub.shape, bool)
    else:
        M2 = np.asarray(mod_mask, bool); assert M2.shape == outer_sub.shape
    M3 = outer_sub3 & M2[:, :, None]
    sig_unit = Sigma_sub.copy()                       # sigma at eps_base
    if static_deps != 0.0:
        Eps_sub_base[M3] = eps0 * (eps_base + static_deps)
        Sigma_sub[M3] = sig_unit[M3] * (eps_base + static_deps) / eps_base
    C1s = 1 / (Eps_static / dt + Sigma_3D / 2); C2s = Eps_static / dt - Sigma_3D / 2
    C1x, C2x = C1s.copy(), C2s.copy(); C1y, C2y = C1s.copy(), C2s.copy(); C1z, C2z = C1s.copy(), C2s.copy()
    for C1, C2 in ((C1x, C2x), (C1y, C2y), (C1z, C2z)):
        C1[SX, SY, SZ] = 1 / (Eps_sub_base / dt + Sigma_sub / 2)
        C2[SX, SY, SZ] = Eps_sub_base / dt - Sigma_sub / 2

    def interface(eps_sub_val):
        eps_if = 0.5 * (eps0 + eps_sub_val)
        sig_if = 0.5 * sigma_sub * (eps_sub_val / (eps0 * eps_base))
        return 1 / (eps_if / dt + sig_if / 2), eps_if / dt - sig_if / 2
    C1i, C2i = interface(eps0 * eps_base)
    C1i_s = np.full(M2.shape, C1i); C2i_s = np.full(M2.shape, C2i)
    if static_deps != 0.0:
        a, b = interface(eps0 * (eps_base + static_deps)); C1i_s[M2] = a; C2i_s[M2] = b
    for C1, C2 in ((C1x, C2x), (C1y, C2y)):
        C1[SX, SY, z_patch - 1] = C1i_s; C2[SX, SY, z_patch - 1] = C2i_s
    eps_sub_static = Eps_sub_base.copy(); sig_sub_static = Sigma_sub.copy()

    shp = (Nx, Ny, Nz)
    Ex, Ey, Ez, Hx, Hy, Hz = (np.zeros(shp) for _ in range(6))
    psiE = [np.zeros(shp) for _ in range(6)]   # Ezy Eyz Exz Ezx Eyx Exy
    psiH = [np.zeros(shp) for _ in range(6)]   # Hzy Hyz Hxz Hzx Hyx Hxy

    # ---- Huygens box ----
    hx1 = max(px1 - hb_lat, npml + 3); hx2 = min(px2 + hb_lat, Nx - npml - 3)
    hy1 = max(py1 - hb_lat, npml + 3); hy2 = min(py2 + hb_lat, Ny - npml - 3)
    hz1 = (max(z_gnd - hb_bot, npml + 3) if closed else z_gnd + hb_bot); hz2 = min(z_patch + hb_top, Nz - npml)
    nx_h, ny_h, nz_h = hx2 - hx1 + 1, hy2 - hy1 + 1, hz2 - hz1 + 1
    P = nx_h * ny_h + 2 * ny_h * nz_h + 2 * nx_h * nz_h + (nx_h * ny_h if closed else 0)
    HXS, HYS, HZS = rng(hx1, hx2), rng(hy1, hy2), rng(hz1, hz2)

    if dft_freqs is None:
        dft_freqs = np.array([f0])
    dft_freqs = np.asarray(dft_freqs, dtype=float)
    Nf = len(dft_freqs)
    if dft_window is None:
        dft_window = np.ones(Nt)
        tl = mround(0.05 * Nt)
        dft_window[Nt - tl:] = 0.5 * (1 + np.cos(np.pi * np.arange(1, tl + 1) / tl))
    W = np.atleast_2d(np.asarray(dft_window, dtype=float)); Nw = W.shape[0]
    accJ = np.zeros((Nw, Nf, 3, P), dtype=np.complex128)
    accM = np.zeros((Nw, Nf, 3, P), dtype=np.complex128)
    Jbuf = np.zeros((3, P)); Mbuf = np.zeros((3, P))

    v_feed = np.zeros(Nt); i_feed = np.zeros(Nt); energy = np.zeros(Nt)
    en_acc = dict(mod=0.0, mod_box=0.0, diel=0.0, diel_box=0.0, src=0.0, n=0)
    z_src = z_gnd + 1
    zs0, zp0, zg0 = z_src - 1, z_patch - 1, z_gnd - 1
    PX, PY = rng(px1, px2), rng(py1, py2)
    ix_min = max(2, feed_x - r_loop); ix_max = min(Nx - 1, feed_x + r_loop)
    iy_min = max(2, feed_y - r_loop); iy_max = min(Ny - 1, feed_y + r_loop)
    cHc = dt / mu0
    vol = dx * dy * dz
    Eps_vol = np.ones(shp) * eps0
    Eps_vol[SX, SY, SZ] = eps0 * eps_base
    for z in range(z_gnd, z_patch + 1):
        Eps_vol[:, :, z - 1][coax] = eps_coax
    omega = 2 * np.pi * dft_freqs
    # resistive voltage source (revision): Norton equivalent in the gap cells
    gap_sel = (slice(None), slice(None), zs0)
    if src_R is not None:
        n_gap = int(inner.sum())
        R_cell = src_R * n_gap
        sig_g = dz / (R_cell * dx * dy)
        C1g = 1.0 / (eps0 / dt + sig_g / 2); C2g = eps0 / dt - sig_g / 2
        C1z[:, :, zs0][inner] = C1g; C2z[:, :, zs0][inner] = C2g

    t_start = _time.time()
    for n in range(Nt):
        nn = n + 1
        update_H(Ex, Ey, Ez, Hx, Hy, Hz, *psiE,
                 bH_x, cH_x, kH_x, mH_x, bH_y, cH_y, kH_y, mH_y, bH_z, cH_z, kH_z, mH_z,
                 dx, dy, dz, cHc)
        if delta_eps != 0:
            e_new = eps0 * (eps_base + delta_eps * np.sin(2 * np.pi * f_mod * nn * dt))
            e_old = eps0 * (eps_base + delta_eps * np.sin(2 * np.pi * f_mod * (nn - 1) * dt))
            sl_new = eps_sub_static.copy(); sl_new[M3] = e_new
            sl_old = eps_sub_static.copy(); sl_old[M3] = e_old
            sg_new = sig_sub_static.copy(); sg_new[M3] = sig_unit[M3] * (e_new / (eps0 * eps_base))
            sg_old = sig_sub_static.copy(); sg_old[M3] = sig_unit[M3] * (e_old / (eps0 * eps_base))
            C1n = 1 / (sl_new / dt + sg_new / 2); C2o = sl_old / dt - sg_old / 2
            C1i_n = C1i_s.copy(); C1i_n[M2] = interface(e_new)[0]
            C2i_o = C2i_s.copy(); C2i_o[M2] = interface(e_old)[1]
            for C1, C2 in ((C1x, C2x), (C1y, C2y), (C1z, C2z)):
                C1[SX, SY, SZ] = C1n; C2[SX, SY, SZ] = C2o
            for C1, C2 in ((C1x, C2x), (C1y, C2y)):
                C1[SX, SY, zp0] = C1i_n; C2[SX, SY, zp0] = C2i_o
            if src_R is not None:
                C1z[:, :, zs0][inner] = C1g; C2z[:, :, zs0][inner] = C2g
        do_en = energy_window is not None and energy_window[n]
        if do_en:
            Eo = [A[SX, SY, SZ].copy() for A in (Ex, Ey, Ez)]          # E^n
            Eoi = [A[SX, SY, zp0].copy() for A in (Ex, Ey)]
            ez_gap_n = Ez[:, :, zs0][inner].copy()
        update_E(Ex, Ey, Ez, Hx, Hy, Hz, *psiH,
                 bE_x, cE_x, kE_x, mE_x, bE_y, cE_y, kE_y, mE_y, bE_z, cE_z, kE_z, mE_z,
                 C1x, C2x, C1y, C2y, C1z, C2z, dx, dy, dz)

        # PEC
        Ex[SX, SY, zg0] = 0; Ey[SX, SY, zg0] = 0
        Ex[PX, PY, zp0][patch_mask] = 0; Ey[PX, PY, zp0][patch_mask] = 0
        for z in range(z_gnd, z_patch + 1):
            Ex[:, :, z - 1][inner] = 0; Ey[:, :, z - 1][inner] = 0
            if z == z_gnd:
                Ex[:, :, z - 1][outer] = 0; Ey[:, :, z - 1][outer] = 0
        # source + pin (fixed: pin ends at z_patch-1)
        if src_R is None:
            Ez[:, :, zs0][inner] += src[n] / dz
            if do_en:   # impressed current J_s = -eps0*src/(dz dt); power = -J_s . (E^n + E^{n+1})/2
                en_acc["src"] += np.sum(eps0 * src[n] / (dz * dt) * 0.5 * (ez_gap_n + Ez[:, :, zs0][inner])) * dx * dy * dz
        else:
            Ez[:, :, zs0][inner] -= C1g * src[n] / (R_cell * dx * dy)
        for z in range(z_gnd, z_patch):
            if z != z_src:
                Ez[:, :, z - 1][inner] = 0
        if do_en:
            # discrete Poynting theorem: modulation term 1/2 d(eps)/dt E^n.E^{n+1}, loss sigma |E^{n+1/2}|^2
            EE = sum(Eo[c] * A[SX, SY, SZ] for c, A in enumerate((Ex, Ey, Ez)))
            EEi = sum(Eoi[c] * A[SX, SY, zp0] for c, A in enumerate((Ex, Ey)))
            EA = sum((0.5 * (Eo[c] + A[SX, SY, SZ])) ** 2 for c, A in enumerate((Ex, Ey, Ez)))
            EAi = sum((0.5 * (Eoi[c] + A[SX, SY, zp0])) ** 2 for c, A in enumerate((Ex, Ey)))
            if delta_eps != 0:
                deps = eps0 * delta_eps * (np.sin(2 * np.pi * f_mod * nn * dt) - np.sin(2 * np.pi * f_mod * (nn - 1) * dt)) / dt
                rho = (e_new + e_old) / (2 * eps0 * eps_base)
            else:
                deps = 0.0; rho = (eps_base + static_deps) / eps_base
            pm = -0.5 * deps * EE * M3                     # power from modulation into field (W/m^3)
            pmi = -0.5 * 0.5 * deps * EEi * M2              # interface nodes: eps_if = (eps0+eps)/2
            sg = np.where(M3, sig_unit * rho, sig_sub_static)
            sgi = 0.5 * sigma_sub * np.where(M2, rho, 1.0)
            pdl = sg * EA; pdli = sgi * EAi
            inb = np.zeros(M2.shape, bool)
            inb[max(hx1 - SX.start - 1, 0):hx2 - SX.start, max(hy1 - SY.start - 1, 0):hy2 - SY.start] = True
            for key, arr, arri in (("mod", pm, pmi), ("diel", pdl, pdli)):
                tot = (arr.sum() + arri.sum()) * vol
                ins = (arr[inb].sum() + arri[inb].sum()) * vol
                en_acc[key] += tot; en_acc[key + "_box"] += ins
            en_acc["n"] += 1
        # port V, I (fixed: V across the single gap cell)
        v_feed[n] = -Ez[feed_x - 1, feed_y - 1, zs0] * dz
        i_feed[n] = (Hx[ix_min - 1:ix_max, iy_min - 2, zs0].sum() * dx
                     + Hy[ix_max - 1, iy_min - 1:iy_max, zs0].sum() * dy
                     - Hx[ix_min - 1:ix_max, iy_max - 1, zs0].sum() * dx
                     - Hy[ix_min - 2, iy_min - 1:iy_max, zs0].sum() * dy)

        # Huygens currents (same layout / ordering as the script)
        idx = 0
        F = lambda a: a.ravel(order="F")
        n1 = nx_h * ny_h
        Jbuf[:, :] = 0; Mbuf[:, :] = 0
        Jbuf[0, idx:idx + n1] = -F(0.5 * (Hy[HXS, HYS, hz2 - 1] + Hy[HXS, HYS, hz2 - 2]))
        Jbuf[1, idx:idx + n1] = F(0.5 * (Hx[HXS, HYS, hz2 - 1] + Hx[HXS, HYS, hz2 - 2]))
        Mbuf[0, idx:idx + n1] = F(Ey[HXS, HYS, hz2 - 1])
        Mbuf[1, idx:idx + n1] = -F(Ex[HXS, HYS, hz2 - 1])
        idx += n1
        n2 = ny_h * nz_h
        Jbuf[1, idx:idx + n2] = -F(0.5 * (Hz[hx2 - 1, HYS, HZS] + Hz[hx2 - 2, HYS, HZS]))
        Jbuf[2, idx:idx + n2] = F(0.5 * (Hy[hx2 - 1, HYS, HZS] + Hy[hx2 - 2, HYS, HZS]))
        Mbuf[1, idx:idx + n2] = F(Ez[hx2 - 1, HYS, HZS])
        Mbuf[2, idx:idx + n2] = -F(Ey[hx2 - 1, HYS, HZS])
        idx += n2
        Jbuf[1, idx:idx + n2] = F(0.5 * (Hz[hx1 - 1, HYS, HZS] + Hz[hx1, HYS, HZS]))
        Jbuf[2, idx:idx + n2] = -F(0.5 * (Hy[hx1 - 1, HYS, HZS] + Hy[hx1, HYS, HZS]))
        Mbuf[1, idx:idx + n2] = -F(Ez[hx1 - 1, HYS, HZS])
        Mbuf[2, idx:idx + n2] = F(Ey[hx1 - 1, HYS, HZS])
        idx += n2
        n3 = nx_h * nz_h
        Jbuf[0, idx:idx + n3] = F(0.5 * (Hz[HXS, hy2 - 1, HZS] + Hz[HXS, hy2 - 2, HZS]))
        Jbuf[2, idx:idx + n3] = -F(0.5 * (Hx[HXS, hy2 - 1, HZS] + Hx[HXS, hy2 - 2, HZS]))
        Mbuf[0, idx:idx + n3] = -F(Ez[HXS, hy2 - 1, HZS])
        Mbuf[2, idx:idx + n3] = F(Ex[HXS, hy2 - 1, HZS])
        idx += n3
        Jbuf[0, idx:idx + n3] = -F(0.5 * (Hz[HXS, hy1 - 1, HZS] + Hz[HXS, hy1, HZS]))
        Jbuf[2, idx:idx + n3] = F(0.5 * (Hx[HXS, hy1 - 1, HZS] + Hx[HXS, hy1, HZS]))
        Mbuf[0, idx:idx + n3] = F(Ez[HXS, hy1 - 1, HZS])
        Mbuf[2, idx:idx + n3] = -F(Ex[HXS, hy1 - 1, HZS])
        if closed:   # bottom face, outward normal -z
            idx += n3
            Jbuf[0, idx:idx + n1] = F(0.5 * (Hy[HXS, HYS, hz1 - 1] + Hy[HXS, HYS, hz1 - 2]))
            Jbuf[1, idx:idx + n1] = -F(0.5 * (Hx[HXS, HYS, hz1 - 1] + Hx[HXS, HYS, hz1 - 2]))
            Mbuf[0, idx:idx + n1] = -F(Ey[HXS, HYS, hz1 - 1])
            Mbuf[1, idx:idx + n1] = F(Ex[HXS, HYS, hz1 - 1])

        wn = W[:, n]
        if wn.any():
            # the script stores float32 samples; keep that rounding for parity
            J32 = Jbuf.astype(np.float32).astype(np.float64); M32 = Mbuf.astype(np.float32).astype(np.float64)
            ph_n = np.exp(-1j * omega * n * dt)
            for iw in range(Nw):
                if wn[iw] != 0.0:
                    dft_accumulate(accJ[iw], accM[iw], J32, M32, ph_n, wn[iw])

        if nn % 500 == 0:
            Eps_now = Eps_vol
            U_E = 0.5 * np.sum(Eps_now * (Ex ** 2 + Ey ** 2 + Ez ** 2)) * vol
            U_H = 0.5 * mu0 * np.sum(Hx ** 2 + Hy ** 2 + Hz ** 2) * vol
            energy[n] = U_E + U_H
        if verbose and nn % progress_every == 0:
            el = _time.time() - t_start
            print(f"  step {nn}/{Nt}  {el:.0f}s  ({el/nn*1e3:.1f} ms/step)", flush=True)

    # surface geometry
    ax_x = np.arange(hx1, hx2 + 1) * dx; ax_y = np.arange(hy1, hy2 + 1) * dy; ax_z = np.arange(hz1, hz2 + 1) * dz
    Fo = lambda a: a.ravel(order="F")
    X_top, Y_top = np.meshgrid(ax_x, ax_y, indexing="ij")
    Y_px, Z_px = np.meshgrid(ax_y, ax_z, indexing="ij")
    X_py, Z_py = np.meshgrid(ax_x, ax_z, indexing="ij")
    surf_X = np.concatenate([Fo(X_top), np.full(Y_px.size, hx2 * dx), np.full(Y_px.size, hx1 * dx), Fo(X_py), Fo(X_py)])
    surf_Y = np.concatenate([Fo(Y_top), Fo(Y_px), Fo(Y_px), np.full(X_py.size, hy2 * dy), np.full(X_py.size, hy1 * dy)])
    surf_Z = np.concatenate([np.full(X_top.size, hz2 * dz), Fo(Z_px), Fo(Z_px), Fo(Z_py), Fo(Z_py)])
    dS = np.concatenate([np.full(X_top.size, dx * dy), np.full(2 * Y_px.size, dy * dz), np.full(2 * X_py.size, dx * dz)])
    surf_face = np.concatenate([np.full(X_top.size, 0), np.full(Y_px.size, 1), np.full(Y_px.size, 2),
                                np.full(X_py.size, 3), np.full(X_py.size, 4)])
    if closed:
        surf_face = np.concatenate([surf_face, np.full(X_top.size, 5)])
        surf_X = np.concatenate([surf_X, Fo(X_top)]); surf_Y = np.concatenate([surf_Y, Fo(Y_top)])
        surf_Z = np.concatenate([surf_Z, np.full(X_top.size, hz1 * dz)]); dS = np.concatenate([dS, np.full(X_top.size, dx * dy)])

    if Nw == 1:
        accJ, accM = accJ[0], accM[0]
    return dict(time=time, src=src, v_feed=v_feed, i_feed=i_feed, energy=energy,
                dft_freqs=dft_freqs, accJ=accJ, accM=accM, dft_window=dft_window, n_windows=Nw, closed=closed,
                energy_acc={k: (v / max(en_acc["n"], 1) if k != "n" else v) for k, v in en_acc.items()},
                surf_face=surf_face, z_patch_m=z_patch * dz,
                surf_X=surf_X, surf_Y=surf_Y, surf_Z=surf_Z, dS=dS,
                z_ground=z_gnd * dz, dt=dt, Nt=Nt, f0=f0, delta_eps=delta_eps, f_mod=f_mod,
                runtime=_time.time() - t_start, model={k: v for k, v in g.items()
                                                      if isinstance(v, (int, float, np.integer, np.floating))})


# --------------------------------------------------------------------------
# Post-processing
# --------------------------------------------------------------------------
def port_spectrum(res, N_fft=None, Z_ref=50.0, window=None):
    v, i, dt, Nt = res["v_feed"], res["i_feed"], res["dt"], res["Nt"]
    if window is None:
        window = np.ones(Nt); tl = mround(0.05 * Nt)
        window[Nt - tl:] = 0.5 * (1 + np.cos(np.pi * np.arange(1, tl + 1) / tl))
    if N_fft is None:
        N_fft = 2 ** int(np.ceil(np.log2(Nt * 4)))
    V = np.fft.fft(v * window, N_fft)[:N_fft // 2 + 1]
    I = np.fft.fft(i * window, N_fft)[:N_fft // 2 + 1]
    f = np.arange(N_fft // 2 + 1) / (dt * N_fft)
    I = I * np.exp(1j * np.pi * f * dt)
    Vinc = 0.5 * (V + Z_ref * I); Vref = 0.5 * (V - Z_ref * I)
    S11 = Vref / (Vinc + 1e-16)
    return dict(f=f, V=V, I=I, S11=S11, S11_dB=20 * np.log10(np.maximum(np.abs(S11), 1e-5)),
                Zin=V / (I + 1e-16), Pacc=0.5 * np.real(V * np.conj(I)), Pinc=np.abs(Vinc) ** 2 / (2 * Z_ref))


def dft_at(res, k):
    """Js, Ms phasors at DFT frequency index k with the Yee dt/2 correction on Js."""
    f = res["dft_freqs"][k]
    return res["accJ"][k] * np.exp(1j * np.pi * f * res["dt"]), res["accM"][k]


def far_field(res, k, theta, phi):
    """E_theta, E_phi (r = 1 m) over theta x phi grid at DFT frequency index k."""
    f = res["dft_freqs"][k]
    k0 = 2 * np.pi * f / c0
    Js, Ms = dft_at(res, k)
    dS = res["dS"]; sX, sY, sZ = res["surf_X"], res["surf_Y"], res["surf_Z"]
    sZi = 2 * res["z_ground"] - sZ
    JdS = Js * dS[None]; MdS = Ms * dS[None]
    img = 0.0 if res.get("closed", False) else 1.0      # closed box: no ground image
    Jimg = img * np.array([-1.0, -1.0, 1.0])[:, None] * JdS
    Mimg = img * np.array([1.0, 1.0, -1.0])[:, None] * MdS
    cp, sp = np.cos(phi), np.sin(phi)
    Et = np.zeros((len(phi), len(theta)), complex); Ep = np.zeros_like(Et)
    sf = 1j * k0 * np.exp(-1j * k0) / (4 * np.pi)
    for ti, th in enumerate(theta):
        ct, st = np.cos(th), np.sin(th)
        a = np.outer(st * cp, sX) + np.outer(st * sp, sY)
        pt = np.exp(1j * k0 * (a + ct * sZ[None])); pti = np.exp(1j * k0 * (a + ct * sZi[None]))
        N = pt @ JdS.T + pti @ Jimg.T; L = pt @ MdS.T + pti @ Mimg.T
        Nth = N[:, 0] * ct * cp + N[:, 1] * ct * sp - N[:, 2] * st; Nph = -N[:, 0] * sp + N[:, 1] * cp
        Lth = L[:, 0] * ct * cp + L[:, 1] * ct * sp - L[:, 2] * st; Lph = -L[:, 0] * sp + L[:, 1] * cp
        Et[:, ti] = -sf * (Lph + eta0 * Nth); Ep[:, ti] = sf * (Lth - eta0 * Nph)
    return Et, Ep


FACE_N = np.array([[0, 0, 1], [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, -1]], float)


def box_flux(res, k):
    """Time-averaged Poynting flux 1/2 Re(E x H*) . n through the Huygens surface at DFT index k,
    from the sampled equivalent currents (J = n x H, M = -n x E). Returns (total, flux through the
    parts of the lateral faces below the patch plane, i.e. within the substrate)."""
    Js, Ms = dft_at(res, k)
    nrm = FACE_N[res["surf_face"]].T                  # (3, P)
    Et = np.cross(nrm.T, Ms.T).T                      # n x M = E_t
    Ht = np.cross(Js.T, nrm.T).T                      # J x n = H_t
    # trapezoidal weights on each face (edge samples are shared between faces)
    wgt = res["dS"].astype(float).copy()
    crd = {"X": res["surf_X"], "Y": res["surf_Y"], "Z": res["surf_Z"]}
    axes = {0: "XY", 5: "XY", 1: "YZ", 2: "YZ", 3: "XZ", 4: "XZ"}
    for f, ax in axes.items():
        sel = res["surf_face"] == f
        if not sel.any(): continue
        for a in ax:
            c = crd[a][sel]; edge = np.isclose(c, c.min()) | np.isclose(c, c.max())
            ws = wgt[sel]; ws[edge] *= 0.5; wgt[sel] = ws
    S = 0.5 * np.real(np.sum(np.cross(Et.T, np.conj(Ht).T).T * nrm, axis=0)) * wgt
    lat = (res["surf_face"] >= 1) & (res["surf_face"] <= 4) & (res["surf_Z"] < res["z_patch_m"] - 1e-12)
    return float(S.sum()), float(S[lat].sum())


def radiated_power(Et, Ep, theta, phi):
    tr = getattr(np, "trapezoid", None) or np.trapz
    U = (np.abs(Et) ** 2 + np.abs(Ep) ** 2) / (2 * eta0)
    return tr(tr(U * np.sin(theta)[None], theta, axis=1), phi), U


def broadside(res, k):
    """Broadside (theta=0) Ex, Ey at DFT index k (r = 1 m)."""
    Et, Ep = far_field(res, k, np.array([0.0]), np.array([0.0]))
    return Et[0, 0], Ep[0, 0]   # theta_hat = x, phi_hat = y at phi = 0


def axial_ratio(Ex, Ey):
    d = np.angle(Ey) - np.angle(Ex)
    E1, E2 = np.abs(Ex), np.abs(Ey)
    r = np.sqrt(E1 ** 4 + E2 ** 4 + 2 * E1 ** 2 * E2 ** 2 * np.cos(2 * d))
    OA = np.sqrt(0.5 * (E1 ** 2 + E2 ** 2 + r)); OB = np.sqrt(np.maximum(0.5 * (E1 ** 2 + E2 ** 2 - r), 0))
    return 20 * np.log10(np.maximum(OA, OB) / (np.minimum(OA, OB) + 1e-16))


def cp_components(Ex, Ey):
    """RHCP, LHCP amplitudes (IEEE, propagation along +z, e^{+j w t} phasors).
    E = E_R (x - j y)/sqrt2 + E_L (x + j y)/sqrt2  =>  E_R = (Ex + j Ey)/sqrt2, E_L = (Ex - j Ey)/sqrt2.
    (The original MATLAB code had these swapped.)"""
    return (Ex + 1j * Ey) / np.sqrt(2), (Ex - 1j * Ey) / np.sqrt(2)
