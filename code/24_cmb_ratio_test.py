"""
CMB-epoch ratio test: what does z = 1100 correspond to in our DDD
cosmogenetic simulation?

In standard cosmology, z = 1100 (CMB last scattering) means the
universe was 1100x smaller (scale factor a ~ 1/1100). In DDD the
lattice does not expand; what changes is the local content R or T^2.

Using the t<->a calibration of Paper VIII:
    a(t) = a* * (t / t_peak)   with a* = 0.69, t_peak = 15
We have:
    a = 1     (today)  -> t_today ≈ t_peak / a* ≈ 21.7
    a = 1/1101 (CMB)   -> t_CMB ≈ t_peak * a_CMB / a* ≈ 0.0197

This script extracts <T^2>(t) and <R>(t) from our headline simulation
at these two times and computes the ratio. The ratio is then
compared to standard cosmology expectations:

  - If T^2 ~ rho_DE (density-like), ratio ~ a^(-3) ~ 1100^3
  - If T^2 ~ rho_rad (radiation-like), ratio ~ a^(-4) ~ 1100^4
  - If T^2 ~ T_CMB^2 (temperature-squared), ratio ~ 1100^2
  - In DDD via 1 + z ~ (T^2_e/T^2_o)^eta, ratio depends on eta

The point is *not* to claim DDD predicts CMB --- Paper VIII
explicitly does not address pre-recombination physics. The point is
to compute the ratio that the cascade actually produces, and see
how it compares to standard cosmology.
"""
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from pathlib import Path
import time as _time

np.random.seed(2024)

N_TARGET = 8000; L_BOX = 50.0; R_LINK = 2.5; D_MIN = 1.0; DIM = 3
SIGMA = 3.90; T_S = 15.0; ALPHA = 0.4
KAPPA_TRAP = 8e-4; R_TRAP = 0.05
DT = 5e-3; T_MAX = 80.0; N_STEPS = int(T_MAX/DT)
EPS_E = 0.65; SEED_AMP = 0.22

# Calibration
A_STAR = 0.69
T_PEAK = 15.0
Z_CMB = 1100.0

T_TODAY = T_PEAK / A_STAR              # t for a=1 today
A_CMB = 1.0 / (1.0 + Z_CMB)            # ~ 0.000909
T_CMB = T_PEAK * A_CMB / A_STAR        # ~ 0.0198

print(f"Calibration t<->a: a = {A_STAR} * (t / {T_PEAK})")
print(f"  Today (a=1):      t_today = {T_TODAY:.3f}")
print(f"  CMB (z={Z_CMB}):  t_CMB   = {T_CMB:.5f}")
print(f"  -> Need <T^2>(t={T_CMB:.3f}) and <T^2>(t={T_TODAY:.2f})")

# Build lattice
print(f"\nBuilding lattice N~{N_TARGET}...")
cell_size = D_MIN / np.sqrt(3)
grid = {}; positions_list = []; attempts = 0
while len(positions_list) < N_TARGET and attempts < N_TARGET * 30:
    c = np.random.uniform(0, L_BOX, size=DIM)
    cx, cy, cz = (c / cell_size).astype(int)
    ok = True
    for dx in range(-1, 2):
        for dy in range(-1, 2):
            for dz in range(-1, 2):
                key = (cx+dx, cy+dy, cz+dz)
                if key in grid:
                    for p in grid[key]:
                        if np.linalg.norm(p - c) < D_MIN: ok = False; break
                    if not ok: break
                if not ok: break
            if not ok: break
        if not ok: break
    if ok:
        positions_list.append(c)
        grid.setdefault((cx,cy,cz), []).append(c)
    attempts += 1
positions = np.array(positions_list)
N = len(positions)
print(f"  N = {N}")

center = np.array([L_BOX/2]*3)
r_node = np.linalg.norm(positions - center, axis=1)
src_amp = np.exp(-r_node**2 / SIGMA**2)
tree = cKDTree(positions)
pairs = tree.query_pairs(r=R_LINK, output_type='ndarray')
i_arr = pairs[:,0]; j_arr = pairs[:,1]
n_links = len(pairs)
i_full = np.concatenate([i_arr, j_arr])
j_full = np.concatenate([j_arr, i_arr])
W = csr_matrix((np.ones(2*n_links), (i_full, j_full)), shape=(N, N))
D_diag = np.array(W.sum(axis=1)).flatten()
D_avg = float(D_diag.mean())

# Run variant (E) at working point, with VERY DENSE early-time sampling
# We need t_CMB ≈ 0.02, so sample every few timesteps for the first second
log_times = np.unique(np.concatenate([
    np.linspace(0.005, 0.5, 100),    # very dense early sampling
    np.linspace(0.5, 5.0, 50),
    np.linspace(5.0, 22.0, 50),
    np.linspace(22.0, T_MAX, 30),
]))

R = SEED_AMP * src_amp
T = np.zeros(N); I = np.zeros(N)
hist_t, hist_T2, hist_Rmean, hist_Rmax = [], [], [], []
log_idx = 0
print(f"\nRunning variant (E) with dense early-time sampling...")
t0 = _time.time()
for step in range(N_STEPS):
    t_now = step * DT
    if t_now < T_S:
        active = R > R_TRAP
        R[active] = R[active] + DT * EPS_E * R[active]
    R_diff_ij = R[j_arr] - R[i_arr]
    flux_in_i = np.maximum(R_diff_ij,  0.0)
    flux_in_j = np.maximum(-R_diff_ij, 0.0)
    dR = np.zeros(N)
    np.add.at(dR, i_arr,  R_diff_ij)
    np.add.at(dR, j_arr, -R_diff_ij)
    R = R + DT * ALPHA * dR / max(D_avg, 1)
    R = np.maximum(R, 0.0)
    T_inflow = np.zeros(N)
    np.add.at(T_inflow, i_arr, flux_in_i)
    np.add.at(T_inflow, j_arr, flux_in_j)
    T = ALPHA * T_inflow / max(D_avg, 1)
    excess_R = np.maximum(R - R_TRAP, 0.0)
    trap_drive = np.minimum(np.maximum(KAPPA_TRAP * excess_R * T * DT, 0), 0.1)
    I = np.sqrt(np.maximum(I**2 + trap_drive * T**2, 0))
    if log_idx < len(log_times) and t_now >= log_times[log_idx]:
        hist_t.append(t_now)
        hist_T2.append(float((T**2).sum()))
        hist_Rmean.append(float(R.mean()))
        hist_Rmax.append(float(R.max()))
        log_idx += 1
print(f"  done in {_time.time()-t0:.1f}s")

t_arr = np.array(hist_t)
T2_arr = np.array(hist_T2)
Rmean_arr = np.array(hist_Rmean)
Rmax_arr = np.array(hist_Rmax)

# Interpolate to get values at t_CMB and t_today
def interp_at(t_query, t_arr, val_arr):
    return float(np.interp(t_query, t_arr, val_arr))

T2_CMB   = interp_at(T_CMB,   t_arr, T2_arr)
T2_today = interp_at(T_TODAY, t_arr, T2_arr)
T2_peak  = T2_arr.max()
R_CMB    = interp_at(T_CMB,   t_arr, Rmean_arr)
R_today  = interp_at(T_TODAY, t_arr, Rmean_arr)
Rmax_CMB = interp_at(T_CMB,   t_arr, Rmax_arr)
Rmax_today = interp_at(T_TODAY, t_arr, Rmax_arr)

print(f"\n=== Cascade values at the two epochs ===")
print(f"  At t_CMB   = {T_CMB:.5f} (z={Z_CMB}, a=1/{Z_CMB+1:.0f}):")
print(f"    sum(T^2)   = {T2_CMB:.4e}")
print(f"    mean(R)    = {R_CMB:.4e}")
print(f"    max(R)     = {Rmax_CMB:.4e}")
print(f"  At t_peak  = {T_PEAK:.3f} (z*={1/A_STAR-1:.2f}, a*={A_STAR}):")
print(f"    sum(T^2)   = {T2_peak:.4e}")
print(f"  At t_today = {T_TODAY:.3f} (a=1):")
print(f"    sum(T^2)   = {T2_today:.4e}")
print(f"    mean(R)    = {R_today:.4e}")
print(f"    max(R)     = {Rmax_today:.4e}")

print(f"\n=== Ratios CMB/today (DDD simulation) ===")
ratio_T2 = T2_CMB / T2_today if T2_today > 0 else float('nan')
ratio_R  = R_CMB / R_today  if R_today > 0 else float('nan')
ratio_Rmax = Rmax_CMB / Rmax_today if Rmax_today > 0 else float('nan')
print(f"  sum(T^2)_CMB / sum(T^2)_today  = {ratio_T2:.4e}")
print(f"  mean(R)_CMB / mean(R)_today    = {ratio_R:.4e}")
print(f"  max(R)_CMB / max(R)_today      = {ratio_Rmax:.4e}")

print(f"\n=== Standard cosmology expectations ===")
print(f"  CMB temperature ratio T_CMB/T_today = (1+z) = {Z_CMB+1:.0f}")
print(f"  Radiation density ratio rho_rad,CMB/rho_rad,today = (1+z)^4 = {(Z_CMB+1)**4:.3e}")
print(f"  Matter density ratio   rho_m,CMB/rho_m,today    = (1+z)^3 = {(Z_CMB+1)**3:.3e}")

print(f"\n=== Interpretation ===")
print(f"In standard cosmology, the CMB epoch is HOTTER (higher density).")
print(f"In our DDD cosmogenetic simulation, t = t_CMB ≈ 0.02 is just after")
print(f"the cosmogenetic epoch START -- the cascade is just beginning,")
print(f"so T^2 at t_CMB is MUCH SMALLER than T^2 today. The ratio is")
print(f"INVERTED with respect to standard cosmology.")
print(f"This is consistent with Paper VIII's explicit statement that we")
print(f"do not model pre-recombination physics; the late-time cascade we")
print(f"compute does not reach the CMB epoch in any meaningful sense.")
