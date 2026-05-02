"""
Cosmogenesis from a Discrete Lattice — Headline Simulation
==============================================================

This script reproduces the headline result of the paper
"Cosmogenesis from a Discrete Lattice: Reproducing the DESI DR2
Evolving Dark Energy Signal from a Single Founding Impulse".

Run time: about 30 seconds on a single CPU core.

Output: data/simulation_best.npz containing all the time-series data
needed to reproduce the figures.

Usage:
    python 01_simulation.py
"""

import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from scipy.signal import savgol_filter
import time
import os

# ============================================================
# REPRODUCIBILITY
# ============================================================
np.random.seed(2024)

# ============================================================
# LATTICE PARAMETERS
# ============================================================
N_TARGET = 8000
L_BOX = 50.0
R_LINK = 2.5
D_MIN = 1.0
DIM = 3

# ============================================================
# DYNAMICAL PARAMETERS (headline configuration)
# ============================================================
S_0 = 0.05             # initial source pumping rate
GAMMA = 0.415          # exponential growth rate of the source (optimized)
SIGMA = 3.90           # spatial width of the founding impulse (optimized)
T_S = 15.0             # source duration
ALPHA_FILL = 0.4       # diffusion coefficient between neighbors
KAPPA_TRAP = 8e-4      # self-trapping rate
R_TRAP = 0.05          # self-trapping threshold

# ============================================================
# INTEGRATION PARAMETERS
# ============================================================
DT = 5e-3
T_MAX = 80.0
N_STEPS = int(T_MAX / DT)

OUTPUT_DIR = '../data' if os.path.exists('../data') else 'data'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# 1. BUILD POISSON-DISK LATTICE
# ============================================================

print("Building 3D Poisson-disk lattice...")
t0 = time.time()

cell_size = D_MIN / np.sqrt(3)
grid = {}
positions_list = []
attempts = 0
max_attempts = N_TARGET * 30

while len(positions_list) < N_TARGET and attempts < max_attempts:
    candidate = np.random.uniform(0, L_BOX, size=DIM)
    cx, cy, cz = (candidate / cell_size).astype(int)
    ok = True
    for dx in range(-1, 2):
        for dy in range(-1, 2):
            for dz in range(-1, 2):
                key = (cx+dx, cy+dy, cz+dz)
                if key in grid:
                    for p in grid[key]:
                        if np.linalg.norm(p - candidate) < D_MIN:
                            ok = False
                            break
                    if not ok: break
                if not ok: break
            if not ok: break
        if not ok: break
    if ok:
        positions_list.append(candidate)
        key = (cx, cy, cz)
        if key not in grid:
            grid[key] = []
        grid[key].append(candidate)
    attempts += 1

positions = np.array(positions_list)
N = len(positions)
print(f"  N = {N} nodes ({attempts} attempts, {time.time()-t0:.1f}s)")

# ============================================================
# 2. BUILD THE GRAPH
# ============================================================

tree = cKDTree(positions)
pairs = tree.query_pairs(r=R_LINK, output_type='ndarray')
i_arr = pairs[:, 0]
j_arr = pairs[:, 1]
n_links = len(pairs)

i_full = np.concatenate([i_arr, j_arr])
j_full = np.concatenate([j_arr, i_arr])
W = csr_matrix((np.ones(2 * n_links), (i_full, j_full)), shape=(N, N))
D_diag = np.array(W.sum(axis=1)).flatten()
D_AVG = float(D_diag.mean())

print(f"  Links: {n_links}, mean degree: {D_AVG:.2f}")

# ============================================================
# 3. INITIAL CONDITIONS: empty lattice
# ============================================================

R = np.zeros(N)
T = np.zeros(N)
I = np.zeros(N)

center = np.array([L_BOX/2] * 3)
distances_to_center = np.linalg.norm(positions - center, axis=1)
source_profile = np.exp(-(distances_to_center / SIGMA)**2)

inner_mask = distances_to_center < L_BOX / 4

# ============================================================
# 4. LOGGING SCHEDULE
# ============================================================

log_times = np.unique(np.concatenate([
    np.linspace(0.02, 1, 50),
    np.linspace(1, 15, 150),
    np.linspace(15, 30, 60),
    np.linspace(30, T_MAX, 40)
]))

hist = {
    't': [], 'sum_R': [], 'sum_T2': [], 'sum_I2': [],
    'sum_T2_inner': [], 'sum_I2_inner': [],
    'R_max': [], 'extent': []
}

# ============================================================
# 5. INTEGRATE THE DYNAMICS
# ============================================================

print(f"\nIntegrating dynamics ({N_STEPS} steps)...")
t_start = time.time()
log_idx = 0

for step in range(N_STEPS):
    t_now = step * DT
    
    # Source: localized exponential pumping
    if t_now < T_S:
        rate_now = S_0 * np.exp(GAMMA * t_now)
        R = R + DT * rate_now * source_profile
    
    # Asymmetric flux between neighbors
    R_diff_ij = R[j_arr] - R[i_arr]
    flux_in_i = np.maximum(R_diff_ij, 0)
    flux_in_j = np.maximum(-R_diff_ij, 0)
    
    # Standard Laplacian update for R
    dR = np.zeros(N)
    np.add.at(dR, i_arr, R_diff_ij)
    np.add.at(dR, j_arr, -R_diff_ij)
    R = R + DT * ALPHA_FILL * dR / max(D_AVG, 1)
    
    # T = total inflow
    T_inflow = np.zeros(N)
    np.add.at(T_inflow, i_arr, flux_in_i)
    np.add.at(T_inflow, j_arr, flux_in_j)
    T = ALPHA_FILL * T_inflow / max(D_AVG, 1)
    
    # Self-trapping into bound patterns
    excess_R = np.maximum(R - R_TRAP, 0)
    trap_drive = np.minimum(np.maximum(KAPPA_TRAP * excess_R * T * DT, 0), 0.1)
    delta_I2 = trap_drive * T**2
    I = np.sqrt(np.maximum(I**2 + delta_I2, 0))
    
    # Logging
    if log_idx < len(log_times) and t_now >= log_times[log_idx]:
        T2 = T**2
        I2 = I**2
        hist['t'].append(t_now)
        hist['sum_R'].append(float(R.sum()))
        hist['sum_T2'].append(float(T2.sum()))
        hist['sum_I2'].append(float(I2.sum()))
        hist['sum_T2_inner'].append(float(T2[inner_mask].sum()))
        hist['sum_I2_inner'].append(float(I2[inner_mask].sum()))
        hist['R_max'].append(float(R.max()))
        if R.sum() > 1e-12:
            w_pos = R / R.sum()
            com = (positions * w_pos[:, None]).sum(axis=0)
            var_pos = ((positions - com)**2).sum(axis=1)
            hist['extent'].append(float(np.sqrt((var_pos * w_pos).sum())))
        else:
            hist['extent'].append(0.0)
        log_idx += 1
    
    if step % (N_STEPS // 10) == 0:
        T2 = T**2
        I2 = I**2
        print(f"  t={t_now:6.2f}  <R>={R.mean():.4f}  R_max={R.max():.3f}  "
              f"sum T^2={T2.sum():.4f}  sum I^2={I2.sum():.5f}")

print(f"\nIntegration finished in {time.time()-t_start:.1f}s")

# ============================================================
# 6. POST-PROCESSING: extract m_eff(t)
# ============================================================

t_arr = np.array(hist['t'])
sum_T2 = np.array(hist['sum_T2'])
sum_I2 = np.array(hist['sum_I2'])
sum_T2_in = np.array(hist['sum_T2_inner'])
sum_I2_in = np.array(hist['sum_I2_inner'])

peak_idx = int(np.argmax(sum_T2))
t_peak = float(t_arr[peak_idx])
print(f"\nT^2 peak at t = {t_peak:.3f}")

valid = (t_arr > 0.05) & (sum_T2 > 1e-12)
t_v = t_arr[valid]
T2_v = sum_T2[valid]
ln_t = np.log(t_v)
ln_T2 = np.log(T2_v)
win = min(21, len(ln_T2) // 2 * 2 - 1)
ln_T2_smooth = savgol_filter(ln_T2, win, 3)
m_eff = -np.gradient(ln_T2_smooth, ln_t)

pre_mask = t_v < t_peak * 0.7
post_mask = t_v > t_peak * 3
m_pre = float(np.median(m_eff[pre_mask])) if pre_mask.sum() > 3 else None
m_post = float(np.median(m_eff[post_mask])) if post_mask.sum() > 5 else None

ratio = sum_I2 / np.maximum(sum_T2, 1e-12)
post_peak = np.arange(peak_idx, len(t_arr))
if len(post_peak) > 5:
    ratio_post = ratio[post_peak]
    idx_046 = int(np.argmin(np.abs(ratio_post - 0.46)))
    t_at_046 = float(t_arr[post_peak[idx_046]])
else:
    t_at_046 = float(t_arr[-1])

# ============================================================
# 7. PRINT SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"  m_pre  (phantom phase)  = {m_pre:+.3f}    DESI target: -3.72")
print(f"  m_post (today)          = {m_post:+.3f}    DESI target: +1.65")
print(f"  t at I^2/T^2 = 0.46     = {t_at_046:.2f}")
print(f"  T^2 peak time           = {t_peak:.3f}")

# ============================================================
# 8. SAVE
# ============================================================

output_path = os.path.join(OUTPUT_DIR, 'simulation_best.npz')
np.savez(output_path,
         t=t_arr,
         sum_T2=sum_T2,
         sum_I2=sum_I2,
         sum_T2_in=sum_T2_in,
         sum_I2_in=sum_I2_in,
         t_m=t_v,
         m_eff=m_eff,
         t_peak=t_peak,
         m_pre=m_pre,
         m_post=m_post,
         t_at_046=t_at_046,
         positions=positions,
         N=N,
         L_box=L_BOX,
         dt=DT)
print(f"\nResults saved to {output_path}")
