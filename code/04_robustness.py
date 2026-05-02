"""
Robustness analysis: ensemble of simulation realizations
==========================================================

Run the headline simulation across multiple random seeds (different
Poisson-disk lattice realizations) to estimate statistical error bars
on m_pre, m_post, t_peak and t_at_046.

This is essential for the manuscript: a single realization could be a
fluke; the ensemble shows the result is robust.

Run time: about 5 minutes for 10 realizations.
"""

import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from scipy.signal import savgol_filter
import os
import time
import json

# ============================================================
# CONFIGURATION (same as headline run)
# ============================================================
N_TARGET = 8000
L_BOX = 50.0
R_LINK = 2.5
D_MIN = 1.0
DIM = 3

S_0 = 0.05
GAMMA = 0.415
SIGMA = 3.90
T_S = 15.0
ALPHA_FILL = 0.4
KAPPA_TRAP = 8e-4
R_TRAP = 0.05

DT = 5e-3
T_MAX = 80.0
N_STEPS = int(T_MAX / DT)

N_REALIZATIONS = 10  # number of independent lattice realizations

OUTPUT_DIR = '../data' if os.path.exists('../data') else 'data'
os.makedirs(OUTPUT_DIR, exist_ok=True)


def build_lattice(seed):
    """Build a Poisson-disk lattice with the given random seed."""
    np.random.seed(seed)
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
                                ok = False; break
                        if not ok: break
                    if not ok: break
                if not ok: break
            if not ok: break
        if ok:
            positions_list.append(candidate)
            key = (cx, cy, cz)
            if key not in grid: grid[key] = []
            grid[key].append(candidate)
        attempts += 1
    return np.array(positions_list)


def run_one_realization(seed):
    """Run one simulation, return summary diagnostics."""
    positions = build_lattice(seed)
    N = len(positions)
    
    tree = cKDTree(positions)
    pairs = tree.query_pairs(r=R_LINK, output_type='ndarray')
    i_arr = pairs[:, 0]
    j_arr = pairs[:, 1]
    n_links = len(pairs)
    
    i_full = np.concatenate([i_arr, j_arr])
    j_full = np.concatenate([j_arr, i_arr])
    W = csr_matrix((np.ones(2 * n_links), (i_full, j_full)), shape=(N, N))
    D_diag = np.array(W.sum(axis=1)).flatten()
    D_avg = float(D_diag.mean())
    
    R = np.zeros(N); T = np.zeros(N); I = np.zeros(N)
    center = np.array([L_BOX/2] * 3)
    distances_to_center = np.linalg.norm(positions - center, axis=1)
    source_profile = np.exp(-(distances_to_center / SIGMA)**2)
    inner_mask = distances_to_center < L_BOX / 4
    
    log_times = np.unique(np.concatenate([
        np.linspace(0.02, 1, 50),
        np.linspace(1, 15, 150),
        np.linspace(15, 30, 60),
        np.linspace(30, T_MAX, 40)
    ]))
    
    hist_t = []
    hist_T2 = []
    hist_I2 = []
    log_idx = 0
    
    for step in range(N_STEPS):
        t_now = step * DT
        if t_now < T_S:
            R = R + DT * S_0 * np.exp(GAMMA * t_now) * source_profile
        
        R_diff_ij = R[j_arr] - R[i_arr]
        flux_in_i = np.maximum(R_diff_ij, 0)
        flux_in_j = np.maximum(-R_diff_ij, 0)
        
        dR = np.zeros(N)
        np.add.at(dR, i_arr, R_diff_ij)
        np.add.at(dR, j_arr, -R_diff_ij)
        R = R + DT * ALPHA_FILL * dR / max(D_avg, 1)
        
        T_inflow = np.zeros(N)
        np.add.at(T_inflow, i_arr, flux_in_i)
        np.add.at(T_inflow, j_arr, flux_in_j)
        T = ALPHA_FILL * T_inflow / max(D_avg, 1)
        
        excess_R = np.maximum(R - R_TRAP, 0)
        trap_drive = np.minimum(np.maximum(KAPPA_TRAP * excess_R * T * DT, 0), 0.1)
        I = np.sqrt(np.maximum(I**2 + trap_drive * T**2, 0))
        
        if log_idx < len(log_times) and t_now >= log_times[log_idx]:
            T2 = T**2; I2 = I**2
            hist_t.append(t_now)
            hist_T2.append(float(T2.sum()))
            hist_I2.append(float(I2.sum()))
            log_idx += 1
    
    t_arr = np.array(hist_t)
    T2_arr = np.array(hist_T2)
    I2_arr = np.array(hist_I2)
    
    if T2_arr.max() <= 0:
        return None
    
    peak_idx = int(np.argmax(T2_arr))
    t_peak = float(t_arr[peak_idx])
    
    valid = (t_arr > 0.05) & (T2_arr > 1e-12)
    t_v = t_arr[valid]
    T2_v = T2_arr[valid]
    if len(T2_v) < 21:
        return None
    win = min(21, len(T2_v) // 2 * 2 - 1)
    ln_T2_smooth = savgol_filter(np.log(T2_v), win, 3)
    m_eff = -np.gradient(ln_T2_smooth, np.log(t_v))
    
    pre_mask = t_v < t_peak * 0.7
    post_mask = t_v > t_peak * 3
    m_pre = float(np.median(m_eff[pre_mask])) if pre_mask.sum() > 3 else None
    m_post = float(np.median(m_eff[post_mask])) if post_mask.sum() > 5 else None
    
    ratio = I2_arr / np.maximum(T2_arr, 1e-12)
    post_peak_indices = np.arange(peak_idx, len(t_arr))
    if len(post_peak_indices) > 5:
        idx_046 = int(np.argmin(np.abs(ratio[post_peak_indices] - 0.46)))
        t_at_046 = float(t_arr[post_peak_indices[idx_046]])
    else:
        t_at_046 = None
    
    return {
        'seed': seed,
        'N': N,
        'n_links': n_links,
        'mean_degree': D_avg,
        't_peak': t_peak,
        'T2_peak': float(T2_arr.max()),
        'm_pre': m_pre,
        'm_post': m_post,
        't_at_046': t_at_046,
        # Save trajectories too for later analysis
        't_arr': t_arr.tolist(),
        'T2_arr': T2_arr.tolist(),
        'I2_arr': I2_arr.tolist(),
    }


# ============================================================
# RUN ENSEMBLE
# ============================================================

print(f"Running {N_REALIZATIONS} independent realizations...")
print(f"(this will take a few minutes)")

t_global = time.time()
results = []

for k in range(N_REALIZATIONS):
    seed = 2024 + k * 1000
    print(f"\n--- Realization {k+1}/{N_REALIZATIONS}, seed = {seed} ---")
    t0 = time.time()
    r = run_one_realization(seed)
    if r is None:
        print(f"  FAILED")
        continue
    print(f"  N = {r['N']}, links = {r['n_links']}")
    print(f"  t_peak = {r['t_peak']:.3f}")
    print(f"  m_pre  = {r['m_pre']:+.3f}    m_post = {r['m_post']:+.3f}")
    if r['t_at_046'] is not None:
        print(f"  t at I^2/T^2 = 0.46:  {r['t_at_046']:.2f}")
    print(f"  ({time.time()-t0:.1f}s)")
    results.append(r)

print(f"\nTotal time: {time.time()-t_global:.1f}s")

# ============================================================
# STATISTICAL SUMMARY
# ============================================================

m_pre_values = np.array([r['m_pre'] for r in results if r['m_pre'] is not None])
m_post_values = np.array([r['m_post'] for r in results if r['m_post'] is not None])
t_peak_values = np.array([r['t_peak'] for r in results])
t_046_values = np.array([r['t_at_046'] for r in results if r['t_at_046'] is not None])

print("\n" + "=" * 60)
print(f"STATISTICAL SUMMARY  ({len(results)} realizations)")
print("=" * 60)
print(f"\n  m_pre  (phantom phase)   = {m_pre_values.mean():+.3f} ± {m_pre_values.std():.3f}")
print(f"  m_post (today)           = {m_post_values.mean():+.3f} ± {m_post_values.std():.3f}")
print(f"  t_peak                   = {t_peak_values.mean():.3f} ± {t_peak_values.std():.3f}")
print(f"  t at I^2/T^2 = 0.46      = {t_046_values.mean():.2f} ± {t_046_values.std():.2f}")
print()
print(f"  DESI targets:  m_inf = -3.72,  m_0 = +1.65")
print()

# Consistency with DESI (in standard deviations)
n_sig_pre = abs(m_pre_values.mean() - (-3.72)) / m_pre_values.std() if m_pre_values.std() > 0 else 0
n_sig_post = abs(m_post_values.mean() - 1.65) / m_post_values.std() if m_post_values.std() > 0 else 0
print(f"  Distance from DESI:")
print(f"    m_pre:  {n_sig_pre:.2f} sigma")
print(f"    m_post: {n_sig_post:.2f} sigma")

# ============================================================
# SAVE
# ============================================================

# Save full ensemble (without trajectories for compactness)
summary = {
    'n_realizations': len(results),
    'm_pre_values': m_pre_values.tolist(),
    'm_post_values': m_post_values.tolist(),
    't_peak_values': t_peak_values.tolist(),
    't_046_values': t_046_values.tolist(),
    'm_pre_mean': float(m_pre_values.mean()),
    'm_pre_std': float(m_pre_values.std()),
    'm_post_mean': float(m_post_values.mean()),
    'm_post_std': float(m_post_values.std()),
    't_peak_mean': float(t_peak_values.mean()),
    't_peak_std': float(t_peak_values.std()),
    't_046_mean': float(t_046_values.mean()) if len(t_046_values) > 0 else None,
    't_046_std': float(t_046_values.std()) if len(t_046_values) > 0 else None,
    'desi_target_m_inf': -3.72,
    'desi_target_m_0': 1.65,
    'parameters': {
        'N_TARGET': N_TARGET, 'L_BOX': L_BOX, 'R_LINK': R_LINK, 'D_MIN': D_MIN,
        'S_0': S_0, 'GAMMA': GAMMA, 'SIGMA': SIGMA, 'T_S': T_S,
        'ALPHA_FILL': ALPHA_FILL, 'KAPPA_TRAP': KAPPA_TRAP, 'R_TRAP': R_TRAP,
        'DT': DT, 'T_MAX': T_MAX
    }
}

with open(os.path.join(OUTPUT_DIR, 'ensemble_summary.json'), 'w') as f:
    json.dump(summary, f, indent=2)

# Also save numpy arrays for plotting
np.savez(os.path.join(OUTPUT_DIR, 'ensemble.npz'),
         m_pre=m_pre_values, m_post=m_post_values,
         t_peak=t_peak_values, t_046=t_046_values,
         t_trajectories=[np.array(r['t_arr']) for r in results],
         T2_trajectories=[np.array(r['T2_arr']) for r in results],
         I2_trajectories=[np.array(r['I2_arr']) for r in results])

print(f"\nResults saved to {OUTPUT_DIR}/ensemble_summary.json and ensemble.npz")
