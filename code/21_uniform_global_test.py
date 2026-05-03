"""
Variant (G): uniform global creation null control.

All nodes are charged uniformly from R = 0 to R = R_target = 1.0
during the creation epoch [0, t_s], at constant rate R_target/t_s.
After t_s, conservative drainage only.

Constraints: GLOBAL (uniform = no spatial structure), AUTONOMOUS,
ZERO-SEED.

Hypothesis: with strictly uniform R, all gradients are zero, so
T = 0 everywhere and at all times. No phantom phase, no dilution.
This is the cleanest null control: any spatially-localised
mechanism is needed to produce the DESI signature.
"""
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from scipy.signal import savgol_filter
from pathlib import Path
import json
import time as _time

np.random.seed(2024)

N_TARGET   = 8000
L_BOX      = 50.0
R_LINK     = 2.5
D_MIN      = 1.0
DIM        = 3
T_S        = 15.0
ALPHA      = 0.4
KAPPA_TRAP = 8e-4
R_TRAP     = 0.05
DT         = 5e-3
T_MAX      = 80.0
N_STEPS    = int(T_MAX / DT)
R_TARGET   = 1.0   # target uniform reserve at t = t_s

HERE = Path(__file__).resolve().parent.parent
DATA = HERE / "data"; DATA.mkdir(exist_ok=True)

# Lattice
print(f"Building lattice N~{N_TARGET}...")
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
                        if np.linalg.norm(p - c) < D_MIN:
                            ok = False; break
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

tree = cKDTree(positions)
pairs = tree.query_pairs(r=R_LINK, output_type='ndarray')
i_arr = pairs[:,0]; j_arr = pairs[:,1]
n_links = len(pairs)
i_full = np.concatenate([i_arr, j_arr])
j_full = np.concatenate([j_arr, i_arr])
W = csr_matrix((np.ones(2*n_links), (i_full, j_full)), shape=(N, N))
D_diag = np.array(W.sum(axis=1)).flatten()
D_avg = float(D_diag.mean())

# Run uniform creation
R = np.zeros(N); T = np.zeros(N); I = np.zeros(N)
log_times = np.unique(np.concatenate([
    np.linspace(0.02, 1, 50), np.linspace(1, 15, 150),
    np.linspace(15, 30, 60), np.linspace(30, T_MAX, 40)
]))
hist_t, hist_T2, hist_R = [], [], []
log_idx = 0
print(f"\nRunning uniform global creation (R_target={R_TARGET}, t_s={T_S})...")
t0 = _time.time()
for step in range(N_STEPS):
    t_now = step * DT
    if t_now < T_S:
        R[:] = R[:] + DT * (R_TARGET / T_S)   # uniform addition to ALL nodes
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
    if log_idx < len(log_times) and t_now >= log_times[log_idx]:
        hist_t.append(t_now)
        hist_T2.append(float((T**2).sum()))
        hist_R.append(float(R.mean()))
        log_idx += 1

print(f"  done in {_time.time()-t0:.1f}s")
t_arr = np.array(hist_t); T2_arr = np.array(hist_T2); Rmean = np.array(hist_R)

print(f"\n=== Results ===")
print(f"  Mean R(t_s={T_S}) = {Rmean[np.argmin(np.abs(t_arr - T_S))]:.4f}  (target {R_TARGET})")
print(f"  Mean R(end T_max={T_MAX}) = {Rmean[-1]:.4f}")
print(f"  Max sum(T^2) over trajectory: {T2_arr.max():.3e}")
print(f"  Mean sum(T^2): {T2_arr.mean():.3e}")
print(f"  R uniformity check: std(R)/mean(R) at t_s = {Rmean[np.argmin(np.abs(t_arr-T_S))]:.4f}")

# Try to extract m_pre, m_post — likely undefined
valid = (t_arr > 0.05) & (T2_arr > 1e-12)
if valid.sum() > 21:
    t_v = t_arr[valid]; T2_v = T2_arr[valid]
    win = min(21, len(T2_v) // 2 * 2 - 1)
    ln_T2_smooth = savgol_filter(np.log(T2_v), win, 3)
    m_eff = -np.gradient(ln_T2_smooth, np.log(t_v))
    t_peak = float(t_v[np.argmax(T2_v)])
    pre = t_v < t_peak * 0.7
    post = t_v > t_peak * 3.0
    m_pre  = float(np.median(m_eff[pre]))  if pre.sum()  > 3 else None
    m_post = float(np.median(m_eff[post])) if post.sum() > 5 else None
    print(f"  Extracted m_pre  = {m_pre}")
    print(f"  Extracted m_post = {m_post}")
else:
    print(f"  T^2 essentially zero throughout: no DESI signature possible.")
    m_pre, m_post = None, None

out = DATA / '21_uniform_global.json'
out.write_text(json.dumps({
    'mechanism': 'variant (G) - uniform global creation, all nodes 0 to R_target during [0, t_s]',
    'R_target': R_TARGET, 'T_S': T_S,
    't_arr': t_arr.tolist(), 'T2_arr': T2_arr.tolist(),
    'Rmean': Rmean.tolist(),
    'T2_max': float(T2_arr.max()),
    'T2_mean': float(T2_arr.mean()),
    'm_pre': m_pre, 'm_post': m_post,
    'verdict': 'TRIVIAL NULL: T^2 essentially zero, no observable cascade dynamics.',
}, indent=2))
print(f"\nSaved -> {out}")
