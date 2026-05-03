"""
Test the conjecture  ε · t_s  =  π²  exactly.

Measured at t_s=15:  ε ≈ 0.65   →  ε·t_s ≈ 9.75
Predicted by π²:                    ε·t_s = 9.870

These differ by ~1%, within single-seed noise.
To discriminate, we test the SAME m_pre signature at a different t_s,
scaling ε to preserve ε·t_s.

Configurations tested (all with SEED=0.22, N=8000, single seed):

  t_s = 15.0  ε = π²/15 = 0.6580      (precise prediction)
  t_s = 15.0  ε = 9.75/15 = 0.6500    (current measurement)
  t_s = 15.0  ε = 9.50/15 = 0.6333    (lower bound)
  t_s = 30.0  ε = π²/30 = 0.3290      (precise prediction)
  t_s = 30.0  ε = 9.75/30 = 0.3250    (current measurement)
  t_s = 30.0  ε = 9.50/30 = 0.3167    (lower bound)

If ε·t_s is the genuine invariant, all six runs should give similar
m_pre. If furthermore the value is precisely π² = 9.870, the runs at
ε = π²/t_s should match DESI -3.72 exactly while the others should
deviate by ~0.05 in m_pre.

Output: data/18_pi_squared_test.json
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
SIGMA      = 3.90
ALPHA      = 0.4
KAPPA_TRAP = 8e-4
R_TRAP     = 0.05
DT         = 5e-3
SEED_AMP   = 0.22

PI2 = np.pi**2
DESI_M_INF = -3.72

HERE = Path(__file__).resolve().parent.parent
DATA = HERE / "data"; DATA.mkdir(exist_ok=True)

# Build lattice
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


def run_one(t_s, eps, t_max=80.0):
    n_steps = int(t_max / DT)
    log_times = np.unique(np.concatenate([
        np.linspace(0.02, 1, 50),
        np.linspace(1, t_s, 150),
        np.linspace(t_s, 2*t_s, 60),
        np.linspace(2*t_s, t_max, 40)
    ]))
    R = SEED_AMP * src_amp
    T = np.zeros(N); I = np.zeros(N)
    hist_t = []; hist_T2 = []
    log_idx = 0
    for step in range(n_steps):
        t_now = step * DT
        if t_now < t_s:
            active = R > R_TRAP
            R[active] = R[active] + DT * eps * R[active]
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
            hist_t.append(t_now); hist_T2.append(float((T**2).sum())); log_idx += 1
    t_arr = np.array(hist_t); T2_arr = np.array(hist_T2)
    valid = (t_arr > 0.05) & (T2_arr > 1e-12)
    t_v = t_arr[valid]; T2_v = T2_arr[valid]
    if len(T2_v) < 21: return None, None
    win = min(21, len(T2_v) // 2 * 2 - 1)
    ln_T2_smooth = savgol_filter(np.log(T2_v), win, 3)
    m_eff = -np.gradient(ln_T2_smooth, np.log(t_v))
    t_peak = float(t_v[np.argmax(T2_v)])
    pre = t_v < t_peak * 0.7
    post = t_v > t_peak * 3.0
    m_pre  = float(np.median(m_eff[pre]))  if pre.sum()  > 3 else None
    m_post = float(np.median(m_eff[post])) if post.sum() > 5 else None
    return m_pre, m_post


# Six runs
configs = [
    ('t_s=15  ε=π²/15 =0.6580',  15.0, PI2/15.0),
    ('t_s=15  ε=9.75/15=0.6500', 15.0, 9.75/15),
    ('t_s=15  ε=9.50/15=0.6333', 15.0, 9.50/15),
    ('t_s=30  ε=π²/30 =0.3290',  30.0, PI2/30.0),
    ('t_s=30  ε=9.75/30=0.3250', 30.0, 9.75/30),
    ('t_s=30  ε=9.50/30=0.3167', 30.0, 9.50/30),
]

print(f"\nRunning {len(configs)} variants...")
print(f"  DESI target m_inf = {DESI_M_INF}")
results = []
for name, t_s, eps in configs:
    t0 = _time.time()
    m_pre, m_post = run_one(t_s=t_s, eps=eps,
                            t_max=max(80.0, 4*t_s))
    elapsed = _time.time() - t0
    prod = eps * t_s
    delta = abs(m_pre - DESI_M_INF) if m_pre is not None else None
    print(f"  {name:30s}  ε·t_s = {prod:.4f}  "
          f"m_pre={m_pre:+.3f}  m_post={m_post:+.3f}  "
          f"|Δm_pre|={delta:.3f}  ({elapsed:.1f}s)")
    results.append({'name': name, 't_s': t_s, 'eps': eps,
                    'eps_x_ts': prod,
                    'm_pre': m_pre, 'm_post': m_post,
                    'delta_m_pre_DESI': delta})

# Summary
print(f"\n=== Summary ===")
print(f"  π² = {PI2:.4f}")
print(f"  If ε·t_s = π² is the exact invariant, the two π² runs (t_s=15, 30)")
print(f"  should give identical m_pre, both close to DESI -3.72.")

out = DATA / '18_pi_squared_test.json'
out.write_text(json.dumps({
    'pi_squared': PI2,
    'desi_m_inf': DESI_M_INF,
    'lattice': {'N': N, 'D_avg': D_avg},
    'results': results,
}, indent=2))
print(f"\nSaved -> {out}")
