"""
Variant (E) ensemble test at best-fit parameters.

Runs 10 independent realisations of the autocatalysis rule
   dR_i/dt |_source = eps * R_i      for R_i > R_TRAP, t < t_s
at the tuned values (SEED=0.22, EPS=0.65, N=8000).

Compares to the exponential ensemble (Paper VIII headline):
   m_pre  = -3.737 ± 0.035
   m_post = +1.593 ± 0.176
and to DESI DR2 targets (m_inf = -3.72, m_0 = +1.65).

Output: data/16_autosource_ensemble.json + console
"""
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from scipy.signal import savgol_filter
from pathlib import Path
import json
import time as _time

# Match 04_robustness.py
N_TARGET    = 8000
L_BOX       = 50.0
R_LINK      = 2.5
D_MIN       = 1.0
DIM         = 3
SIGMA       = 3.90
T_S         = 15.0
ALPHA_FILL  = 0.4
KAPPA_TRAP  = 8e-4
R_TRAP      = 0.05
DT          = 5e-3
T_MAX       = 80.0
N_STEPS     = int(T_MAX / DT)
N_REALS     = 10

# Variant (E) tuned best-fit (from 15_autosource_tuning.py)
SEED_AMP = 0.22
EPS      = 0.65

HERE = Path(__file__).resolve().parent.parent
DATA = HERE / "data"; DATA.mkdir(exist_ok=True)


def build_lattice(seed):
    np.random.seed(seed)
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
    return np.array(positions_list)


def run_one(seed):
    pos = build_lattice(seed)
    N = len(pos)
    center = np.array([L_BOX/2]*3)
    r_node = np.linalg.norm(pos - center, axis=1)
    src_amp = np.exp(-r_node**2 / SIGMA**2)
    tree = cKDTree(pos)
    pairs = tree.query_pairs(r=R_LINK, output_type='ndarray')
    i_arr = pairs[:,0]; j_arr = pairs[:,1]
    n_links = len(pairs)
    i_full = np.concatenate([i_arr, j_arr])
    j_full = np.concatenate([j_arr, i_arr])
    W = csr_matrix((np.ones(2*n_links), (i_full, j_full)), shape=(N, N))
    D_diag = np.array(W.sum(axis=1)).flatten()
    D_avg = float(D_diag.mean())

    # Initial seed (replaces the Gaussian source amplitude profile)
    R = SEED_AMP * src_amp
    T = np.zeros(N); I = np.zeros(N)
    log_times = np.unique(np.concatenate([
        np.linspace(0.02, 1, 50),
        np.linspace(1, 15, 150),
        np.linspace(15, 30, 60),
        np.linspace(30, T_MAX, 40)
    ]))
    hist_t = []; hist_T2 = []
    log_idx = 0
    for step in range(N_STEPS):
        t_now = step * DT
        if t_now < T_S:
            active = R > R_TRAP
            R[active] = R[active] + DT * EPS * R[active]
        R_diff_ij = R[j_arr] - R[i_arr]
        flux_in_i = np.maximum(R_diff_ij,  0.0)
        flux_in_j = np.maximum(-R_diff_ij, 0.0)
        dR = np.zeros(N)
        np.add.at(dR, i_arr,  R_diff_ij)
        np.add.at(dR, j_arr, -R_diff_ij)
        R = R + DT * ALPHA_FILL * dR / max(D_avg, 1)
        R = np.maximum(R, 0.0)
        T_inflow = np.zeros(N)
        np.add.at(T_inflow, i_arr, flux_in_i)
        np.add.at(T_inflow, j_arr, flux_in_j)
        T = ALPHA_FILL * T_inflow / max(D_avg, 1)
        excess_R = np.maximum(R - R_TRAP, 0.0)
        trap_drive = np.minimum(np.maximum(KAPPA_TRAP * excess_R * T * DT, 0), 0.1)
        I = np.sqrt(np.maximum(I**2 + trap_drive * T**2, 0))
        if log_idx < len(log_times) and t_now >= log_times[log_idx]:
            hist_t.append(t_now); hist_T2.append(float((T**2).sum())); log_idx += 1

    t_arr = np.array(hist_t); T2_arr = np.array(hist_T2)
    valid = (t_arr > 0.05) & (T2_arr > 1e-12)
    t_v = t_arr[valid]; T2_v = T2_arr[valid]
    win = min(21, len(T2_v) // 2 * 2 - 1)
    ln_T2_smooth = savgol_filter(np.log(T2_v), win, 3)
    m_eff = -np.gradient(ln_T2_smooth, np.log(t_v))
    t_peak = float(t_v[np.argmax(T2_v)])
    pre = t_v < t_peak * 0.7
    post = t_v > t_peak * 3.0
    m_pre  = float(np.median(m_eff[pre]))
    m_post = float(np.median(m_eff[post]))
    return {'seed': seed, 'N': N, 'D_avg': D_avg, 't_peak': t_peak,
            'T2_peak': float(T2_arr.max()),
            'm_pre': m_pre, 'm_post': m_post}


print(f"=== Variant (E) ensemble (N={N_TARGET}, {N_REALS} realisations) ===")
print(f"  SEED_AMP = {SEED_AMP}, EPS = {EPS}")
print(f"  DESI: m_inf = -3.72,  m_0 = +1.65")
print(f"  Headline ensemble (Config A): m_pre = -3.737 ± 0.035, m_post = +1.593 ± 0.176")
print()

results = []
for k in range(N_REALS):
    seed = 2024 + k * 1000
    t0 = _time.time()
    r = run_one(seed)
    print(f"  realisation {k+1:>2d}/{N_REALS}  seed={seed}  N={r['N']}  "
          f"m_pre={r['m_pre']:+.3f}  m_post={r['m_post']:+.3f}  "
          f"({_time.time()-t0:.1f}s)")
    results.append(r)

m_pre  = np.array([r['m_pre']  for r in results])
m_post = np.array([r['m_post'] for r in results])

print()
print("=" * 60)
print(f"  m_pre  (variant E) : {m_pre.mean():+.3f} ± {m_pre.std():.3f}")
print(f"  m_post (variant E) : {m_post.mean():+.3f} ± {m_post.std():.3f}")
print()
print(f"  DESI target         : m_inf = -3.72,  m_0 = +1.65")
print(f"  Headline (Config A) : m_pre = -3.737 ± 0.035")
print(f"                       m_post = +1.593 ± 0.176")
print()
sig_pre  = abs(m_pre.mean() - (-3.72))  / m_pre.std()  if m_pre.std()  > 0 else 0
sig_post = abs(m_post.mean() - 1.65) / m_post.std() if m_post.std() > 0 else 0
print(f"  Distance from DESI:")
print(f"    m_pre  : {sig_pre:.2f} sigma")
print(f"    m_post : {sig_post:.2f} sigma")
print("=" * 60)

out = DATA / '16_autosource_ensemble.json'
out.write_text(json.dumps({
    'mechanism':  'variant (E) - autocatalysis dR/dt|_active = eps * R',
    'parameters': {
        'SEED_AMP': SEED_AMP, 'EPS': EPS,
        'N_TARGET': N_TARGET, 'T_S': T_S, 'SIGMA': SIGMA,
        'ALPHA_FILL': ALPHA_FILL, 'KAPPA_TRAP': KAPPA_TRAP,
        'R_TRAP': R_TRAP, 'DT': DT, 'T_MAX': T_MAX,
        'N_REALIZATIONS': N_REALS,
    },
    'results': results,
    'm_pre_mean':   float(m_pre.mean()),
    'm_pre_std':    float(m_pre.std()),
    'm_post_mean':  float(m_post.mean()),
    'm_post_std':   float(m_post.std()),
    'desi_target_m_inf': -3.72,
    'desi_target_m_0':    1.65,
    'reference_headline_m_pre_mean':  -3.737,
    'reference_headline_m_pre_std':    0.035,
    'reference_headline_m_post_mean': +1.593,
    'reference_headline_m_post_std':   0.176,
}, indent=2))
print(f"\nSaved -> {out}")
