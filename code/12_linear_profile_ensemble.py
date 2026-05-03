"""
Linear profile ensemble — 10 realisations at N=8000.

Tests whether the LINEAR source profile S(t) = (2 E_tot / t_s^2) * t
reproduces DESI DR2 target values (m_inf = -3.72, m_0 = +1.65) as
robustly as the exponential profile does.

Methodology mirrors 04_robustness.py exactly:
  - Same N=8000 Poisson-disk lattice
  - Same conservative drainage rule with edge-wise T formula
  - Same self-trapping
  - Same time sampling (log-spaced)
  - Same m_eff extraction (savgol_filter window=21, median over t < 0.7 t_peak)

Output: data/12_linear_ensemble.json + console summary
"""
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from scipy.signal import savgol_filter
from pathlib import Path
import json
import time as _time

# Match 04_robustness.py exactly
N_TARGET    = 8000
L_BOX       = 50.0
R_LINK      = 2.5
D_MIN       = 1.0
DIM         = 3

# Linear profile: amplitude calibrated to give same total integrated
# energy as the exponential baseline (S0=0.05, gamma=0.415, t_s=15)
S0_EXP      = 0.05
GAMMA_EXP   = 0.415
T_S         = 15.0
SIGMA       = 3.90
ALPHA_FILL  = 0.4
KAPPA_TRAP  = 8e-4
R_TRAP      = 0.05
DT          = 5e-3
T_MAX       = 80.0
N_STEPS     = int(T_MAX / DT)
N_REALIZATIONS = 10

# Total integrated energy of the exponential reference
E_TOT = S0_EXP * (np.exp(GAMMA_EXP * T_S) - 1) / GAMMA_EXP
# Linear amplitude: ∫_0^{t_s} A * t dt = E_TOT  ⇒  A = 2 E_TOT / t_s^2
A_LIN = 2 * E_TOT / T_S**2

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


def run_one_realization(seed):
    pos = build_lattice(seed)
    N = len(pos)
    center = np.array([L_BOX/2]*3)
    r_node = np.linalg.norm(pos - center, axis=1)

    tree = cKDTree(pos)
    pairs = tree.query_pairs(r=R_LINK, output_type='ndarray')
    i_arr = pairs[:,0]; j_arr = pairs[:,1]
    n_links = len(pairs)
    i_full = np.concatenate([i_arr, j_arr])
    j_full = np.concatenate([j_arr, i_arr])
    W = csr_matrix((np.ones(2*n_links), (i_full, j_full)), shape=(N, N))
    D_diag = np.array(W.sum(axis=1)).flatten()
    D_avg = float(D_diag.mean())

    R = np.zeros(N); T = np.zeros(N); I = np.zeros(N)
    src_amp = np.exp(-r_node**2 / SIGMA**2)
    log_times = np.unique(np.concatenate([
        np.linspace(0.02, 1, 50),
        np.linspace(1, 15, 150),
        np.linspace(15, 30, 60),
        np.linspace(30, T_MAX, 40)
    ]))
    hist_t = []; hist_T2 = []; log_idx = 0
    for step in range(N_STEPS):
        t_now = step * DT
        if t_now < T_S:
            # LINEAR profile: S(t) = A_LIN * t
            R = R + DT * A_LIN * t_now * src_amp
        R_diff_ij = R[j_arr] - R[i_arr]
        flux_in_i = np.maximum(R_diff_ij,  0.0)
        flux_in_j = np.maximum(-R_diff_ij, 0.0)
        dR = np.zeros(N)
        np.add.at(dR, i_arr,  R_diff_ij)
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


print(f"=== Linear profile ensemble (N={N_TARGET}, {N_REALIZATIONS} realisations) ===")
print(f"  Linear amplitude A = 2 E_tot / t_s^2 = {A_LIN:.6f}")
print(f"  E_tot reference (from exponential) = {E_TOT:.4f}")
print(f"  DESI targets: m_inf = -3.72, m_0 = +1.65")
print()

results = []
for k in range(N_REALIZATIONS):
    seed = 2024 + k * 1000
    t0 = _time.time()
    r = run_one_realization(seed)
    print(f"  realisation {k+1:>2d}/{N_REALIZATIONS}  seed={seed}  "
          f"N={r['N']}  m_pre={r['m_pre']:+.3f}  m_post={r['m_post']:+.3f}  "
          f"({_time.time()-t0:.1f}s)")
    results.append(r)

m_pre_arr  = np.array([r['m_pre']  for r in results])
m_post_arr = np.array([r['m_post'] for r in results])

print()
print("=" * 60)
print(f"  m_pre  (phantom plateau):  {m_pre_arr.mean():+.3f} ± {m_pre_arr.std():.3f}")
print(f"  m_post (today)         :  {m_post_arr.mean():+.3f} ± {m_post_arr.std():.3f}")
print()
print(f"  DESI targets         : m_inf = -3.72, m_0 = +1.65")
n_sig_pre  = abs(m_pre_arr.mean()  - (-3.72)) / m_pre_arr.std()  if m_pre_arr.std()  > 0 else 0
n_sig_post = abs(m_post_arr.mean() -   1.65 ) / m_post_arr.std() if m_post_arr.std() > 0 else 0
print(f"  Distance from DESI:")
print(f"    m_pre  : {n_sig_pre:.2f} sigma")
print(f"    m_post : {n_sig_post:.2f} sigma")
print("=" * 60)

# Save
out = DATA / '12_linear_ensemble.json'
out.write_text(json.dumps({
    'profile':   'linear S(t) = A_LIN * t',
    'A_LIN':     A_LIN,
    'E_TOT_ref': E_TOT,
    'parameters': {
        'N_TARGET': N_TARGET, 'L_BOX': L_BOX, 'R_LINK': R_LINK,
        'D_MIN': D_MIN, 'T_S': T_S, 'SIGMA': SIGMA,
        'ALPHA_FILL': ALPHA_FILL, 'KAPPA_TRAP': KAPPA_TRAP,
        'R_TRAP': R_TRAP, 'DT': DT, 'T_MAX': T_MAX,
        'N_REALIZATIONS': N_REALIZATIONS,
    },
    'results':       results,
    'm_pre_mean':    float(m_pre_arr.mean()),
    'm_pre_std':     float(m_pre_arr.std()),
    'm_post_mean':   float(m_post_arr.mean()),
    'm_post_std':    float(m_post_arr.std()),
    'desi_target_m_inf': -3.72,
    'desi_target_m_0':    1.65,
}, indent=2))
print(f"\nSaved -> {out}")
