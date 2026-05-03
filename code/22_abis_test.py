"""
Variant (A.bis): self-tuned amplitude with Gauss-fixed spatial template.

Replaces the externally-imposed exponential calendar of (A):
   S(r,t) = S_0 * e^(gamma t) * Gauss(r/sigma)
by a self-tuned amplitude:
   S(r,t) = eps_tilde * R_center(t) * Gauss(r/sigma),  for t < t_s

where R_center(t) is the value of R at the node closest to Omega.

Constraints: GAUSS-TEMPLATED (spatial profile fixed by Omega's
position), AUTONOMOUS (no explicit t in rule), SEEDED (small initial
core to bootstrap; cannot use zero-seed because R_center(0) = 0
would freeze the rule).

Hypothesis (Stan): the exp(gamma t) in (A) is simply the natural
amplitude needed to overcome diffusion losses; an autonomous rule
that auto-adjusts amplitude to current R_center should reproduce
DESI without needing an explicit calendar. (A.bis) tests this.

If (A.bis) reproduces DESI, it provides a concrete bridge from (A)
[externally-templated] to (E) [state-adaptive]: same physics,
different parametrisation.
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
T_S        = 15.0
ALPHA      = 0.4
KAPPA_TRAP = 8e-4
R_TRAP     = 0.05
DT         = 5e-3
T_MAX      = 80.0
N_STEPS    = int(T_MAX / DT)

# Bootstrap seed at center (cannot be zero or rule freezes)
R_CENTER_SEED = 0.05   # at the central node only; other nodes start at 0

EPS_VALUES = [0.60, 0.65, 0.68, 0.70]  # finer scan around best fit

DESI_M_INF = -3.72
DESI_M_0   = +1.65

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
# Find the node closest to Omega = lattice center
idx_center = int(np.argmin(r_node))
print(f"  central node idx = {idx_center}, distance to Omega = {r_node[idx_center]:.3f}")

tree = cKDTree(positions)
pairs = tree.query_pairs(r=R_LINK, output_type='ndarray')
i_arr = pairs[:,0]; j_arr = pairs[:,1]
n_links = len(pairs)
i_full = np.concatenate([i_arr, j_arr])
j_full = np.concatenate([j_arr, i_arr])
W = csr_matrix((np.ones(2*n_links), (i_full, j_full)), shape=(N, N))
D_diag = np.array(W.sum(axis=1)).flatten()
D_avg = float(D_diag.mean())

log_times = np.unique(np.concatenate([
    np.linspace(0.02, 1, 50),
    np.linspace(1, 15, 150),
    np.linspace(15, 30, 60),
    np.linspace(30, T_MAX, 40)
]))


def run_abis(eps_tilde):
    R = np.zeros(N)
    R[idx_center] = R_CENTER_SEED
    T = np.zeros(N); I = np.zeros(N)
    hist_t = []; hist_T2 = []; hist_Rcenter = []
    log_idx = 0
    t0 = _time.time()
    for step in range(N_STEPS):
        t_now = step * DT
        if t_now < T_S:
            R_center = R[idx_center]
            R = R + DT * eps_tilde * R_center * src_amp
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
            hist_Rcenter.append(float(R[idx_center]))
            log_idx += 1
    t_arr = np.array(hist_t); T2_arr = np.array(hist_T2); Rc = np.array(hist_Rcenter)
    valid = (t_arr > 0.05) & (T2_arr > 1e-12)
    t_v = t_arr[valid]; T2_v = T2_arr[valid]
    if len(T2_v) < 21:
        return None, None, None, Rc[-1] if len(Rc) > 0 else 0
    win = min(21, len(T2_v) // 2 * 2 - 1)
    ln_T2_smooth = savgol_filter(np.log(T2_v), win, 3)
    m_eff = -np.gradient(ln_T2_smooth, np.log(t_v))
    t_peak = float(t_v[np.argmax(T2_v)])
    pre = t_v < t_peak * 0.7
    post = t_v > t_peak * 3.0
    m_pre  = float(np.median(m_eff[pre]))  if pre.sum()  > 3 else None
    m_post = float(np.median(m_eff[post])) if post.sum() > 5 else None
    return m_pre, m_post, t_peak, Rc.max()


# Scan
print(f"\n=== Scanning variant (A.bis): eps_tilde scan ===")
print(f"  Bootstrap seed at center: R_CENTER_SEED = {R_CENTER_SEED}")
print(f"  All other nodes start at R = 0")
print(f"  Rule: dR/dt|source = eps_tilde * R_center(t) * Gauss(r/sigma)")
print()
results = {}
for eps in EPS_VALUES:
    t0 = _time.time()
    m_pre, m_post, t_peak, R_max = run_abis(eps)
    elapsed = _time.time() - t0
    if m_pre is not None:
        d = ((m_pre - DESI_M_INF)**2 + (m_post - DESI_M_0)**2)**0.5
        tag = '***' if d < 0.5 else (' * ' if d < 1.0 else '   ')
        print(f"  eps_tilde={eps:.3f}  t_peak={t_peak:5.2f}  "
              f"m_pre={m_pre:+.3f}  m_post={m_post:+.3f}  "
              f"R_center_max={R_max:.2e}  d_DESI={d:.2f}  {tag}({elapsed:.1f}s)")
        results[eps] = {'m_pre': m_pre, 'm_post': m_post,
                        't_peak': t_peak, 'R_max': R_max, 'd_DESI': d}
    else:
        print(f"  eps_tilde={eps:.3f}  insufficient T^2 dynamic range")
        results[eps] = None

# Find best
viable = {k:v for k,v in results.items() if v is not None}
if viable:
    best_eps = min(viable.keys(), key=lambda k: viable[k]['d_DESI'])
    best = viable[best_eps]
    print(f"\n=== Best fit (A.bis) ===")
    print(f"  eps_tilde* = {best_eps:.3f}")
    print(f"  m_pre  = {best['m_pre']:+.3f}  (DESI {DESI_M_INF})")
    print(f"  m_post = {best['m_post']:+.3f}  (DESI {DESI_M_0})")
    print(f"  distance = {best['d_DESI']:.3f}")

# Save
out = DATA / '22_abis_test.json'
out.write_text(json.dumps({
    'mechanism': 'variant (A.bis) - self-tuned amplitude with Gauss template',
    'rule': 'dR/dt|source = eps_tilde * R[idx_center] * Gauss(r/sigma) for t < t_s',
    'constraints': 'Gauss-templated, autonomous, seeded (bootstrap)',
    'parameters': {
        'N': N, 'D_avg': D_avg, 'T_S': T_S, 'SIGMA': SIGMA,
        'ALPHA': ALPHA, 'R_CENTER_SEED': R_CENTER_SEED,
    },
    'eps_scan': {str(k): v for k,v in results.items()},
    'desi_m_inf': DESI_M_INF, 'desi_m_0': DESI_M_0,
}, indent=2))
print(f"\nSaved -> {out}")
