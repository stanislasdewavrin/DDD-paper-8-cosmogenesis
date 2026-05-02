"""
Test the combined scaling:
  sigma = 0.078 * L_box       (already validated)
  gamma * L_box = const ?     OR gamma * sqrt(D_avg) * L = const ?

For the reference (L=50, gamma=0.415):
  gamma * L = 20.75
  gamma / L = 0.0083

Hypothesis test 1: gamma = 20.75 / L
"""

import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from scipy.signal import savgol_filter

S_0 = 0.05
T_S = 15.0
ALPHA_FILL = 0.4
KAPPA_TRAP = 8e-4
R_TRAP = 0.05
DT = 5e-3
T_MAX = 80.0
N_STEPS = int(T_MAX / DT)
R_LINK = 2.5
D_MIN = 1.0


def build_lattice(N_target, L_box, seed=2024):
    np.random.seed(seed)
    cell_size = D_MIN / np.sqrt(3)
    grid = {}
    positions_list = []
    attempts = 0
    while len(positions_list) < N_target and attempts < N_target * 30:
        candidate = np.random.uniform(0, L_box, size=3)
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


def run_one(positions, gamma, sigma):
    N = len(positions)
    tree = cKDTree(positions)
    pairs = tree.query_pairs(r=R_LINK, output_type='ndarray')
    i_arr = pairs[:, 0]; j_arr = pairs[:, 1]
    n_links = len(pairs)
    i_full = np.concatenate([i_arr, j_arr])
    j_full = np.concatenate([j_arr, i_arr])
    W = csr_matrix((np.ones(2 * n_links), (i_full, j_full)), shape=(N, N))
    D_avg = float(np.array(W.sum(axis=1)).flatten().mean())
    
    R = np.zeros(N); T = np.zeros(N); I = np.zeros(N)
    L_box_local = positions.max() + 1.0
    center = np.array([L_box_local/2] * 3)
    distances_to_center = np.linalg.norm(positions - center, axis=1)
    source_profile = np.exp(-(distances_to_center / sigma)**2)
    
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
            R = R + DT * S_0 * np.exp(gamma * t_now) * source_profile
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
            T2 = T**2
            hist_t.append(t_now)
            hist_T2.append(float(T2.sum()))
            log_idx += 1
    
    t_arr = np.array(hist_t)
    T2_arr = np.array(hist_T2)
    if T2_arr.max() <= 0: return None
    peak_idx = int(np.argmax(T2_arr))
    t_peak = float(t_arr[peak_idx])
    valid = (t_arr > 0.05) & (T2_arr > 1e-12)
    if valid.sum() < 21: return None
    t_v = t_arr[valid]; T2_v = T2_arr[valid]
    win = min(21, len(T2_v) // 2 * 2 - 1)
    ln_T2_smooth = savgol_filter(np.log(T2_v), win, 3)
    m_eff = -np.gradient(ln_T2_smooth, np.log(t_v))
    pre = t_v < t_peak * 0.7
    post = t_v > t_peak * 3
    m_pre = float(np.median(m_eff[pre])) if pre.sum() > 3 else None
    m_post = float(np.median(m_eff[post])) if post.sum() > 5 else None
    return m_pre, m_post


# Reference values
SIGMA_OVER_L = 3.90 / 50.0  # 0.0780

# Test different scaling laws for gamma
# Reference: gamma=0.415 at L=50

print("Combined scaling test (sigma = 0.078*L kept, gamma scaling varied)")
print(f"DESI target: m_inf = -3.72, m_0 = +1.65")
print()

rho_target = 0.064

# Try multiple hypotheses for gamma scaling
hypotheses = [
    ('gamma = 0.415 (constant)', lambda L: 0.415),
    ('gamma * L = 20.75 (1/L)',  lambda L: 20.75 / L),
    ('gamma = 0.415 * (50/L)^0.5 (1/sqrt(L))', lambda L: 0.415 * (50/L)**0.5),
    ('gamma * L^2 = 1037.5 (1/L^2)', lambda L: 1037.5 / L**2),
]

for hyp_name, gamma_func in hypotheses:
    print(f"\n--- {hyp_name} ---")
    print(f"{'N':>5} {'L_box':>6} {'gamma':>6} {'sigma':>6} {'  m_pre':>8} {'  m_post':>8} {'err_pre%':>9} {'err_post%':>10}", flush=True)
    
    for N_target in [4000, 8000, 12000, 16000]:
        L_box = (N_target / rho_target) ** (1/3)
        gamma = gamma_func(L_box)
        sigma = SIGMA_OVER_L * L_box
        
        positions = build_lattice(N_target, L_box)
        r = run_one(positions, gamma, sigma)
        if r is None:
            print(f"{len(positions):>5} {L_box:>6.2f} FAIL", flush=True)
            continue
        m_pre, m_post = r
        err_pre = abs((m_pre - (-3.72)) / 3.72) * 100
        err_post = abs((m_post - 1.65) / 1.65) * 100
        print(f"{len(positions):>5} {L_box:>6.2f} {gamma:>6.3f} {sigma:>6.2f} {m_pre:>+8.3f} {m_post:>+8.3f} {err_pre:>8.2f}% {err_post:>9.2f}%", flush=True)
