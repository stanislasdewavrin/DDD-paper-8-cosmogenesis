"""
Variant (H): delta-function initial condition, pure drainage.

The cleanest "Big Bang" model: at t = 0, a single point at Omega
holds all the universe's energy E_total (concentrated as a Kronecker
delta on the central node). For t > 0, no source whatsoever --- pure
conservative drainage of Paper~I.

Constraints:
  - Spatial template: SINGLE-POINT (delta-function at Omega; no
    template, no Gaussian smearing, no spread).
  - Time-dependence: AUTONOMOUS (no source rule of any kind for t > 0).
  - Initial condition: STRONGLY SEEDED (single node holds all energy).

This is the cleanest possible "no source, just initial condition"
formulation. There is no local non-conservation in the *dynamics*:
the asymmetry that drives the cascade is entirely encoded in the
initial condition.

Hypothesis: the cascade will spread outward from the delta peak,
T^2 will briefly grow as more nodes are activated, then decay
monotonically. m_eff will be positive throughout (no phantom phase),
because there is no growth mechanism --- only redistribution of the
initial energy.

If true, this confirms that producing the DESI phantom signature
requires *some* form of energy injection beyond t = 0, even if it is
restricted to a finite epoch.
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
SIGMA      = 3.90  # only for output reference
T_S        = 15.0  # only for output reference (no source here)
ALPHA      = 0.4
KAPPA_TRAP = 8e-4
R_TRAP     = 0.05
DT         = 5e-3
T_MAX      = 80.0
N_STEPS    = int(T_MAX / DT)

# Total energy = same as the cumulative integral of the exponential
# source in variant (A) at the working point
S0_EXP = 0.05; GAMMA = 0.415
E_TOT = S0_EXP * (np.exp(GAMMA * T_S) - 1) / GAMMA

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

# Initial condition: ALL energy on a single node
R = np.zeros(N)
R[idx_center] = E_TOT
print(f"\n=== Variant (H): delta-function initial condition ===")
print(f"  R(t=0)[Omega] = {E_TOT:.4f}  (all other nodes R = 0)")
print(f"  No source for t > 0: pure conservative drainage only.")

T = np.zeros(N); I = np.zeros(N)
hist_t, hist_T2, hist_R, hist_Rmax = [], [], [], []
log_idx = 0
t0 = _time.time()
for step in range(N_STEPS):
    t_now = step * DT
    # NO SOURCE TERM
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
        hist_R.append(float(R.sum()))
        hist_Rmax.append(float(R.max()))
        log_idx += 1

print(f"  done in {_time.time()-t0:.1f}s")
t_arr = np.array(hist_t); T2_arr = np.array(hist_T2)
R_total = np.array(hist_R); R_max = np.array(hist_Rmax)

# Analyse
print(f"\n=== Results ===")
print(f"  Total R conservation check:")
print(f"    R_total(t=0) = {E_TOT:.4f}")
print(f"    R_total(t=t_s) = {R_total[np.argmin(np.abs(t_arr - T_S))]:.4f}")
print(f"    R_total(t_end) = {R_total[-1]:.4f}")
print(f"  Peak <T^2> in trajectory: {T2_arr.max():.3e} at t = {t_arr[np.argmax(T2_arr)]:.3f}")
print(f"  R_max(t_end) = {R_max[-1]:.4f}  (started at {E_TOT:.2f})")

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
    print(f"\n  Extracted: t_peak = {t_peak:.3f}  m_pre = {m_pre}  m_post = {m_post}")
    if m_pre is not None and m_pre < 0:
        print(f"  PHANTOM phase detected (m_pre < 0)!")
    elif m_pre is not None:
        print(f"  No phantom phase (m_pre = {m_pre:+.2f} > 0): cascade only decays.")
    if m_post is not None:
        d_DESI = abs(m_post - DESI_M_0)
        print(f"  m_post vs DESI {DESI_M_0}: |Δ| = {d_DESI:.3f}")
else:
    print(f"  Insufficient T^2 dynamic range for m_eff extraction.")
    m_pre, m_post, t_peak = None, None, None

out = DATA / '23_delta_init.json'
out.write_text(json.dumps({
    'mechanism': 'variant (H) - delta-function initial condition, pure drainage',
    'rule': 'R[Omega](t=0) = E_TOT; R[other](t=0) = 0; no source for t > 0',
    'constraints': 'Single-point, autonomous (no source ever), strongly seeded',
    'parameters': {'N': N, 'D_avg': D_avg, 'E_TOT': E_TOT, 'idx_center': idx_center},
    't_arr': t_arr.tolist(), 'T2_arr': T2_arr.tolist(),
    'R_total_evolution': R_total.tolist(),
    'R_max_evolution': R_max.tolist(),
    'm_pre': m_pre, 'm_post': m_post, 't_peak': t_peak,
    'desi_m_inf': DESI_M_INF, 'desi_m_0': DESI_M_0,
}, indent=2))
print(f"\nSaved -> {out}")
