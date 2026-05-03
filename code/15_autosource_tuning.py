"""
Tune auto-source variant (E) to match DESI DR2 exactly.

Variant (E) replaces the externally-imposed S_0 e^(gamma t) source
of the headline run with a local autocatalysis rule:

   for t < t_s and for each node with R_i > R_TRAP :
       dR_i/dt |_source  =  eps * R_i

The cosmogenetic exponential growth then emerges from substrate
self-feedback (each active node amplifies its own R at rate eps).
The two free parameters of variant (E) are:
   - SEED_AMP : initial Gaussian seed amplitude
   - EPS      : local autocatalysis rate

We scan (SEED_AMP, EPS) on a 5x5 grid, run each, extract m_pre and
m_post via the same protocol as 04_robustness.py, and report the
combination that best matches DESI's (-3.72, +1.65).

Also tests variant (F) at the best point: a hybrid where the
autocatalysis is also weighted by the original Gaussian profile,
       dR_i/dt |_source  =  eps * R_i * Gauss(r_i / sigma)
which restores the spatial concentration of (A).

Output: figures/fig_autosource_tuning.{pdf,png}
        data/15_autosource_tuning.json
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from scipy.signal import savgol_filter
from pathlib import Path
import json
import time as _time

np.random.seed(2024)

# Lattice + drainage parameters (match 04)
N_TARGET   = 8000
L_BOX      = 50.0
R_LINK     = 2.5
D_MIN      = 1.0
DIM        = 3
SIGMA      = 3.90
T_S        = 15.0
ALPHA_FILL = 0.4
KAPPA_TRAP = 8e-4
R_TRAP     = 0.05
DT         = 5e-3
T_MAX      = 80.0
N_STEPS    = int(T_MAX / DT)

# Scan grid
SEED_VALUES = np.array([0.18, 0.22, 0.26])
EPS_VALUES  = np.array([0.55, 0.65, 0.75])

# DESI targets
DESI_M_INF = -3.72
DESI_M_0   = +1.65

HERE = Path(__file__).resolve().parent.parent
FIG  = HERE / "figures"; FIG.mkdir(exist_ok=True)
DATA = HERE / "data";    DATA.mkdir(exist_ok=True)

# =============================================================
# Build lattice (once)
# =============================================================
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

log_times = np.unique(np.concatenate([
    np.linspace(0.02, 1, 50),
    np.linspace(1, 15, 150),
    np.linspace(15, 30, 60),
    np.linspace(30, T_MAX, 40)
]))


def run_variantE(seed_amp, eps, hybrid_F=False):
    """One run of variant (E) (or (F) if hybrid_F=True).
    Returns (m_pre, m_post, t_peak, T2_peak).
    """
    R = seed_amp * src_amp
    T = np.zeros(N); I = np.zeros(N)
    hist_t = []; hist_T2 = []
    log_idx = 0
    for step in range(N_STEPS):
        t_now = step * DT
        if t_now < T_S:
            active = R > R_TRAP
            if hybrid_F:
                R[active] = R[active] + DT * eps * R[active] * src_amp[active]
            else:
                R[active] = R[active] + DT * eps * R[active]
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
            hist_t.append(t_now)
            hist_T2.append(float((T**2).sum()))
            log_idx += 1
    t_arr = np.array(hist_t); T2_arr = np.array(hist_T2)
    if T2_arr.max() <= 0:
        return None, None, None, None
    valid = (t_arr > 0.05) & (T2_arr > 1e-12)
    t_v = t_arr[valid]; T2_v = T2_arr[valid]
    if len(T2_v) < 21:
        return None, None, None, float(T2_arr.max())
    win = min(21, len(T2_v)//2*2 - 1)
    ln_T2_smooth = savgol_filter(np.log(T2_v), win, 3)
    m_eff = -np.gradient(ln_T2_smooth, np.log(t_v))
    t_peak = float(t_v[np.argmax(T2_v)])
    pre  = t_v < t_peak * 0.7
    post = t_v > t_peak * 3.0
    m_pre  = float(np.median(m_eff[pre]))  if pre.sum()  > 3 else None
    m_post = float(np.median(m_eff[post])) if post.sum() > 5 else None
    return m_pre, m_post, t_peak, float(T2_arr.max())


# =============================================================
# 2D grid scan of variant (E)
# =============================================================
print(f"\nScanning variant (E): {len(SEED_VALUES)} x {len(EPS_VALUES)} = "
      f"{len(SEED_VALUES)*len(EPS_VALUES)} runs...")

m_pre_grid  = np.full((len(SEED_VALUES), len(EPS_VALUES)), np.nan)
m_post_grid = np.full((len(SEED_VALUES), len(EPS_VALUES)), np.nan)
peak_grid   = np.full((len(SEED_VALUES), len(EPS_VALUES)), np.nan)

t_scan = _time.time()
for i, seed in enumerate(SEED_VALUES):
    for j, eps in enumerate(EPS_VALUES):
        t0 = _time.time()
        m_pre, m_post, t_peak, T2_pk = run_variantE(seed, eps)
        elapsed = _time.time() - t0
        if m_pre is not None:
            m_pre_grid[i,j]  = m_pre
            m_post_grid[i,j] = m_post
            peak_grid[i,j]   = T2_pk
            tag = ' '
            d = ((m_pre - DESI_M_INF)**2 + (m_post - DESI_M_0)**2)**0.5
            if d < 0.5: tag = '***'
            elif d < 1.0: tag = ' * '
            print(f"  SEED={seed:5.2f}  EPS={eps:5.2f}  "
                  f"m_pre={m_pre:+.2f}  m_post={m_post:+.2f}  "
                  f"d_DESI={d:5.2f}  T2_pk={T2_pk:.1e}  {tag}({elapsed:.1f}s)")
        else:
            print(f"  SEED={seed:5.2f}  EPS={eps:5.2f}  "
                  f"insufficient T^2 dynamic range  ({elapsed:.1f}s)")
print(f"\nTotal scan time: {_time.time()-t_scan:.1f}s")

# =============================================================
# Find best (SEED, EPS) combination
# =============================================================
distances = np.sqrt((m_pre_grid - DESI_M_INF)**2 + (m_post_grid - DESI_M_0)**2)
i_best, j_best = np.unravel_index(np.nanargmin(distances), distances.shape)
seed_best = SEED_VALUES[i_best]
eps_best  = EPS_VALUES[j_best]
print(f"\n=== Best fit ===")
print(f"  SEED = {seed_best:.3f}, EPS = {eps_best:.3f}")
print(f"  m_pre  = {m_pre_grid[i_best,j_best]:+.3f}  (DESI {DESI_M_INF:+.2f})")
print(f"  m_post = {m_post_grid[i_best,j_best]:+.3f}  (DESI {DESI_M_0:+.2f})")
print(f"  distance = {distances[i_best,j_best]:.3f}")

# =============================================================
# Test variant (F) at the best (SEED, EPS) point
# =============================================================
print(f"\nTesting variant (F) [hybrid: eps * R * Gauss(r)] at SEED={seed_best}, EPS={eps_best}:")
mF_pre, mF_post, tF_peak, T2F_pk = run_variantE(seed_best, eps_best, hybrid_F=True)
print(f"  variant F: m_pre = {mF_pre:+.3f}  m_post = {mF_post:+.3f}")

# Also tune variant F separately
print(f"\nQuick variant (F) sweep over EPS at SEED={seed_best}:")
F_results = {}
for eps in EPS_VALUES:
    mp, mq, tp, _ = run_variantE(seed_best, eps, hybrid_F=True)
    if mp is not None:
        F_results[float(eps)] = (mp, mq)
        d = ((mp - DESI_M_INF)**2 + (mq - DESI_M_0)**2)**0.5
        print(f"  EPS={eps:.2f}  m_pre={mp:+.2f}  m_post={mq:+.2f}  d_DESI={d:.2f}")

# =============================================================
# Plot 2D heatmaps
# =============================================================
fig, axes = plt.subplots(1, 3, figsize=(16.0, 5.0))
ax_p, ax_q, ax_d = axes

extent = [EPS_VALUES.min()-0.1, EPS_VALUES.max()+0.1,
          SEED_VALUES.min()-0.02, SEED_VALUES.max()+0.02]

# Panel 1: m_pre heatmap
im1 = ax_p.imshow(m_pre_grid, origin='lower', aspect='auto',
                  extent=[EPS_VALUES.min(), EPS_VALUES.max(),
                          SEED_VALUES.min(), SEED_VALUES.max()],
                  cmap='RdBu_r', vmin=-8, vmax=2)
plt.colorbar(im1, ax=ax_p, label=r'$m_\infty$', fraction=0.045)
# annotate cells
for i, seed in enumerate(SEED_VALUES):
    for j, eps in enumerate(EPS_VALUES):
        if np.isfinite(m_pre_grid[i,j]):
            ax_p.text(eps, seed, f'{m_pre_grid[i,j]:.1f}',
                      ha='center', va='center', fontsize=9,
                      color='white' if abs(m_pre_grid[i,j]) > 4 else 'black')
# Mark DESI target contour
ax_p.contour(EPS_VALUES, SEED_VALUES, m_pre_grid, levels=[DESI_M_INF],
             colors='lime', linewidths=2.5)
ax_p.plot(eps_best, seed_best, marker='*', color='gold',
          markersize=18, markeredgecolor='black', markeredgewidth=1.5,
          label=f'best: ({eps_best},{seed_best})')
ax_p.set_xlabel(r'$\epsilon$ (autocatalysis rate)')
ax_p.set_ylabel(r'SEED amplitude')
ax_p.set_title(r'(A) $m_\infty$ — DESI target $-3.72$ (lime contour)',
               fontsize=11, fontweight='bold')
ax_p.legend(loc='upper right', fontsize=8)

# Panel 2: m_post heatmap
im2 = ax_q.imshow(m_post_grid, origin='lower', aspect='auto',
                  extent=[EPS_VALUES.min(), EPS_VALUES.max(),
                          SEED_VALUES.min(), SEED_VALUES.max()],
                  cmap='RdBu_r', vmin=0, vmax=3)
plt.colorbar(im2, ax=ax_q, label=r'$m_0$', fraction=0.045)
for i, seed in enumerate(SEED_VALUES):
    for j, eps in enumerate(EPS_VALUES):
        if np.isfinite(m_post_grid[i,j]):
            ax_q.text(eps, seed, f'{m_post_grid[i,j]:.2f}',
                      ha='center', va='center', fontsize=9,
                      color='white' if abs(m_post_grid[i,j]-1.5) > 1 else 'black')
ax_q.contour(EPS_VALUES, SEED_VALUES, m_post_grid, levels=[DESI_M_0],
             colors='lime', linewidths=2.5)
ax_q.plot(eps_best, seed_best, marker='*', color='gold',
          markersize=18, markeredgecolor='black', markeredgewidth=1.5)
ax_q.set_xlabel(r'$\epsilon$')
ax_q.set_title(r'(B) $m_0$ — DESI target $+1.65$ (lime contour)',
               fontsize=11, fontweight='bold')

# Panel 3: combined distance to DESI
im3 = ax_d.imshow(distances, origin='lower', aspect='auto',
                  extent=[EPS_VALUES.min(), EPS_VALUES.max(),
                          SEED_VALUES.min(), SEED_VALUES.max()],
                  cmap='viridis_r')
plt.colorbar(im3, ax=ax_d, label=r'$\sqrt{(\Delta m_\infty)^2+(\Delta m_0)^2}$',
             fraction=0.045)
for i, seed in enumerate(SEED_VALUES):
    for j, eps in enumerate(EPS_VALUES):
        if np.isfinite(distances[i,j]):
            ax_d.text(eps, seed, f'{distances[i,j]:.2f}',
                      ha='center', va='center', fontsize=9,
                      color='white' if distances[i,j] > 2 else 'black')
ax_d.plot(eps_best, seed_best, marker='*', color='gold',
          markersize=18, markeredgecolor='black', markeredgewidth=1.5,
          label=f'best: d={distances[i_best,j_best]:.2f}')
ax_d.set_xlabel(r'$\epsilon$')
ax_d.set_title(r'(C) Distance to DESI target',
               fontsize=11, fontweight='bold')
ax_d.legend(loc='upper right', fontsize=8)

fig.suptitle(r'\textbf{Variant (E) tuning:} can autocatalysis dR/dt = $\epsilon$ R '
             r'reproduce DESI without an externally-imposed exponential?',
             fontsize=12.0, y=1.02)
fig.tight_layout()
fig.savefig(FIG / 'fig_autosource_tuning.pdf', bbox_inches='tight')
fig.savefig(FIG / 'fig_autosource_tuning.png', dpi=150, bbox_inches='tight')
print(f"\nSaved -> {FIG/'fig_autosource_tuning.pdf'}")

# Save numerical results
out = DATA / '15_autosource_tuning.json'
out.write_text(json.dumps({
    'parameters': {'N': N, 'D_avg': D_avg, 'T_S': T_S, 'R_TRAP': R_TRAP,
                   'SEED_values': SEED_VALUES.tolist(),
                   'EPS_values':  EPS_VALUES.tolist()},
    'm_pre_grid':  m_pre_grid.tolist(),
    'm_post_grid': m_post_grid.tolist(),
    'distance_grid': distances.tolist(),
    'best': {
        'seed':   float(seed_best),
        'eps':    float(eps_best),
        'm_pre':  float(m_pre_grid[i_best,j_best]),
        'm_post': float(m_post_grid[i_best,j_best]),
        'distance': float(distances[i_best,j_best]),
    },
    'F_at_best_seed': {str(k): list(v) for k,v in F_results.items()},
    'desi_target_m_inf': DESI_M_INF,
    'desi_target_m_0':    DESI_M_0,
}, indent=2))
print(f"Saved -> {out}")
