"""
Test the conjecture  ε = 2d · α / D̄

If ε is the critical autocatalysis rate set by the discrete Laplacian
eigenvalue (criticality hypothesis), it should scale linearly with α
at fixed substrate (D̄, d). The slope predicted by the criticality
argument is 2d/D̄ ≈ 6/3.73 ≈ 1.61 in 3D Poisson-disk.

Test: for each α in {0.2, 0.3, 0.4, 0.5, 0.8}, scan ε at three values
near the predicted ε* = 1.61·α, and use linear interpolation in ε to
find the ε* that gives m_pre = -3.72 (DESI target). Plot ε* vs α.

Output: figures/fig_eps_alpha_scaling.{pdf,png}
        data/17_eps_alpha_scaling.json
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

N_TARGET   = 8000
L_BOX      = 50.0
R_LINK     = 2.5
D_MIN      = 1.0
DIM        = 3
SIGMA      = 3.90
T_S        = 15.0
KAPPA_TRAP = 8e-4
R_TRAP     = 0.05
DT         = 5e-3
T_MAX      = 80.0
N_STEPS    = int(T_MAX / DT)
SEED_AMP   = 0.22  # fixed (controls m_post; we only target m_pre here)

# Test grid: vary α, scan ε around predicted critical value
ALPHA_VALUES = [0.2, 0.3, 0.4, 0.5, 0.8]
SLOPE_PRED   = 1.61   # = 2d/D_avg with D_avg=3.73, d=3
DESI_M_INF   = -3.72

HERE = Path(__file__).resolve().parent.parent
FIG  = HERE / "figures"; FIG.mkdir(exist_ok=True)
DATA = HERE / "data";    DATA.mkdir(exist_ok=True)

# Build lattice once
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
print(f"  D_avg = {D_avg:.3f},  predicted slope 2d/D_avg = {6/D_avg:.3f}")

log_times = np.unique(np.concatenate([
    np.linspace(0.02, 1, 50),
    np.linspace(1, 15, 150),
    np.linspace(15, 30, 60),
    np.linspace(30, T_MAX, 40)
]))


def run_one(alpha, eps):
    R = SEED_AMP * src_amp
    T = np.zeros(N); I = np.zeros(N)
    hist_t = []; hist_T2 = []
    log_idx = 0
    for step in range(N_STEPS):
        t_now = step * DT
        if t_now < T_S:
            active = R > R_TRAP
            R[active] = R[active] + DT * eps * R[active]
        R_diff_ij = R[j_arr] - R[i_arr]
        flux_in_i = np.maximum(R_diff_ij,  0.0)
        flux_in_j = np.maximum(-R_diff_ij, 0.0)
        dR = np.zeros(N)
        np.add.at(dR, i_arr,  R_diff_ij)
        np.add.at(dR, j_arr, -R_diff_ij)
        R = R + DT * alpha * dR / max(D_avg, 1)
        R = np.maximum(R, 0.0)
        T_inflow = np.zeros(N)
        np.add.at(T_inflow, i_arr, flux_in_i)
        np.add.at(T_inflow, j_arr, flux_in_j)
        T = alpha * T_inflow / max(D_avg, 1)
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


# For each alpha, scan 3 eps around predicted critical
print(f"\nScan: 5 alpha × 3 eps = 15 runs...")
results = {}
t_scan = _time.time()
for alpha in ALPHA_VALUES:
    eps_predicted = SLOPE_PRED * alpha
    eps_scan = [0.7 * eps_predicted, 1.0 * eps_predicted, 1.3 * eps_predicted]
    runs = []
    for eps in eps_scan:
        t0 = _time.time()
        m_pre, m_post = run_one(alpha, eps)
        runs.append({'eps': eps, 'm_pre': m_pre, 'm_post': m_post})
        print(f"  alpha={alpha:.2f}  eps={eps:.3f}  m_pre={m_pre:+.3f}  "
              f"m_post={m_post:+.3f}  ({_time.time()-t0:.1f}s)")
    results[alpha] = runs
print(f"Total: {_time.time()-t_scan:.1f}s")

# For each alpha, find eps* that gives m_pre = -3.72 by linear interpolation
print(f"\nLinear interpolation to find eps* (target m_pre = {DESI_M_INF}):")
eps_star_dict = {}
for alpha, runs in results.items():
    eps_arr = np.array([r['eps'] for r in runs])
    mp_arr  = np.array([r['m_pre'] for r in runs])
    # Sort by m_pre (since m_pre is monotone in eps: more eps → more negative m_pre)
    order = np.argsort(mp_arr)
    mp_s = mp_arr[order]; eps_s = eps_arr[order]
    # Interpolate: find eps where m_pre = DESI_M_INF
    if mp_s.min() <= DESI_M_INF <= mp_s.max():
        eps_star = float(np.interp(DESI_M_INF, mp_s, eps_s))
        eps_star_dict[alpha] = eps_star
        print(f"  alpha={alpha:.2f}  eps* = {eps_star:.3f}  "
              f"(predicted by 2d/D_avg×alpha = {SLOPE_PRED * alpha:.3f})")
    else:
        eps_star_dict[alpha] = None
        print(f"  alpha={alpha:.2f}  m_pre range [{mp_s.min():.2f},{mp_s.max():.2f}] "
              f"does not bracket {DESI_M_INF}")

# Linear fit eps* vs alpha
alpha_arr = np.array([a for a in eps_star_dict if eps_star_dict[a] is not None])
eps_arr   = np.array([eps_star_dict[a] for a in alpha_arr])
if len(alpha_arr) >= 2:
    coef = np.polyfit(alpha_arr, eps_arr, 1)
    slope = float(coef[0]); intercept = float(coef[1])
    pred = slope * alpha_arr + intercept
    r2 = 1 - ((eps_arr - pred)**2).sum() / ((eps_arr - eps_arr.mean())**2).sum()
    print(f"\nLinear fit: eps* = {slope:.3f} × alpha + {intercept:+.3f}  R² = {r2:.4f}")
    print(f"Predicted slope (criticality):  2d/D_avg = {6/D_avg:.3f}")
    print(f"Slope ratio: measured/predicted = {slope/(6/D_avg):.3f}")
else:
    slope, intercept, r2 = None, None, None

# Plot
fig, ax = plt.subplots(figsize=(8.5, 6.0))
# All scan data (faint)
for alpha, runs in results.items():
    eps_arr_run = [r['eps'] for r in runs]
    mp_arr_run  = [r['m_pre'] for r in runs]
    ax.scatter([alpha]*len(runs), eps_arr_run,
               c=mp_arr_run, cmap='RdBu_r', vmin=-7, vmax=-1,
               s=80, alpha=0.7, edgecolors='black', linewidths=0.5)
# Best-fit eps* points (large stars)
ax.plot(alpha_arr, eps_arr, marker='*', markersize=22,
        markerfacecolor='gold', markeredgecolor='black',
        markeredgewidth=1.2, linestyle='none',
        label=r'$\epsilon^*$ giving $m_\infty = -3.72$', zorder=10)
# Linear fit
if slope is not None:
    a_line = np.linspace(0, 1.0, 50)
    eps_fit = slope * a_line + intercept
    ax.plot(a_line, eps_fit, 'k-', lw=2,
            label=fr'fit: $\epsilon^* = {slope:.2f}\,\alpha + {intercept:+.2f}$, $R^2={r2:.3f}$')
# Predicted line eps = 2d/D_avg × alpha
ax.plot(a_line, (6/D_avg) * a_line, 'g--', lw=2,
        label=fr'criticality prediction: $\epsilon^* = (2d/\bar D)\,\alpha = {6/D_avg:.2f}\,\alpha$')
ax.set_xlabel(r'drainage rate $\alpha$', fontsize=12)
ax.set_ylabel(r'critical autocatalysis $\epsilon^*$ (matching DESI $m_\infty$)', fontsize=12)
ax.set_title(r'Test of criticality conjecture: $\epsilon^* \propto \alpha$ ?',
             fontsize=12, fontweight='bold')
ax.legend(loc='lower right', fontsize=10, framealpha=0.95)
ax.grid(alpha=0.3)
ax.set_xlim(0, 0.9)
ax.set_ylim(0, max(1.5, eps_arr.max()*1.1) if len(eps_arr) > 0 else 1.5)
cbar = plt.colorbar(ax.collections[0], ax=ax, label=r'$m_\infty$ (scan)',
                    fraction=0.04, pad=0.02)
fig.tight_layout()
fig.savefig(FIG / 'fig_eps_alpha_scaling.pdf', bbox_inches='tight')
fig.savefig(FIG / 'fig_eps_alpha_scaling.png', dpi=150, bbox_inches='tight')
print(f"\nSaved -> {FIG/'fig_eps_alpha_scaling.pdf'}")

out = DATA / '17_eps_alpha_scaling.json'
out.write_text(json.dumps({
    'D_avg': D_avg,
    'predicted_slope_2d_over_D_avg': 6/D_avg,
    'desi_m_inf_target': DESI_M_INF,
    'scan': {str(a): r for a,r in results.items()},
    'eps_star_per_alpha': {str(a): v for a,v in eps_star_dict.items()},
    'fit': {'slope': slope, 'intercept': intercept, 'R2': r2}
        if slope is not None else None,
}, indent=2))
print(f"Saved -> {out}")
