"""
Scan ε(t_s) at constant DESI target m_inf = -3.72.

For each t_s in {10, 15, 20, 25, 30}, scan ε at three values around the
prediction π²/t_s, then linearly interpolate to find ε*(t_s) that gives
m_pre = -3.72.

Compare to:
  - π²/t_s     (precise spectral conjecture)
  - C/t_s      (any simple inverse, with C ≠ π²)
  - C/t_s^x    (other power law)
  - non-power  (complex dependency on t_s and other parameters)

Output: figures/fig_eps_ts_scan.{pdf,png}
        data/19_eps_ts_scan.json
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
ALPHA      = 0.4
KAPPA_TRAP = 8e-4
R_TRAP     = 0.05
DT         = 5e-3
SEED_AMP   = 0.22

PI2 = np.pi**2
DESI_M_INF = -3.72
TS_VALUES = [10.0, 15.0, 20.0, 25.0, 30.0]

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


def run_one(t_s, eps, t_max=None):
    if t_max is None:
        t_max = max(60.0, 3.5*t_s)
    n_steps = int(t_max / DT)
    log_times = np.unique(np.concatenate([
        np.linspace(0.02, 1, 50),
        np.linspace(1, t_s, 100),
        np.linspace(t_s, 2*t_s, 50),
        np.linspace(2*t_s, t_max, 30)
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
    if len(T2_v) < 21: return None
    win = min(21, len(T2_v) // 2 * 2 - 1)
    ln_T2_smooth = savgol_filter(np.log(T2_v), win, 3)
    m_eff = -np.gradient(ln_T2_smooth, np.log(t_v))
    t_peak = float(t_v[np.argmax(T2_v)])
    pre = t_v < t_peak * 0.7
    return float(np.median(m_eff[pre])) if pre.sum() > 3 else None


print(f"\n=== Scan: 5 t_s × 3 eps = 15 runs ===")
print(f"  DESI target m_inf = {DESI_M_INF}")
print(f"  Predictions: ε = π²/t_s = {PI2:.4f}/t_s\n")

results = {}
t_scan = _time.time()
for t_s in TS_VALUES:
    eps_pred = PI2 / t_s
    eps_scan = [eps_pred * 0.85, eps_pred, eps_pred * 1.15]
    runs = []
    for eps in eps_scan:
        t0 = _time.time()
        m_pre = run_one(t_s=t_s, eps=eps)
        elapsed = _time.time() - t0
        runs.append({'eps': float(eps), 'm_pre': m_pre})
        marker = ' '
        if m_pre is not None and abs(m_pre - DESI_M_INF) < 0.05:
            marker = '***'
        elif m_pre is not None and abs(m_pre - DESI_M_INF) < 0.15:
            marker = ' * '
        print(f"  t_s={t_s:5.1f}  ε={eps:.4f}  ε·t_s={eps*t_s:.4f}  "
              f"m_pre={m_pre:+.3f}  Δ={abs(m_pre-DESI_M_INF) if m_pre else 0:.3f}  "
              f"{marker}({elapsed:.1f}s)")
    results[t_s] = runs

total_time = _time.time() - t_scan
print(f"\nScan total: {total_time:.1f}s")

# Linear interpolation for each t_s
print(f"\n=== ε*(t_s) for m_pre = {DESI_M_INF} ===")
ts_arr = []; eps_star_arr = []; product_arr = []
for t_s, runs in results.items():
    eps_arr = np.array([r['eps'] for r in runs])
    mp_arr  = np.array([r['m_pre'] for r in runs])
    order = np.argsort(mp_arr)
    mp_s, eps_s = mp_arr[order], eps_arr[order]
    if mp_s.min() <= DESI_M_INF <= mp_s.max():
        eps_star = float(np.interp(DESI_M_INF, mp_s, eps_s))
        product  = eps_star * t_s
        eps_pi2  = PI2 / t_s
        ts_arr.append(t_s); eps_star_arr.append(eps_star); product_arr.append(product)
        rel_pi2  = (eps_star / eps_pi2 - 1) * 100
        print(f"  t_s={t_s:5.1f}  ε*={eps_star:.4f}  ε·t_s={product:.4f}  "
              f"vs π²={PI2:.4f}  (rel diff {rel_pi2:+.2f}%)")
    else:
        print(f"  t_s={t_s:5.1f}  out of bracket")

ts_arr = np.array(ts_arr); eps_star_arr = np.array(eps_star_arr)
product_arr = np.array(product_arr)

# Power-law fit  ε* = C * t_s^x
if len(ts_arr) >= 2:
    coef = np.polyfit(np.log(ts_arr), np.log(eps_star_arr), 1)
    x_fit = float(coef[0]); logC = float(coef[1])
    C_fit = float(np.exp(logC))
    print(f"\n=== Power-law fit: ε* = C × t_s^x ===")
    print(f"  C = {C_fit:.4f}")
    print(f"  x = {x_fit:+.4f}     (predicted by π²/t_s : x = -1)")
    print(f"  Compare C to π² = {PI2:.4f}")

# Plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5.0))
ax_e, ax_p = axes

# Panel 1: ε* vs t_s, with predictions
ts_line = np.linspace(8, 35, 100)
ax_e.plot(ts_line, PI2/ts_line, 'g--', lw=2,
          label=fr'prediction: $\epsilon = \pi^2/t_s = {PI2:.3f}/t_s$')
if len(ts_arr) >= 2:
    ax_e.plot(ts_line, C_fit * ts_line**x_fit, 'r:', lw=2,
              label=fr'fit: $\epsilon = {C_fit:.3f} \cdot t_s^{{{x_fit:+.3f}}}$')
ax_e.plot(ts_arr, eps_star_arr, marker='*', markersize=18,
          markerfacecolor='gold', markeredgecolor='black',
          markeredgewidth=1.2, linestyle='none',
          label=r'measured $\epsilon^*$ (DESI-matching)', zorder=10)
# Add scan data faintly
for t_s, runs in results.items():
    for r in runs:
        if r['m_pre'] is not None:
            color = 'red' if r['m_pre'] < DESI_M_INF else 'blue'
            ax_e.scatter(t_s, r['eps'], c='gray', s=30, alpha=0.4, zorder=5)
ax_e.set_xlabel(r'$t_s$ (source phase duration)', fontsize=12)
ax_e.set_ylabel(r'$\epsilon^*$ (matching DESI)', fontsize=12)
ax_e.set_title(r'$\epsilon^*$ vs $t_s$', fontsize=12, fontweight='bold')
ax_e.set_xscale('log'); ax_e.set_yscale('log')
ax_e.legend(loc='upper right', fontsize=10)
ax_e.grid(alpha=0.3, which='both')

# Panel 2: product ε*·t_s vs t_s
ax_p.axhline(PI2, color='green', linestyle='--', lw=2,
             label=fr'$\pi^2 = {PI2:.4f}$')
ax_p.plot(ts_arr, product_arr, marker='*', markersize=18,
          markerfacecolor='gold', markeredgecolor='black',
          markeredgewidth=1.2, linestyle='-',
          label=r'measured $\epsilon^* \cdot t_s$', zorder=10)
ax_p.set_xlabel(r'$t_s$', fontsize=12)
ax_p.set_ylabel(r'$\epsilon^* \cdot t_s$ (product, would be invariant if $\epsilon \propto 1/t_s$)',
                fontsize=11)
ax_p.set_title(r'Product $\epsilon^* \cdot t_s$ — invariant or not?',
               fontsize=12, fontweight='bold')
ax_p.set_xlim(8, 32)
ax_p.legend(loc='upper left', fontsize=10)
ax_p.grid(alpha=0.3)

fig.suptitle(r'\textbf{Test of $\epsilon^* = \pi^2/t_s$ at $m_\infty^{\rm DESI} = -3.72$}',
             fontsize=12.5, y=1.01)
fig.tight_layout()
fig.savefig(FIG / 'fig_eps_ts_scan.pdf', bbox_inches='tight')
fig.savefig(FIG / 'fig_eps_ts_scan.png', dpi=150, bbox_inches='tight')
print(f"\nSaved -> {FIG/'fig_eps_ts_scan.pdf'}")

out = DATA / '19_eps_ts_scan.json'
out.write_text(json.dumps({
    'pi_squared': PI2,
    'desi_m_inf': DESI_M_INF,
    'lattice': {'N': N, 'D_avg': D_avg},
    'scan': {str(k): v for k,v in results.items()},
    'eps_star_per_ts': dict(zip([str(t) for t in ts_arr.tolist()],
                                 eps_star_arr.tolist())),
    'product_eps_ts':  dict(zip([str(t) for t in ts_arr.tolist()],
                                 product_arr.tolist())),
    'powerlaw_fit': {'C': C_fit, 'x': x_fit} if len(ts_arr) >= 2 else None,
}, indent=2))
print(f"Saved -> {out}")
