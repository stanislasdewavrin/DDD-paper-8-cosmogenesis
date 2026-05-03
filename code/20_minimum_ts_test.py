"""
Minimum t_s test: how short can the cosmogenetic epoch be while
still reproducing the DESI phantom signature m_inf = -3.72?

For each t_s in {5, 8, 10, 12, 15, 20}, scan ε at three values around
the previously measured calibration (ε ~ 9.75/t_s) and try to find
ε* such that m_pre = -3.72.

If the minimum bracketed m_pre value (most negative) is greater than
-3.72 (less phantom than DESI), then no ε can reach DESI at this t_s
→ this t_s is BELOW the minimum.

Theoretical guides:
  - Diffusive propagation time over scale σ:
       τ_diff = σ²/(6D_eff) ≈ 24 (with α/D̄ = 0.107)
  - FKPP front speed in autocatalytic-diffusive system:
       v = 2√(ε·D_eff)  ≈ 0.53 lattice units/time
  - Time for cascade to propagate beyond source extent σ:
       t_min ≈ σ/v  ≈ 7.4

So we predict t_s ≈ 7-10 to be the lower bound.

Output: figures/fig_minimum_ts.{pdf,png}
        data/20_minimum_ts.json
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

DESI_M_INF = -3.72
TS_VALUES = [5.0, 8.0, 10.0, 12.0, 15.0, 20.0]

HERE = Path(__file__).resolve().parent.parent
FIG  = HERE / "figures"; FIG.mkdir(exist_ok=True)
DATA = HERE / "data";    DATA.mkdir(exist_ok=True)

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

D_eff = ALPHA / D_avg  # diffusion coefficient (per link)
print(f"  D_avg={D_avg:.2f}, D_eff = α/D_avg = {D_eff:.4f}")
print(f"  τ_diff = σ²/(6 D_eff) = {SIGMA**2/(6*D_eff):.1f}")
print(f"  Predicted FKPP front speed v = 2√(εD) (with ε≈0.65) = "
      f"{2*np.sqrt(0.65*D_eff):.3f}")
print(f"  Predicted minimum t_s ≈ σ/v = {SIGMA/(2*np.sqrt(0.65*D_eff)):.1f}")


def run_one(t_s, eps, t_max=None):
    if t_max is None:
        t_max = max(60.0, 4*t_s)
    n_steps = int(t_max / DT)
    log_times = np.unique(np.concatenate([
        np.linspace(0.02, 1, 50),
        np.linspace(1, 15, 150),
        np.linspace(15, 30, 60),
        np.linspace(30, t_max, 40)
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
    if len(T2_v) < 21: return None, None, None
    win = min(21, len(T2_v) // 2 * 2 - 1)
    ln_T2_smooth = savgol_filter(np.log(T2_v), win, 3)
    m_eff = -np.gradient(ln_T2_smooth, np.log(t_v))
    t_peak = float(t_v[np.argmax(T2_v)])
    pre = t_v < t_peak * 0.7
    post = t_v > t_peak * 3.0
    m_pre  = float(np.median(m_eff[pre]))  if pre.sum()  > 3 else None
    m_post = float(np.median(m_eff[post])) if post.sum() > 5 else None
    return m_pre, m_post, t_peak


# For each t_s, scan 3 eps values: {0.5, 1.0, 1.5} × baseline
print(f"\n=== Scan: 6 t_s × 3 eps = 18 runs ===")
print(f"  Baseline ε(t_s) ≈ 9.75/t_s\n")
results = {}
t_scan = _time.time()
for t_s in TS_VALUES:
    eps_base = 9.75 / t_s
    eps_scan = [0.7 * eps_base, 1.0 * eps_base, 1.5 * eps_base]
    runs = []
    for eps in eps_scan:
        t0 = _time.time()
        m_pre, m_post, t_peak = run_one(t_s=t_s, eps=eps)
        elapsed = _time.time() - t0
        runs.append({'eps': float(eps), 'm_pre': m_pre,
                     'm_post': m_post, 't_peak': t_peak})
        print(f"  t_s={t_s:5.1f}  ε={eps:.4f}  "
              f"m_pre={m_pre:+.3f}  m_post={m_post:+.3f}  "
              f"t_peak={t_peak:5.2f}  ({elapsed:.1f}s)")
    results[t_s] = runs
print(f"\nScan total: {_time.time()-t_scan:.1f}s")

# Analyse: for each t_s, find min(m_pre) reached and compare to DESI -3.72
print(f"\n=== Can each t_s reach m_pre = {DESI_M_INF}? ===")
ts_arr = []; mpre_min = []; mpre_at_baseline = []; viability = []
eps_star_arr = []
for t_s, runs in results.items():
    mp_arr  = np.array([r['m_pre'] for r in runs])
    eps_arr = np.array([r['eps']   for r in runs])
    mp_min  = float(mp_arr.min())  # most negative reached
    mp_base = float(mp_arr[1])     # at eps_baseline (1.0×)
    mpre_min.append(mp_min)
    mpre_at_baseline.append(mp_base)
    ts_arr.append(t_s)
    if mp_arr.min() <= DESI_M_INF <= mp_arr.max():
        order = np.argsort(mp_arr)
        eps_star = float(np.interp(DESI_M_INF, mp_arr[order], eps_arr[order]))
        viability.append('viable')
        eps_star_arr.append(eps_star)
        print(f"  t_s={t_s:5.1f}  reaches DESI: ε* = {eps_star:.3f}  "
              f"(m_pre range [{mp_arr.min():+.2f}, {mp_arr.max():+.2f}])")
    elif mp_min > DESI_M_INF:
        viability.append('TOO SHORT')
        eps_star_arr.append(None)
        print(f"  t_s={t_s:5.1f}  CANNOT reach DESI: best m_pre={mp_min:+.2f} > {DESI_M_INF}  "
              f"→ t_s too short")
    else:
        viability.append('only with smaller ε')
        eps_star_arr.append(None)
        print(f"  t_s={t_s:5.1f}  baseline ε already too negative  "
              f"(m_pre min = {mp_min:+.2f})  → could match with smaller ε")

# Plot
fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.0))
ax_m, ax_e = axes

# Left: m_pre minimum (deepest phantom reachable) vs t_s
ts_a = np.array(ts_arr)
mp_a = np.array(mpre_min)
mb_a = np.array(mpre_at_baseline)
ax_m.axhline(DESI_M_INF, color='red', linestyle='--', lw=2,
             label=fr'DESI $m_\infty = {DESI_M_INF}$')
ax_m.plot(ts_a, mp_a, marker='o', markersize=10,
          markerfacecolor='gold', markeredgecolor='black', linestyle='-',
          label='deepest $m_{\\rm pre}$ reachable')
ax_m.plot(ts_a, mb_a, marker='s', markersize=8,
          markerfacecolor='lightblue', markeredgecolor='black', linestyle=':',
          label='at baseline ε = 9.75/t_s')
# Theoretical prediction for t_s_min
t_min_pred = SIGMA / (2*np.sqrt(0.65*D_eff))
ax_m.axvline(t_min_pred, color='green', linestyle=':', lw=2,
             label=fr'predicted $t_{{s,\min}} \approx \sigma/v = {t_min_pred:.1f}$')
ax_m.set_xlabel(r'$t_s$ (source phase duration)', fontsize=12)
ax_m.set_ylabel(r'$m_{\rm pre}$', fontsize=12)
ax_m.set_title(r'Deepest phantom reachable as function of $t_s$',
               fontsize=12, fontweight='bold')
ax_m.legend(loc='lower right', fontsize=10, framealpha=0.95)
ax_m.grid(alpha=0.3)
ax_m.set_ylim(min(-7, mp_a.min()*1.05), 0)

# Right: ε* vs t_s for viable t_s
viable_ts = np.array([ts_arr[i] for i in range(len(ts_arr)) if eps_star_arr[i] is not None])
viable_eps = np.array([eps_star_arr[i] for i in range(len(eps_star_arr)) if eps_star_arr[i] is not None])
ax_e.plot(viable_ts, viable_eps, marker='*', markersize=18,
          markerfacecolor='gold', markeredgecolor='black',
          markeredgewidth=1.2, linestyle='-', label=r'$\epsilon^*$ giving DESI')
ts_line = np.linspace(5, 25, 100)
ax_e.plot(ts_line, np.pi**2/ts_line, 'g--', lw=1.5, alpha=0.7,
          label=r'$\pi^2/t_s$')
ax_e.plot(ts_line, 9.75/ts_line, 'b:', lw=1.5, alpha=0.7,
          label=r'$9.75/t_s$ (baseline)')
ax_e.set_xlabel(r'$t_s$', fontsize=12)
ax_e.set_ylabel(r'$\epsilon^*$', fontsize=12)
ax_e.set_title(r'$\epsilon^*$ vs $t_s$ for DESI matching',
               fontsize=12, fontweight='bold')
ax_e.legend(loc='upper right', fontsize=10)
ax_e.grid(alpha=0.3)
ax_e.set_xlim(5, 25)

fig.suptitle(r'\textbf{Minimum $t_s$ test:} '
             r'how short can the cosmogenetic epoch be?',
             fontsize=12.5, y=1.02)
fig.tight_layout()
fig.savefig(FIG / 'fig_minimum_ts.pdf', bbox_inches='tight')
fig.savefig(FIG / 'fig_minimum_ts.png', dpi=150, bbox_inches='tight')
print(f"\nSaved -> {FIG/'fig_minimum_ts.pdf'}")

out = DATA / '20_minimum_ts.json'
out.write_text(json.dumps({
    'lattice': {'N': N, 'D_avg': D_avg},
    'D_eff_alpha_over_D_avg': D_eff,
    'predicted_t_s_min': t_min_pred,
    'desi_m_inf': DESI_M_INF,
    'scan': {str(k): v for k,v in results.items()},
    'mpre_min_per_ts': dict(zip([str(t) for t in ts_arr], mpre_min)),
    'eps_star_per_ts': {str(t): e for t,e in zip(ts_arr, eps_star_arr) if e is not None},
}, indent=2))
print(f"Saved -> {out}")
