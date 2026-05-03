"""
Branching origin of gamma --- substrate-only test.

Tests the hypothesis: can the DDD substrate alone (without explicit
time-dependent source) produce the cosmogenetic phantom-quintessence
signature, via a branching/cascade mechanism that yields exponential
growth of N_active(t)?

Three configurations are compared:

  (A) STANDARD COSMOGENETIC SOURCE  (control = headline run)
      R(0) = 0,  S(t) = S_0 exp(gamma t)  for t < t_s
      Expected: m_inf ≈ -3.72,  m_0 ≈ +1.65   (DESI-matching)

  (B) IMPULSE INITIAL CONDITION, NO SOURCE  (test of substrate alone)
      R(0) = R_init * Gaussian(r/sigma) with the same total integrated
      energy as the exponential cumulative source ; no source in time.
      Pure conservative drainage thereafter.
      Question: does <T^2>(t) still grow then fall in a phantom-like
      pattern? Or does it just decay monotonically?

  (C) SUSTAINED CONSTANT SOURCE                (intermediate test)
      R(0) = 0,  S(t) = S_const for t < t_s   (already in profile scan)
      For comparison.

For each we compute and report:
  - <T^2>(t) trajectory
  - N_active(t) = number of nodes with R_i > R_TRAP
  - log-fit of N_active(t) in early phase: is it exponential? at what beta?
  - m_eff = -d ln <T^2> / d ln t
  - extracted m_inf, m_0 (matching 04_robustness.py extraction)

Output:
  figures/fig_branching_test.{pdf,png}
  data/13_branching_test.json
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

# Parameters (match 04_robustness.py)
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

# Exponential reference
S0_EXP = 0.05
GAMMA  = 0.415
E_TOT  = S0_EXP * (np.exp(GAMMA * T_S) - 1) / GAMMA

HERE = Path(__file__).resolve().parent.parent
FIG  = HERE / "figures"; FIG.mkdir(exist_ok=True)
DATA = HERE / "data";    DATA.mkdir(exist_ok=True)

# =============================================================
# Build lattice (shared)
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

# Calibrate R_init for config B so that initial energy matches E_TOT
# (mathematically, ∫ R_init * Gauss = R_init * π^(3/2) σ^3 ≈
#  R_init * 5.57 * 59.3 ≈ 330 * R_init for sigma=3.9)
gauss_integral = float(np.sum(src_amp))   # nodes within Gaussian
R_INIT = E_TOT / gauss_integral if gauss_integral > 0 else 0
print(f"  Lattice: D_avg = {D_avg:.2f}, sigma node integral = {gauss_integral:.1f}")
print(f"  Reference E_tot = {E_TOT:.4f}")
print(f"  Config B: R_init = {R_INIT:.4f} (calibrated to match E_tot)")

log_times = np.unique(np.concatenate([
    np.linspace(0.02, 1, 50),
    np.linspace(1, 15, 150),
    np.linspace(15, 30, 60),
    np.linspace(30, T_MAX, 40)
]))


def run_config(name, source_fn, R0_fn):
    """Run cascade with given initial condition and source function.
    source_fn(t) returns scalar (multiplied by src_amp at each node, for t < T_S).
    R0_fn(src_amp) returns initial R array.
    Returns dict with t, T2, N_active, total_R."""
    t0 = _time.time()
    R = R0_fn(src_amp)
    T = np.zeros(N); I = np.zeros(N)
    hist_t = []; hist_T2 = []; hist_Nact = []; hist_Rtot = []
    log_idx = 0
    for step in range(N_STEPS):
        t_now = step * DT
        if t_now < T_S and source_fn is not None:
            R = R + DT * source_fn(t_now) * src_amp
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
            hist_Nact.append(int((R > R_TRAP).sum()))
            hist_Rtot.append(float(R.sum()))
            log_idx += 1
    print(f"  {name:30s} done in {_time.time()-t0:.1f}s")
    return {'name': name,
            't':       np.array(hist_t),
            'T2':      np.array(hist_T2),
            'N_act':   np.array(hist_Nact),
            'R_tot':   np.array(hist_Rtot)}


def extract(t_arr, T2_arr):
    valid = (t_arr > 0.05) & (T2_arr > 1e-12)
    t_v = t_arr[valid]; T2_v = T2_arr[valid]
    if len(T2_v) < 21:
        return None, None, None, None, None
    win = min(21, len(T2_v) // 2 * 2 - 1)
    ln_T2_smooth = savgol_filter(np.log(T2_v), win, 3)
    m_eff = -np.gradient(ln_T2_smooth, np.log(t_v))
    t_peak = float(t_v[np.argmax(T2_v)])
    pre  = t_v < t_peak * 0.7
    post = t_v > t_peak * 3.0
    m_pre  = float(np.median(m_eff[pre])) if pre.sum() > 3 else None
    m_post = float(np.median(m_eff[post])) if post.sum() > 5 else None
    return t_v, m_eff, t_peak, m_pre, m_post


# =============================================================
# Run three configurations
# =============================================================
print(f"\nIntegrating three configurations for T_MAX={T_MAX}...")
configs = {
    'A_exponential_source': run_config(
        '(A) exp source, R(0)=0',
        source_fn = lambda t: S0_EXP * np.exp(GAMMA * t),
        R0_fn     = lambda amp: np.zeros(N)),
    'B_impulse_only':       run_config(
        '(B) R(0)=Gauss, no source',
        source_fn = None,
        R0_fn     = lambda amp: R_INIT * amp),
    'C_constant_source':    run_config(
        '(C) constant source, R(0)=0',
        source_fn = lambda t: E_TOT / T_S,
        R0_fn     = lambda amp: np.zeros(N)),
}

print("\n--- Extracted m_eff signature per config ---")
for key, c in configs.items():
    t_v, m_eff, t_peak, m_pre, m_post = extract(c['t'], c['T2'])
    c['t_peak']  = t_peak
    c['t_v']     = t_v
    c['m_eff']   = m_eff
    c['m_pre']   = m_pre
    c['m_post']  = m_post
    if m_pre is not None and m_post is not None:
        print(f"  {c['name']:32s}  t_peak={t_peak:5.2f}  "
              f"m_pre={m_pre:+.3f}  m_post={m_post:+.3f}")
    else:
        print(f"  {c['name']:32s}  insufficient data for extraction")

# =============================================================
# Fit N_active(t) to test for exponential growth
# =============================================================
print("\n--- N_active(t) growth test ---")
beta_fits = {}
for key, c in configs.items():
    t_arr = c['t']; N_arr = c['N_act']
    # Fit log(N_active) vs t in early phase where N grows
    grow_mask = (t_arr > 0.5) & (t_arr < min(T_S, c['t_peak'] - 1)) & (N_arr > 5)
    if grow_mask.sum() < 10:
        print(f"  {c['name']:32s}  N grows too little to fit"); continue
    log_N = np.log(N_arr[grow_mask])
    t_fit = t_arr[grow_mask]
    # Linear fit log_N = log_N0 + beta * t
    coef = np.polyfit(t_fit, log_N, 1)
    beta = float(coef[0]); logN0 = float(coef[1])
    # Residual to gauge fit quality (R^2)
    pred = beta * t_fit + logN0
    ss_res = float(((log_N - pred)**2).sum())
    ss_tot = float(((log_N - log_N.mean())**2).sum())
    r2 = 1 - ss_res / max(ss_tot, 1e-30)
    beta_fits[key] = (beta, logN0, r2, t_fit.min(), t_fit.max())
    print(f"  {c['name']:32s}  log-fit β = {beta:+.3f}  "
          f"(window t∈[{t_fit.min():.1f},{t_fit.max():.1f}], R²={r2:.3f})")

# =============================================================
# Plot: 4-panel comparison
# =============================================================
fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.0))
ax_T2 = axes[0,0]
ax_N  = axes[0,1]
ax_m  = axes[1,0]
ax_t  = axes[1,1]

colors = {'A_exponential_source': '#E41A1C',
          'B_impulse_only':       '#377EB8',
          'C_constant_source':    '#4DAF4A'}

# Panel A: <T^2>(t)
for key, c in configs.items():
    ax_T2.plot(c['t'], np.maximum(c['T2'], 1e-10), color=colors[key],
               lw=1.6, label=c['name'])
ax_T2.axvline(T_S, color='cyan', linestyle='--', lw=1, alpha=0.6)
ax_T2.set_xlabel('time $t$', fontsize=11)
ax_T2.set_ylabel(r'$\sum T^2(t)$', fontsize=11)
ax_T2.set_yscale('log')
ax_T2.set_title(r'(A) Cascade trajectory $\sum T^2(t)$', fontsize=11, fontweight='bold')
ax_T2.legend(loc='lower center', fontsize=9, framealpha=0.9)
ax_T2.grid(alpha=0.3, which='both')

# Panel B: N_active(t)
for key, c in configs.items():
    ax_N.plot(c['t'], np.maximum(c['N_act'], 1), color=colors[key],
              lw=1.6, label=c['name'])
    if key in beta_fits:
        beta, logN0, r2, tmin, tmax = beta_fits[key]
        t_line = np.linspace(tmin, tmax, 50)
        N_line = np.exp(beta * t_line + logN0)
        ax_N.plot(t_line, N_line, color=colors[key], linestyle='--', lw=1.0,
                  alpha=0.6, label=fr'  fit: $\beta={beta:+.2f}$, $R^2={r2:.2f}$')
ax_N.axvline(T_S, color='cyan', linestyle='--', lw=1, alpha=0.6)
ax_N.set_xlabel('time $t$', fontsize=11)
ax_N.set_ylabel(r'$N_{\rm active}(t)$', fontsize=11)
ax_N.set_yscale('log')
ax_N.set_title(r'(B) Active node count + exponential fits',
               fontsize=11, fontweight='bold')
ax_N.legend(loc='lower right', fontsize=8, framealpha=0.9)
ax_N.grid(alpha=0.3, which='both')

# Panel C: m_eff(t)
ax_m.axhline(0, color='red', linestyle='--', lw=1.0)
ax_m.fill_between([0, T_MAX], -6, 0, color='blue', alpha=0.07)
ax_m.fill_between([0, T_MAX], 0, 5, color='orange', alpha=0.07)
ax_m.text(2, -5, 'phantom\n($m<0$)', fontsize=9, color='blue', alpha=0.85)
ax_m.text(55, 4, 'quintessence\n($m>0$)', fontsize=9, color='darkorange', alpha=0.85)
for key, c in configs.items():
    if c['t_v'] is not None:
        ax_m.plot(c['t_v'], c['m_eff'], color=colors[key], lw=1.6, label=c['name'])
ax_m.axhline(-3.72, color='black', linestyle=':', lw=1, alpha=0.6)
ax_m.text(75, -3.72, '$m_\\infty^{\\rm DESI}$', fontsize=8, color='black', va='center', ha='right')
ax_m.axhline(+1.65, color='black', linestyle=':', lw=1, alpha=0.6)
ax_m.text(75, +1.65, '$m_0^{\\rm DESI}$', fontsize=8, color='black', va='center', ha='right')
ax_m.set_xlabel('time $t$', fontsize=11)
ax_m.set_ylabel(r'$m_{\rm eff}(t) = -d\,\ln\sum T^2/d\,\ln t$', fontsize=11)
ax_m.set_xlim(0, T_MAX); ax_m.set_ylim(-6, 5)
ax_m.set_title(r'(C) Effective dilution exponent',
               fontsize=11, fontweight='bold')
ax_m.legend(loc='upper right', fontsize=8, framealpha=0.9)
ax_m.grid(alpha=0.3)

# Panel D: summary table
ax_t.axis('off')
def fmt(v, sign=True):
    if v is None or not np.isfinite(v): return 'N/A'
    return (f'${v:+.2f}$' if sign else f'${v:.2f}$')
def fmt_int(v):
    if v is None: return 'N/A'
    return f'${int(v)}$'
rows = [['Config', '$m_\\infty$', '$m_0$', r'$\beta_{\rm fit}$', r'$R^2$']]
for key, c in configs.items():
    if key in beta_fits:
        beta, logN0, r2, _, _ = beta_fits[key]
    else:
        beta, r2 = None, None
    rows.append([c['name'].split(')')[0]+')',
                 fmt(c['m_pre']),
                 fmt(c['m_post']),
                 fmt(beta), fmt(r2, sign=False)])
rows.append(['DESI DR2', '$-3.72$', '$+1.65$', '---', '---'])
rows.append([r'$\gamma$ ref (input)', '---', '---', f'${GAMMA:+.3f}$', '---'])

t_obj = ax_t.table(cellText=rows[1:], colLabels=rows[0],
                   loc='center', cellLoc='center')
t_obj.auto_set_font_size(False)
t_obj.set_fontsize(10); t_obj.scale(1.0, 1.7)
for j in range(len(rows[0])):
    t_obj[(0, j)].set_facecolor('#cfe2f3')
    t_obj[(0, j)].set_text_props(weight='bold')
# Highlight DESI row (penultimate) and gamma ref row (last)
for j in range(len(rows[0])):
    t_obj[(len(rows)-2, j)].set_facecolor('#ffd966')
    t_obj[(len(rows)-2, j)].set_text_props(weight='bold')
    t_obj[(len(rows)-1, j)].set_facecolor('#d9e1f2')
    t_obj[(len(rows)-1, j)].set_text_props(style='italic')
ax_t.set_title('(D) Summary --- does substrate alone reproduce DESI?',
               fontsize=11, fontweight='bold', y=0.95)

fig.suptitle(r'\textbf{Branching test:} can the substrate alone (no time-explicit source)'
             r' produce the cosmogenetic DESI signature?',
             fontsize=12.5, y=1.005)
fig.tight_layout()
fig.savefig(FIG / 'fig_branching_test.pdf', bbox_inches='tight')
fig.savefig(FIG / 'fig_branching_test.png', dpi=150, bbox_inches='tight')
print(f"\nSaved -> {FIG/'fig_branching_test.pdf'}")

# Save numerical results
out = DATA / '13_branching_test.json'
out.write_text(json.dumps({
    'parameters': {
        'N': N, 'D_avg': D_avg, 'T_S': T_S, 'GAMMA_ref': GAMMA,
        'E_TOT_ref': E_TOT, 'R_INIT_for_B': R_INIT,
    },
    'configs': {key: {
        'name': c['name'],
        't_peak': c['t_peak'],
        'm_pre':  c['m_pre'],
        'm_post': c['m_post'],
        'beta_fit': beta_fits.get(key, (None,None,None,None,None))[0],
        'beta_R2':  beta_fits.get(key, (None,None,None,None,None))[2],
    } for key, c in configs.items()},
    'desi_target_m_inf': -3.72,
    'desi_target_m_0':    1.65,
}, indent=2))
print(f"Saved -> {out}")
