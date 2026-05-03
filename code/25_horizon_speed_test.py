"""
Front propagation speed test for the cosmological horizon problem.

Measure how fast the cosmogenetic cascade front propagates through
the DDD substrate, and as a function of what:
  - constant in time (linear FKPP-type front)?
  - dependent on local T^2 at the front?
  - dependent on the global T^2 background?

We run two independent experiments:

  Exp A -- Pulse on background:
    Initialize R_i = R_bg uniform; perturb by adding eps at the
    central node. Watch the perturbation spread. Measure the front
    speed v_pulse(R_bg) for several R_bg values.
    -> Tests whether front speed depends on background R level.
    -> Linear DDD (Paper I): v should be constant in R_bg.

  Exp B -- Cosmogenetic cascade:
    Run variant (E) cosmogenesis (seed + autocatalysis), and record
    r_front(t) = furthest node above R_trap as function of time.
    Compute v(t) = dr_front/dt.
    -> If v(t) varies, identify the dependence on local T^2 or R.

Output: figures/fig_horizon_speed.{pdf,png}
        data/25_horizon_speed.json

The relevance to the horizon problem (Paper XII):
  At z = 1100 (CMB), standard cosmology requires causal contact
  across ~93 Gly today, while the standard light cone in 380,000 years
  is only ~1.2 Mly -- a factor of ~10^5. Solving this in DDD without
  inflation requires the effective propagation speed during the
  high-T^2 cosmogenetic epoch to exceed today's c by ~10^5. We test
  here whether the substrate's bare drainage rule provides any such
  enhancement, and if not, what additional ingredient (nonlinear
  mobility, coupled clock rate, etc.) would be needed.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from pathlib import Path
import json
import time as _time

np.random.seed(2024)

N_TARGET = 6000   # smaller for speed; lattice density unchanged
L_BOX = 50.0; R_LINK = 2.5; D_MIN = 1.0; DIM = 3
SIGMA = 3.90; T_S = 15.0; ALPHA = 0.4
KAPPA_TRAP = 8e-4; R_TRAP = 0.05
DT = 5e-3
EPS_E = 0.65; SEED_AMP = 0.22

HERE = Path(__file__).resolve().parent.parent
FIG = HERE / "figures"; FIG.mkdir(exist_ok=True)
DATA = HERE / "data";   DATA.mkdir(exist_ok=True)

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
                        if np.linalg.norm(p - c) < D_MIN: ok = False; break
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
idx_center = int(np.argmin(r_node))
tree = cKDTree(positions)
pairs = tree.query_pairs(r=R_LINK, output_type='ndarray')
i_arr = pairs[:,0]; j_arr = pairs[:,1]
n_links = len(pairs)
i_full = np.concatenate([i_arr, j_arr])
j_full = np.concatenate([j_arr, i_arr])
W = csr_matrix((np.ones(2*n_links), (i_full, j_full)), shape=(N, N))
D_diag = np.array(W.sum(axis=1)).flatten()
D_avg = float(D_diag.mean())


# =============================================================
# Exp A: pulse on uniform background, scan R_bg
# =============================================================
print("\n=== Exp A: pulse on uniform background ===")
# For each R_bg, set R = R_bg uniform + small perturbation at center
# Measure how fast the perturbation front spreads
# Front criterion: r_front = max distance from center where |dR| > thr

R_BG_VALUES = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
PULSE_AMP   = 1.0   # added at center on top of R_bg
T_PULSE_MAX = 8.0
N_STEPS_PULSE = int(T_PULSE_MAX / DT)
SAMPLE_EVERY_PULSE = 80

results_A = {}
for R_BG in R_BG_VALUES:
    R = np.full(N, R_BG, dtype=float)
    R[idx_center] += PULSE_AMP   # initial perturbation
    R0_node = R.copy()           # save initial state to detect perturbation
    front_radius = []; times = []
    t0 = _time.time()
    for step in range(N_STEPS_PULSE):
        t_now = step * DT
        # NO source: pure drainage spreading the perturbation
        R_diff_ij = R[j_arr] - R[i_arr]
        dR = np.zeros(N)
        np.add.at(dR, i_arr,  R_diff_ij)
        np.add.at(dR, j_arr, -R_diff_ij)
        R = R + DT * ALPHA * dR / max(D_avg, 1)
        R = np.maximum(R, 0.0)
        if step % SAMPLE_EVERY_PULSE == 0:
            # Front = furthest node where |R - R_BG| > 1% of pulse
            perturbation = np.abs(R - R_BG)
            mask = perturbation > 0.01 * PULSE_AMP
            if mask.any():
                r_max = float(r_node[mask].max())
            else:
                r_max = 0.0
            front_radius.append(r_max); times.append(t_now)
    times = np.array(times); front_radius = np.array(front_radius)
    # Fit r_front = c * t^p (log-log linear fit)
    valid = (times > 0.5) & (front_radius > 0)
    if valid.sum() >= 3:
        log_t = np.log(times[valid]); log_r = np.log(front_radius[valid])
        slope, intercept = np.polyfit(log_t, log_r, 1)
        # Extract effective speed at t = mid simulation
        r_at_mid = np.interp(T_PULSE_MAX/2, times, front_radius)
        v_eff = r_at_mid / (T_PULSE_MAX/2)  # average speed over half-time
    else:
        slope, intercept, v_eff = np.nan, np.nan, np.nan
    print(f"  R_BG = {R_BG:8.4g}  fit r ~ t^{slope:.3f}  v_eff(t={T_PULSE_MAX/2}) = {v_eff:.4f}  ({_time.time()-t0:.1f}s)")
    results_A[R_BG] = {'times': times.tolist(), 'r_front': front_radius.tolist(),
                       'slope_powerlaw': float(slope) if np.isfinite(slope) else None,
                       'v_eff': float(v_eff) if np.isfinite(v_eff) else None}


# =============================================================
# Exp B: cosmogenetic cascade with autocatalysis (variant E)
# =============================================================
print("\n=== Exp B: cosmogenetic cascade front ===")
T_MAX_B = 30.0
N_STEPS_B = int(T_MAX_B / DT)
SAMPLE_EVERY_B = 10  # dense sampling

R = SEED_AMP * src_amp
T = np.zeros(N); I = np.zeros(N)
hist_t, hist_T2, hist_rfront, hist_Rmax = [], [], [], []
t0 = _time.time()
for step in range(N_STEPS_B):
    t_now = step * DT
    if t_now < T_S:
        active = R > R_TRAP
        R[active] = R[active] + DT * EPS_E * R[active]
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
    if step % SAMPLE_EVERY_B == 0:
        # Front = furthest active node (R > R_trap)
        mask = R > R_TRAP
        r_front = float(r_node[mask].max()) if mask.any() else 0.0
        hist_t.append(t_now)
        hist_T2.append(float((T**2).sum()))
        hist_rfront.append(r_front)
        hist_Rmax.append(float(R.max()))
print(f"  done in {_time.time()-t0:.1f}s")

t_b = np.array(hist_t)
T2_b = np.array(hist_T2)
rfront_b = np.array(hist_rfront)
Rmax_b = np.array(hist_Rmax)

# Compute instantaneous front speed v(t) = dr_front/dt
v_b = np.gradient(rfront_b, t_b)
# Smooth using window of 5
def smooth_box(x, w=5):
    return np.convolve(x, np.ones(w)/w, mode='same')
v_b_sm = smooth_box(v_b, 7)

# Print front speed at several characteristic times
for t_target in [1.0, 5.0, 10.0, 14.5, 16.0, 22.0]:
    idx = int(np.argmin(np.abs(t_b - t_target)))
    print(f"  t = {t_target:5.1f}  r_front = {rfront_b[idx]:6.2f}  "
          f"v_inst ~ {v_b_sm[idx]:+.4f}  T^2 = {T2_b[idx]:.3e}  "
          f"R_max = {Rmax_b[idx]:.3e}")

# Save numerical
out = DATA / '25_horizon_speed.json'
out.write_text(json.dumps({
    'lattice': {'N': N, 'D_avg': D_avg},
    'exp_A_R_BG_values': R_BG_VALUES,
    'exp_A_results': {str(k): v for k,v in results_A.items()},
    'exp_B_t': t_b.tolist(),
    'exp_B_T2': T2_b.tolist(),
    'exp_B_rfront': rfront_b.tolist(),
    'exp_B_v_inst_smoothed': v_b_sm.tolist(),
    'exp_B_Rmax': Rmax_b.tolist(),
}, indent=2))
print(f"\nSaved -> {out}")

# =============================================================
# Plot
# =============================================================
fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.0))
ax_pulse = axes[0,0]
ax_v_R   = axes[0,1]
ax_cascade = axes[1,0]
ax_v_T2  = axes[1,1]

# Panel A1: pulse propagation r(t) for various R_BG
import matplotlib.cm as cm
colors = cm.viridis(np.linspace(0.05, 0.95, len(R_BG_VALUES)))
for i, R_BG in enumerate(R_BG_VALUES):
    res = results_A[R_BG]
    ax_pulse.plot(res['times'], res['r_front'], color=colors[i], lw=1.5,
                  label=fr'$R_{{\rm bg}} = {R_BG}$')
ax_pulse.set_xlabel(r'time $t$', fontsize=11)
ax_pulse.set_ylabel(r'$r_{\rm front}$ (perturbation)', fontsize=11)
ax_pulse.set_title('(A) Pulse propagation on uniform background',
                   fontsize=11, fontweight='bold')
ax_pulse.legend(loc='lower right', fontsize=8)
ax_pulse.grid(alpha=0.3)

# Panel A2: effective speed vs R_BG
v_effs = [results_A[R]['v_eff'] for R in R_BG_VALUES]
ax_v_R.plot(R_BG_VALUES, v_effs, marker='*', markersize=14,
            markerfacecolor='gold', markeredgecolor='black',
            linestyle='-', lw=1.5)
ax_v_R.set_xscale('log'); ax_v_R.set_xlabel(r'$R_{\rm bg}$', fontsize=11)
ax_v_R.set_ylabel(r'$v_{\rm eff}$ (lattice units / lattice tick)', fontsize=11)
ax_v_R.set_title('(B) Effective front speed vs background $R$',
                 fontsize=11, fontweight='bold')
ax_v_R.grid(alpha=0.3)

# Panel B1: cascade front r_front(t) and T^2(t)
ax_cascade.plot(t_b, rfront_b, 'b-', lw=1.8, label=r'$r_{\rm front}(t)$ (active region)')
ax_cascade.set_xlabel(r'time $t$', fontsize=11)
ax_cascade.set_ylabel(r'$r_{\rm front}$', fontsize=11, color='blue')
ax_cascade.tick_params(axis='y', labelcolor='blue')
ax_cascade.axvline(T_S, color='cyan', linestyle='--', lw=1, alpha=0.6,
                   label=f'$t_s = {T_S}$')
ax_cascade.set_title('(C) Cascade front propagation (variant E)',
                     fontsize=11, fontweight='bold')
ax_cascade.legend(loc='upper left', fontsize=9)
ax_cascade.grid(alpha=0.3)
ax_cascade2 = ax_cascade.twinx()
ax_cascade2.plot(t_b, T2_b, 'r:', lw=1.5, label=r'$\sum T^2(t)$')
ax_cascade2.set_ylabel(r'$\sum T^2$', fontsize=11, color='red')
ax_cascade2.set_yscale('log')
ax_cascade2.tick_params(axis='y', labelcolor='red')

# Panel B2: instantaneous front speed vs T^2(t)
mask_pos = (v_b_sm > 0.001) & (t_b > 0.5)
ax_v_T2.scatter(T2_b[mask_pos], v_b_sm[mask_pos], c=t_b[mask_pos],
                cmap='plasma', s=10)
cbar = plt.colorbar(ax_v_T2.collections[0], ax=ax_v_T2, label='time t')
ax_v_T2.set_xlabel(r'$\sum T^2(t)$', fontsize=11)
ax_v_T2.set_ylabel(r'$v_{\rm front}(t) = dr/dt$', fontsize=11)
ax_v_T2.set_xscale('log')
ax_v_T2.set_title(r'(D) Front speed vs $\sum T^2$ during cascade',
                  fontsize=11, fontweight='bold')
ax_v_T2.grid(alpha=0.3, which='both')

fig.suptitle(r'\textbf{Front propagation speed vs substrate state:} '
             r'is there a $c_{\rm eff}(T^2)$ dependence?',
             fontsize=12, y=1.005)
fig.tight_layout()
fig.savefig(FIG / 'fig_horizon_speed.pdf', bbox_inches='tight')
fig.savefig(FIG / 'fig_horizon_speed.png', dpi=150, bbox_inches='tight')
print(f"Saved -> {FIG/'fig_horizon_speed.pdf'}")
