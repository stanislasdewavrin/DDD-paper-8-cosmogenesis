"""
Phase heatmap of the DDD cosmogenetic cascade.

Produces a multi-panel figure showing the chronology of the cascade:

  Panel A: heatmap T^2(t, r_bin) — the cascade front sweeping outward
           through the lattice as the source pumps energy.

  Panel B: m_eff(t) on shared time axis, with phantom region (m<0)
           and quintessence region (m>0) shaded; m=0 line marks the
           phantom crossing w = -1.

  Panel C: w(z) = m/3 - 1 on a redshift axis (using the t<->a
           calibration of the paper), showing the three phases:
           phantom, crossing, quintessence; DESI DR2 best-fit overlaid.

Output:
  figures/fig_phase_heatmap.{pdf,png}
  data/07_phase_heatmap.npz
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.spatial import cKDTree
from scipy.signal import savgol_filter
from pathlib import Path
import os

np.random.seed(2024)

# ============================================================
# Parameters (matching headline run; smaller N for speed)
# ============================================================
N_TARGET = 4000  # half the headline lattice for ~10s runtime
L_BOX = 50.0
R_LINK = 2.5
D_MIN = 1.0
DIM = 3

S_0 = 0.05
GAMMA = 0.415
SIGMA = 3.90
T_S = 15.0
ALPHA_FILL = 0.4
KAPPA_TRAP = 8e-4
R_TRAP = 0.05

DT = 5e-3
T_MAX = 80.0
N_STEPS = int(T_MAX / DT)
SAMPLE_EVERY = 80   # save heatmap snapshot every N steps -> 200 frames

HERE = Path(__file__).resolve().parent.parent
FIG = HERE / "figures"; FIG.mkdir(exist_ok=True)
DATA = HERE / "data"; DATA.mkdir(exist_ok=True)

# ============================================================
# Build Poisson-disk lattice
# ============================================================
print(f"Building lattice N~{N_TARGET}...")
cell_size = D_MIN / np.sqrt(3)
grid = {}
positions_list = []
attempts = 0
max_attempts = N_TARGET * 30
while len(positions_list) < N_TARGET and attempts < max_attempts:
    candidate = np.random.uniform(0, L_BOX, size=DIM)
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
        grid.setdefault(key, []).append(candidate)
    attempts += 1
positions = np.array(positions_list)
N = len(positions)
print(f"  N = {N}")

# Centre and radial coordinates
center = np.array([L_BOX/2]*3)
r_node = np.linalg.norm(positions - center, axis=1)

# Adjacency as sparse matrix (vectorised path)
from scipy.sparse import csr_matrix
tree = cKDTree(positions)
pairs = tree.query_pairs(R_LINK, output_type='ndarray')
# Symmetrise
i_arr = pairs[:,0]
j_arr = pairs[:,1]
n_links = len(pairs)
edges_i = np.concatenate([i_arr, j_arr])
edges_j = np.concatenate([j_arr, i_arr])
adj = csr_matrix((np.ones(len(edges_i)), (edges_i, edges_j)), shape=(N, N))
deg = np.asarray(adj.sum(axis=1)).ravel()
mean_deg = float(deg.mean())
print(f"  mean degree = {mean_deg:.2f}")

# Source profile (centred)
source_amp = np.exp(-r_node**2 / SIGMA**2)

# Radial bins for heatmap
N_BINS = 30
r_edges = np.linspace(0, L_BOX/2, N_BINS+1)
r_centers = 0.5*(r_edges[:-1] + r_edges[1:])
bin_idx = np.digitize(r_node, r_edges) - 1
bin_idx = np.clip(bin_idx, 0, N_BINS-1)

# ============================================================
# Time integration
# ============================================================
R = np.zeros(N)
I2 = np.zeros(N)
heatmap = []   # list of (T2 binned by radius)
times = []
sum_T2 = []
sum_I2 = []

print(f"Integrating for T_MAX={T_MAX} ({N_STEPS} steps)...")
import time as _time
t0 = _time.time()
coef = ALPHA_FILL / mean_deg
# Pre-bin counts for radial averaging
bin_counts = np.bincount(bin_idx, minlength=N_BINS).astype(float)
bin_counts_safe = np.maximum(bin_counts, 1)

for step in range(N_STEPS):
    t = step * DT
    # Source phase
    if t < T_S:
        R += DT * S_0 * np.exp(GAMMA * t) * source_amp

    # Per-edge symmetric drainage (matches paper.tex eqn and 04_robustness.py exactly)
    R_diff_ij = R[j_arr] - R[i_arr]
    flux_in_i = np.maximum(R_diff_ij,  0.0)
    flux_in_j = np.maximum(-R_diff_ij, 0.0)
    # R update: dR_i = sum_{j~i} (R_j - R_i)
    dR_per_node = np.zeros(N)
    np.add.at(dR_per_node, i_arr,  R_diff_ij)
    np.add.at(dR_per_node, j_arr, -R_diff_ij)
    R += DT * coef * dR_per_node
    R = np.maximum(R, 0.0)
    # Correct T_i = (alpha/D_avg) * sum_{j~i} max(R_j - R_i, 0)
    T_inflow = np.zeros(N)
    np.add.at(T_inflow, i_arr, flux_in_i)
    np.add.at(T_inflow, j_arr, flux_in_j)
    T = coef * T_inflow

    # Self-trapping
    above = np.maximum(R - R_TRAP, 0.0)
    I2 += DT * KAPPA_TRAP * above * T**3

    if step % SAMPLE_EVERY == 0:
        T2 = T**2
        binned_sum = np.bincount(bin_idx, weights=T2, minlength=N_BINS)
        binned = binned_sum / bin_counts_safe
        heatmap.append(binned)
        times.append(t)
        sum_T2.append(float(T2.sum()))
        sum_I2.append(float(I2.sum()))
        if step % (SAMPLE_EVERY*10) == 0:
            print(f"  step={step:>6d} t={t:6.2f} sum_T2={sum_T2[-1]:.3e}")

print(f"  done in {_time.time()-t0:.1f}s")

heatmap = np.array(heatmap)   # (N_frames, N_BINS)
times = np.array(times)
sum_T2 = np.array(sum_T2)
sum_I2 = np.array(sum_I2)

# Compute m_eff(t) and w(z)
log_T2 = np.log(np.maximum(sum_T2, 1e-15))
log_t = np.log(np.maximum(times, 1e-3))
# Smooth derivative
win = max(7, 2*(len(log_T2)//30) + 1)
if len(log_T2) >= win:
    log_T2s = savgol_filter(log_T2, win, 3)
else:
    log_T2s = log_T2
m_eff = -np.gradient(log_T2s, log_t)
w_eff = m_eff/3 - 1

# t <-> a calibration: peak T2 -> a* = 0.69 (DESI DR2)
peak_idx = int(np.argmax(sum_T2))
t_peak = times[peak_idx]
A_STAR = 0.69
# Linear-in-t map: a = A_STAR * (t / t_peak)
a_arr = A_STAR * (times / t_peak)
z_arr = 1.0/np.maximum(a_arr, 1e-3) - 1

# ============================================================
# Plot
# ============================================================
fig = plt.figure(figsize=(12, 9))
gs = fig.add_gridspec(3, 1, height_ratios=[3, 1.2, 1.2], hspace=0.35)

# === Panel A: heatmap T^2(t, r) ===
ax1 = fig.add_subplot(gs[0])
hm = heatmap.T  # (N_BINS, N_frames)
hm_plot = np.where(hm > 0, hm, np.nan)
vmin = max(hm[hm > 0].min(), 1e-8) if (hm > 0).any() else 1e-8
vmax = hm.max()
im = ax1.imshow(hm_plot, aspect='auto', origin='lower',
                extent=[times.min(), times.max(),
                        r_edges[0], r_edges[-1]],
                norm=LogNorm(vmin=vmin, vmax=vmax),
                cmap='magma')
cbar = plt.colorbar(im, ax=ax1, fraction=0.04, pad=0.02)
cbar.set_label(r'$\langle T^2 \rangle$ per shell', fontsize=10)

ax1.axvline(T_S, color='cyan', linestyle='--', lw=1.5, alpha=0.8)
ax1.text(T_S+0.5, L_BOX/2*0.95, 'source ends\n$t_s=15$',
         color='cyan', fontsize=9, va='top')
ax1.axvline(t_peak, color='lime', linestyle=':', lw=1.5, alpha=0.8)
ax1.text(t_peak+0.5, L_BOX/2*0.7,
         f'peak $T^2$\n$t^*={t_peak:.1f}$\n$\\to a^*={A_STAR}$',
         color='lime', fontsize=9, va='top')
ax1.axhline(SIGMA, color='white', linestyle=':', lw=1.0, alpha=0.5)
ax1.text(0.5, SIGMA+0.5, f'source extent $\\sigma={SIGMA}$',
         color='white', fontsize=8)

ax1.set_ylabel('Radial distance from centre $r$', fontsize=11)
ax1.set_xlabel('simulation time $t$', fontsize=10)
ax1.set_title(r'(A) Cascade front: $\langle T^2 \rangle$ per radial shell vs simulation time',
              fontsize=12, fontweight='bold')

# === Panel B: m_eff(t) ===
ax2 = fig.add_subplot(gs[1], sharex=ax1)
ax2.plot(times, m_eff, 'b-', lw=1.6, label=r'$m_{\rm eff}(t)$')
ax2.axhline(0, color='k', lw=0.8, linestyle='--')
ax2.fill_between(times, 0, m_eff, where=(m_eff < 0),
                 color='blue', alpha=0.15, label=r'phantom ($w<-1$)')
ax2.fill_between(times, 0, m_eff, where=(m_eff > 0),
                 color='orange', alpha=0.15, label=r'quintessence ($w>-1$)')
ax2.axvline(T_S, color='cyan', linestyle='--', lw=1, alpha=0.6)
ax2.axvline(t_peak, color='lime', linestyle=':', lw=1, alpha=0.6)
ax2.set_ylabel(r'$m_{\rm eff}(t)$', fontsize=11)
ax2.legend(loc='upper right', fontsize=8, framealpha=0.9, ncol=3)
ax2.grid(alpha=0.3)
ax2.set_ylim(min(-5, m_eff.min()*1.1), max(3, m_eff.max()*1.1))

# === Panel C: w(z) ===
ax3 = fig.add_subplot(gs[2])
mask = (z_arr > 0) & (z_arr < 3) & np.isfinite(w_eff)
ax3.plot(z_arr[mask], w_eff[mask], 'b-', lw=1.6, label=r'DDD simulation')
ax3.axhline(-1, color='red', linestyle='--', lw=1.2,
            label=r'phantom divide $w=-1$')
ax3.fill_between(z_arr[mask], -2, w_eff[mask],
                 where=w_eff[mask] < -1,
                 color='blue', alpha=0.15)
ax3.fill_between(z_arr[mask], w_eff[mask], 1,
                 where=w_eff[mask] > -1,
                 color='orange', alpha=0.15)
# DESI DR2 reference (CPL with w0, wa)
w0_desi, wa_desi = -0.45, -1.79
a_grid = np.linspace(0.2, 1, 200)
z_grid = 1/a_grid - 1
w_desi = w0_desi + wa_desi*(1-a_grid)
ax3.plot(z_grid, w_desi, 'k--', lw=1.2, alpha=0.7,
         label='DESI DR2 CPL fit')
ax3.set_xlim(0, 2.0)
ax3.set_ylim(-2.5, 0.5)
ax3.set_xlabel(r'Redshift $z$  (via $a = 0.69\,t/t^*$)', fontsize=11)
ax3.set_ylabel(r'$w(z) = m/3 - 1$', fontsize=11)
ax3.legend(loc='lower right', fontsize=8, framealpha=0.9)
ax3.grid(alpha=0.3)

fig.suptitle(r'Cosmogenetic cascade: front, $m_{\rm eff}(t)$, and phantom crossing',
             fontsize=12, fontweight='bold', y=0.995)
fig.savefig(FIG / 'fig_phase_heatmap.pdf', bbox_inches='tight')
fig.savefig(FIG / 'fig_phase_heatmap.png', dpi=150, bbox_inches='tight')
print(f"Saved -> {FIG/'fig_phase_heatmap.pdf'}")

np.savez(DATA / '07_phase_heatmap.npz', times=times, r_centers=r_centers, heatmap=heatmap,
         sum_T2=sum_T2, sum_I2=sum_I2, m_eff=m_eff, w_eff=w_eff,
         z_arr=z_arr, t_peak=t_peak, T_S=T_S, A_STAR=A_STAR)
print(f"Saved -> {DATA/'07_phase_heatmap.npz'}")
