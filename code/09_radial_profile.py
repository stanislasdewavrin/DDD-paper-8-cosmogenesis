"""
Isotropic radial profile of the cosmogenetic cascade.

Replaces the noisy 2D slab projection with the physically meaningful
quantity: the spherically-averaged R(r, t).

Two panels:
  Panel A: R(r) at 8 snapshot times — coloured curves showing the
           cascade front propagating outward and settling.
  Panel B: heatmap R(r, t) on (radius, time) axes, smoother analogue
           of the radial-shell heatmap of fig 2 but on the *reserve*
           rather than on T^2 — shows the front linearly sweeping
           outward through the lattice.

Output: figures/fig_radial_profile.{pdf,png}
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from pathlib import Path
import time as _time

np.random.seed(2024)

# Same parameters as 07/08
N_TARGET = 4000
L_BOX = 50.0
R_LINK = 2.5
D_MIN = 1.0
DIM = 3
S_0 = 0.05
GAMMA = 0.415
SIGMA = 3.90
T_S = 15.0
ALPHA_FILL = 0.4
DT = 5e-3
T_MAX = 80.0
N_STEPS = int(T_MAX / DT)
SAMPLE_EVERY = 40

HERE = Path(__file__).resolve().parent.parent
FIG = HERE / "figures"; FIG.mkdir(exist_ok=True)

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

center = np.array([L_BOX/2]*3)
r_node = np.linalg.norm(positions - center, axis=1)

# Adjacency
tree = cKDTree(positions)
pairs = tree.query_pairs(R_LINK, output_type='ndarray')
edges_i = np.concatenate([pairs[:,0], pairs[:,1]])
edges_j = np.concatenate([pairs[:,1], pairs[:,0]])
adj = csr_matrix((np.ones(len(edges_i)), (edges_i, edges_j)), shape=(N, N))
deg = np.asarray(adj.sum(axis=1)).ravel()
mean_deg = float(deg.mean())

source_amp = np.exp(-r_node**2 / SIGMA**2)
coef = ALPHA_FILL / mean_deg

# Radial bins (finer than fig 2)
N_BINS = 40
r_edges = np.linspace(0, L_BOX/2, N_BINS+1)
r_centers = 0.5*(r_edges[:-1] + r_edges[1:])
bin_idx = np.digitize(r_node, r_edges) - 1
bin_idx = np.clip(bin_idx, 0, N_BINS-1)
bin_counts = np.bincount(bin_idx, minlength=N_BINS).astype(float)
bin_counts_safe = np.maximum(bin_counts, 1)

SNAPSHOT_TIMES = [3.0, 7.0, 11.0, 15.0, 18.0, 25.0, 40.0, 70.0]
snap_indices = sorted({int(round(t/DT)) for t in SNAPSHOT_TIMES})

# Time integration with full radial heatmap recording
R = np.zeros(N)
heatmap = []   # R radial profile every SAMPLE_EVERY steps
times_full = []
snap_curves = {}  # exact snapshot times

print(f"Integrating for T_MAX={T_MAX}...")
t0 = _time.time()
for step in range(N_STEPS):
    t = step * DT
    if t < T_S:
        R += DT * S_0 * np.exp(GAMMA * t) * source_amp
    sumR_neigh = adj @ R
    flux_net = coef * (sumR_neigh - deg * R)
    R += DT * flux_net
    R = np.maximum(R, 0.0)

    if step % SAMPLE_EVERY == 0:
        binned_sum = np.bincount(bin_idx, weights=R, minlength=N_BINS)
        binned = binned_sum / bin_counts_safe
        heatmap.append(binned)
        times_full.append(t)

    if step in snap_indices:
        binned_sum = np.bincount(bin_idx, weights=R, minlength=N_BINS)
        snap_curves[t] = binned_sum / bin_counts_safe

print(f"  done in {_time.time()-t0:.1f}s")

heatmap = np.array(heatmap)
times_full = np.array(times_full)

# ============================================================
# Plot — 2 panels
# ============================================================
fig = plt.figure(figsize=(13, 5.5))
gs = fig.add_gridspec(1, 2, width_ratios=[1.1, 1.4], wspace=0.28)

# === Panel A: R(r) at 8 snapshot times ===
ax1 = fig.add_subplot(gs[0])
times_sorted = sorted(snap_curves.keys())
cmap = plt.cm.plasma
colors = cmap(np.linspace(0.05, 0.95, len(times_sorted)))
for i, t in enumerate(times_sorted):
    R_r = snap_curves[t]
    if t < T_S:
        phase = 'source'
    elif t < T_S + 5:
        phase = 'cross.'
    else:
        phase = 'dilut.'
    ax1.plot(r_centers, np.maximum(R_r, 1e-6), '-', lw=2.0,
             color=colors[i],
             label=fr'$t={t:.0f}$  ({phase})')
ax1.axvline(SIGMA, color='cyan', linestyle='--', lw=1.0, alpha=0.8)
ax1.text(SIGMA + 0.5, 1e-5, r'source extent $\sigma$',
         color='cyan', fontsize=8, rotation=90, va='bottom')
ax1.set_xlabel(r'radius $r$ from $\Omega$', fontsize=11)
ax1.set_ylabel(r'spherically-averaged reserve $\langle R(r)\rangle$',
               fontsize=11)
ax1.set_yscale('log')
ax1.set_xlim(0, L_BOX/2)
ax1.set_ylim(1e-5, 60)
ax1.legend(loc='lower left', fontsize=8, ncol=2, framealpha=0.95)
ax1.grid(alpha=0.3, which='both')
ax1.set_title(r'(A) Radial profile $\langle R(r)\rangle$ at 8 snapshot times',
              fontsize=12, fontweight='bold')

# === Panel B: heatmap R(r, t) ===
ax2 = fig.add_subplot(gs[1])
hm = heatmap.T  # (N_BINS, N_frames)
hm_plot = np.maximum(hm, 1e-5)
im = ax2.imshow(hm_plot, aspect='auto', origin='lower',
                extent=[times_full.min(), times_full.max(),
                        r_edges[0], r_edges[-1]],
                norm=LogNorm(vmin=1e-3, vmax=hm.max()),
                cmap='magma',
                interpolation='bilinear')
cbar = plt.colorbar(im, ax=ax2, fraction=0.04, pad=0.02)
cbar.set_label(r'$\langle R(r)\rangle$', fontsize=10)
ax2.axvline(T_S, color='cyan', linestyle='--', lw=1.5, alpha=0.85)
ax2.text(T_S + 0.6, L_BOX/2 * 0.92,
         '$t_s = 15$\nsource ends',
         color='cyan', fontsize=9, va='top')
ax2.axhline(SIGMA, color='cyan', linestyle=':', lw=1.0, alpha=0.6)
ax2.text(0.5, SIGMA + 0.5, r'$\sigma$', color='cyan', fontsize=9)
# Cascade-front line: r_front(t) ~ sqrt(D t) (illustrative diffusion law)
t_front = np.linspace(0.5, T_MAX, 200)
D_eff = ALPHA_FILL / 6
r_front = np.sqrt(6 * D_eff * t_front) * 1.5  # heuristic constant
ax2.plot(t_front, r_front, color='lime', linestyle=':', lw=1.2, alpha=0.7,
         label=r'$r_{\rm front}(t)\propto\sqrt{t}$')
ax2.legend(loc='upper right', fontsize=8, framealpha=0.85)
ax2.set_xlabel('simulation time $t$', fontsize=11)
ax2.set_ylabel(r'radius $r$ from $\Omega$', fontsize=11)
ax2.set_title(r'(B) Heatmap $\langle R(r,t)\rangle$: cascade front sweeps outward',
              fontsize=12, fontweight='bold')

fig.suptitle(r'Cosmogenetic cascade in isotropic radial coordinates',
             fontsize=13, fontweight='bold', y=1.02)
fig.savefig(FIG / 'fig_radial_profile.pdf', bbox_inches='tight')
fig.savefig(FIG / 'fig_radial_profile.png', dpi=150, bbox_inches='tight')
print(f"Saved -> {FIG/'fig_radial_profile.pdf'}")
