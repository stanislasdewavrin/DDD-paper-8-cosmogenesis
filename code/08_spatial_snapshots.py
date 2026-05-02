"""
Spatial 2D snapshots of the cascade around the founding impulse.

Produces a 2x4 grid of 2D scatter maps centred on the founding impulse
(located at the lattice centre Omega = (L/2, L/2, L/2)).
At each snapshot time we project nodes within a thin slab around the
median z-plane onto the xy-plane and colour them by R(t).

Snapshots span the full chronology:
  - early filling (source still active)
  - peak of T^2
  - early dilution
  - quintessence asymptote

Output: figures/fig_spatial_snapshots.{pdf,png}
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from pathlib import Path
import time as _time

np.random.seed(2024)

# Same parameters as 07_phase_heatmap.py
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
KAPPA_TRAP = 8e-4
R_TRAP = 0.05
DT = 5e-3
T_MAX = 80.0
N_STEPS = int(T_MAX / DT)

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
print(f"  mean degree = {mean_deg:.2f}")

source_amp = np.exp(-r_node**2 / SIGMA**2)
coef = ALPHA_FILL / mean_deg

# Times at which to take spatial snapshots
SNAPSHOT_TIMES = [3.0, 7.0, 11.0, 15.0, 18.0, 25.0, 40.0, 70.0]
snap_indices = [int(round(t/DT)) for t in SNAPSHOT_TIMES]
snap_indices = sorted(set(snap_indices))

# ============================================================
# Time integration
# ============================================================
R = np.zeros(N)
I2 = np.zeros(N)
snapshots = {}

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

    if step in snap_indices:
        snapshots[t] = R.copy()
        print(f"  snapshot at t={t:.2f}: R range [{R.min():.3e}, {R.max():.3e}]")

print(f"  done in {_time.time()-t0:.1f}s")

# ============================================================
# Plot 2x4 grid
# ============================================================
# Project onto xy-plane: take a slab in z near centre
slab_thickness = 8.0  # lattice units (thicker for denser projection)
in_slab = np.abs(positions[:,2] - L_BOX/2) < slab_thickness/2
print(f"  slab nodes: {in_slab.sum()} / {N}")

# Centre coordinates (relative to Omega)
xc = positions[in_slab, 0] - L_BOX/2
yc = positions[in_slab, 1] - L_BOX/2

# Common colour scale across all snapshots (log)
all_R = np.concatenate([snapshots[t][in_slab] for t in sorted(snapshots)])
nonzero = all_R[all_R > 0]
if nonzero.size:
    vmin_g = max(nonzero.min(), 1e-6)
    vmax_g = all_R.max()
else:
    vmin_g, vmax_g = 1e-6, 1.0

from scipy.interpolate import griddata
# Build a regular xy grid for interpolation
half = L_BOX/2
GRID_N = 120
gx = np.linspace(-half, half, GRID_N)
gy = np.linspace(-half, half, GRID_N)
GX, GY = np.meshgrid(gx, gy)

fig, axes = plt.subplots(2, 4, figsize=(15, 8.0))
axes = axes.flatten()
times_sorted = sorted(snapshots.keys())
for k, ax in enumerate(axes):
    if k >= len(times_sorted):
        ax.axis('off'); continue
    t = times_sorted[k]
    Rs = snapshots[t][in_slab]
    # Interpolate scattered nodes to regular grid (linear within convex hull,
    # nearest-neighbour for fill)
    Z_lin = griddata(np.column_stack([xc, yc]), Rs, (GX, GY),
                     method='linear', fill_value=np.nan)
    Z_nn = griddata(np.column_stack([xc, yc]), Rs, (GX, GY),
                    method='nearest')
    Z = np.where(np.isnan(Z_lin), Z_nn, Z_lin)
    Z = np.maximum(Z, vmin_g)
    sc = ax.imshow(Z, origin='lower', extent=[-half, half, -half, half],
                   norm=LogNorm(vmin=vmin_g, vmax=vmax_g),
                   cmap='magma', aspect='equal',
                   interpolation='bilinear')
    # Overlay actual node positions as small dots for visual confirmation
    ax.scatter(xc, yc, c='white', s=2, alpha=0.25, edgecolors='none')
    # Mark Omega (founding impulse centre)
    ax.plot(0, 0, marker='+', color='cyan', markersize=12, mew=2)
    # Source extent circle
    theta = np.linspace(0, 2*np.pi, 100)
    ax.plot(SIGMA*np.cos(theta), SIGMA*np.sin(theta),
            color='cyan', linestyle='--', lw=0.8, alpha=0.7)
    # Phase label
    if t < T_S:
        phase = 'source ON  ($w<-1$)'; col = 'lightcoral'
    elif t < T_S + 5:
        phase = 'crossing $w \\to -1$'; col = 'gold'
    else:
        phase = 'dilution  ($w>-1$)'; col = 'skyblue'
    ax.set_title(f'$t = {t:.1f}$  —  {phase}', fontsize=11, color=col,
                 fontweight='bold')
    ax.set_xlim(-L_BOX/2, L_BOX/2)
    ax.set_ylim(-L_BOX/2, L_BOX/2)
    ax.set_aspect('equal')
    ax.set_facecolor('#0a0a0a')
    if k % 4 == 0:
        ax.set_ylabel('$y - y_\\Omega$', fontsize=10)
    if k >= 4:
        ax.set_xlabel('$x - x_\\Omega$', fontsize=10)
    ax.tick_params(labelsize=8)

# Single shared colorbar
cbar = fig.colorbar(sc, ax=axes.tolist(), fraction=0.025, pad=0.02,
                    aspect=50, shrink=0.85)
cbar.set_label(r'reserve $R$ at node (slab around $z = z_\Omega$)',
               fontsize=10)

fig.suptitle(r'Spatial dynamics of the cosmogenetic cascade around $\Omega$ '
             '(2D slab projection, 8 snapshots)',
             fontsize=13, fontweight='bold', y=0.99)
fig.savefig(FIG / 'fig_spatial_snapshots.pdf', bbox_inches='tight')
fig.savefig(FIG / 'fig_spatial_snapshots.png', dpi=150, bbox_inches='tight')
print(f"Saved -> {FIG/'fig_spatial_snapshots.pdf'}")
