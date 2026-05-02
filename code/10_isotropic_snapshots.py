"""
Isotropic 2D snapshots of the cascade.

Two figures, each as 2x4 grid of 8 spatial maps:

  fig_iso_R_snapshots:    R(r) reconstructed isotropically from the
                          spherical average — smooth concentric rings,
                          no Poisson-disk noise.

  fig_iso_front_snapshots: T^2(r) reconstructed isotropically — shows
                          the *position of the cascade front* (where T^2
                          is non-zero, i.e. where the matter front is
                          actively propagating), with a marker on the
                          peak r_front(t).

Both use radial averaging followed by 2D reconstruction by symmetry.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from pathlib import Path
import time as _time

np.random.seed(2024)

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

tree = cKDTree(positions)
pairs = tree.query_pairs(R_LINK, output_type='ndarray')
edges_i = np.concatenate([pairs[:,0], pairs[:,1]])
edges_j = np.concatenate([pairs[:,1], pairs[:,0]])
adj = csr_matrix((np.ones(len(edges_i)), (edges_i, edges_j)), shape=(N, N))
deg = np.asarray(adj.sum(axis=1)).ravel()
mean_deg = float(deg.mean())

source_amp = np.exp(-r_node**2 / SIGMA**2)
coef = ALPHA_FILL / mean_deg

# Radial bins for averaging
N_BINS = 50
r_edges = np.linspace(0, L_BOX/2, N_BINS+1)
r_centers = 0.5*(r_edges[:-1] + r_edges[1:])
bin_idx = np.digitize(r_node, r_edges) - 1
bin_idx = np.clip(bin_idx, 0, N_BINS-1)
bin_counts = np.bincount(bin_idx, minlength=N_BINS).astype(float)
bin_counts_safe = np.maximum(bin_counts, 1)

SNAPSHOT_TIMES = [3.0, 7.0, 11.0, 15.0, 18.0, 25.0, 40.0, 70.0]
snap_indices = sorted({int(round(t/DT)) for t in SNAPSHOT_TIMES})

R = np.zeros(N)
snap_R = {}     # snapshot of R(r)
snap_T2 = {}    # snapshot of T^2(r)

print(f"Integrating for T_MAX={T_MAX}...")
t0 = _time.time()
for step in range(N_STEPS):
    t = step * DT
    if t < T_S:
        R += DT * S_0 * np.exp(GAMMA * t) * source_amp
    sumR_neigh = adj @ R
    flux_net = coef * (sumR_neigh - deg * R)
    T = np.maximum(flux_net, 0.0)
    R += DT * flux_net
    R = np.maximum(R, 0.0)

    if step in snap_indices:
        # radial averages
        rR_sum = np.bincount(bin_idx, weights=R, minlength=N_BINS)
        T2 = T**2
        rT2_sum = np.bincount(bin_idx, weights=T2, minlength=N_BINS)
        snap_R[t] = rR_sum / bin_counts_safe
        snap_T2[t] = rT2_sum / bin_counts_safe

print(f"  done in {_time.time()-t0:.1f}s")


def reconstruct_2D(r_centers, profile, half=L_BOX/2, GRID=200):
    """Build a 2D xy image by isotropic reconstruction from R(r)."""
    g = np.linspace(-half, half, GRID)
    GX, GY = np.meshgrid(g, g)
    Rgrid = np.sqrt(GX**2 + GY**2)
    # Interpolate radial profile onto Rgrid (linear); fill edge with last value
    Z = np.interp(Rgrid.ravel(), r_centers, profile,
                  left=profile[0], right=profile[-1]).reshape(Rgrid.shape)
    return g, Z


def plot_grid(title, snap_dict, vmin, vmax, cmap, label,
              outname, mark_front=False):
    """8-panel 2x4 grid of isotropic 2D maps."""
    fig, axes = plt.subplots(2, 4, figsize=(15, 8.0))
    axes = axes.flatten()
    times_sorted = sorted(snap_dict.keys())
    for k, ax in enumerate(axes):
        if k >= len(times_sorted):
            ax.axis('off'); continue
        t = times_sorted[k]
        prof = snap_dict[t]
        g, Z = reconstruct_2D(r_centers, prof)
        Z_plot = np.maximum(Z, vmin)
        sc = ax.imshow(Z_plot, origin='lower',
                       extent=[g.min(), g.max(), g.min(), g.max()],
                       norm=LogNorm(vmin=vmin, vmax=vmax),
                       cmap=cmap, aspect='equal',
                       interpolation='bilinear')
        # Mark Omega
        ax.plot(0, 0, marker='+', color='cyan', markersize=12, mew=2)
        # Source extent circle
        theta = np.linspace(0, 2*np.pi, 100)
        ax.plot(SIGMA*np.cos(theta), SIGMA*np.sin(theta),
                color='cyan', linestyle='--', lw=0.8, alpha=0.7)
        # Optional: mark front position (peak of profile)
        if mark_front and prof.max() > vmin*10:
            r_peak = r_centers[int(np.argmax(prof))]
            ax.plot(r_peak*np.cos(theta), r_peak*np.sin(theta),
                    color='lime', linestyle=':', lw=1.6, alpha=0.95,
                    label=f'front: $r_{{\\max}}={r_peak:.1f}$')
            ax.legend(loc='lower right', fontsize=7, framealpha=0.9)
        # Phase label
        if t < T_S:
            phase = 'source ON  ($w<-1$)'; col = 'lightcoral'
        elif t < T_S + 5:
            phase = 'crossing $w \\to -1$'; col = 'gold'
        else:
            phase = 'dilution  ($w>-1$)'; col = 'skyblue'
        ax.set_title(f'$t = {t:.1f}$  —  {phase}', fontsize=11,
                     color=col, fontweight='bold')
        ax.set_xlim(g.min(), g.max())
        ax.set_ylim(g.min(), g.max())
        ax.set_aspect('equal')
        ax.set_facecolor('#0a0a0a')
        if k % 4 == 0:
            ax.set_ylabel('$y - y_\\Omega$', fontsize=10)
        if k >= 4:
            ax.set_xlabel('$x - x_\\Omega$', fontsize=10)
        ax.tick_params(labelsize=8)
    cbar = fig.colorbar(sc, ax=axes.tolist(), fraction=0.025, pad=0.02,
                        aspect=50, shrink=0.85)
    cbar.set_label(label, fontsize=10)
    fig.suptitle(title, fontsize=13, fontweight='bold', y=0.99)
    fig.savefig(FIG / f'{outname}.pdf', bbox_inches='tight')
    fig.savefig(FIG / f'{outname}.png', dpi=150, bbox_inches='tight')
    print(f"Saved -> {FIG/(outname+'.pdf')}")


# Common colour scales
all_R = np.concatenate(list(snap_R.values()))
vmin_R = max(all_R[all_R > 0].min() if (all_R > 0).any() else 1e-6, 1e-5)
vmax_R = all_R.max()

all_T2 = np.concatenate(list(snap_T2.values()))
nz_T2 = all_T2[all_T2 > 0]
if nz_T2.size:
    vmin_T2 = max(nz_T2.min(), 1e-12)
    vmax_T2 = nz_T2.max()
else:
    vmin_T2, vmax_T2 = 1e-12, 1.0

# === Figure 1: isotropic R(x,y,t) ===
plot_grid(
    title=r'Isotropic reconstruction of $R(x,y)$ around $\Omega$ — 8 snapshots',
    snap_dict=snap_R, vmin=vmin_R, vmax=vmax_R,
    cmap='magma', label=r'reserve $R$ (isotropic from $\langle R(r)\rangle$)',
    outname='fig_iso_R_snapshots',
    mark_front=False,
)

# === Figure 2: isotropic T^2(x,y,t) — shows the matter front ===
plot_grid(
    title=r'Isotropic reconstruction of $T^2(x,y)$: position of the matter front (lime ring)',
    snap_dict=snap_T2, vmin=vmin_T2, vmax=vmax_T2,
    cmap='inferno', label=r'translational $T^2$ (isotropic from $\langle T^2(r)\rangle$)',
    outname='fig_iso_T2_front',
    mark_front=True,
)
