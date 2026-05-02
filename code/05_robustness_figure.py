"""
Generate the robustness figure (ensemble of realizations) for the paper.

This figure shows that the result is statistically robust across different
random Poisson-disk lattice realizations.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import json
import os

mpl.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Computer Modern Roman', 'DejaVu Serif'],
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 12,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'mathtext.fontset': 'cm',
    'lines.linewidth': 1.8,
    'axes.linewidth': 0.8,
})

DATA_DIR = '../data' if os.path.exists('../data') else 'data'
OUT_DIR = '../figures' if os.path.exists('../figures') else 'figures'

with open(os.path.join(DATA_DIR, 'ensemble_summary.json'), 'r') as f:
    summary = json.load(f)

m_pre = np.array(summary['m_pre_values'])
m_post = np.array(summary['m_post_values'])
n = summary['n_realizations']

# Load full trajectories
data = np.load(os.path.join(DATA_DIR, 'ensemble.npz'), allow_pickle=True)
t_trajs = data['t_trajectories']
T2_trajs = data['T2_trajectories']
I2_trajs = data['I2_trajectories']

# ============================================================
# Figure: 2 panels
#   Left: T^2 trajectories of all realizations
#   Right: scatter of (m_pre, m_post) with DESI target
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# --- LEFT: trajectories
ax = axes[0]
for i in range(len(t_trajs)):
    t = t_trajs[i]
    T2 = T2_trajs[i]
    mask = T2 > 1e-10
    ax.loglog(t[mask], T2[mask], 'b-', alpha=0.3, linewidth=1.2)

# Mean trajectory (interpolated on common grid)
t_common = np.logspace(-1, np.log10(70), 200)
T2_grid = []
for i in range(len(t_trajs)):
    T2_interp = np.interp(t_common, t_trajs[i], T2_trajs[i])
    T2_grid.append(T2_interp)
T2_grid = np.array(T2_grid)
T2_mean = np.exp(np.mean(np.log(T2_grid + 1e-30), axis=0))
T2_std_log = np.std(np.log(T2_grid + 1e-30), axis=0)
T2_low = np.exp(np.log(T2_mean) - T2_std_log)
T2_high = np.exp(np.log(T2_mean) + T2_std_log)

mask_mean = T2_mean > 1e-10
ax.loglog(t_common[mask_mean], T2_mean[mask_mean], 'b-', linewidth=2.5,
          label=fr'mean of {n} realizations')
ax.fill_between(t_common[mask_mean], T2_low[mask_mean], T2_high[mask_mean],
                alpha=0.2, color='blue', label=r'$\pm 1\sigma$ band')

ax.set_xlabel(r'simulation time $t$')
ax.set_ylabel(r'$\sum_i T_i^2$')
ax.set_title(r'Trajectories across {} realizations'.format(n))
ax.legend(loc='lower center', frameon=True, framealpha=0.95)
ax.grid(True, which='both', alpha=0.3)

# --- RIGHT: scatter of (m_pre, m_post)
ax = axes[1]
ax.scatter(m_pre, m_post, c='blue', s=80, alpha=0.7, edgecolors='black', 
           linewidth=0.8, label='realizations')

# Mean point with error bars
m_pre_mean = m_pre.mean()
m_post_mean = m_post.mean()
m_pre_std = m_pre.std()
m_post_std = m_post.std()

ax.errorbar(m_pre_mean, m_post_mean, xerr=m_pre_std, yerr=m_post_std,
            fmt='bs', markersize=12, capsize=5, capthick=2, linewidth=2,
            label=fr'mean $\pm 1\sigma$: $({m_pre_mean:.2f} \pm {m_pre_std:.2f}, {m_post_mean:.2f} \pm {m_post_std:.2f})$',
            zorder=5)

# DESI target
ax.scatter([-3.72], [1.65], marker='*', s=400, c='red', edgecolors='black',
           linewidth=1.5, label='DESI DR2 target', zorder=6)

# Connecting line
ax.plot([m_pre_mean, -3.72], [m_post_mean, 1.65], 'k:', alpha=0.5, linewidth=1)

ax.set_xlabel(r'$m_\infty$ (phantom phase)')
ax.set_ylabel(r'$m_0$ (today)')
ax.set_title(r'Statistical agreement with DESI DR2')
ax.legend(loc='upper right', frameon=True, framealpha=0.95)
ax.grid(True, alpha=0.3)

# Set reasonable limits
ax.set_xlim(min(m_pre.min(), -3.85) - 0.05, max(m_pre.max(), -3.5) + 0.05)
ax.set_ylim(min(m_post.min(), 1.5) - 0.1, max(m_post.max(), 1.85) + 0.1)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'fig_robustness.pdf'), bbox_inches='tight')
plt.savefig(os.path.join(OUT_DIR, 'fig_robustness.png'), dpi=150, bbox_inches='tight')
plt.close()

print(f"Figure saved: {OUT_DIR}/fig_robustness.pdf")
print()
print(f"Summary across {n} realizations:")
print(f"  m_pre  = {m_pre_mean:+.3f} ± {m_pre_std:.3f}")
print(f"  m_post = {m_post_mean:+.3f} ± {m_post_std:.3f}")
print(f"  Distance from DESI target:")
print(f"    m_pre:  {abs(m_pre_mean - (-3.72))/m_pre_std:.2f} sigma")
print(f"    m_post: {abs(m_post_mean - 1.65)/m_post_std:.2f} sigma")
