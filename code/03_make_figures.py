"""
Generate the publication figures.

Run AFTER 01_simulation.py.
Requires: matplotlib, numpy, scipy.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
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
os.makedirs(OUT_DIR, exist_ok=True)

data = np.load(os.path.join(DATA_DIR, 'simulation_best.npz'))
t = data['t']
sum_T2 = data['sum_T2']
sum_I2 = data['sum_I2']
t_m = data['t_m']
m_eff = data['m_eff']
t_peak = float(data['t_peak'])
m_pre = float(data['m_pre'])
m_post = float(data['m_post'])
t_at_046 = float(data['t_at_046'])

# ============================================================
# FIGURE 1: Time evolution log-log
# ============================================================

fig, ax = plt.subplots(figsize=(7, 4.5))
mask_T = sum_T2 > 1e-10
mask_I = sum_I2 > 1e-12
ax.loglog(t[mask_T], sum_T2[mask_T], 'b-', linewidth=2,
          label=r'$\sum_i T_i^2$ (translational)')
ax.loglog(t[mask_I], sum_I2[mask_I], 'r-', linewidth=2,
          label=r'$\sum_i I_i^2$ (bound patterns)')
ax.axvline(t_peak, color='green', linestyle='--', alpha=0.6,
           label=fr'peak $t_*={t_peak:.1f}$')
ax.axvline(t_at_046, color='purple', linestyle=':', alpha=0.6,
           label=fr'$I^2/T^2 = 0.46$ at $t={t_at_046:.1f}$')
ax.set_xlabel(r'simulation time $t$')
ax.set_ylabel(r'amplitude')
ax.legend(loc='lower center', frameon=True, framealpha=0.95)
ax.grid(True, which='both', alpha=0.3)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/fig_evolution.pdf', bbox_inches='tight')
plt.savefig(f'{OUT_DIR}/fig_evolution.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"  fig_evolution saved")

# ============================================================
# FIGURE 2: m_eff(t)
# ============================================================

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.semilogx(t_m, m_eff, 'b-', linewidth=2, alpha=0.85, label=r'simulation $m_{\rm eff}(t)$')
ax.axhline(0, color='black', linestyle='-', alpha=0.4, linewidth=1)
ax.axhline(-3.72, color='red', linestyle='--', alpha=0.7, label=r'DESI $m_\infty = -3.72$')
ax.axhline(1.65, color='orange', linestyle='--', alpha=0.7, label=r'DESI $m_0 = +1.65$')
ax.axvline(t_peak, color='green', linestyle=':', alpha=0.5)
xlim = (t_m.min(), t_m.max())
ax.fill_between([xlim[0], t_peak], -6, 5, alpha=0.08, color='red')
ax.fill_between([t_peak, xlim[1]], -6, 5, alpha=0.08, color='blue')
ax.text(t_peak * 0.15, -4.5, 'phantom\n(filling)', fontsize=10,
        ha='center', color='darkred', style='italic')
ax.text(t_peak * 5, -4.5, 'quintessence\n(dilution)', fontsize=10,
        ha='center', color='darkblue', style='italic')
ax.set_xlabel(r'simulation time $t$')
ax.set_ylabel(r'effective dilution exponent $m_{\rm eff} = -d\ln T^2 / d\ln t$')
ax.legend(loc='upper left', frameon=True, framealpha=0.95)
ax.grid(True, which='both', alpha=0.3)
ax.set_ylim(-6, 5)
ax.set_xlim(xlim)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/fig_m_eff.pdf', bbox_inches='tight')
plt.savefig(f'{OUT_DIR}/fig_m_eff.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"  fig_m_eff saved")

# ============================================================
# FIGURE 3: Comparison w(z) DDD vs DESI
# ============================================================

def w_from_m(a, m_inf, m_0, p=1.0):
    return (m_inf - (m_inf - m_0) * a**p) / 3 - 1

a_arr = np.linspace(0.25, 1.0, 300)
z_arr = 1/a_arr - 1
w_desi = w_from_m(a_arr, -3.72, 1.65)
w_ttd = w_from_m(a_arr, m_pre, m_post)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(z_arr, w_desi, 'k-', linewidth=2.5,
        label=fr'DESI DR2: $m_\infty = -3.72$, $m_0 = +1.65$')
ax.plot(z_arr, w_ttd, 'b--', linewidth=2.5, dashes=(5,3),
        label=fr'DDD simulation: $m_\infty = {m_pre:+.2f}$, $m_0 = {m_post:+.2f}$')
ax.axhline(-1, color='red', linestyle=':', linewidth=2, alpha=0.7,
           label=r'$\Lambda$CDM ($w = -1$)')
ax.fill_between(z_arr, -1, w_ttd, where=(w_ttd < -1), alpha=0.10, color='red')
ax.fill_between(z_arr, -1, w_ttd, where=(w_ttd > -1), alpha=0.10, color='blue')

z_cross = 1 / (m_pre / (m_pre - m_post)) - 1
ax.axvline(z_cross, color='green', linestyle=':', alpha=0.5)
ax.annotate(fr'phantom crossing $z \simeq {z_cross:.2f}$',
            xy=(z_cross, -1), xytext=(z_cross + 0.6, -0.5),
            fontsize=10, ha='left',
            arrowprops=dict(arrowstyle='->', alpha=0.5))

ax.plot(0, w_from_m(1.0, -3.72, 1.65), 'ko', markersize=8, zorder=5)
ax.plot(0, w_from_m(1.0, m_pre, m_post), 'b^', markersize=9, zorder=5)
ax.annotate(r'today', xy=(0.01, -0.45), xytext=(0.3, -0.15),
            fontsize=10, arrowprops=dict(arrowstyle='->', alpha=0.5))

ax.set_xlabel(r'redshift $z$')
ax.set_ylabel(r'dark-energy equation of state $w(z)$')
ax.legend(loc='lower left', frameon=True, framealpha=0.95)
ax.grid(True, alpha=0.3)
ax.invert_xaxis()
ax.set_xlim(2.5, -0.1)
ax.set_ylim(-2.3, 0.05)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/fig_comparison.pdf', bbox_inches='tight')
plt.savefig(f'{OUT_DIR}/fig_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"  fig_comparison saved")

# ============================================================
# FIGURE 4: Dark energy density rho_DE(z)
# ============================================================

def rho_DE(a, m_inf, m_0, p=1.0):
    a_arr = np.atleast_1d(a)
    rho = np.zeros_like(a_arr, dtype=float)
    a_grid = np.linspace(0.005, 1.0, 8000)
    m_grid = m_inf - (m_inf - m_0) * a_grid**p
    for i, av in enumerate(a_arr):
        mask = (a_grid >= av) & (a_grid <= 1.0)
        if mask.sum() < 2:
            rho[i] = 1.0
            continue
        integrand = m_grid[mask] / a_grid[mask]
        rho[i] = np.exp(np.trapezoid(integrand, a_grid[mask]))
    return rho

a_plot = np.linspace(0.18, 1.0, 200)
z_plot = 1/a_plot - 1
rho_sim = rho_DE(a_plot, m_pre, m_post)
rho_desi = rho_DE(a_plot, -3.72, 1.65)

fig, ax = plt.subplots(figsize=(8.5, 5.0))
ax.plot(z_plot, rho_sim, 'b-', linewidth=2.5,
        label=fr'DDD simulation ($m_\infty={m_pre:+.2f}$, $m_0={m_post:+.2f}$)')
ax.plot(z_plot, rho_desi, 'k--', linewidth=2.5, dashes=(5,3),
        label='DESI DR2 fit')
ax.axhline(1, color='gray', linestyle=':', linewidth=1.5, alpha=0.6,
           label=r'$\Lambda$CDM (constant)')
idx_sim = np.argmax(rho_sim)
idx_desi = np.argmax(rho_desi)
ax.axvline(z_plot[idx_sim], color='blue', linestyle=':', alpha=0.5)
ax.axvline(z_plot[idx_desi], color='black', linestyle=':', alpha=0.5)
# Combined peak annotation pushed below the curve to avoid overlap with legend
y_peak_max = max(rho_sim[idx_sim], rho_desi[idx_desi])
ax.annotate(fr'peak (sim) $z = {z_plot[idx_sim]:.2f}$',
            xy=(z_plot[idx_sim], rho_sim[idx_sim]),
            xytext=(z_plot[idx_sim] + 0.7, 0.35),
            fontsize=9, color='blue',
            arrowprops=dict(arrowstyle='->', alpha=0.6, color='blue'))
ax.annotate(fr'peak (DESI) $z = {z_plot[idx_desi]:.2f}$',
            xy=(z_plot[idx_desi], rho_desi[idx_desi]),
            xytext=(z_plot[idx_desi] + 1.6, 0.55),
            fontsize=9, color='black',
            arrowprops=dict(arrowstyle='->', alpha=0.6))
ax.set_xlabel(r'redshift $z$')
ax.set_ylabel(r'$\rho_{\rm DE}(z)\, /\, \rho_{\rm DE}(0)$')
# Place legend in upper-LEFT (high-z, low-rho corner — empty region)
ax.legend(loc='upper left', frameon=True, framealpha=0.95)
ax.grid(True, alpha=0.3)
ax.invert_xaxis()
ax.set_xlim(4, -0.1)
# Pad y-axis upward so peak markers + legend never collide
ax.set_ylim(0.0, max(1.35, y_peak_max * 1.15))
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/fig_rho_DE.pdf', bbox_inches='tight')
plt.savefig(f'{OUT_DIR}/fig_rho_DE.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"  fig_rho_DE saved")

print(f"\nAll figures saved to {OUT_DIR}/")
, max(1.35, y_peak_max * 1.15))
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/fig_rho_DE.pdf', bbox_inches='tight')
plt.savefig(f'{OUT_DIR}/fig_rho_DE.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"  fig_rho_DE saved")
