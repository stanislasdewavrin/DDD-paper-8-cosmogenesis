"""
Analyze the simulation result and print key metrics.

Run AFTER 01_simulation.py.
"""

import numpy as np
import os

DATA_DIR = '../data' if os.path.exists('../data') else 'data'
data_path = os.path.join(DATA_DIR, 'simulation_best.npz')

if not os.path.exists(data_path):
    print(f"ERROR: {data_path} not found.")
    print("Run 01_simulation.py first.")
    raise SystemExit(1)

data = np.load(data_path)

t = data['t']
sum_T2 = data['sum_T2']
sum_I2 = data['sum_I2']
m_eff = data['m_eff']
t_m = data['t_m']
t_peak = float(data['t_peak'])
m_pre = float(data['m_pre'])
m_post = float(data['m_post'])
t_at_046 = float(data['t_at_046'])

# DESI DR2 targets
M_INF_DESI = -3.72
M_0_DESI = +1.65

print("=" * 60)
print("ANALYSIS — Cosmogenesis from a Discrete Lattice")
print("=" * 60)

print(f"\nLattice: N = {int(data['N'])} nodes, L = {float(data['L_box'])}")
print(f"Time step: dt = {float(data['dt'])}")

print(f"\n--- Time evolution ---")
print(f"  T^2 peak time:   t_peak  = {t_peak:.3f}")
print(f"  T^2 peak value:           = {sum_T2.max():.4f}")
print(f"  Final T^2:                = {sum_T2[-1]:.5f}")
print(f"  Final I^2:                = {sum_I2[-1]:.5f}")

print(f"\n--- Effective dilution exponent ---")
print(f"  Pre-peak  (phantom):  m_inf = {m_pre:+.3f}    DESI target: {M_INF_DESI:+.2f}")
print(f"  Post-peak (today):    m_0   = {m_post:+.3f}    DESI target: {M_0_DESI:+.2f}")
print(f"  Deviation m_inf:                       {abs((m_pre - M_INF_DESI)/M_INF_DESI):.1%}")
print(f"  Deviation m_0:                         {abs((m_post - M_0_DESI)/M_0_DESI):.1%}")

print(f"\n--- Cosmological mapping ---")
print(f"  Time when ratio I^2/T^2 = 0.46:  t = {t_at_046:.2f}")
print(f"  This identifies a = 1 (today) in the simulation.")

# Predicted phantom-crossing redshift assuming linear m(a)
a_cross = m_pre / (m_pre - m_post)
z_cross = 1 / a_cross - 1 if a_cross > 0 else None
print(f"  Phantom crossing predicted at:   a* = {a_cross:.3f}")
print(f"                                   z* = {z_cross:.3f}     DESI: ~0.45")

# w today
w_today_sim = m_post / 3 - 1
w_today_desi = M_0_DESI / 3 - 1
print(f"\n  w(z=0):  simulation = {w_today_sim:+.3f}    DESI = {w_today_desi:+.3f}")

print(f"\n--- Summary ---")
print(f"{'Quantity':<28s}{'Simulation':>14s}{'DESI target':>14s}{'Deviation':>12s}")
print("-" * 68)
print(f"{'m_inf (phantom phase)':<28s}{m_pre:>+14.2f}{M_INF_DESI:>+14.2f}{abs((m_pre-M_INF_DESI)/M_INF_DESI):>11.1%}")
print(f"{'m_0 (today)':<28s}{m_post:>+14.2f}{M_0_DESI:>+14.2f}{abs((m_post-M_0_DESI)/M_0_DESI):>11.1%}")
print(f"{'w(z=0)':<28s}{w_today_sim:>+14.2f}{w_today_desi:>+14.2f}{abs((w_today_sim-w_today_desi)/w_today_desi):>11.1%}")
print(f"{'Phantom crossing z*':<28s}{z_cross:>14.2f}{0.45:>14.2f}{abs((z_cross-0.45)/0.45):>11.1%}")
