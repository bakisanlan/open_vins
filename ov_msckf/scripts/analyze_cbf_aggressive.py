#!/usr/bin/env python3
"""
Diagnostic analysis of CBF aggressiveness at curvature points.

Investigates:
1. Why CBF over-corrects (μ << 0 → large v_safe boost) at corners
2. Why g(x) largest component aligns with movement direction
3. Whether there's a "wind-up" effect in the objective/gradient
"""

import sys
import os
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# ── Find the latest log with CBF data ──
log_dir = os.path.expanduser("~/cbf_logs")
csvs = sorted(glob.glob(os.path.join(log_dir, "cbf_log_*.csv")),
              key=os.path.getmtime, reverse=True)

# Pick the latest file that has mu_margin column (new format)
df = None
log_path = None
for f in csvs:
    tmp = pd.read_csv(f)
    if 'mu_margin' in tmp.columns and tmp['mu_margin'].notna().sum() > 10:
        df = tmp
        log_path = f
        break

if df is None:
    print("No log with CBF data (mu_margin) found.")
    sys.exit(1)

print(f"Analyzing: {log_path}")
print(f"  Rows: {len(df)},  CBF-active rows (mu not NaN): {df['mu_margin'].notna().sum()}")

# ── Filter to only CBF-active rows ──
cbf = df.dropna(subset=['mu_margin']).copy()
cbf = cbf.reset_index(drop=True)
t = cbf['timestamp'].values

print(f"\n{'='*70}")
print("1. MARGIN (μ) STATISTICS")
print(f"{'='*70}")
mu = cbf['mu_margin'].values
print(f"  μ min  = {mu.min():.4f}")
print(f"  μ max  = {mu.max():.4f}")
print(f"  μ mean = {mu.mean():.4f}")
print(f"  μ std  = {mu.std():.4f}")
print(f"  % of time μ < 0 (CBF active): {100*np.mean(mu < 0):.1f}%")
print(f"  % of time μ < -1 (very negative): {100*np.mean(mu < -1):.1f}%")

print(f"\n{'='*70}")
print("2. SLACK VARIABLE (δ_T) STATISTICS")
print(f"{'='*70}")
delta = cbf['delta_T'].values
print(f"  δ_T min  = {delta.min():.4f}")
print(f"  δ_T max  = {delta.max():.4f}")
print(f"  δ_T mean = {delta.mean():.4f}")
print(f"  % of time δ_T > 0 (slack used): {100*np.mean(delta > 1e-6):.1f}%")

print(f"\n{'='*70}")
print("3. VELOCITY ANALYSIS")
print(f"{'='*70}")
v_nom_norm = cbf['v_nom_norm'].values
v_cbf_norm = cbf['v_cbf_norm'].values
v_nom_x = cbf['v_nom_x'].values
v_nom_y = cbf['v_nom_y'].values
v_cbf_x = cbf['v_cbf_x'].values
v_cbf_y = cbf['v_cbf_y'].values

# Velocity modification magnitude
dv = v_cbf_norm - v_nom_norm
print(f"  ||v_cbf|| - ||v_nom|| : min={dv.min():.3f}, max={dv.max():.3f}, mean={dv.mean():.3f}")
print(f"  max ||v_cbf|| = {v_cbf_norm.max():.3f}")
print(f"  max ||v_nom|| = {v_nom_norm.max():.3f}")

# Find the moments of maximum speed-up
speed_up_idx = np.argsort(v_cbf_norm - v_nom_norm)[-10:]
print(f"\n  Top 10 speed-up moments (t, v_nom_norm, v_cbf_norm, mu, delta_T):")
for i in speed_up_idx:
    print(f"    t={t[i]:.1f}s  ||v_nom||={v_nom_norm[i]:.3f}  ||v_cbf||={v_cbf_norm[i]:.3f}"
          f"  μ={mu[i]:.4f}  δ_T={delta[i]:.4f}")

# ── Compute approximate g(x) from the CBF correction ──
# v_safe = v_nom - (μ / gᵀ W⁻¹ g) W⁻¹ g  when μ < 0
# So the correction vector is: Δv = v_safe - v_nom = -(μ / gᵀ W⁻¹ g) W⁻¹ g
# Direction of Δv tells us the direction of W⁻¹ g

dv_x = v_cbf_x - v_nom_x
dv_y = v_cbf_y - v_nom_y
dv_z = cbf['v_cbf_z'].values - cbf['v_nom_z'].values
dv_mag = np.sqrt(dv_x**2 + dv_y**2 + dv_z**2)

print(f"\n{'='*70}")
print("4. CBF CORRECTION DIRECTION ANALYSIS")
print(f"{'='*70}")
# When correction is significant
active_mask = dv_mag > 0.1
if active_mask.sum() > 0:
    # Angle between correction and nominal velocity
    for i in np.where(active_mask)[0][:20]:
        v_n = np.array([v_nom_x[i], v_nom_y[i], cbf['v_nom_z'].values[i]])
        dv_vec = np.array([dv_x[i], dv_y[i], dv_z[i]])
        v_n_norm = np.linalg.norm(v_n)
        dv_norm = np.linalg.norm(dv_vec)
        if v_n_norm > 0.1 and dv_norm > 0.1:
            cos_angle = np.dot(v_n, dv_vec) / (v_n_norm * dv_norm)
            angle_deg = np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))
            print(f"  t={t[i]:.1f}s: v_nom=[{v_n[0]:.2f},{v_n[1]:.2f},{v_n[2]:.2f}] "
                  f"Δv=[{dv_x[i]:.2f},{dv_y[i]:.2f},{dv_z[i]:.2f}] "
                  f"angle={angle_deg:.1f}° μ={mu[i]:.4f}")

print(f"\n{'='*70}")
print("5. LOGDET & DRIFT ANALYSIS AT CORNERS")
print(f"{'='*70}")
logdet = cbf['logdet'].values
f_drift = cbf['f_drift'].values
print(f"  logdet: min={logdet.min():.4f}  max={logdet.max():.4f}  mean={logdet.mean():.4f}")
print(f"  f_drift: min={f_drift.min():.4f}  max={f_drift.max():.4f}  mean={f_drift.mean():.4f}")

# Find sudden logdet drops
dlogdet = np.diff(logdet)
big_drops = np.where(dlogdet < -0.3)[0]
if len(big_drops) > 0:
    print(f"  Sudden logdet drops (>0.3): {len(big_drops)} events")
    for i in big_drops[:10]:
        print(f"    t={t[i]:.1f}s: logdet {logdet[i]:.4f} → {logdet[i+1]:.4f} (Δ={dlogdet[i]:.4f})"
              f"  v_nom=[{v_nom_x[i]:.2f},{v_nom_y[i]:.2f}]")

# ── Key diagnostic: g(x) direction vs movement direction at high-correction moments ──
print(f"\n{'='*70}")
print("6. BARRIER FUNCTION h(x) DURING INTERVENTION")
print(f"{'='*70}")
h = logdet - cbf['logdet_threshold'].values
print(f"  h(x) = logdet - h_min")
print(f"  h min  = {h.min():.4f}")
print(f"  h max  = {h.max():.4f}")
print(f"  h mean = {h.mean():.4f}")
print(f"  % time h < 0 (unsafe): {100*np.mean(h < 0):.1f}%")

# During high-correction moments, what's h?
if active_mask.sum() > 0:
    h_active = h[active_mask]
    print(f"  h during CBF active: min={h_active.min():.4f} max={h_active.max():.4f} mean={h_active.mean():.4f}")

# ── Tri counts ──
print(f"\n{'='*70}")
print("7. TRIANGULATION FEATURE COUNTS")
print(f"{'='*70}")
tri_tried = cbf['tri_tried'].values
tri_success = cbf['tri_success'].values
print(f"  tri_tried:   min={tri_tried.min()}, max={tri_tried.max()}, mean={tri_tried.mean():.1f}")
print(f"  tri_success: min={tri_success.min()}, max={tri_success.max()}, mean={tri_success.mean():.1f}")
if tri_tried.max() > 0:
    ratio = tri_success / np.maximum(tri_tried, 1)
    print(f"  success rate: min={ratio.min():.2%}, max={ratio.max():.2%}, mean={ratio.mean():.2%}")

# ══════════════════════════════════════════════════════════════════════
# DIAGNOSTIC PLOTS
# ══════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(18, 22))
gs = GridSpec(6, 2, figure=fig, hspace=0.35, wspace=0.25)

# ── 1. μ (margin) over time ──
ax1 = fig.add_subplot(gs[0, :])
ax1.plot(t, mu, 'b-', lw=0.8, alpha=0.8, label='μ = f + gᵀv_nom + γh')
ax1.axhline(0, color='r', ls='--', lw=1.5, label='μ = 0 (boundary)')
ax1.fill_between(t, mu, 0, where=(mu < 0), alpha=0.15, color='red', label='CBF active region')
ax1.set_ylabel('μ (margin)')
ax1.set_xlabel('time (s)')
ax1.set_title('CBF Margin μ(t) — negative triggers intervention')
ax1.legend(loc='upper right')
ax1.grid(True, alpha=0.3)

# ── 2. Velocity norms ──
ax2 = fig.add_subplot(gs[1, :])
ax2.plot(t, v_nom_norm, 'g-', lw=1.0, alpha=0.8, label='||v_nom||')
ax2.plot(t, v_cbf_norm, 'r-', lw=1.0, alpha=0.8, label='||v_cbf||')
ax2.fill_between(t, v_nom_norm, v_cbf_norm, alpha=0.15, color='orange', label='correction')
ax2.set_ylabel('speed (m/s)')
ax2.set_xlabel('time (s)')
ax2.set_title('Nominal vs CBF Velocity Magnitude')
ax2.legend(loc='upper right')
ax2.grid(True, alpha=0.3)

# ── 3. logdet and barrier h(x) ──
ax3 = fig.add_subplot(gs[2, 0])
ax3.plot(t, logdet, 'b-', lw=0.8, label='logdet (ℓ̄)')
ax3.axhline(cbf['logdet_threshold'].values[0], color='r', ls='--', lw=1.5, label='h_min')
ax3.set_ylabel('logdet')
ax3.set_title('Observability Metric logdet(M)')
ax3.legend()
ax3.grid(True, alpha=0.3)

ax3b = fig.add_subplot(gs[2, 1])
ax3b.plot(t, h, 'b-', lw=0.8, label='h(x) = ℓ̄ - h_min')
ax3b.axhline(0, color='r', ls='--', lw=1.5)
ax3b.set_ylabel('h(x)')
ax3b.set_title('Barrier Function h(x) — negative = unsafe')
ax3b.legend()
ax3b.grid(True, alpha=0.3)

# ── 4. f_drift and its relationship with μ ──
ax4 = fig.add_subplot(gs[3, 0])
ax4.plot(t, f_drift, 'purple', lw=0.8, label='f(x) drift')
ax4.set_ylabel('f(x)')
ax4.set_title('Drift Term f(x)')
ax4.legend()
ax4.grid(True, alpha=0.3)

ax4b = fig.add_subplot(gs[3, 1])
ax4b.plot(t, delta, 'orange', lw=0.8, label='δ_T (slack)')
ax4b.axhline(0, color='k', ls='--', lw=0.5)
ax4b.set_ylabel('δ_T')
ax4b.set_title('Slack Variable δ_T')
ax4b.legend()
ax4b.grid(True, alpha=0.3)

# ── 5. v_nom components vs v_cbf components ──
ax5a = fig.add_subplot(gs[4, 0])
ax5a.plot(t, v_nom_x, 'g-', lw=0.8, alpha=0.7, label='v_nom_x')
ax5a.plot(t, v_cbf_x, 'r-', lw=0.8, alpha=0.7, label='v_cbf_x')
ax5a.set_ylabel('vx (m/s)')
ax5a.set_title('X-component: Nominal vs CBF')
ax5a.legend()
ax5a.grid(True, alpha=0.3)

ax5b = fig.add_subplot(gs[4, 1])
ax5b.plot(t, v_nom_y, 'g-', lw=0.8, alpha=0.7, label='v_nom_y')
ax5b.plot(t, v_cbf_y, 'r-', lw=0.8, alpha=0.7, label='v_cbf_y')
ax5b.set_ylabel('vy (m/s)')
ax5b.set_title('Y-component: Nominal vs CBF')
ax5b.legend()
ax5b.grid(True, alpha=0.3)

# ── 6. Correction magnitude + direction angle ──
ax6 = fig.add_subplot(gs[5, 0])
ax6.plot(t, dv_mag, 'r-', lw=0.8, label='||Δv|| = ||v_cbf - v_nom||')
ax6.set_ylabel('||Δv|| (m/s)')
ax6.set_xlabel('time (s)')
ax6.set_title('CBF Correction Magnitude')
ax6.legend()
ax6.grid(True, alpha=0.3)

# Angle between correction and nominal velocity
angles = np.full(len(t), np.nan)
for i in range(len(t)):
    v_n = np.array([v_nom_x[i], v_nom_y[i], cbf['v_nom_z'].values[i]])
    dv_vec = np.array([dv_x[i], dv_y[i], dv_z[i]])
    vn_n = np.linalg.norm(v_n)
    dv_n = np.linalg.norm(dv_vec)
    if vn_n > 0.3 and dv_n > 0.05:
        cos_a = np.dot(v_n, dv_vec) / (vn_n * dv_n)
        angles[i] = np.degrees(np.arccos(np.clip(cos_a, -1, 1)))

ax6b = fig.add_subplot(gs[5, 1])
valid_angles = ~np.isnan(angles)
if valid_angles.sum() > 0:
    ax6b.scatter(t[valid_angles], angles[valid_angles], s=2, c='blue', alpha=0.5)
    ax6b.axhline(0, color='k', ls='--', lw=0.5)
    ax6b.axhline(180, color='k', ls='--', lw=0.5)
ax6b.set_ylabel('angle (°)')
ax6b.set_xlabel('time (s)')
ax6b.set_title('Angle between Δv and v_nom (0°=along, 180°=opposing)')
ax6b.set_ylim(-10, 200)
ax6b.grid(True, alpha=0.3)

fig.suptitle(f'CBF Aggressiveness Diagnostic — {os.path.basename(log_path)}', fontsize=14, fontweight='bold')
out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fig_cbf_diagnostic.pdf')
fig.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"\nDiagnostic plot saved to: {out_path}")

# ══════════════════════════════════════════════════════════════════════
# KEY INSIGHT: Decompose the margin μ = f + gᵀv_nom + γh
# to see which term dominates the negative margin
# ══════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("8. MARGIN DECOMPOSITION: μ = f + gᵀv_nom + γh")
print(f"{'='*70}")
# We have f_drift and h, but not g directly. However:
# μ = f + gᵀv_nom + γh  →  gᵀv_nom = μ - f - γh
# We need gamma from config
gamma = None
try:
    import yaml
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', 'config', 'cbf_config.yaml')
    with open(config_path) as fh:
        cfg = yaml.safe_load(fh)
    params = cfg.get('/**', {}).get('ros__parameters', {})
    gamma = params.get('cbf_gamma', 1.0)
    w_par = params.get('cbf_w_parallel', 1.0)
    w_perp = params.get('cbf_w_perpendicular', 10.0)
    p_T = params.get('cbf_p_T', 100.0)
    print(f"  Config: γ={gamma}, w_∥={w_par}, w_⊥={w_perp}, p_T={p_T}")
except:
    gamma = 1.0
    print(f"  Could not read config, using default γ={gamma}")

gamma_h = gamma * h
g_dot_v = mu - f_drift - gamma_h

print(f"\n  Term breakdown (when μ < 0):")
neg_mask = mu < 0
if neg_mask.sum() > 0:
    print(f"    f(x):     mean={f_drift[neg_mask].mean():.4f}  std={f_drift[neg_mask].std():.4f}")
    print(f"    γ·h(x):   mean={gamma_h[neg_mask].mean():.4f}  std={gamma_h[neg_mask].std():.4f}")
    print(f"    gᵀv_nom:  mean={g_dot_v[neg_mask].mean():.4f}  std={g_dot_v[neg_mask].std():.4f}")
    print(f"    μ total:   mean={mu[neg_mask].mean():.4f}")
    print(f"\n  → The dominant negative contributor is: ", end="")
    terms = {'f(x)': f_drift[neg_mask].mean(),
             'γh': gamma_h[neg_mask].mean(),
             'gᵀv_nom': g_dot_v[neg_mask].mean()}
    most_neg = min(terms, key=terms.get)
    print(f"{most_neg} = {terms[most_neg]:.4f}")
    
    # ── CBF closed-form correction magnitude ──
    # v_safe = v_nom - (μ / (gᵀ W⁻¹ g)) W⁻¹ g
    # The correction magnitude is |μ| / (gᵀ W⁻¹ g) * ||W⁻¹ g||
    # With isotropic W (w_par = w_perp = w), this simplifies to |μ| / (||g||² / w) * ||g||/w = |μ|*w/||g||
    # With anisotropic W, the correction is direction-dependent
    print(f"\n  → With large |μ|, the correction |μ|/(gᵀW⁻¹g) scales LINEARLY with |μ|.")
    print(f"  → If gᵀv_nom dominates μ, then |μ| grows with speed, creating")
    print(f"    amplified corrections at higher velocities (corners with direction change).")
else:
    print("  No CBF interventions detected in this log.")

print(f"\n{'='*70}")
print("9. ANALYSIS SUMMARY")
print(f"{'='*70}")
print("""
HYPOTHESIS 1 — g(x) alignment with movement direction:
  The gradient g(x) = ∂h/∂v measures how velocity affects the barrier.
  Features are triangulated from camera views along the flight direction.
  Moving faster along a direction creates more baseline for features
  ahead, so ∂h/∂v_x is naturally large when moving in x.

HYPOTHESIS 2 — Aggressive correction at curvature:
  At corners, v_nom changes direction abruptly. The term gᵀv_nom can
  swing rapidly from positive to very negative because:
  (a) g(x) lags behind (EMA smoothing on g from C++ side)
  (b) v_nom changes direction but g still points in old direction
  This creates a large negative μ → large |correction|.
  The correction is proportional to |μ|, not just sign(μ).

HYPOTHESIS 3 — Wind-up effect:
  The EMA smoothing of g(x), logdet, and drift in UpdaterMSCKF.cpp
  creates temporal inertia. When the drone changes direction:
  - g(x) still reflects the old direction (lagged gradient)
  - The new v_nom is nearly perpendicular to old g(x)
  - gᵀv_nom drops suddenly → μ goes very negative
  - The CBF applies a large correction in the (lagged) g direction
  - This ACCELERATES the drone along the old movement direction
  This is NOT a CBF design flaw — it's the EMA smoothing creating
  a "memory" of the old gradient direction.
""")
