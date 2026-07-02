#!/usr/bin/env python3
"""
Trajectory Comparison Plotter
=============================

Compares the actual simulated drone trajectory (from the CBF log) 
with the nominal ideal reference trajectory (from the B-Spline definition).

Usage:
    python3 plot_sim_traj.py [initial_yaw_deg]
"""

import os
import sys
import glob
import yaml
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# Import trajectory classes from the existing node
try:
    from bspline_traj_node import (BSplineTrajectory, StraightTrajectory,
                                    CircleTrajectory, ClimbingStraightTrajectory)
except ImportError:
    # Allow running from other directories
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from bspline_traj_node import (BSplineTrajectory, StraightTrajectory,
                                    CircleTrajectory, ClimbingStraightTrajectory)


def setup_style():
    """Configure matplotlib for publication-quality output."""
    try:
        mpl.font_manager.findfont("Times New Roman", fallback_to_default=False)
        font_family = "Times New Roman"
    except Exception:
        font_family = "serif"

    plt.rcParams.update({
        "font.family": font_family,
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 13,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
        "lines.linewidth": 1.6,
        "axes.grid": True,
        "grid.alpha": 0.35,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })

def find_latest_log(log_dir: str = None) -> str:
    """Return the path to the most recently modified CBF CSV log.

    Searches scripts/logs/flight_*/cbf_log_*.csv first,
    then falls back to ~/cbf_logs/ for legacy logs.
    """
    files = []
    scripts_dir = os.path.dirname(os.path.abspath(__file__))
    flight_pattern = os.path.join(scripts_dir, "logs", "flight_*", "cbf_log_*.csv")
    files += glob.glob(flight_pattern)
    legacy_dir = os.path.expanduser(log_dir or "~/cbf_logs")
    files += glob.glob(os.path.join(legacy_dir, "cbf_log_*.csv"))
    files = sorted(files, key=os.path.getmtime)
    if not files:
        print(f"No CBF log files found in {flight_pattern} or {legacy_dir}")
        sys.exit(1)
    return files[-1]

def load_log(path: str) -> dict:
    data = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding="utf-8")
    return {name: np.array(data[name], dtype=float) for name in data.dtype.names}

def load_bspline_config() -> tuple:
    """Load config parameters from the bspline config file."""
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "config", "bspline_traj_config.yaml")
    config_path = os.path.normpath(config_path)
    
    side, radius, speed, climb_speed = 20.0, 30.0, 3.0, 1.0  # defaults
    traj_type = "square"
    if os.path.isfile(config_path):
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        params = cfg.get("/**", {}).get("ros__parameters", {})
        side = params.get("side_length", side)
        radius = params.get("radius", radius)
        speed = params.get("speed", speed)
        climb_speed = params.get("climb_speed", climb_speed)
        traj_type = params.get("traj_type", traj_type)
        print(f"Loaded config: type={traj_type}, side_length={side}m, radius={radius}m, "
              f"speed={speed}m/s, climb_speed={climb_speed}m/s")
    else:
        print(f"Config not found at {config_path}, using defaults")
        
    return traj_type, side, radius, speed, climb_speed

def main():
    setup_style()
    
    # 1. Load latest CBF log
    log_path = find_latest_log()
    print(f"Reading log: {log_path}")
    d = load_log(log_path)
    
    # Extract actual trajectory
    t = d["timestamp"]
    x = d["drone_x"]
    y = d["drone_y"]
    z = d["drone_z"]
    v_nom_norm = d["v_nom_norm"]
    
    # Filter valid positions
    valid = ~(np.isnan(x) | np.isnan(y) | np.isnan(z))
    if valid.sum() == 0:
        print("No valid position data in log.")
        sys.exit(1)
        
    t_v = t[valid]
    x_v = x[valid]
    y_v = y[valid]
    z_v = z[valid]
    vnom_v = v_nom_norm[valid]
    
    # Find when nominal velocity first became active (> 0.05 m/s)
    active_indices = np.where(vnom_v > 0.05)[0]
    if len(active_indices) == 0:
        print("Trajectory never became active (v_nom always 0).")
        # Just plot whatever we have
        start_idx = 0
    else:
        start_idx = active_indices[0]
        
    # The simulation origin is the drone position when it started moving
    origin_x = x_v[start_idx]
    origin_y = y_v[start_idx]
    origin_z = z_v[start_idx]
    t_start = t_v[start_idx]
    
    # 2. Generate Nominal Trajectory (in local frame)
    traj_type, side, radius, speed, climb_speed = load_bspline_config()
    
    if traj_type == "straight":
        traj = StraightTrajectory(side, speed)
    elif traj_type == "climb":
        traj = ClimbingStraightTrajectory(side, speed, climb_speed)
    elif traj_type == "circle":
        traj = CircleTrajectory(radius, speed)
    else:
        traj = BSplineTrajectory.make_square(side, speed)
    
    # Evaluate nominal trajectory points over the duration of the active log
    duration = t_v[-1] - t_start
    if duration <= 0:
        duration = traj.period
        
    t_nom = np.linspace(0, duration, 500)
    nom_positions = np.array([traj.evaluate(ti)[0] for ti in t_nom])
    
    # Also evaluate exactly at the drone's time steps to get exact expected positions
    t_eval = t_v[start_idx:] - t_start
    exact_nom = np.array([traj.evaluate(ti)[0] for ti in t_eval])
    
    # ── ALIGNMENT: Rotate OpenVINS local frame to MAVROS ENU frame ──
    # Instead of guessing from the trajectory (which includes CBF drift),
    # we use the known initial yaw of the drone in Gazebo.
    # Default is 0 degrees (facing East).
    initial_yaw_deg = 0.0
    if len(sys.argv) > 1:
        try:
            initial_yaw_deg = float(sys.argv[1])
        except ValueError:
            pass
            
    actual_yaw = np.radians(initial_yaw_deg)
    
    c, s = np.cos(actual_yaw), np.sin(actual_yaw)
    R = np.array([[c, -s], [s, c]])

    # Rotate and translate continuous plotting points (XY only, Z passes through)
    nom_xy_rot = nom_positions[:, :2] @ R.T
    nom_x = origin_x + nom_xy_rot[:, 0]
    nom_y = origin_y + nom_xy_rot[:, 1]
    nom_z = origin_z + nom_positions[:, 2]
    
    # Rotate and translate exact evaluation points
    exact_nom_xy_rot = exact_nom[:, :2] @ R.T
    exact_nom_x = origin_x + exact_nom_xy_rot[:, 0]
    exact_nom_y = origin_y + exact_nom_xy_rot[:, 1]
    exact_nom_z = origin_z + exact_nom[:, 2]
    
    # Compute 3D error
    err_x = x_v[start_idx:] - exact_nom_x
    err_y = y_v[start_idx:] - exact_nom_y
    err_z = z_v[start_idx:] - exact_nom_z
    err_norm = np.sqrt(err_x**2 + err_y**2 + err_z**2)
    err_lateral = np.sqrt(err_x**2 + err_y**2)
    avg_error = np.mean(err_norm)
    max_error = np.max(err_norm)
    
    # ── Choose plot layout based on trajectory type ──
    is_3d = traj_type in ["climb"]
    
    if is_3d:
        _plot_3d(t_eval, nom_x, nom_y, nom_z, x_v, y_v, z_v,
                 origin_x, origin_y, origin_z, start_idx,
                 err_norm, err_lateral, err_z, avg_error, max_error, traj_type)
    else:
        _plot_2d(t_eval, nom_x, nom_y, x_v, y_v,
                 origin_x, origin_y, start_idx,
                 err_norm, avg_error, max_error, traj_type)


def _plot_2d(t_eval, nom_x, nom_y, x_v, y_v,
             origin_x, origin_y, start_idx,
             err_norm, avg_error, max_error, traj_type):
    """Original 2-panel XY plot for planar trajectories."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Panel 1: XY Plot
    ax = axes[0]
    ax.plot(nom_x, nom_y, color='gray', linestyle='--', linewidth=2.5, label='Nominal Trajectory')
    ax.plot(x_v, y_v, color='#2563EB', linewidth=2, label='Actual (Simulation)')
    ax.scatter([origin_x], [origin_y], c='#16A34A', s=80, marker='o', zorder=5, edgecolors='k', label='Start')
    ax.scatter([x_v[-1]], [y_v[-1]], c='#DC2626', s=80, marker='s', zorder=5, edgecolors='k', label='End')
    
    ax.set_aspect('equal')
    ax.set_xlabel('East [m]')
    ax.set_ylabel('North [m]')
    ax.set_title(f'Nominal vs. Simulation Trajectory ({traj_type})')
    ax.legend(loc='best')
    
    # Panel 2: Tracking Error Over Time
    ax = axes[1]
    ax.plot(t_eval, err_norm, color='#DC2626', linewidth=1.8, label='Tracking Error')
    ax.axhline(avg_error, color='gray', linestyle='--', label=f'Avg Error: {avg_error:.2f} m')
    ax.set_xlabel('Time since start [s]')
    ax.set_ylabel('Position Error [m]')
    ax.set_title(f'Trajectory Tracking Error (Max: {max_error:.2f} m)')
    ax.legend(loc='best')
    
    plt.tight_layout()
    save_path = "fig_traj_comparison.pdf"
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")
    plt.show()


def _plot_3d(t_eval, nom_x, nom_y, nom_z, x_v, y_v, z_v,
             origin_x, origin_y, origin_z, start_idx,
             err_norm, err_lateral, err_z, avg_error, max_error, traj_type):
    """3-panel plot for climbing trajectories: 3D view + lateral error + altitude error."""
    fig = plt.figure(figsize=(18, 6))
    
    # ── Panel 1: 3D trajectory ──
    ax3d = fig.add_subplot(131, projection='3d')
    
    ax3d.plot(nom_x, nom_y, nom_z, color='gray', linestyle='--', linewidth=2.5,
              label='Nominal', zorder=2)
    ax3d.plot(x_v, y_v, z_v, color='#2563EB', linewidth=2,
              label='Actual', zorder=3)
    
    # Start / end markers
    ax3d.scatter([origin_x], [origin_y], [origin_z],
                 c='#16A34A', s=100, marker='o', edgecolors='k', zorder=5, label='Start')
    ax3d.scatter([x_v[-1]], [y_v[-1]], [z_v[-1]],
                 c='#DC2626', s=100, marker='s', edgecolors='k', zorder=5, label='End')
    
    ax3d.set_xlabel('East [m]')
    ax3d.set_ylabel('North [m]')
    ax3d.set_zlabel('Altitude [m]')
    ax3d.set_title(f'3D Trajectory ({traj_type})')
    ax3d.legend(loc='upper left', fontsize=9)
    
    # Equal aspect ratio for 3D
    all_x = np.concatenate([nom_x, x_v])
    all_y = np.concatenate([nom_y, y_v])
    all_z = np.concatenate([nom_z, z_v])
    max_range = max(all_x.max() - all_x.min(),
                    all_y.max() - all_y.min(),
                    all_z.max() - all_z.min()) / 2.0
    mid_x = (all_x.max() + all_x.min()) / 2.0
    mid_y = (all_y.max() + all_y.min()) / 2.0
    mid_z = (all_z.max() + all_z.min()) / 2.0
    ax3d.set_xlim(mid_x - max_range, mid_x + max_range)
    ax3d.set_ylim(mid_y - max_range, mid_y + max_range)
    ax3d.set_zlim(mid_z - max_range, mid_z + max_range)
    
    # ── Panel 2: Lateral error ──
    ax_lat = fig.add_subplot(132)
    ax_lat.plot(t_eval, err_lateral, color='#7C3AED', linewidth=1.8, label='Lateral Error')
    ax_lat.axhline(np.mean(err_lateral), color='gray', linestyle='--',
                   label=f'Avg: {np.mean(err_lateral):.2f} m')
    ax_lat.set_xlabel('Time since start [s]')
    ax_lat.set_ylabel('Lateral Error [m]')
    ax_lat.set_title(f'Lateral Deviation (Max: {np.max(err_lateral):.2f} m)')
    ax_lat.legend(loc='best')
    
    # ── Panel 3: Altitude error ──
    ax_alt = fig.add_subplot(133)
    ax_alt.plot(t_eval, err_z, color='#DC2626', linewidth=1.8, label='Altitude Error (Δz)')
    ax_alt.axhline(0, color='k', linewidth=0.5)
    ax_alt.axhline(np.mean(err_z), color='gray', linestyle='--',
                   label=f'Avg: {np.mean(err_z):.2f} m')
    ax_alt.set_xlabel('Time since start [s]')
    ax_alt.set_ylabel('Altitude Error [m]')
    ax_alt.set_title(f'Altitude Tracking Error')
    ax_alt.legend(loc='best')
    
    plt.tight_layout()
    save_path = "fig_traj_comparison.pdf"
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")
    plt.show()


if __name__ == "__main__":
    main()
