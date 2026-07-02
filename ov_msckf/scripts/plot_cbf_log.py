#!/usr/bin/env python3
"""
CBF Log Plotter — Publication-quality figures from CBF safety filter logs.
==========================================================================

Reads the latest (or specified) CSV log produced by cbf_safety_filter_node.py
and generates four journal-article-quality figures:

  1. Log-det metric vs. time with safety threshold
  2. 2×3 velocity comparison grid (v_nom vs v_cbf, per axis + norm)
  3. 3D trajectory with v_nom / v_cbf vector quivers
  4. Histogram of SLAM feature counts

Usage:
    python3 plot_cbf_log.py                           # latest log in ~/cbf_logs
    python3 plot_cbf_log.py --log path/to/log.csv     # specific log file
    python3 plot_cbf_log.py --arrow_freq 2.0           # velocity arrows every 0.5s
    python3 plot_cbf_log.py --save                     # save figures as PDF
    python3 plot_cbf_log.py --save --format png        # save as PNG instead
"""

import argparse
import glob
import os
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3D projection)
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Default path to drone STL mesh (iris quadrotor)
_DEFAULT_STL = os.path.join(
    os.path.expanduser("~/ros2_ws/src/ardupilot_gazebo/models/"
                       "iris_with_standoffs/meshes/iris_collision.stl"))

# ──────────────────────────────────────────────────────────────────────────────
# Global style: Times New Roman, journal-grade
# ──────────────────────────────────────────────────────────────────────────────
def setup_style():
    """Configure matplotlib for publication-quality output."""
    # Try Times New Roman; fall back to serif if unavailable
    try:
        mpl.font_manager.findfont("Times New Roman", fallback_to_default=False)
        font_family = "Times New Roman"
    except Exception:
        font_family = "serif"

    plt.rcParams.update({
        # Font
        "font.family": font_family,
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
        # Lines
        "lines.linewidth": 1.4,
        "lines.markersize": 4,
        # Axes
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "grid.alpha": 0.35,
        "grid.linewidth": 0.5,
        # Figure
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        # Ticks
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,
    })


# ──────────────────────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────────────────────
def load_log(path: str) -> dict:
    """Load a CBF CSV log file into a dict of numpy arrays."""
    data = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding="utf-8")
    return {name: np.array(data[name], dtype=float) for name in data.dtype.names}


def find_latest_log(log_dir: str = None) -> str:
    """Return the path to the most recently modified CBF CSV log.

    Searches scripts/logs/flight_*/cbf_log_*.csv first,
    then falls back to ~/cbf_logs/ for legacy logs.
    """
    files = []
    # Primary: flight log directories next to this script
    scripts_dir = os.path.dirname(os.path.abspath(__file__))
    flight_pattern = os.path.join(scripts_dir, "logs", "flight_*", "cbf_log_*.csv")
    files += glob.glob(flight_pattern)
    # Fallback: legacy ~/cbf_logs/
    legacy_dir = os.path.expanduser(log_dir or "~/cbf_logs")
    files += glob.glob(os.path.join(legacy_dir, "cbf_log_*.csv"))
    files = sorted(files, key=os.path.getmtime)
    if not files:
        print(f"No CBF log files found in {flight_pattern} or {legacy_dir}")
        sys.exit(1)
    return files[-1]


# ──────────────────────────────────────────────────────────────────────────────
# Color palette (curated for legibility on white background)
# ──────────────────────────────────────────────────────────────────────────────
C = {
    "blue":    "#2563EB",
    "red":     "#DC2626",
    "green":   "#16A34A",
    "orange":  "#EA580C",
    "purple":  "#7C3AED",
    "gray":    "#6B7280",
    "cyan":    "#0891B2",
    "black":   "#1F2937",
}


# ──────────────────────────────────────────────────────────────────────────────
# Figure 1: Log-det vs Time
# ──────────────────────────────────────────────────────────────────────────────
def fig_logdet(d, save=False, fmt="pdf"):
    fig, ax = plt.subplots(figsize=(7, 3.2))

    ax.plot(d["timestamp"], d["logdet"],
            color=C["blue"], linewidth=1.6, label=r"$\bar{\ell}(x)$  (log-det)")

    # Constant threshold
    thresh = d["logdet_threshold"][0]
    ax.axhline(thresh, color=C["red"], linestyle="--", linewidth=1.4,
               label=rf"$h_{{\min}} = {thresh:.2f}$")

    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Log-det Metric")
    ax.set_title("Observability Metric (Log-det) vs. Time")
    ax.legend(loc="best", framealpha=0.9, edgecolor="none")
    fig.tight_layout()

    if save:
        fig.savefig(f"fig_logdet.{fmt}")
        print(f"  Saved fig_logdet.{fmt}")
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# Figure 2: Velocity Comparison Grid (2×3)
# ──────────────────────────────────────────────────────────────────────────────
def fig_velocity(d, save=False, fmt="pdf"):
    fig = plt.figure(figsize=(10, 5.5))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.35)

    t = d["timestamp"]

    # Row 1: per-axis comparisons
    axes_labels = [
        ("v_nom_x", "v_cbf_x", r"$v_x$", "X-axis Velocity [m/s]"),
        ("v_nom_y", "v_cbf_y", r"$v_y$", "Y-axis Velocity [m/s]"),
        ("v_nom_z", "v_cbf_z", r"$v_z$", "Z-axis Velocity [m/s]"),
    ]
    for col, (nom_key, cbf_key, title, ylabel) in enumerate(axes_labels):
        ax = fig.add_subplot(gs[0, col])
        ax.plot(t, d[nom_key], color=C["blue"], linewidth=1.4,
                label=r"$v_{\mathrm{nom}}$")
        ax.plot(t, d[cbf_key], color=C["red"], linewidth=1.4,
                label=r"$v_{\mathrm{cbf}}$", alpha=0.85)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        if col == 0:
            ax.legend(loc="best", framealpha=0.9, edgecolor="none")
        ax.set_xlabel("Time [s]")

    # Row 2: norm comparison — spanning all 3 columns
    ax_norm = fig.add_subplot(gs[1, :])
    ax_norm.plot(t, d["v_nom_norm"], color=C["blue"], linewidth=1.6,
                 label=r"$\| v_{\mathrm{nom}} \|$")
    ax_norm.plot(t, d["v_cbf_norm"], color=C["red"], linewidth=1.6,
                 label=r"$\| v_{\mathrm{cbf}} \|$", alpha=0.85)
    ax_norm.set_xlabel("Time [s]")
    ax_norm.set_ylabel("Velocity Norm [m/s]")
    ax_norm.set_title("Velocity Norm Comparison")
    ax_norm.legend(loc="best", framealpha=0.9, edgecolor="none")

    fig.suptitle("Nominal vs. CBF-Corrected Velocity", fontsize=14, y=1.01)
    fig.tight_layout()

    if save:
        fig.savefig(f"fig_velocity.{fmt}")
        print(f"  Saved fig_velocity.{fmt}")
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# STL mesh loader (lightweight, binary/ASCII)
# ──────────────────────────────────────────────────────────────────────────────
def _load_stl(path):
    """Load an STL file and return (N, 3, 3) array of triangle vertices."""
    try:
        from stl import mesh as stl_mesh  # numpy-stl
        m = stl_mesh.Mesh.from_file(path)
        return m.vectors  # (N, 3, 3)
    except ImportError:
        print("  [INFO] numpy-stl not installed — drone mesh will not be shown.")
        print("         Install with: pip install numpy-stl")
        return None
    except Exception as e:
        print(f"  [WARN] Could not load STL: {e}")
        return None


def _add_drone_mesh(ax, stl_verts, pos, scale=1.0, color='#6B7280', alpha=0.45):
    """Add a drone STL mesh at position `pos` to the 3D axis."""
    if stl_verts is None:
        return
    # Center the mesh at origin, then scale and translate
    centroid = stl_verts.reshape(-1, 3).mean(axis=0)
    verts = (stl_verts - centroid) * scale + np.array(pos)
    poly = Poly3DCollection(verts, alpha=alpha, facecolor=color,
                            edgecolor='#9CA3AF', linewidth=0.15)
    ax.add_collection3d(poly)


# ──────────────────────────────────────────────────────────────────────────────
# Shared trajectory data preparation
# ──────────────────────────────────────────────────────────────────────────────
def _prepare_traj_data(d, arrow_freq, stl_path):
    """Compute all shared trajectory/arrow/mesh data for both figures."""
    x, y, z = d["drone_x"], d["drone_y"], d["drone_z"]
    t = d["timestamp"]

    valid = ~(np.isnan(x) | np.isnan(y) | np.isnan(z))
    if valid.sum() < 2:
        return None

    x, y, z = x[valid], y[valid], z[valid]
    t_valid = t[valid]

    # Load drone STL
    mesh_path = stl_path or _DEFAULT_STL
    stl_verts = _load_stl(mesh_path) if os.path.isfile(mesh_path) else None
    traj_range = max(np.ptp(x), np.ptp(y), np.ptp(z), 1e-6)
    mesh_scale = traj_range * 0.05  # 5% of trajectory span (larger drones)

    # Arrow sample indices
    dt_mean = np.mean(np.diff(t_valid)) if len(t_valid) > 1 else 1.0
    step = max(1, int(round(1.0 / (arrow_freq * dt_mean))))
    idx = np.arange(0, len(t_valid), step)
    valid_indices = np.where(valid)[0]
    orig_idx = valid_indices[idx]

    vn_x, vn_y, vn_z = d["v_nom_x"][orig_idx], d["v_nom_y"][orig_idx], d["v_nom_z"][orig_idx]
    vc_x, vc_y, vc_z = d["v_cbf_x"][orig_idx], d["v_cbf_y"][orig_idx], d["v_cbf_z"][orig_idx]
    vc_norm = d["v_cbf_norm"][orig_idx]
    ar_x, ar_y, ar_z = x[idx], y[idx], z[idx]

    max_vel = max(np.max(np.abs(vn_x) + np.abs(vn_y) + np.abs(vn_z)),
                  np.max(np.abs(vc_x) + np.abs(vc_y) + np.abs(vc_z)), 1e-6)
    scale = 0.08 * traj_range / max_vel

    cmap = plt.cm.YlOrRd
    cnorm = Normalize(vmin=vc_norm.min(), vmax=max(vc_norm.max(), 1e-6))
    vc_colors = cmap(cnorm(vc_norm))

    return dict(
        x=x, y=y, z=z, t_valid=t_valid, idx=idx, stl_verts=stl_verts,
        mesh_scale=mesh_scale, traj_range=traj_range,
        vn_x=vn_x, vn_y=vn_y, vn_z=vn_z,
        vc_x=vc_x, vc_y=vc_y, vc_z=vc_z, vc_norm=vc_norm,
        ar_x=ar_x, ar_y=ar_y, ar_z=ar_z,
        scale=scale, cmap=cmap, cnorm=cnorm, vc_colors=vc_colors,
    )


def _draw_on_ax(ax, td, title, xlabel, ylabel, elev, azim, is_proj):
    """Draw trajectory, drone meshes, and velocity arrows on a 3D axis."""
    x, y, z = td["x"], td["y"], td["z"]
    ar_x, ar_y, ar_z = td["ar_x"], td["ar_y"], td["ar_z"]
    vn_x, vn_y, vn_z = td["vn_x"], td["vn_y"], td["vn_z"]
    vc_x, vc_y, vc_z = td["vc_x"], td["vc_y"], td["vc_z"]
    scale = td["scale"]
    vc_colors = td["vc_colors"]
    idx = td["idx"]

    # Trajectory line
    ax.plot(x, y, z, color=C["black"], linewidth=1.4, alpha=0.7, label="Trajectory")
    ax.scatter(x[0], y[0], z[0], c=C["green"], s=60, marker="o",
               zorder=5, label="Start", edgecolors="k", linewidths=0.5)
    ax.scatter(x[-1], y[-1], z[-1], c=C["red"], s=60, marker="s",
               zorder=5, label="End", edgecolors="k", linewidths=0.5)

    # v_nom arrows (fixed blue) — mounted on drone positions
    ax.quiver(ar_x, ar_y, ar_z,
              vn_x * scale, vn_y * scale, vn_z * scale,
              color=C["blue"], alpha=0.75, linewidth=2.0,
              arrow_length_ratio=0.12)

    # v_cbf arrows (colormap) — mounted on drone positions
    for i in range(len(idx)):
        ax.quiver(ar_x[i], ar_y[i], ar_z[i],
                  vc_x[i] * scale, vc_y[i] * scale, vc_z[i] * scale,
                  color=vc_colors[i], alpha=0.85, linewidth=2.0,
                  arrow_length_ratio=0.12)

    # Remove grids and panes
    ax.grid(False)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('none')
    ax.yaxis.pane.set_edgecolor('none')
    ax.zaxis.pane.set_edgecolor('none')

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=12)
    ax.view_init(elev=elev, azim=azim)

    if is_proj:
        if elev == 90:        # XY top-down
            ax.set_zlabel("")
            ax.zaxis.set_ticklabels([])
        elif azim == -90:     # XZ front
            ax.set_zlabel("Up [m]")
        else:                 # YZ side
            ax.set_zlabel("Up [m]")


# ──────────────────────────────────────────────────────────────────────────────
# Figure 3a: Orthographic Views (1×3)
# ──────────────────────────────────────────────────────────────────────────────
def fig_traj_orthographic(d, arrow_freq=0.5, stl_path=None, save=False, fmt="pdf"):
    """1×3 figure with XY, XZ, YZ orthographic views."""
    td = _prepare_traj_data(d, arrow_freq, stl_path)
    if td is None:
        print("  [WARN] Insufficient drone position data — skipping orthographic views.")
        return None

    from matplotlib.lines import Line2D

    views = [
        ("XY View (Top-down)",  "East [m]",  "North [m]", 90, -90, True),
        ("XZ View (Front)",     "East [m]",  "Up [m]",    0,  -90, True),
        ("YZ View (Side)",      "North [m]", "Up [m]",    0,    0, True),
    ]

    fig = plt.figure(figsize=(18, 6))
    axes = []

    for col, (title, xl, yl, elev, azim, is_proj) in enumerate(views):
        ax = fig.add_subplot(1, 3, col + 1, projection="3d")
        _draw_on_ax(ax, td, title, xl, yl, elev, azim, is_proj)
        axes.append(ax)

    # Shared legend on first axis
    cmap, cnorm = td["cmap"], td["cnorm"]
    legend_handles = [
        Line2D([0], [0], color=C["black"], linewidth=1.4, label="Trajectory"),
        Line2D([0], [0], color=C["green"], marker="o", linestyle="",
               markersize=7, label="Start"),
        Line2D([0], [0], color=C["red"], marker="s", linestyle="",
               markersize=7, label="End"),
        Line2D([0], [0], color=C["blue"], linewidth=2.5,
               label=r"$v_{\mathrm{nom}}$"),
        Line2D([0], [0], color=cmap(0.6), linewidth=2.5,
               label=r"$v_{\mathrm{cbf}}$"),
    ]
    axes[0].legend(handles=legend_handles, loc="upper left",
                   framealpha=0.9, edgecolor="none", fontsize=9)

    # Colorbar
    sm = ScalarMappable(cmap=cmap, norm=cnorm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, shrink=0.6, pad=0.04, aspect=20)
    cbar.set_label(r"$\| v_{\mathrm{cbf}} \|$ [m/s]")

    fig.suptitle("Drone Trajectory — Orthographic Views", fontsize=14, y=0.98)
    fig.tight_layout()

    if save:
        fig.savefig(f"fig_traj_ortho.{fmt}")
        print(f"  Saved fig_traj_ortho.{fmt}")
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# Figure 3b: Isometric View (single large plot)
# ──────────────────────────────────────────────────────────────────────────────
def fig_traj_isometric(d, arrow_freq=0.5, stl_path=None, save=False, fmt="pdf"):
    """Single large isometric 3D trajectory view."""
    td = _prepare_traj_data(d, arrow_freq, stl_path)
    if td is None:
        print("  [WARN] Insufficient drone position data — skipping isometric view.")
        return None

    from matplotlib.lines import Line2D

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")

    _draw_on_ax(ax, td, "Isometric View", "East [m]", "North [m]",
                elev=25, azim=-55, is_proj=False)
    ax.set_zlabel("Up [m]")

    # Legend
    cmap, cnorm = td["cmap"], td["cnorm"]
    legend_handles = [
        Line2D([0], [0], color=C["black"], linewidth=1.4, label="Trajectory"),
        Line2D([0], [0], color=C["green"], marker="o", linestyle="",
               markersize=8, label="Start"),
        Line2D([0], [0], color=C["red"], marker="s", linestyle="",
               markersize=8, label="End"),
        Line2D([0], [0], color=C["blue"], linewidth=2.5,
               label=r"$v_{\mathrm{nom}}$"),
        Line2D([0], [0], color=cmap(0.6), linewidth=2.5,
               label=r"$v_{\mathrm{cbf}}$"),
    ]
    ax.legend(handles=legend_handles, loc="upper left",
              framealpha=0.9, edgecolor="none", fontsize=10)

    # Colorbar
    sm = ScalarMappable(cmap=cmap, norm=cnorm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.55, pad=0.08, aspect=22)
    cbar.set_label(r"$\| v_{\mathrm{cbf}} \|$ [m/s]")

    fig.suptitle("Drone Trajectory with Velocity Vectors", fontsize=14, y=0.96)
    fig.tight_layout()

    if save:
        fig.savefig(f"fig_traj_iso.{fmt}")
        print(f"  Saved fig_traj_iso.{fmt}")
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# Figure 4: SLAM Feature Count — Scatter Plot vs. Time
# ──────────────────────────────────────────────────────────────────────────────
def fig_feature_scatter(d, save=False, fmt="pdf"):
    t = d["timestamp"]
    nf = d["num_slam_features"]

    # Remove NaN
    mask = ~np.isnan(nf)
    t_f, nf_f = t[mask], nf[mask]
    if len(nf_f) == 0:
        print("  [WARN] No SLAM feature data — skipping scatter plot.")
        return None

    fig, ax = plt.subplots(figsize=(7, 3.5))

    ax.scatter(t_f, nf_f, s=12, color=C["cyan"], alpha=0.6,
              edgecolors="none", label="SLAM Features", zorder=2)

    mean_val = np.mean(nf_f)
    ax.axhline(mean_val, color=C["red"], linestyle="--", linewidth=1.4,
               label=rf"Mean = {mean_val:.1f}", zorder=3)

    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Number of SLAM Features")
    ax.set_title("SLAM Feature Count Over Time")
    ax.legend(loc="best", framealpha=0.9, edgecolor="none")
    fig.tight_layout()

    if save:
        fig.savefig(f"fig_feature_scatter.{fmt}")
        print(f"  Saved fig_feature_scatter.{fmt}")
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Generate publication-quality CBF log plots.")
    parser.add_argument("--log", type=str, default=None,
                        help="Path to a specific CSV log file. "
                             "Default: latest in ~/cbf_logs.")
    parser.add_argument("--arrow_freq", type=float, default=0.5,
                        help="Frequency (Hz) of velocity arrows in 3D plot. "
                             "Default: 0.5")
    parser.add_argument("--stl", type=str, default=None,
                        help="Path to drone STL mesh file. "
                             "Default: iris_collision.stl from ardupilot_gazebo.")
    parser.add_argument("--save", action="store_true",
                        help="Save figures to current directory.")
    parser.add_argument("--format", type=str, default="pdf",
                        choices=["pdf", "png", "svg", "eps"],
                        help="Output format when --save is used. Default: pdf")
    args = parser.parse_args()

    # Resolve log path
    log_path = args.log if args.log else find_latest_log()
    print(f"Reading log: {log_path}")
    d = load_log(log_path)
    print(f"  {len(d['timestamp'])} samples, "
          f"t=[{d['timestamp'][0]:.1f}, {d['timestamp'][-1]:.1f}] s")

    setup_style()

    # Generate all figures
    fig_logdet(d, save=args.save, fmt=args.format)
    fig_velocity(d, save=args.save, fmt=args.format)
    fig_traj_orthographic(d, arrow_freq=args.arrow_freq, stl_path=args.stl,
                          save=args.save, fmt=args.format)
    fig_traj_isometric(d, arrow_freq=args.arrow_freq, stl_path=args.stl,
                       save=args.save, fmt=args.format)
    fig_feature_scatter(d, save=args.save, fmt=args.format)

    if not args.save:
        plt.show()


if __name__ == "__main__":
    main()
