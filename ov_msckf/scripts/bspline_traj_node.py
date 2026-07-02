#!/usr/bin/env python3
"""
B-Spline Trajectory Generator for CBF Safety Filter Testing
============================================================

Generates a smooth closed-loop trajectory using cubic B-splines through
user-defined waypoints. Publishes body-frame velocity commands for
ArduPilot/MAVROS via the CBF safety filter pipeline.

Architecture:
  1. Waits for the first OpenVINS odometry to lock the start pose.
  2. Builds a periodic cubic B-spline through square waypoints (with
     midpoints for shape fidelity).
  3. At each control tick:
       - Evaluate B-spline → desired position p_d and velocity v_d
       - Proportional feedback: v_cmd = v_d + Kp · (p_d − p_cur)
       - Rotate v_cmd to body frame using current orientation
       - Publish TwistStamped to the nominal velocity topic

The CBF safety filter intercepts this nominal command and projects it
onto the safe set before forwarding to ArduPilot.
"""

import numpy as np
from scipy.interpolate import make_interp_spline
from scipy.spatial.transform import Rotation

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from nav_msgs.msg import Odometry
from geometry_msgs.msg import TwistStamped
from std_msgs.msg import String
from mavros_msgs.msg import State as MavrosState


# ══════════════════════════════════════════════════════════════════════════
# Trajectories (pure math, no ROS dependency)
# ══════════════════════════════════════════════════════════════════════════
class StraightTrajectory:
    """Straight line trajectory from origin along the +X axis.
    
    Travels a given distance at a constant speed, then stops (v=0).
    """
    def __init__(self, distance: float, speed: float):
        self.distance = distance
        self.speed = speed
        self.duration = distance / speed
        self.period = self.duration * 2  # Plotting compatibility
        self.total_length = distance

    def evaluate(self, t: float):
        """Return (pos_3d, vel_3d) at wall-clock time *t* (seconds)."""
        if t < self.duration:
            pos = np.array([self.speed * t, 0.0, 0.0])
            vel = np.array([self.speed, 0.0, 0.0])
        else:
            pos = np.array([self.distance, 0.0, 0.0])
            vel = np.zeros(3)
        return pos, vel

class ClimbingStraightTrajectory:
    """Straight line with simultaneous climb along the body Z axis.

    Moves forward at *speed* along the drone's heading (+X in trajectory frame)
    while climbing at *climb_speed* along +Z.  No lateral component.
    After covering *distance* horizontally the drone stops completely (v=0).
    """
    def __init__(self, distance: float, speed: float, climb_speed: float = 1.0):
        self.distance = distance
        self.speed = speed
        self.climb_speed = climb_speed
        self.duration = distance / speed
        self.climb_total = climb_speed * self.duration
        self.period = self.duration * 2   # plotting compatibility
        self.total_length = distance

    def evaluate(self, t: float):
        """Return (pos_3d, vel_3d) at wall-clock time *t*."""
        if t < self.duration:
            pos = np.array([self.speed * t, 0.0, self.climb_speed * t])
            vel = np.array([self.speed, 0.0, self.climb_speed])
        else:
            pos = np.array([self.distance, 0.0, self.climb_total])
            vel = np.zeros(3)
        return pos, vel

class CircleTrajectory:
    """Continuous circular trajectory starting from the origin.
    
    Initially moves along the +X axis, curves towards +Y.
    """
    def __init__(self, radius: float, speed: float):
        self.radius = radius
        self.speed = speed
        self.omega = speed / radius
        self.period = 2.0 * np.pi * radius / speed
        self.total_length = 2.0 * np.pi * radius

    def evaluate(self, t: float):
        """Return (pos_3d, vel_3d) at wall-clock time *t* (seconds)."""
        pos = np.array([
            self.radius * np.sin(self.omega * t),
            self.radius * (1.0 - np.cos(self.omega * t)),
            0.0
        ])
        vel = np.array([
            self.speed * np.cos(self.omega * t),
            self.speed * np.sin(self.omega * t),
            0.0
        ])
        return pos, vel

class BSplineTrajectory:
    """Smooth closed-loop trajectory from waypoints using periodic cubic B-splines.

    The trajectory is parameterised by normalised arc-length s ∈ [0, 1),
    which advances at a constant desired speed so that ds/dt = speed / L_total.
    """

    def __init__(self, waypoints_2d: np.ndarray, speed: float = 3.0):
        """
        Args:
            waypoints_2d: (N, 2) array of [x, y] waypoints in a local frame.
                          First and last points MUST be identical (closed loop).
            speed:        Desired traversal speed (m/s).
        """
        assert np.allclose(waypoints_2d[0], waypoints_2d[-1]), \
            "Waypoints must form a closed loop (first == last)."

        self.waypoints = waypoints_2d
        self.speed = speed

        # ── Parameterise by cumulative chord length ──
        diffs = np.diff(waypoints_2d, axis=0)
        chord_lengths = np.linalg.norm(diffs, axis=1)
        s = np.zeros(len(waypoints_2d))
        s[1:] = np.cumsum(chord_lengths)
        self.total_length = s[-1]
        s_norm = s / self.total_length           # ∈ [0, 1]

        # ── Periodic cubic B-spline (C² at the join) ──
        self._spl_x = make_interp_spline(
            s_norm, waypoints_2d[:, 0], k=3, bc_type="periodic")
        self._spl_y = make_interp_spline(
            s_norm, waypoints_2d[:, 1], k=3, bc_type="periodic")

        # Period of one full loop
        self.period = self.total_length / self.speed
        
        # Calculate offset at t=0 so evaluation starts at (0,0)
        self.start_offset = np.array([float(self._spl_x(0.0)), float(self._spl_y(0.0)), 0.0])

    # ------------------------------------------------------------------
    def evaluate(self, t: float):
        """Return (pos_3d, vel_3d) at wall-clock time *t* (seconds).

        pos is [x, y, 0] in the local waypoint frame.
        vel is [vx, vy, 0] in the same frame.
        After one full loop the drone stops at the starting position.
        """
        if t >= self.period:
            return np.zeros(3), np.zeros(3)

        s = (t / self.period) % 1.0           # normalised parameter
        dsdt = 1.0 / self.period              # ds/dt = speed / L

        px = float(self._spl_x(s)) - self.start_offset[0]
        py = float(self._spl_y(s)) - self.start_offset[1]
        vx = float(self._spl_x(s, nu=1)) * dsdt
        vy = float(self._spl_y(s, nu=1)) * dsdt

        return np.array([px, py, 0.0]), np.array([vx, vy, 0.0])

    # ------------------------------------------------------------------
    def plot(self, n_samples: int = 500, save_path: str = None):
        """Visualise the B-spline trajectory: 2D path + velocity profile.

        Args:
            n_samples: number of evaluation points along the loop.
            save_path: if given, save figure to this path instead of showing.
        """
        import matplotlib.pyplot as plt

        t_vals = np.linspace(0, self.period, n_samples, endpoint=False)
        positions = np.array([self.evaluate(t)[0] for t in t_vals])
        velocities = np.array([self.evaluate(t)[1] for t in t_vals])
        speed = np.linalg.norm(velocities[:, :2], axis=1)

        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        fig.suptitle("B-Spline Reference Trajectory", fontsize=14, fontweight="bold")

        # ── Panel 1: 2D path ──
        ax = axes[0]
        ax.plot(positions[:, 0], positions[:, 1], 'b-', lw=2, label='B-spline path')
        # Waypoints must be offset for plotting alignment
        waypoints_offset = self.waypoints - self.start_offset[:2]
        ax.plot(waypoints_offset[:, 0], waypoints_offset[:, 1],
                'ro', ms=8, zorder=5, label='Waypoints')
        # Mark corners (every other point since midpoints are interleaved)
        corners = waypoints_offset[::2]
        for i, c in enumerate(corners[:-1]):  # skip duplicate closing point
            ax.annotate(f'WP{i}', (c[0], c[1]), textcoords='offset points',
                        xytext=(8, 8), fontsize=9, fontweight='bold')
        # Start arrow
        ax.annotate('START', xy=positions[0, :2], fontsize=10, color='green',
                    fontweight='bold', xytext=(10, -15), textcoords='offset points',
                    arrowprops=dict(arrowstyle='->', color='green'))
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title('Path (XY plane)')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)

        # ── Panel 2: velocity components ──
        ax = axes[1]
        ax.plot(t_vals, velocities[:, 0], label='v_x', lw=1.5)
        ax.plot(t_vals, velocities[:, 1], label='v_y', lw=1.5)
        ax.plot(t_vals, speed, 'k--', label='||v||', lw=1.5)
        ax.axhline(self.speed, color='gray', ls=':', alpha=0.5, label=f'target={self.speed:.1f}')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Velocity (m/s)')
        ax.set_title('Velocity Profile')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)

        # ── Panel 3: speed along path ──
        ax = axes[2]
        sc = ax.scatter(positions[:, 0], positions[:, 1],
                        c=speed, cmap='coolwarm', s=8, zorder=3)
        plt.colorbar(sc, ax=ax, label='Speed (m/s)')
        ax.plot(waypoints_offset[:, 0], waypoints_offset[:, 1],
                'ko', ms=5, zorder=5)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title('Speed along Path')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved to {save_path}")
        else:
            plt.show()

    # ------------------------------------------------------------------
    @staticmethod
    def make_square(side_length: float, speed: float = 3.0):
        """Factory: square trajectory with midpoints for shape fidelity."""
        L = side_length
        waypoints = np.array([
            [0.0,  0.0],        # WP0 — corner
            [L/2,  0.0],        # midpoint side 0→1
            [L,    0.0],        # WP1 — corner
            [L,    L/2],        # midpoint side 1→2
            [L,    L],          # WP2 — corner
            [L/2,  L],          # midpoint side 2→3
            [0.0,  L],          # WP3 — corner
            [0.0,  L/2],        # midpoint side 3→0
            [0.0,  0.0],        # close loop
        ])
        return BSplineTrajectory(waypoints, speed)


# ══════════════════════════════════════════════════════════════════════════
# ROS 2 node
# ══════════════════════════════════════════════════════════════════════════
class BSplineTrajectoryNode(Node):
    """ROS 2 node: B-spline trajectory generator with body-frame velocity output."""

    def __init__(self):
        super().__init__("bspline_traj")

        # ── Flight file logger (mirrors all get_logger() output to traj.log) ──
        from flight_logger import setup_flight_logger
        setup_flight_logger(self, "traj.log")

        # ── Parameters ──
        self.declare_parameter("traj_enabled",      False)
        self.declare_parameter("traj_type",         "square")
        self.declare_parameter("side_length",       20.0)    # m
        self.declare_parameter("radius",            30.0)    # m
        self.declare_parameter("speed",             3.0)     # m/s
        self.declare_parameter("climb_speed",       1.0)     # m/s vertical
        self.declare_parameter("kp",                0.5)     # position feedback gain
        self.declare_parameter("rate",              20.0)    # Hz
        self.declare_parameter("hold_altitude",     True)    # no z correction
        self.declare_parameter("topic_pub_cmd_vel", "cmd_vel_nom")
        self.declare_parameter("topic_sub_odom",    "odomimu_enu")

        self.enabled     = self.get_parameter("traj_enabled").value
        self.traj_type   = self.get_parameter("traj_type").value
        side             = self.get_parameter("side_length").value
        radius           = self.get_parameter("radius").value
        speed            = self.get_parameter("speed").value
        climb_speed      = self.get_parameter("climb_speed").value
        self.kp          = self.get_parameter("kp").value
        rate             = self.get_parameter("rate").value
        self.hold_z      = self.get_parameter("hold_altitude").value
        topic_pub        = self.get_parameter("topic_pub_cmd_vel").value
        topic_sub_odom   = self.get_parameter("topic_sub_odom").value

        # ── Build trajectory ──
        if self.traj_type == "straight":
            self.traj = StraightTrajectory(side, speed)
        elif self.traj_type == "climb":
            self.traj = ClimbingStraightTrajectory(side, speed, climb_speed)
            self.hold_z = False  # must pass Z velocity through
        elif self.traj_type == "circle":
            self.traj = CircleTrajectory(radius, speed)
        else:
            self.traj = BSplineTrajectory.make_square(side, speed)

        # ── State ──
        self.origin      = None          # starting position (global frame)
        self.current_pos = None          # latest position (global frame)
        self.R_GtoI      = np.eye(3)     # rotation global → body
        self.t_start     = None          # wall-clock time of trajectory start
        self.drone_mode  = ""            # current flight mode from MAVROS

        # ── QoS ──
        qos_be = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST, depth=5)

        # ── Subscriber ──
        self.sub_odom = self.create_subscription(
            Odometry, topic_sub_odom, self._cb_odom, qos_be)
        self.sub_mavros_state = self.create_subscription(
            MavrosState, '/mavros/state', self._cb_mavros_state, 10)

        # ── Publishers ──
        self.pub_cmd = self.create_publisher(TwistStamped, topic_pub, 10)
        self.pub_status = self.create_publisher(String, "traj/status", 10)

        # ── Control timer ──
        self.timer = self.create_timer(1.0 / rate, self._timer_cb)

        self.get_logger().info(
            f"Trajectory Node: type={self.traj_type} dist/side={side:.1f}m rad={radius:.1f}m "
            f"speed={speed:.1f}m/s climb={climb_speed:.1f}m/s "
            f"Kp={self.kp:.2f} hold_z={self.hold_z} enabled={self.enabled}"
        )

    # ------------------------------------------------------------------
    def _cb_odom(self, msg: Odometry):
        """Extract current position and orientation from OpenVINS odometry."""
        p = msg.pose.pose.position
        self.current_pos = np.array([p.x, p.y, p.z])

        q = msg.pose.pose.orientation
        qvec = [q.x, q.y, q.z, q.w]
        if np.any(np.isnan(qvec)) or (q.x**2 + q.y**2 + q.z**2 + q.w**2) < 1e-10:
            return  # uninitialised quaternion
        rot = Rotation.from_quat(qvec)
        self.R_GtoI = rot.as_matrix().T          # R_ItoG → R_GtoI

    # ------------------------------------------------------------------
    def _cb_mavros_state(self, msg: MavrosState):
        """Track current flight mode from /mavros/state."""
        prev = self.drone_mode
        self.drone_mode = msg.mode
        if prev != "GUIDED" and msg.mode == "GUIDED":
            self.get_logger().info("Drone entered GUIDED mode – trajectory generation armed")

    # ------------------------------------------------------------------
    def _timer_cb(self):
        """Evaluate trajectory and publish body-frame velocity command."""
        if not self.enabled or self.current_pos is None or self.drone_mode != "GUIDED":
            return

        # Start clock on first enabled tick
        if self.t_start is None:
            self.t_start = self.get_clock().now().nanoseconds * 1e-9
            # Lock origin and rotation precisely when trajectory is enabled
            self.origin = self.current_pos.copy()
            if self.traj_type in ["straight", "circle", "climb"]:
                # Align to the drone's *current heading*
                self.R_traj_to_G = self.R_GtoI.T.copy()
            else:
                # Square trajectory runs in the standard OpenVINS global frame
                self.R_traj_to_G = np.eye(3)
                
            self.get_logger().info(
                f"Trajectory tracking started! Origin: "
                f"[{self.origin[0]:.2f}, {self.origin[1]:.2f}, {self.origin[2]:.2f}]"
            )

        t = self.get_clock().now().nanoseconds * 1e-9 - self.t_start

        # ── Desired state from trajectory (local trajectory frame) ──
        p_des_local, v_des_local = self.traj.evaluate(t)

        # Shift to global frame (rotate by initial heading if straight)
        p_des_global = self.origin + self.R_traj_to_G @ p_des_local
        v_des_global = self.R_traj_to_G @ v_des_local

        # ── Position error in global frame ──
        e_pos = p_des_global - self.current_pos
        if self.hold_z:
            e_pos[2] = 0.0

        # ── Velocity command: feedforward + proportional feedback ──
        v_cmd_global = v_des_global +0* self.kp * e_pos

        # ── Rotate to body (IMU) frame ──
        v_cmd_body = self.R_GtoI @ v_cmd_global

        # ── Publish TwistStamped ──
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "base_link"
        msg.twist.linear.x = float(v_cmd_body[0])
        msg.twist.linear.y = float(v_cmd_body[1])
        msg.twist.linear.z = float(v_cmd_body[2])
        self.pub_cmd.publish(msg)

        # ── Status (throttled) ──
        dist = np.linalg.norm(e_pos)
        s_pct = ((t / self.traj.period) % 1.0) * 100
        status = (
            f"s={s_pct:.0f}% | err={dist:.2f}m | "
            f"v_body=[{v_cmd_body[0]:.2f},{v_cmd_body[1]:.2f},{v_cmd_body[2]:.2f}]"
        )
        status_msg = String()
        status_msg.data = status
        self.pub_status.publish(status_msg)
        self.get_logger().info(f"[BSpline] {status}", throttle_duration_sec=0.5)


# ══════════════════════════════════════════════════════════════════════════
def main(args=None):
    rclpy.init(args=args)
    node = BSplineTrajectoryNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    import sys
    if "--plot" in sys.argv:
        # Standalone visualisation — read from YAML config (no ROS needed)
        import os, yaml

        # Locate config file (next to this script's parent config dir)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(script_dir, "..", "config", "bspline_traj_config.yaml")
        config_path = os.path.normpath(config_path)

        side = 20.0
        spd = 3.0
        if os.path.isfile(config_path):
            with open(config_path) as f:
                cfg = yaml.safe_load(f)
            params = cfg.get("/**", {}).get("ros__parameters", {})
            side = params.get("side_length", side)
            spd = params.get("speed", spd)
            print(f"Loaded config from {config_path}")
        else:
            print(f"Config not found at {config_path}, using defaults")

        # CLI overrides still work
        for arg in sys.argv:
            if arg.startswith("--side="):
                side = float(arg.split("=")[1])
            if arg.startswith("--speed="):
                spd = float(arg.split("=")[1])

        traj = BSplineTrajectory.make_square(side, spd)
        print(f"Square: side={side}m  speed={spd}m/s  period={traj.period:.1f}s  "
              f"total_length={traj.total_length:.1f}m")
        traj.plot()
    else:
        main()
