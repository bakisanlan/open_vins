#!/usr/bin/env python3
"""
Square Trajectory Nominal Velocity Generator
=============================================

Generates a closed-square trajectory in the horizontal plane with alternating
slow (0-2 m/s) and fast (5-10 m/s) segments for testing the CBF safety filter.

Architecture:
  - Waits for the first OpenVINS odometry message to lock the start pose.
  - Defines four waypoints relative to that start position.
  - A proportional pursuit controller computes the desired velocity in global
    (ENU) frame and rotates it into the body (IMU) frame.
  - Publishes geometry_msgs/TwistStamped to the nominal velocity topic, which
    the CBF safety filter node reads and corrects before forwarding to ArduPilot.

Square layout (top view, NED/ENU x=North, y=East or x=East, y=North):

    WP3 ←─(fast)─── WP2
     │                │
  (slow)           (fast)
     │                │
    WP0 ──(slow)──→ WP1

Segment speeds:
  WP0→WP1 : slow  (traj_slow_speed)
  WP1→WP2 : fast  (traj_fast_speed)
  WP2→WP3 : slow  (traj_slow_speed)
  WP3→WP0 : fast  (traj_fast_speed)

Parameters (configurable via traj_config.yaml):
  traj_side_length   : square side in metres         (default 10.0)
  traj_slow_speed    : speed for slow segments (m/s)  (default 1.5)
  traj_fast_speed    : speed for fast segments (m/s)  (default 7.0)
  traj_accept_radius : waypoint acceptance radius (m) (default 1.2)
  traj_repeat        : loop trajectory indefinitely   (default true)
  traj_hold_altitude : do not touch z component       (default true)
  topic_odom         : OpenVINS odometry topic
  topic_cmd_vel_nom  : nominal velocity output topic
"""

import numpy as np
from scipy.spatial.transform import Rotation

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from nav_msgs.msg import Odometry
from geometry_msgs.msg import TwistStamped
from std_msgs.msg import String


class SquareTrajNode(Node):
    """Generates a square nominal velocity trajectory for CBF testing."""

    def __init__(self):
        super().__init__("square_traj_node")

        # ── Parameters ──
        self.declare_parameter("traj_enabled",       True)
        self.declare_parameter("traj_side_length",   10.0)
        self.declare_parameter("traj_slow_speed",     1.5)
        self.declare_parameter("traj_fast_speed",     7.0)
        self.declare_parameter("traj_accept_radius",  1.2)
        self.declare_parameter("traj_repeat",         True)
        self.declare_parameter("traj_hold_altitude",  True)
        self.declare_parameter("topic_odom",          "odomimu")
        self.declare_parameter("topic_cmd_vel_nom",   "cmd_vel_nom")

        self.enabled = self.get_parameter("traj_enabled").value

        L    = self.get_parameter("traj_side_length").value
        self.v_slow  = self.get_parameter("traj_slow_speed").value
        self.v_fast  = self.get_parameter("traj_fast_speed").value
        self.r_acc   = self.get_parameter("traj_accept_radius").value
        self.repeat  = self.get_parameter("traj_repeat").value
        self.hold_z  = self.get_parameter("traj_hold_altitude").value
        topic_odom   = self.get_parameter("topic_odom").value
        topic_nom    = self.get_parameter("topic_cmd_vel_nom").value

        if not self.enabled:
            self.get_logger().warn(
                "[SquareTraj] DISABLED — publishing zero velocity. "
                "Set traj_enabled: true in traj_config.yaml to activate."
            )

        # ── Square waypoints (relative offsets, in global ENU frame) ──
        # Waypoints and their *approach* speeds (speed from previous WP to this WP)
        # [dx, dy, dz from start position]
        self._wp_offsets = np.array([
            [0.0,  0.0, 0.0],   # WP0 (start / home)
            [L,    0.0, 0.0],   # WP1 → slow segment WP0→WP1
            [L,    L,   0.0],   # WP2 → fast segment WP1→WP2
            [0.0,  L,   0.0],   # WP3 → slow segment WP2→WP3
        ])
        self._seg_speeds = [
            self.v_slow,  # WP0 → WP1
            self.v_fast,  # WP1 → WP2
            self.v_slow,  # WP2 → WP3
            self.v_fast,  # WP3 → WP0 (return)
        ]

        # ── State ──
        self.origin_pos = None      # global ENU position at start
        self.current_pos = None     # latest global ENU position
        self.R_ItoG = np.eye(3)    # rotation IMU→global
        self.wp_idx = 0            # current waypoint index
        self.done = False

        self.get_logger().info(
            f"Square traj: L={L:.1f}m, slow={self.v_slow:.1f}m/s, "
            f"fast={self.v_fast:.1f}m/s, accept_r={self.r_acc:.1f}m, "
            f"repeat={self.repeat}"
        )

        # ── QoS ──
        qos_be = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )
        qos_rel = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )

        # ── Subscribers ──
        self.sub_odom = self.create_subscription(
            Odometry, topic_odom, self._cb_odom, qos_be
        )

        # ── Publishers ──
        self.pub_nom = self.create_publisher(TwistStamped, topic_nom, qos_rel)
        self.pub_status = self.create_publisher(String, "traj/status", 10)

        # ── Control timer (20 Hz) ──
        self.timer = self.create_timer(0.05, self._control_loop)

        self.get_logger().info(f"Trajectory node ready, listening on '{topic_odom}'...")

    # ──────────────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────────────
    def _current_wp_global(self):
        """Return the current target waypoint in global ENU frame."""
        return self.origin_pos + self._wp_offsets[self.wp_idx]

    def _next_wp_idx(self):
        """Advance waypoint index, returning True if trajectory completed."""
        self.wp_idx = (self.wp_idx + 1) % len(self._wp_offsets)
        if self.wp_idx == 0:
            if not self.repeat:
                return True   # done
            self.get_logger().info("Square loop complete — restarting.")
        seg = self.wp_idx % len(self._seg_speeds)
        v = self._seg_speeds[seg]
        self.get_logger().info(
            f"  → WP{self.wp_idx} | segment speed: {v:.1f} m/s"
        )
        return False

    # ──────────────────────────────────────────────────────────────────────
    # Callbacks
    # ──────────────────────────────────────────────────────────────────────
    def _cb_odom(self, msg: Odometry):
        """Extract position and orientation from OpenVINS odometry."""
        p = msg.pose.pose.position
        pos_global = np.array([p.x, p.y, p.z])

        # Lock starting position on first message
        if self.origin_pos is None:
            self.origin_pos = pos_global.copy()
            self.get_logger().info(
                f"Origin locked at [{pos_global[0]:.2f}, "
                f"{pos_global[1]:.2f}, {pos_global[2]:.2f}]"
            )

        self.current_pos = pos_global

        # Extract rotation IMU → global
        q = msg.pose.pose.orientation
        rot = Rotation.from_quat([q.x, q.y, q.z, q.w])
        self.R_ItoG = rot.as_matrix()

    # ──────────────────────────────────────────────────────────────────────
    # Control loop
    # ──────────────────────────────────────────────────────────────────────
    def _control_loop(self):
        """Compute and publish nominal velocity at 20 Hz."""
        if not self.enabled:
            self._publish_zero()
            return

        if self.current_pos is None or self.origin_pos is None:
            return   # waiting for first odometry

        if self.done:
            # Trajectory finished — publish zero command to hold position
            self._publish_zero()
            return

        # ── Current target waypoint ──
        wp_global = self._current_wp_global()
        error_global = wp_global - self.current_pos

        if self.hold_z:
            error_global[2] = 0.0   # do not chase altitude in this controller

        dist = np.linalg.norm(error_global)

        # ── Waypoint acceptance ──
        if dist < self.r_acc:
            self.get_logger().info(
                f"Reached WP{self.wp_idx} "
                f"(dist={dist:.2f}m < {self.r_acc:.1f}m)"
            )
            self.done = self._next_wp_idx()
            return

        # ── Segment speed ──
        seg = self.wp_idx % len(self._seg_speeds)
        v_target = self._seg_speeds[seg]

        # Smooth approach: reduce speed linearly inside 3× acceptance radius
        slow_zone = 3.0 * self.r_acc
        if dist < slow_zone:
            v_target = v_target * (dist / slow_zone)
            v_target = max(v_target, 0.2)   # minimum creep speed

        # ── Unit vector toward waypoint ──
        direction_global = error_global / (dist + 1e-9)
        v_nom_global = v_target * direction_global

        # ── Rotate to body (IMU) frame ──
        R_GtoI = self.R_ItoG.T
        v_nom_body = R_GtoI @ v_nom_global

        # ── Publish TwistStamped ──
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "base_link"
        msg.twist.linear.x = float(v_nom_body[0])
        msg.twist.linear.y = float(v_nom_body[1])
        msg.twist.linear.z = float(v_nom_body[2])
        # No angular velocity command
        msg.twist.angular.x = 0.0
        msg.twist.angular.y = 0.0
        msg.twist.angular.z = 0.0
        self.pub_nom.publish(msg)

        # ── Status ──
        status_msg = String()
        status_msg.data = (
            f"WP{self.wp_idx} | dist={dist:.2f}m | "
            f"v_nom={v_target:.2f}m/s | "
            f"v_body=[{v_nom_body[0]:.2f},{v_nom_body[1]:.2f},{v_nom_body[2]:.2f}]"
        )
        self.pub_status.publish(status_msg)
        self.get_logger().info(status_msg.data, throttle_duration_sec=1.0)

    def _publish_zero(self):
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "base_link"
        self.pub_nom.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = SquareTrajNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
