#!/usr/bin/env python3
"""
CBF Safety Filter Node for OpenVINS — with live sliding plot
=============================================================

Subscribes to CBF observability metrics from OpenVINS, runs the closed-form
CBF-QP, and displays a real-time sliding-window plot of mean_logdet and
the corrected velocity magnitude.

Topics, gains, and plot window are all configurable via cbf_config.yaml.
"""

import atexit
import collections
import csv
import os
import signal
import subprocess
import threading
from datetime import datetime

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from rclpy.callback_groups import ReentrantCallbackGroup, MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor

from std_msgs.msg import Float64, Bool
from geometry_msgs.msg import Vector3Stamped, TwistStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2
from scipy.spatial.transform import Rotation


class CbfSafetyFilterNode(Node):
    """ROS 2 node: CBF safety filter + live 2-panel sliding plot."""

    def __init__(self):
        super().__init__("cbf_safety_filter")

        # ── Flight file logger (mirrors all get_logger() output to cbf.log) ──
        from flight_logger import setup_flight_logger
        setup_flight_logger(self, "cbf.log")

        # ── CBF parameters ──
        self.declare_parameter("cbf_enabled", True)
        self.declare_parameter("cbf_gamma", 1.0)
        self.declare_parameter("cbf_h_min", -5.0)
        self.declare_parameter("plot_enabled", True)         # live plot on/off
        self.declare_parameter("plot_show_logdet", True)      # show logdet panel
        self.declare_parameter("plot_show_g", False)          # show g(x) gradient panel
        self.declare_parameter("plot_show_velocity", True)    # show velocity panel
        self.declare_parameter("plot_show_features", False)   # show feature count panel
        self.declare_parameter("plot_window_sec", 5.0)        # time window in seconds

        # ── Nominal-direction-weighted CBF-QP metric (TANGO-VIO Eq. 37-46) ──
        # W_B = w_∥ P_∥ + w_⊥ P_⊥  where:
        #   t_B = v_nom / ‖v_nom‖   (unit nominal direction, body frame)
        #   P_∥ = t_B t_Bᵀ          (tangent projection)
        #   P_⊥ = I − t_B t_Bᵀ      (perpendicular projection)
        #
        # The condition 0 < w_∥ < w_⊥ penalises lateral deviations more
        # than along-track speed changes, so the CBF preferentially modifies
        # speed before heading.  W_B⁻¹ = (1/w_∥) P_∥ + (1/w_⊥) P_⊥.
        #
        # cbf_w_parallel      : tangent weight (smaller → allow more speed change)
        # cbf_w_perpendicular : lateral weight (larger → resist heading change)
        # cbf_v_epsilon       : low-speed threshold below which W_B = w_∥ I
        self.declare_parameter("cbf_w_parallel", 1.0)
        self.declare_parameter("cbf_w_perpendicular", 10.0)
        self.declare_parameter("cbf_v_epsilon", 0.1)

        # Soft-CBF slack penalty p_T.
        # Controls the trade-off between trajectory preservation and
        # triangulation maintenance.  The relaxed CBF condition is:
        #   f + gᵀ v_B + γ h + δ_T ≥ 0,   δ_T ≥ 0
        # with cost  ½ ‖v-v_nom‖²_W + (p_T/2) δ_T².
        #
        #   p_T → ∞  : hard CBF (no relaxation, full correction)
        #   p_T large: small slack, close to hard CBF
        #   p_T small: large slack, velocity stays close to nominal
        #
        # Set very large (e.g. 1e6) to recover the hard-CBF behaviour.
        self.declare_parameter("cbf_p_T", 100.0)

        # Output smoothing: EMA on v_safe
        # 1.0 = no smoothing (raw CBF), 0.1 = very smooth
        self.declare_parameter("cbf_output_alpha", 0.3)

        # Maximum speed and acceleration limits for v_safe
        self.declare_parameter("cbf_v_safe_max", 5.0)   # m/s
        self.declare_parameter("cbf_acc_max", 2.0)       # m/s²

        # Minimum SLAM feature threshold.
        # If the number of SLAM features drops below this, CBF forces the drone
        # to stop (v_safe = 0) regardless of the margin.
        # Set to -1 to disable this feature-count guard.
        self.declare_parameter("cbf_min_features", 10)

        # ── Logging parameters ──
        self.declare_parameter("log_enabled", True)

        # ── Topic names (configurable via YAML) ──
        self.declare_parameter("topic_sub_mean_logdet", "cbf/mean_logdet")
        self.declare_parameter("topic_sub_g", "cbf/g")
        self.declare_parameter("topic_sub_drift", "cbf/drift")
        self.declare_parameter("topic_pub_v_safe", "cbf/v_safe")
        self.declare_parameter("topic_pub_active", "cbf/active")
        self.declare_parameter("topic_pub_barrier", "cbf/barrier")
        self.declare_parameter("topic_sub_odom", "odomimu")
        self.declare_parameter("topic_sub_num_features", "cbf/num_features")
        # Nominal velocity in (from guidance/autopilot) and output to ArduPilot
        self.declare_parameter("topic_sub_cmd_vel_nom", "cmd_vel_nom")
        self.declare_parameter("topic_pub_cmd_vel_ap", "/ap/cmd_vel")

        # Read parameters
        self.cbf_enabled = self.get_parameter("cbf_enabled").value
        self.gamma = self.get_parameter("cbf_gamma").value
        self.h_min = self.get_parameter("cbf_h_min").value
        self.output_alpha = self.get_parameter("cbf_output_alpha").value
        self.min_features = self.get_parameter("cbf_min_features").value
        self.plot_window_sec = self.get_parameter("plot_window_sec").value
        self.plot_enabled = self.get_parameter("plot_enabled").value
        self.show_logdet = self.get_parameter("plot_show_logdet").value
        self.show_g = self.get_parameter("plot_show_g").value
        self.show_velocity = self.get_parameter("plot_show_velocity").value
        self.show_features = self.get_parameter("plot_show_features").value

        # Weighted CBF-QP metric
        self.w_par = self.get_parameter("cbf_w_parallel").value
        self.w_perp = self.get_parameter("cbf_w_perpendicular").value
        self.v_epsilon = self.get_parameter("cbf_v_epsilon").value
        self.p_T = self.get_parameter("cbf_p_T").value
        self.v_safe_max = self.get_parameter("cbf_v_safe_max").value
        self.acc_max = self.get_parameter("cbf_acc_max").value

        topic_sub_logdet = self.get_parameter("topic_sub_mean_logdet").value
        topic_sub_g = self.get_parameter("topic_sub_g").value
        topic_sub_drift = self.get_parameter("topic_sub_drift").value
        topic_pub_v_safe = self.get_parameter("topic_pub_v_safe").value
        topic_pub_active = self.get_parameter("topic_pub_active").value
        topic_pub_barrier = self.get_parameter("topic_pub_barrier").value
        topic_sub_odom = self.get_parameter("topic_sub_odom").value
        topic_sub_nfeat = self.get_parameter("topic_sub_num_features").value
        topic_sub_cmd_vel_nom = self.get_parameter("topic_sub_cmd_vel_nom").value
        topic_pub_cmd_vel_ap = self.get_parameter("topic_pub_cmd_vel_ap").value

        # ── Logging setup ──
        self.log_enabled = self.get_parameter("log_enabled").value
        self._log_file = None
        self._csv_writer = None
        self._drone_pos = [float('nan')] * 3      # latest ENU position
        self._num_slam_features = 0                # latest SLAM feature count
        self._tri_tried = 0                        # latest triangulation attempts
        self._tri_success = 0                      # latest triangulation successes

        if self.log_enabled:
            log_dir = getattr(self, '_flight_dir', os.path.expanduser('~/cbf_logs'))
            os.makedirs(log_dir, exist_ok=True)
            ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_path = os.path.join(log_dir, f"cbf_log_{ts_str}.csv")
            self._log_file = open(log_path, 'w', newline='')
            self._csv_writer = csv.writer(self._log_file)
            self._csv_writer.writerow([
                'timestamp',
                'logdet', 'logdet_threshold',
                'v_nom_x', 'v_nom_y', 'v_nom_z', 'v_nom_norm',
                'v_cbf_x', 'v_cbf_y', 'v_cbf_z', 'v_cbf_norm',
                'drone_x', 'drone_y', 'drone_z',
                'num_slam_features',
                'tri_tried', 'tri_success',
                'mu_margin', 'delta_T', 'f_drift',
            ])
            self._log_file.flush()
            self.get_logger().info(f"Logging to: {log_path}")
            # Register cleanup
            atexit.register(self._close_log)

        self.get_logger().info(
            f"CBF Safety Filter: enabled={self.cbf_enabled}, "
            f"gamma={self.gamma:.2f}, h_min={self.h_min:.2f}, "
            f"W_B: w_∥={self.w_par:.2f}, w_⊥={self.w_perp:.2f}, "
            f"ε_v={self.v_epsilon:.3f}, "
            f"plot_window={self.plot_window_sec:.1f}s"
        )

        # ── Live state ──
        self.mean_logdet = None
        self.g_vec = None
        self.g_header = None
        self.drift = 0.0                   # f(x) drift from past poses
        self._last_g_time = None           # timestamp of last g(x) message
        self._last_logdet_time = None      # timestamp of last logdet message
        self._cbf_timeout_sec = 2.0        # reset to None if no msg within this window
        self.v_nom_body = np.zeros(3)      # latest nominal velocity in body frame
        self.v_nom_angular = np.zeros(3)   # latest nominal angular velocity (pass-through)
        self._last_v_nom_time = None       # timestamp of last cmd_vel_nom message

        # ── Plot data buffers (large safety cap, time-based trimming in plot) ──
        N = 2000  # safety cap to prevent unbounded memory growth
        self.ts_logdet = collections.deque(maxlen=N)     # time axis
        self.ys_logdet = collections.deque(maxlen=N)     # mean_logdet values
        self.ys_hmin = collections.deque(maxlen=N)       # h_min reference line
        self.ys_barrier = collections.deque(maxlen=N)    # h = logdet - h_min

        self.ts_vel = collections.deque(maxlen=N)        # time axis
        self.ys_vx = collections.deque(maxlen=N)         # v_safe.x
        self.ys_vy = collections.deque(maxlen=N)         # v_safe.y
        self.ys_vz = collections.deque(maxlen=N)         # v_safe.z
        self.ys_vmag = collections.deque(maxlen=N)       # ||v_safe||

        self.ys_vnom_x = collections.deque(maxlen=N)     # v_nom.x
        self.ys_vnom_y = collections.deque(maxlen=N)     # v_nom.y
        self.ys_vnom_z = collections.deque(maxlen=N)     # v_nom.z
        self.ys_vnom_mag = collections.deque(maxlen=N)   # ||v_nom||

        self.ys_active = collections.deque(maxlen=N)     # CBF active flag

        self.ts_g = collections.deque(maxlen=N)            # time axis
        self.ys_g_x = collections.deque(maxlen=N)           # g.x
        self.ys_g_y = collections.deque(maxlen=N)           # g.y
        self.ys_g_z = collections.deque(maxlen=N)           # g.z
        self.ys_g_mag = collections.deque(maxlen=N)         # ||g||

        self.ts_odom = collections.deque(maxlen=N)         # time axis
        self.ys_vnorm_odom = collections.deque(maxlen=N)   # ||v_est|| from OpenVINS
        self.R_ItoG = np.eye(3)                            # rotation IMU→global (updated from odom)

        self.ts_nfeat = collections.deque(maxlen=N)        # time axis
        self.ys_nfeat = collections.deque(maxlen=N)        # feature count

        self.t0 = None  # reference time

        # ── Callback groups ──
        # CBF-critical: reentrant so g/logdet/drift are never blocked by odom
        self._cbf_group = ReentrantCallbackGroup()
        # Data providers: mutually exclusive (safe for shared state writes)
        self._data_group = MutuallyExclusiveCallbackGroup()

        # ── QoS ──
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )

        # ── Subscribers ──
        # CBF-critical path: g(x), drift, and logdet → triggers CBF filter + publish
        self.sub_logdet = self.create_subscription(
            Float64, topic_sub_logdet, self._cb_logdet, qos,
            callback_group=self._cbf_group
        )
        self.sub_g = self.create_subscription(
            Vector3Stamped, topic_sub_g, self._cb_g, qos,
            callback_group=self._cbf_group
        )
        self.sub_drift = self.create_subscription(
            Float64, topic_sub_drift, self._cb_drift, qos,
            callback_group=self._cbf_group
        )
        # Data providers: odom, features, nominal velocity
        self.sub_odom = self.create_subscription(
            Odometry, topic_sub_odom, self._cb_odom, qos,
            callback_group=self._data_group
        )
        self.sub_nfeat = self.create_subscription(
            Float64, topic_sub_nfeat, self._cb_num_features, qos,
            callback_group=self._data_group
        )
        # Nominal velocity from guidance — use reliable QoS (guidance commands)
        self.sub_cmd_vel_nom = self.create_subscription(
            TwistStamped, topic_sub_cmd_vel_nom, self._cb_cmd_vel_nom,
            QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                       history=HistoryPolicy.KEEP_LAST, depth=5),
            callback_group=self._data_group
        )

        # ── MAVROS drone position subscriber ──
        self.sub_drone_pos = self.create_subscription(
            Odometry, '/ov_msckf/odomimu_enu',
            self._cb_drone_pos, qos,
            callback_group=self._data_group
        )
        # Enable MAVROS LOCAL_POSITION_NED at 20 Hz
        self._enable_mavros_local_position()

        # ── SLAM feature point cloud subscriber ──
        self.sub_slam_pts = self.create_subscription(
            PointCloud2, '/ov_msckf/points_slam',
            self._cb_slam_points, qos,
            callback_group=self._data_group
        )

        # ── Triangulation count subscribers ──
        self.sub_tri_tried = self.create_subscription(
            Float64, 'cbf/tri_tried', self._cb_tri_tried, qos,
            callback_group=self._data_group
        )
        self.sub_tri_success = self.create_subscription(
            Float64, 'cbf/tri_success', self._cb_tri_success, qos,
            callback_group=self._data_group
        )

        # ── Publishers ──
        self.pub_v_safe = self.create_publisher(Vector3Stamped, topic_pub_v_safe, 10)
        self.pub_active = self.create_publisher(Bool, topic_pub_active, 10)
        self.pub_barrier = self.create_publisher(Float64, topic_pub_barrier, 10)
        self.pub_cmd_vel_ap = self.create_publisher(TwistStamped, topic_pub_cmd_vel_ap, 10)

        # ── Periodic log timer (runs even when CBF is inactive) ──
        self._log_timer = self.create_timer(0.1, self._timer_log_cb)  # 10 Hz
        self._cbf_logged_this_tick = False  # flag to avoid double-logging

        self.get_logger().info("CBF node ready — waiting for topics...")

    # ──────────────────────────────────────────────────────────────────────
    # Callbacks
    # ──────────────────────────────────────────────────────────────────────
    def _cb_logdet(self, msg: Float64):
        self.mean_logdet = msg.data
        self._last_logdet_time = self.get_clock().now().nanoseconds * 1e-9

    def _cb_g(self, msg: Vector3Stamped):
        self.g_vec = np.array([msg.vector.x, msg.vector.y, msg.vector.z])
        self.g_header = msg.header
        self._last_g_time = self.get_clock().now().nanoseconds * 1e-9
        self._run_cbf_filter()

    def _cb_drift(self, msg: Float64):
        self.drift = msg.data

    def _cb_odom(self, msg: Odometry):
        """Receive estimated velocity + orientation from OpenVINS odometry."""
        vx = msg.twist.twist.linear.x
        vy = msg.twist.twist.linear.y
        vz = msg.twist.twist.linear.z
        vnorm = np.sqrt(vx*vx + vy*vy + vz*vz)
        now = self.get_clock().now().nanoseconds * 1e-9
        if self.t0 is None:
            self.t0 = now
        self.ts_odom.append(now - self.t0)
        self.ys_vnorm_odom.append(vnorm)

        # Extract R_ItoG (IMU-to-global rotation) from pose orientation quaternion
        q = msg.pose.pose.orientation
        qvec = [q.x, q.y, q.z, q.w]
        # Guard against zero-norm quaternion (before OpenVINS initializes)
        if np.any(np.isnan(qvec)) or (q.x*q.x + q.y*q.y + q.z*q.z + q.w*q.w) < 1e-10:
            return
        rot = Rotation.from_quat(qvec)
        self.R_ItoG = rot.as_matrix()  # 3×3, maps body→global

    def _cb_num_features(self, msg: Float64):
        """Receive number of features used for CBF computation."""
        now = self.get_clock().now().nanoseconds * 1e-9
        if self.t0 is None:
            self.t0 = now
        self.ts_nfeat.append(now - self.t0)
        self.ys_nfeat.append(int(msg.data))

    def _cb_drone_pos(self, msg: Odometry):
        """Receive drone ENU position from MAVROS global_position/local."""
        self._drone_pos = [
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            msg.pose.pose.position.z,
        ]

    def _cb_slam_points(self, msg: PointCloud2):
        """Receive SLAM feature point cloud — extract count from width field."""
        self._num_slam_features = msg.width * msg.height

    def _cb_tri_tried(self, msg: Float64):
        """Receive number of features attempted for triangulation."""
        self._tri_tried = int(msg.data)

    def _cb_tri_success(self, msg: Float64):
        """Receive number of features successfully triangulated."""
        self._tri_success = int(msg.data)

    def _enable_mavros_local_position(self):
        """Call MAVROS set_message_interval service to publish LOCAL_POSITION_NED at 20 Hz."""
        try:
            cmd = (
                'ros2 service call /mavros/set_message_interval '
                'mavros_msgs/srv/MessageInterval '
                '"{message_id: 33,  message_rate: 50.0}"'
            )
            subprocess.Popen(
                cmd, shell=True,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            self.get_logger().info("Requested MAVROS LOCAL_POSITION_NED at 20 Hz.")
        except Exception as e:
            self.get_logger().warn(
                f"Could not call set_message_interval: {e}. "
                "Drone position may not be available."
            )

    def _close_log(self):
        """Flush and close the CSV log file."""
        if self._log_file is not None and not self._log_file.closed:
            self._log_file.flush()
            self._log_file.close()
            self.get_logger().info("CSV log file closed.")

    def _timer_log_cb(self):
        """Periodic fallback logger — runs at 10 Hz even when CBF g(x) is not arriving."""
        if not self.log_enabled:
            return
        # If the CBF path already logged this tick, skip to avoid duplicates
        if self._cbf_logged_this_tick:
            self._cbf_logged_this_tick = False
            return
        # Write a row with NaN for CBF-dependent fields
        # Skip if t0 hasn't been set yet (no CBF data received at all)
        if self.t0 is None:
            return
        t = self.get_clock().now().nanoseconds * 1e-9 - self.t0
        v_nom = self.v_nom_body.copy()
        nan3 = np.array([float('nan')] * 3)
        self._write_log_row(
            t, float('nan'), self.h_min, v_nom, nan3,
            float('nan'), float('nan'), float('nan')
        )

    def _write_log_row(self, t, logdet, h_min, v_nom, v_safe, mu, delta_T, f_drift):
        """Write a single row to the CSV log."""
        if self._csv_writer is None:
            return
        self._csv_writer.writerow([
            f"{t:.6f}",
            f"{logdet:.6f}", f"{h_min:.6f}",
            f"{v_nom[0]:.6f}", f"{v_nom[1]:.6f}", f"{v_nom[2]:.6f}",
            f"{np.linalg.norm(v_nom):.6f}",
            f"{v_safe[0]:.6f}", f"{v_safe[1]:.6f}", f"{v_safe[2]:.6f}",
            f"{np.linalg.norm(v_safe):.6f}",
            f"{self._drone_pos[0]:.6f}", f"{self._drone_pos[1]:.6f}",
            f"{self._drone_pos[2]:.6f}",
            self._num_slam_features,
            self._tri_tried,
            self._tri_success,
            f"{mu:.6f}",
            f"{delta_T:.6f}",
            f"{f_drift:.6f}",
        ])
        # Flush periodically (every ~50 rows) for safety without perf hit
        if not hasattr(self, '_log_row_count'):
            self._log_row_count = 0
        self._log_row_count += 1
        if self._log_row_count % 50 == 0:
            self._log_file.flush()

    def _cb_cmd_vel_nom(self, msg: TwistStamped):
        """Receive nominal velocity command from guidance system (body frame)."""
        self._last_v_nom_time = self.get_clock().now().nanoseconds * 1e-9

        self.v_nom_body = np.array([
            msg.twist.linear.x,
            msg.twist.linear.y,
            msg.twist.linear.z,
        ])
        self.v_nom_angular = np.array([
            msg.twist.angular.x,
            msg.twist.angular.y,
            msg.twist.angular.z,
        ])

        # Check staleness: reset CBF data if not received within timeout
        now_sec = self.get_clock().now().nanoseconds * 1e-9
        if self._last_g_time is not None and (now_sec - self._last_g_time) > self._cbf_timeout_sec:
            self.g_vec = None
            self._last_g_time = None
        if self._last_logdet_time is not None and (now_sec - self._last_logdet_time) > self._cbf_timeout_sec:
            self.mean_logdet = None
            self._last_logdet_time = None

        # Pass-through: forward nominal to ArduPilot when CBF is not yet active
        if self.g_vec is None or self.mean_logdet is None:
            self.get_logger().info('CBF not ready — passing through nominal velocity', throttle_duration_sec=1.0)
            ap_msg = TwistStamped()
            ap_msg.header = msg.header
            ap_msg.header.frame_id = "base_link"
            ap_msg.twist = msg.twist
            self.pub_cmd_vel_ap.publish(ap_msg)

    # ──────────────────────────────────────────────────────────────────────
    # Core CBF logic
    # ──────────────────────────────────────────────────────────────────────
    def _run_cbf_filter(self):
        if self.mean_logdet is None or self.g_vec is None:
            return
        # Gate on fresh v_nom: don't publish if no cmd_vel_nom within timeout
        now_check = self.get_clock().now().nanoseconds * 1e-9
        if self._last_v_nom_time is None or (now_check - self._last_v_nom_time) > self._cbf_timeout_sec:
            self.get_logger().info('v_nom stale or absent — not publishing v_safe', throttle_duration_sec=1.0)
            return

        now = self.get_clock().now().nanoseconds * 1e-9
        if self.t0 is None:
            self.t0 = now
        t = now - self.t0

        # Barrier value
        h = self.mean_logdet - self.h_min

        # Publish barrier
        barrier_msg = Float64()
        barrier_msg.data = h
        self.pub_barrier.publish(barrier_msg)

        # Use latest nominal velocity (body frame) — falls back to zero if no guidance yet
        v_nom = self.v_nom_body.copy()

        g_norm_sq = np.dot(self.g_vec, self.g_vec)
        cbf_active = False
        delta_T = 0.0           # soft-CBF slack (0 when inactive)
        v_safe = v_nom.copy()
        self._v_safe_prev = getattr(self, '_v_safe_prev', None)  # lazy init

        # Compute margin for logging (even when CBF is disabled)
        margin = self.drift + np.dot(self.g_vec, v_nom) + self.gamma * h

        if self.cbf_enabled and g_norm_sq > 1e-20:
            # ── Weighted CBF-QP (TANGO-VIO Eq. 37-46) ─────────────────────
            #
            #   QP:  min ½ ‖v_B − v_nom‖²_{W_B}   s.t.  f + gᵀ v_B + γ h ≥ 0
            #
            #   W_B = w_∥ P_∥ + w_⊥ P_⊥
            #   P_∥ = t_B t_Bᵀ,  P_⊥ = I − t_B t_Bᵀ,  t_B = v_nom/‖v_nom‖
            #
            #   Closed-form (Eq. 42):
            #     μ = f + gᵀ v_nom + γ h
            #     if μ ≥ 0:  v* = v_nom
            #     if μ < 0:  v* = v_nom − (μ / (gᵀ W_B⁻¹ g)) W_B⁻¹ g
            #
            #   W_B⁻¹ = (1/w_∥) P_∥ + (1/w_⊥) P_⊥
            #   Low-speed fallback: W_B = w_∥ I  ⇒  W_B⁻¹ = (1/w_∥) I

            # Build W_B⁻¹ from instantaneous nominal direction
            v_nom_norm = np.linalg.norm(v_nom)
            if v_nom_norm > self.v_epsilon:
                # Normal: direction-dependent weighting
                t_B = v_nom / v_nom_norm                            # unit tangent
                P_par = np.outer(t_B, t_B)                          # P_∥ = t_B t_Bᵀ
                P_perp = np.eye(3) - P_par                          # P_⊥ = I - P_∥
                W_inv = (1.0 / self.w_par) * P_par + (1.0 / self.w_perp) * P_perp
            else:
                # Low-speed fallback (Eq. 46): isotropic W_B = w_∥ I
                W_inv = (1.0 / self.w_par) * np.eye(3)

            # Weighted gradient: W_B⁻¹ g(x)
            W_inv_g = W_inv @ self.g_vec
            # Denominator: gᵀ W_B⁻¹ g
            gT_Winv_g = np.dot(self.g_vec, W_inv_g)

            if margin < 0 and gT_Winv_g > 1e-20:
                # Soft-CBF: denominator includes slack penalty 1/p_T
                # v* = v_nom − (μ / (S(x) + 1/p_T)) W_B⁻¹ g
                # δ_T* = −μ / (1 + p_T S(x))
                S_x = gT_Winv_g
                lam = -margin / (S_x + 1.0 / self.p_T)
                v_safe = v_nom + lam * W_inv_g
                delta_T = -margin / (1.0 + self.p_T * S_x)
                cbf_active = True


        # ── Output smoothing (EMA low-pass filter) ──
        if self._v_safe_prev is not None and self.output_alpha < 1.0:
            v_safe = self.output_alpha * v_safe + (1.0 - self.output_alpha) * self._v_safe_prev

        # ── Acceleration limiter ──
        now_acc = self.get_clock().now().nanoseconds * 1e-9
        if self._v_safe_prev is not None and hasattr(self, '_last_cbf_time') and self._last_cbf_time is not None:
            dt_cbf = now_acc - self._last_cbf_time
            if dt_cbf > 0:
                dv = v_safe - self._v_safe_prev
                dv_norm = np.linalg.norm(dv)
                max_dv = self.acc_max * dt_cbf
                if dv_norm > max_dv:
                    v_safe = self._v_safe_prev + dv * (max_dv / dv_norm)
        self._last_cbf_time = now_acc

        # ── Speed clamp ──
        speed = np.linalg.norm(v_safe)
        if speed > self.v_safe_max:
            v_safe = v_safe * (self.v_safe_max / speed)

        self._v_safe_prev = v_safe.copy()

        # ── Publish v_safe as Vector3Stamped (internal monitoring) ──
        v_msg = Vector3Stamped()
        v_msg.header = self.g_header
        v_msg.header.frame_id = "imu"
        v_msg.vector.x = float(v_safe[0])
        v_msg.vector.y = float(v_safe[1])
        v_msg.vector.z = float(v_safe[2])
        self.pub_v_safe.publish(v_msg)

        # ── Publish TwistStamped to ArduPilot flight controller ──
        ap_msg = TwistStamped()
        ap_msg.header = self.g_header
        ap_msg.header.frame_id = "base_link"  # ArduPilot expects body frame
        ap_msg.twist.linear.x = float(v_safe[0])
        ap_msg.twist.linear.y = float(v_safe[1])
        ap_msg.twist.linear.z = float(v_safe[2])
        # Angular velocity passes through unchanged
        ap_msg.twist.angular.x = float(self.v_nom_angular[0]*0)
        ap_msg.twist.angular.y = float(self.v_nom_angular[1]*0)
        ap_msg.twist.angular.z = float(self.v_nom_angular[2]*0)
        self.pub_cmd_vel_ap.publish(ap_msg)

        # Publish active flag
        active_msg = Bool()
        active_msg.data = cbf_active
        self.pub_active.publish(active_msg)

        # ── Store for plot ──
        self.ts_logdet.append(t)
        self.ys_logdet.append(self.mean_logdet)
        self.ys_hmin.append(self.h_min)
        self.ys_barrier.append(h)

        self.ts_vel.append(t)
        self.ys_vx.append(v_safe[0])
        self.ys_vy.append(v_safe[1])
        self.ys_vz.append(v_safe[2])
        self.ys_vmag.append(np.linalg.norm(v_safe))
        self.ys_vnom_x.append(v_nom[0])
        self.ys_vnom_y.append(v_nom[1])
        self.ys_vnom_z.append(v_nom[2])
        self.ys_vnom_mag.append(np.linalg.norm(v_nom))
        self.ys_active.append(1.0 if cbf_active else 0.0)

        self.ts_g.append(t)
        self.ys_g_x.append(self.g_vec[0])
        self.ys_g_y.append(self.g_vec[1])
        self.ys_g_z.append(self.g_vec[2])
        self.ys_g_mag.append(np.linalg.norm(self.g_vec))

        # ── Write to CSV log ──
        if self.log_enabled:
            self._write_log_row(t, self.mean_logdet, self.h_min, v_nom, v_safe, margin, delta_T, self.drift)
            self._cbf_logged_this_tick = True

        # Console log (throttled)
        status = "ACTIVE" if cbf_active else "ok"
        self.get_logger().info(
            f"[CBF] h={h:.4f} | μ={margin:.4f} | δ_T={delta_T:.4f} | f={self.drift:.4f} | "
            f"g=[{self.g_vec[0]:.4f},{self.g_vec[1]:.4f},{self.g_vec[2]:.4f}] "
            f"| ||v_safe||={np.linalg.norm(v_safe):.4f} | {status}",
            throttle_duration_sec=0.5,
        )


# ══════════════════════════════════════════════════════════════════════════
# PyQtGraph sliding plot + ROS spin integration
# ══════════════════════════════════════════════════════════════════════════
def main(args=None):
    rclpy.init(args=args)
    node = CbfSafetyFilterNode()

    # Multi-threaded executor: CBF callbacks are never blocked by odom
    executor = MultiThreadedExecutor(num_threads=3)
    executor.add_node(node)

    if not node.plot_enabled:
        # ── No-plot mode: pure spin for maximum throughput ──
        node.get_logger().info("Plot DISABLED — running headless (max Hz, multi-threaded).")
        try:
            executor.spin()
        except KeyboardInterrupt:
            pass
        finally:
            node._close_log()
            executor.shutdown()
            node.destroy_node()
            rclpy.shutdown()
        return

    # ── Plot mode: pyqtgraph (GPU-accelerated) ──
    panels = []
    if node.show_logdet: panels.append('logdet')
    if node.show_g: panels.append('g')
    if node.show_velocity: panels.append('velocity')
    if node.show_features: panels.append('features')
    node.get_logger().info(f"Plot ENABLED — panels: {panels}")
    import pyqtgraph as pg
    from PyQt5 import QtWidgets, QtCore

    # Dark theme
    pg.setConfigOptions(antialias=True, background='#1e1e1e', foreground='#cccccc')

    app = QtWidgets.QApplication([])
    win = QtWidgets.QMainWindow()
    win.setWindowTitle("CBF Safety Filter — Live Monitor")
    win.resize(1000, 250 * len(panels) if panels else 400)

    central = QtWidgets.QWidget()
    win.setCentralWidget(central)
    layout = QtWidgets.QVBoxLayout(central)
    layout.setSpacing(2)
    layout.setContentsMargins(4, 4, 4, 4)

    # Track curves for conditional update
    c_logdet = c_hmin = c_vnorm = None
    c_g_x = c_g_y = c_g_z = c_g_mag = None
    c_vx = c_vy = c_vz = c_vmag = None
    c_vnom_x = c_vnom_y = c_vnom_z = c_vnom_mag = None
    c_nfeat = None
    p_link_src = None  # first plot for x-axis linking

    # ── Panel: logdet ──
    if node.show_logdet:
        p1 = pg.PlotWidget(title="<span style='color:cyan'>Log-det Metric  &amp;  ||v_est||</span>")
        p1.addLegend(offset=(10, 10), labelTextSize='9pt')
        p1.showGrid(x=True, y=True, alpha=0.3)
        p1.setLabel('left', 'Log-det', color='cyan')
        c_logdet = p1.plot(pen=pg.mkPen('#00ffff', width=2), name='mean logdet (ℓ̄)')
        c_hmin = p1.plot(pen=pg.mkPen('#ff4444', width=1, style=QtCore.Qt.DashLine),
                         name=f'h_min = {node.h_min:.1f}')
        # Secondary ViewBox for velocity norm (right axis)
        p1b = pg.ViewBox()
        p1.scene().addItem(p1b)
        p1.getAxis('right').linkToView(p1b)
        p1b.setXLink(p1)
        p1.getAxis('right').setLabel('||v_est|| (m/s)', color='#ffd43b')
        p1.showAxis('right')
        c_vnorm = pg.PlotCurveItem(pen=pg.mkPen('#ffd43b', width=1.5), name='||v_est||')
        p1b.addItem(c_vnorm)
        def update_p1b_geometry():
            p1b.setGeometry(p1.getViewBox().sceneBoundingRect())
        p1.getViewBox().sigResized.connect(update_p1b_geometry)
        layout.addWidget(p1)
        p_link_src = p1

    # ── Panel: g(x) gradient ──
    if node.show_g:
        p2 = pg.PlotWidget(title="<span style='color:#cccccc'>g(x) Control Gradient</span>")
        p2.addLegend(offset=(10, 10), labelTextSize='9pt')
        p2.showGrid(x=True, y=True, alpha=0.3)
        p2.setLabel('left', 'g(x)')
        c_g_x = p2.plot(pen=pg.mkPen('#ffa94d', width=1.5), name='g.x')
        c_g_y = p2.plot(pen=pg.mkPen('#a9e34b', width=1.5), name='g.y')
        c_g_z = p2.plot(pen=pg.mkPen('#74c0fc', width=1.5), name='g.z')
        c_g_mag = p2.plot(pen=pg.mkPen('#ffffff', width=2), name='||g||')
        layout.addWidget(p2)
        if p_link_src: p2.setXLink(p_link_src)
        else: p_link_src = p2

    # ── Panel: Velocity ──
    if node.show_velocity:
        p3 = pg.PlotWidget(title="<span style='color:#cccccc'>Velocity  (solid=safe, dashed=nom)</span>")
        p3.addLegend(offset=(10, 10), labelTextSize='8pt', colCount=2)
        p3.showGrid(x=True, y=True, alpha=0.3)
        p3.setLabel('left', 'Velocity (m/s)')
        c_vx = p3.plot(pen=pg.mkPen('#ff6b6b', width=1.5), name='v_safe.x')
        c_vy = p3.plot(pen=pg.mkPen('#51cf66', width=1.5), name='v_safe.y')
        c_vz = p3.plot(pen=pg.mkPen('#339af0', width=1.5), name='v_safe.z')
        c_vmag = p3.plot(pen=pg.mkPen('#ffffff', width=2), name='||v_safe||')
        c_vnom_x = p3.plot(pen=pg.mkPen('#ff6b6b', width=1, style=QtCore.Qt.DashLine), name='v_nom.x')
        c_vnom_y = p3.plot(pen=pg.mkPen('#51cf66', width=1, style=QtCore.Qt.DashLine), name='v_nom.y')
        c_vnom_z = p3.plot(pen=pg.mkPen('#339af0', width=1, style=QtCore.Qt.DashLine), name='v_nom.z')
        c_vnom_mag = p3.plot(pen=pg.mkPen('#ffffff', width=1.5, style=QtCore.Qt.DashLine), name='||v_nom||')
        layout.addWidget(p3)
        if p_link_src: p3.setXLink(p_link_src)
        else: p_link_src = p3

    # ── Panel: Feature count ──
    if node.show_features:
        p4 = pg.PlotWidget(title="<span style='color:#cccccc'>Feature Count</span>")
        p4.addLegend(offset=(10, 10), labelTextSize='9pt')
        p4.showGrid(x=True, y=True, alpha=0.3)
        p4.setLabel('left', 'Count')
        p4.setLabel('bottom', 'Time (s)')
        c_nfeat = p4.plot(pen=pg.mkPen('#e599f7', width=2), name='# features')
        layout.addWidget(p4)
        if p_link_src: p4.setXLink(p_link_src)
        else: p_link_src = p4

    def update():
        """Timer callback: update plots only (ROS spins on background thread)."""

        # Helper: deque → numpy array, trimmed to last T seconds
        T = node.plot_window_sec

        def d2a_trim(ts_deque, *data_deques):
            """Return (ts_arr, data_arr1, data_arr2, ...) trimmed to last T seconds.
            Snapshots deques to lists first to avoid race conditions."""
            # Snapshot to avoid mutation during conversion
            ts_list = list(ts_deque)
            data_lists = [list(d) for d in data_deques]
            # Truncate all to the shortest length for consistency
            min_len = min(len(ts_list), *(len(d) for d in data_lists)) if data_lists else len(ts_list)
            ts = np.array(ts_list[:min_len]) if min_len > 0 else np.array([])
            if len(ts) == 0:
                return (ts,) + tuple(np.array([]) for _ in data_deques)
            t_cutoff = ts[-1] - T
            mask = ts >= t_cutoff
            results = [ts[mask]]
            for dl in data_lists:
                arr = np.array(dl[:min_len])
                results.append(arr[mask])
            return tuple(results)

        # Logdet panel
        if c_logdet is not None:
            ts, ld, hm = d2a_trim(node.ts_logdet, node.ys_logdet, node.ys_hmin)
            if len(ts) > 0:
                c_logdet.setData(ts, ld)
                c_hmin.setData(ts, hm)
            ts_o, vn = d2a_trim(node.ts_odom, node.ys_vnorm_odom)
            if len(ts_o) > 0 and c_vnorm is not None:
                c_vnorm.setData(ts_o, vn)

        # g(x) panel
        if c_g_x is not None:
            ts_p, gx, gy, gz, gm = d2a_trim(
                node.ts_g, node.ys_g_x, node.ys_g_y,
                node.ys_g_z, node.ys_g_mag)
            if len(ts_p) > 0:
                c_g_x.setData(ts_p, gx)
                c_g_y.setData(ts_p, gy)
                c_g_z.setData(ts_p, gz)
                c_g_mag.setData(ts_p, gm)

        # Velocity panel
        if c_vx is not None:
            ts_v, vx, vy, vz, vm, nx, ny, nz, nm = d2a_trim(
                node.ts_vel, node.ys_vx, node.ys_vy, node.ys_vz, node.ys_vmag,
                node.ys_vnom_x, node.ys_vnom_y, node.ys_vnom_z, node.ys_vnom_mag)
            if len(ts_v) > 0:
                c_vx.setData(ts_v, vx)
                c_vy.setData(ts_v, vy)
                c_vz.setData(ts_v, vz)
                c_vmag.setData(ts_v, vm)
                c_vnom_x.setData(ts_v, nx)
                c_vnom_y.setData(ts_v, ny)
                c_vnom_z.setData(ts_v, nz)
                c_vnom_mag.setData(ts_v, nm)

        # Feature count panel
        if c_nfeat is not None:
            ts_n, nf = d2a_trim(node.ts_nfeat, node.ys_nfeat)
            if len(ts_n) > 0:
                c_nfeat.setData(ts_n, nf)

    # QTimer for plot refresh only (50ms = ~20 FPS, lightweight)
    timer = QtCore.QTimer()
    timer.timeout.connect(update)
    timer.start(50)

    # Spin the executor on a background daemon thread (full Hz)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()
    node.get_logger().info("ROS executor running on background thread (full Hz).")

    win.show()

    try:
        app.exec_()
    except KeyboardInterrupt:
        pass
    finally:
        node._close_log()
        timer.stop()
        executor.shutdown()
        spin_thread.join(timeout=2.0)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
