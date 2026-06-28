#!/usr/bin/env python3
"""
record_bag_on_arm.py — Arm-triggered rosbag recorder for ov_msckf topics.

Behavior:
  1. Waits for /mavros/state to report armed=True  → starts ros2 bag record
  2. When /mavros/state reports armed=False (disarm) → kills the bag recording
  3. Exits cleanly after one arm/disarm cycle (systemd will NOT restart it)

The bag is saved in the working directory with a timestamped name.
"""

import os
import signal
import subprocess
import sys
from datetime import datetime

import rclpy
from rclpy.node import Node
from mavros_msgs.msg import State


TOPICS = [
    "/mavros/global_position/local",
    "/mavros/global_position/global",
    "/ov_msckf/pathimu",
    "/ov_msckf/pathgt",
    "/ov_msckf/points_msckf",
    "/ov_msckf/points_slam",
    "/ov_msckf/loop_feats",
    "/ov_msckf/odomimu",
]


# ──────────────────────────────────────────────────────────────────
# Set to True to start recording immediately (skip arm-wait).
# Set to False for production (arm-triggered recording).
TEST_MODE = False
# ──────────────────────────────────────────────────────────────────


class ArmTriggeredRecorder(Node):
    def __init__(self):
        super().__init__("arm_triggered_bag_recorder")
        self._bag_proc = None
        self._was_armed = False

        self.create_subscription(State, "/mavros/state", self._state_cb, 10)

        if TEST_MODE:
            self.get_logger().warn("TEST_MODE enabled — recording immediately!")
            self._was_armed = True
            self._start_recording()
        else:
            self.get_logger().info("Waiting for vehicle to arm …")

    # ── state callback ──────────────────────────────────────────────
    def _state_cb(self, msg: State):
        if msg.armed and not self._was_armed:
            # Transition: disarmed → armed
            self._was_armed = True
            self._start_recording()

        elif not msg.armed and self._was_armed:
            # Transition: armed → disarmed
            self.get_logger().info("Vehicle DISARMED — stopping bag recording.")
            self._stop_recording()
            # Shut down cleanly
            rclpy.shutdown()

    # ── recording helpers ───────────────────────────────────────────
    def _start_recording(self):
        timestamp = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
        bag_name = f"flight_{timestamp}"

        cmd = ["ros2", "bag", "record", "-o", bag_name] + TOPICS
        self.get_logger().info(f"Vehicle ARMED — starting bag: {bag_name}")
        self._bag_proc = subprocess.Popen(
            cmd,
            # Inherit env so ROS 2 middleware is reachable
            env=os.environ.copy(),
            # Use a new process group so we can kill the whole tree
            preexec_fn=os.setsid,
        )

    def _stop_recording(self):
        if self._bag_proc is None:
            return
        try:
            # Send SIGINT (Ctrl-C) to the process group for a clean flush
            os.killpg(os.getpgid(self._bag_proc.pid), signal.SIGINT)
            self._bag_proc.wait(timeout=10)
            self.get_logger().info("Bag recording stopped cleanly.")
        except subprocess.TimeoutExpired:
            self.get_logger().warn("Bag did not stop in time — killing.")
            os.killpg(os.getpgid(self._bag_proc.pid), signal.SIGKILL)
            self._bag_proc.wait()
        except Exception as e:
            self.get_logger().error(f"Error stopping bag: {e}")
        finally:
            self._bag_proc = None

    # ── cleanup on external shutdown ────────────────────────────────
    def destroy_node(self):
        self._stop_recording()
        super().destroy_node()


def main():
    rclpy.init()
    node = ArmTriggeredRecorder()

    # Handle SIGTERM/SIGINT from systemd gracefully
    def _shutdown(signum, frame):
        node.get_logger().info(f"Received signal {signum} — shutting down.")
        node._stop_recording()
        rclpy.shutdown()

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    try:
        rclpy.spin(node)
    except Exception:
        pass
    finally:
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()


if __name__ == "__main__":
    main()
