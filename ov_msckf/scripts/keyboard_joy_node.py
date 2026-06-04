#!/usr/bin/env python3
"""
Keyboard Joy Node — ArduPilot RC simulation via /ap/joy
=========================================================

Simulates a joystick by reading keyboard input and publishing
sensor_msgs/Joy to /ap/joy.

Key Mapping:
  W / S          → Throttle up / down  (holds value, incremental)
  ↑ / ↓ Arrow   → Pitch forward / back (returns to 0 on release)
  ← / → Arrow   → Roll left / right    (returns to 0 on release)
  A / D          → Yaw left / right     (returns to 0 on release)
  SPACE          → Throttle → 0 (kill)
  Q / ESC        → Quit

Joy axes layout (sensor_msgs/Joy):
  axes[0] = Roll     [-1=left,  +1=right]
  axes[1] = Pitch    [-1=fwd,   +1=back ]
  axes[2] = Throttle [ 0=low,   +1=high ] (note: starts at 0)
  axes[3] = Yaw      [-1=left,  +1=right]

Run:
  ros2 run ov_msckf keyboard_joy_node.py
"""

import sys
import math
import time
import threading
import curses

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Joy


# ──────────────────────────────────────────────────────────────────────────────
# Shared state (updated by curses loop, read by ROS publisher)
# ──────────────────────────────────────────────────────────────────────────────
class JoyState:
    def __init__(self):
        self.lock = threading.Lock()
        self.roll     = 0.0   # axes[0]
        self.pitch    = 0.0   # axes[1]
        self.throttle = 0.0   # axes[2]
        self.yaw      = 0.0   # axes[3]
        self.quit     = False

    def snapshot(self):
        with self.lock:
            return self.roll, self.pitch, self.throttle, self.yaw

    def set(self, **kwargs):
        with self.lock:
            for k, v in kwargs.items():
                setattr(self, k, v)


# ──────────────────────────────────────────────────────────────────────────────
# ROS2 publisher node
# ──────────────────────────────────────────────────────────────────────────────
class KeyboardJoyNode(Node):
    def __init__(self, state: JoyState):
        super().__init__("keyboard_joy_node")

        self.declare_parameter("topic_joy",        "/ap/joy")
        self.declare_parameter("publish_rate_hz",  20.0)
        self.declare_parameter("throttle_step",    0.02)   # per key-press cycle
        self.declare_parameter("axis_step",        0.05)   # roll/pitch/yaw step

        self.topic_joy    = self.get_parameter("topic_joy").value
        rate_hz           = self.get_parameter("publish_rate_hz").value
        self.thr_step     = self.get_parameter("throttle_step").value
        self.axis_step    = self.get_parameter("axis_step").value

        self.state = state
        self.pub   = self.create_publisher(Joy, self.topic_joy, 10)
        self.timer = self.create_timer(1.0 / rate_hz, self._publish)

        self.get_logger().info(
            f"KeyboardJoy → publishing to '{self.topic_joy}' @ {rate_hz:.0f} Hz"
        )

    def _publish(self):
        roll, pitch, throttle, yaw = self.state.snapshot()

        msg = Joy()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = ""
        # axes: [roll, pitch, throttle, yaw]
        msg.axes    = [float(roll), float(pitch), float(throttle), float(yaw)]
        msg.buttons = []
        self.pub.publish(msg)


# ──────────────────────────────────────────────────────────────────────────────
# Curses UI — runs in main thread
# ──────────────────────────────────────────────────────────────────────────────
def curses_loop(stdscr, state: JoyState, thr_step: float, axis_step: float):
    curses.curs_set(0)
    stdscr.nodelay(True)   # non-blocking getch
    stdscr.timeout(50)     # 20 Hz refresh

    # Color pairs
    curses.start_color()
    curses.init_pair(1, curses.COLOR_CYAN,    curses.COLOR_BLACK)
    curses.init_pair(2, curses.COLOR_GREEN,   curses.COLOR_BLACK)
    curses.init_pair(3, curses.COLOR_YELLOW,  curses.COLOR_BLACK)
    curses.init_pair(4, curses.COLOR_RED,     curses.COLOR_BLACK)
    curses.init_pair(5, curses.COLOR_WHITE,   curses.COLOR_BLACK)

    # Pressed-key tracking for auto-return axes
    keys_held = set()

    def clamp(v, lo=-1.0, hi=1.0):
        return max(lo, min(hi, v))

    def bar(val, lo=-1.0, hi=1.0, width=30):
        frac = (val - lo) / (hi - lo)
        filled = int(frac * width)
        return "█" * filled + "░" * (width - filled)

    while not state.quit:
        # ── Key input ──
        try:
            key = stdscr.getch()
        except Exception:
            key = -1

        with state.lock:
            roll     = state.roll
            pitch    = state.pitch
            throttle = state.throttle
            yaw      = state.yaw

        if key == ord('q') or key == 27:   # q or ESC
            state.quit = True
            break

        # Throttle (holds value)
        if key == ord('w'):
            throttle = clamp(throttle + thr_step, 0.0, 1.0)
        elif key == ord('s'):
            throttle = clamp(throttle - thr_step, 0.0, 1.0)
        elif key == ord(' '):
            throttle = 0.0

        # Roll (arrow keys: left/right)
        if key == curses.KEY_RIGHT:
            roll = clamp(roll + axis_step)
            keys_held.add('roll+')
        elif key == curses.KEY_LEFT:
            roll = clamp(roll - axis_step)
            keys_held.add('roll-')
        else:
            roll = roll * 0.75   # decay when not pressed

        # Pitch (arrow keys: up/down)
        if key == curses.KEY_UP:
            pitch = clamp(pitch + axis_step)
        elif key == curses.KEY_DOWN:
            pitch = clamp(pitch - axis_step)
        else:
            pitch = pitch * 0.75

        # Yaw (A/D)
        if key == ord('d'):
            yaw = clamp(yaw + axis_step)
        elif key == ord('a'):
            yaw = clamp(yaw - axis_step)
        else:
            yaw = yaw * 0.75

        # Snap near-zero to zero
        if abs(roll)  < 0.01: roll  = 0.0
        if abs(pitch) < 0.01: pitch = 0.0
        if abs(yaw)   < 0.01: yaw   = 0.0

        with state.lock:
            state.roll     = roll
            state.pitch    = pitch
            state.throttle = throttle
            state.yaw      = yaw

        # ── Draw UI ──
        stdscr.erase()
        h, w = stdscr.getmaxyx()
        row = 0

        def put(r, c, text, pair=5, bold=False):
            attr = curses.color_pair(pair)
            if bold:
                attr |= curses.A_BOLD
            try:
                stdscr.addstr(r, c, text, attr)
            except curses.error:
                pass

        put(row, 2, "╔══════════════════════════════════════════╗", 1, True); row += 1
        put(row, 2, "║      Keyboard Joy — ArduPilot RC Sim     ║", 1, True); row += 1
        put(row, 2, "╚══════════════════════════════════════════╝", 1, True); row += 1
        row += 1

        # Controls legend
        put(row, 2, "  Controls:", 3, True); row += 1
        put(row, 2, "  W / S        : Throttle  ↑ / ↓  (holds)"); row += 1
        put(row, 2, "  ↑ / ↓ Arrow  : Pitch     fwd / back"); row += 1
        put(row, 2, "  ← / → Arrow  : Roll      left / right"); row += 1
        put(row, 2, "  A / D        : Yaw       left / right"); row += 1
        put(row, 2, "  SPACE        : Throttle kill"); row += 1
        put(row, 2, "  Q / ESC      : Quit"); row += 1
        row += 1

        # Value bars
        put(row, 2, "─" * 44, 1); row += 1

        THR_COLOR = 2 if throttle > 0.3 else 4
        put(row, 2, f"  THROTTLE  {throttle:+.3f}  [{bar(throttle,0,1)}]", THR_COLOR, True); row += 1

        ROLL_COLOR = 3 if abs(roll) > 0.05 else 5
        put(row, 2, f"  ROLL      {roll:+.3f}  [{bar(roll)}]", ROLL_COLOR); row += 1

        PITCH_COLOR = 3 if abs(pitch) > 0.05 else 5
        put(row, 2, f"  PITCH     {pitch:+.3f}  [{bar(pitch)}]", PITCH_COLOR); row += 1

        YAW_COLOR = 3 if abs(yaw) > 0.05 else 5
        put(row, 2, f"  YAW       {yaw:+.3f}  [{bar(yaw)}]", YAW_COLOR); row += 1

        row += 1
        put(row, 2, "─" * 44, 1); row += 1
        put(row, 2, "  Joy axes: [roll, pitch, throttle, yaw]", 5); row += 1
        put(row, 2,
            f"  [{roll:+.3f}, {pitch:+.3f}, {throttle:+.3f}, {yaw:+.3f}]", 2); row += 1

        stdscr.refresh()


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────
def main(args=None):
    rclpy.init(args=args)

    state = JoyState()

    # Spin ROS in a background thread
    node = KeyboardJoyNode(state)
    ros_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    ros_thread.start()

    thr_step  = node.thr_step
    axis_step = node.axis_step

    try:
        curses.wrapper(curses_loop, state, thr_step, axis_step)
    except KeyboardInterrupt:
        pass
    finally:
        state.quit = True
        node.destroy_node()
        rclpy.shutdown()
        ros_thread.join(timeout=2.0)


if __name__ == "__main__":
    main()
