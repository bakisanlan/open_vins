#!/usr/bin/env bash
#
# launch_stack.sh — start the ROS 2 stack in four GNOME-Terminal tabs
# ──────────────────────────────────────────────────────────────────


gnome-terminal \
  --tab --title="RViz2" -- bash -c '
    echo "▶ Launching RViz2…"
    rviz2 -d ~/ros2_ws/src/open_vins/ov_msckf/launch/display_ros2.rviz
  '

gnome-terminal \
  --tab --title="Runner" -- bash -c '
    echo "▶ Running runner.py…"
    python runner.py
  ' 

