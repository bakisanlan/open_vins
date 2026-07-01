#!/usr/bin/env bash
#
# record_bag_ov_msckf_service.sh — wrapper for systemd
# Sources ROS 2 environment then runs the arm-triggered bag recorder.
# Recording starts when vehicle arms and stops when it disarms.

set -o pipefail  # no -e/-u: ROS setup.bash uses unbound variables

export HOME=/home/ituarc
export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:$PATH"

# Source ROS 2 environment (setup.bash uses unbound vars, so no -u)
source /opt/ros/humble/setup.bash
source /home/ituarc/ros2_ws/install/setup.bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Run the arm-triggered bag recorder
exec python3 "$SCRIPT_DIR/record_bag_on_arm.py"
