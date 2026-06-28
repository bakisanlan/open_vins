#!/usr/bin/env bash
# record_bag.sh – record a ros2 bag with predefined topics

# 1) Source ROS 2 (adjust distro if needed)

# 2) (Optional) Source your workspace, if you have one
# source ~/your_ros2_ws/install/setup.bash

# 3) Run the bag recording
ros2 bag record bag_09062025_0 \
  /camera/image_raw \
  /mavros/imu/data_raw \
  /imu/data \
  /mavros/global_position/local \
  /mavros/global_position/global \
  /ov_msckf/trackhist \
  /ov_msckf/pathimu \
  /ov_msckf/points_msckf \
  /ov_msckf/points_slam \
  /ov_msckf/points_aruco \
  /ov_msckf/loop_feats \
  /ov_msckf/points_sim \
  /ov_msckf/odomimu
