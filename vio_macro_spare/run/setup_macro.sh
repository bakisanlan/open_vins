#!/usr/bin/env bash
#
# launch_stack.sh — start the ROS 2 stack in four GNOME-Terminal tabs
# ──────────────────────────────────────────────────────────────────


gnome-terminal \
  --tab --title="MAVROS" -- bash -c '
    echo "▶ Launching MAVROS…"
    ros2 launch mavros apm.launch fcu_url:=/dev/ttyACM0:115200 &
    sleep 5
    ros2 service call /mavros/set_stream_rate mavros_msgs/srv/StreamRate \
        "{stream_id: 0, message_rate: 1, on_off: true}" 
    ros2 service call /mavros/set_message_interval mavros_msgs/srv/MessageInterval \
        "{message_id: 33,  message_rate: 20.0}"
    ros2 service call /mavros/set_message_interval mavros_msgs/srv/MessageInterval \
        "{message_id: 27,  message_rate: 200.0}"
    ros2 service call /mavros/set_message_interval mavros_msgs/srv/MessageInterval \
        "{message_id: 105, message_rate: 200.0}" 
    ros2 service call /mavros/set_message_interval mavros_msgs/srv/MessageInterval \
    "{message_id: 32, message_rate: 50.0}" # /mavros/local_position/pose 
    
    ros2 topic hz /mavros/imu/data_raw -w 200

  '

gnome-terminal \
  --tab --title="Camera driver" -- bash -c '
    echo "▶ Starting camera driver…"
    ros2 launch camera_driver camera_driver.launch.py
  '

gnome-terminal \
  --tab --title="RViz2" -- bash -c '
    echo "▶ Launching RViz2…"
    rviz2 -d ~/ros2_ws/src/open_vins/ov_msckf/launch/display_ros2.rviz
  '

gnome-terminal \
  --tab --title="Runner" -- bash -c '
    echo "▶ Running runner.py…"
    sleep 20
    python runner.py
  ' 

