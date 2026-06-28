#!/bin/bash

export HOME=/home/ituarc
source /opt/ros/humble/setup.bash
source /home/ituarc/ros2_ws/install/setup.bash


ros2 launch mavros apm.launch fcu_url:=/dev/ttyACM0:115200 &
sleep 10

# 2.10 Set MAVLink stream and message rates
ros2 service call /mavros/set_stream_rate mavros_msgs/srv/StreamRate \
    "{stream_id: 0, message_rate: 1, on_off: true}"

ros2 service call /mavros/set_message_interval mavros_msgs/srv/MessageInterval \
    "{message_id: 33,  message_rate: 20.0}"  # IMU scaled

ros2 service call /mavros/set_message_interval mavros_msgs/srv/MessageInterval \
    "{message_id: 27,  message_rate: 200.0}" # Raw IMU

ros2 service call /mavros/set_message_interval mavros_msgs/srv/MessageInterval \
    "{message_id: 31, message_rate: 200.0}" # Attitude quaternion

