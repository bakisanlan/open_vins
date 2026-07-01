#!/usr/bin/env bash
#
# launch_stack_tmux.sh — start the ROS 2 stack in a tmux session with 3 side-by-side panes
# usage: chmod +x launch_stack_tmux.sh && ./launch_stack_tmux.sh

SESSION="ros2stack"

# Kill existing session (if any) and start a new detached session
tmux kill-session -t $SESSION 2>/dev/null
tmux new-session -d -s $SESSION -n main

# Split the window into three side-by-side panes
tmux split-window -h -t $SESSION:0.0
tmux split-window -h -t $SESSION:0.0

# Pane 0 (leftmost): MAVROS
tmux send-keys -t "$SESSION":0.0 \
  'echo "▶ Launching MAVROS…"' C-m \
  'ros2 launch mavros px4.launch fcu_url:=/dev/ttyACM0:115200 &' C-m \
  'sleep 5' C-m \
  'ros2 service call /mavros/set_stream_rate mavros_msgs/srv/StreamRate "{stream_id: 0, message_rate: 1, on_off: true}"' C-m \
  'ros2 service call /mavros/set_message_interval mavros_msgs/srv/MessageInterval "{message_id: 33,  message_rate: 20.0}"' C-m \
  'ros2 service call /mavros/set_message_interval mavros_msgs/srv/MessageInterval "{message_id: 27, message_rate: 200.0}"' C-m \
  'ros2 service call /mavros/set_message_interval mavros_msgs/srv/MessageInterval "{message_id: 105, message_rate: 200.0}"' C-m \
  'ros2 topic hz /mavros/imu/data_raw -w 20' C-m

# Pane 1 (middle): Camera driver
tmux send-keys -t $SESSION:0.1 'echo "▶ Starting camera driver…"' C-m \
                'ros2 launch camera_driver camera_driver.launch.py' C-m

# Arrange the panes evenly
tmux select-layout -t $SESSION:0 even-horizontal

# Attach to the session
tmux attach -t $SESSION

