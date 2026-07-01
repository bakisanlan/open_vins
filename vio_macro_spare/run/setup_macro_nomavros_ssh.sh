#!/usr/bin/env bash
#
# launch_stack_tmux.sh — start the ROS 2 stack in a tmux session with 3 panes (including system monitoring)
# usage: chmod +x launch_stack_tmux.sh && ./launch_stack_tmux.sh

SESSION="ros2stack"

# Kill existing session (if any) and start a new detached session
tmux kill-session -t $SESSION 2>/dev/null
tmux new-session -d -s $SESSION -n main

# Split the window into three panes
tmux split-window -h -t $SESSION:0.0
tmux split-window -v -t $SESSION:0.1  # Create a new pane for system monitoring

# Pane 0 (left): Camera driver
tmux send-keys -t $SESSION:0.0 'echo "▶ Starting camera driver…"' C-m \
		'fan &' C-m \
                'ros2 launch camera_driver camera_driver.launch.py' C-m

# Pane 1 (top-right): Runner
tmux send-keys -t $SESSION:0.1 'echo "▶ Running runner_sh.py…"' C-m \
		'sleep 10'  C-m \
                'python runner_sh.py' C-m

# Pane 2 (bottom-right): System Monitoring (CPU, RAM, Temp, GPU) + Display Log
tmux send-keys -t $SESSION:0.2 \
  'echo "▶ Starting system monitoring (CPU, RAM, Temp, GPU)…"' C-m \
  'while true; do' C-m \
  '  timestamp=$(date +"%Y-%m-%d %H:%M:%S")' C-m \
  '  cpu=$(mpstat 1 1 | awk "/Average:/ {print 100-\$12}")' C-m \
  '  cpu=$(printf "%.2f" "$cpu")' C-m \
  '  ram=$(free -m | awk "NR==2{printf \"%.2f\", \$3*100/\$2 }")' C-m \
  '  ram=$(printf "%.2f" "$ram")' C-m \
  '  temp=$(cat /sys/class/thermal/thermal_zone0/temp)' C-m \
  '  temp=$(printf "%.2f" "$((${temp}/1000))")' C-m \
  '  gpu_load=$(cat /sys/devices/platform/bus@0/17000000.gpu/load 2>/dev/null || echo "0")' C-m \
  '  gpu_util=$(awk "BEGIN{printf \"%.1f\", $gpu_load/10}")' C-m \
  '  gpu_freq=$(cat /sys/devices/platform/bus@0/17000000.gpu/devfreq/17000000.gpu/cur_freq 2>/dev/null || echo "0")' C-m \
  '  gpu_freq_mhz=$(awk "BEGIN{printf \"%.0f\", $gpu_freq/1000000}")' C-m \
  '  echo "$timestamp, CPU: ${cpu}%, RAM: ${ram}%, Temp: ${temp}°C, GPU: ${gpu_util}%, GPU_FREQ: ${gpu_freq_mhz}MHz" >> system_monitor.log' C-m \
  '  sleep 0.5' C-m \
  'done' C-m

# Start tailing the log file to display its contents in Pane 3
tmux send-keys -t $SESSION:0.2 'echo "▶ Showing live log from system_monitor.log…" && tail -f system_monitor.log' C-m

# Arrange the panes evenly
tmux select-layout -t $SESSION:0 even-horizontal

# Attach to the session
tmux attach -t $SESSION

