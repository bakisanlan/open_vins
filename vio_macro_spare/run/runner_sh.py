import subprocess
import time
import os
import signal
import sys

TMUX_SESSION = "ros2stack"
TMUX_WINDOW = "OV_MSCKF"


def signal_handler(signum, frame):
    """Signal handler for graceful shutdown"""
    print(f"\n[!] Received signal {signum}, shutting down...")
    if is_ros2_process_running():
        print("[-] Cleaning up ROS2 processes...")
        stop_ros2_processes()
    print("Yönetim döngüsü sonlandı.")
    sys.exit(0)


def condition_to_start():
    return os.path.exists("START")


def condition_to_stop():
    return not os.path.exists("START")


def is_ros2_process_running():
    """Check if the ROS2 tmux window is active"""
    try:
        # Check for tmux session and window
        result = subprocess.run(
            ["tmux", "list-windows", "-t", TMUX_SESSION],
            capture_output=True,
            text=True,
            check=False
        )
        return TMUX_WINDOW in result.stdout
    except:
        return False


def start_ros2_terminal():
    """Create a tmux session and window for ROS2 launch"""
    # Ensure the tmux session exists
    subprocess.run(
        f"tmux has-session -t {TMUX_SESSION} || tmux new-session -d -s {TMUX_SESSION}",
        shell=True
    )
    # Spawn a new window for the ROS2 node
    cmd = (
        f"tmux new-window -t {TMUX_SESSION} -n {TMUX_WINDOW} '"
        "ros2 launch ov_msckf subscribe.launch.py config:=my_config max_cameras:=1'"
    )
    subprocess.run(cmd, shell=True)
    print(f"[+] tmux window '{TMUX_WINDOW}' created in session '{TMUX_SESSION}'")


def stop_ros2_processes():
    """Kill the tmux window running the ROS2 launch"""
    print("[-] Stopping ROS2 tmux window...")
    try:
        subprocess.run(
            f"tmux kill-window -t {TMUX_SESSION}:{TMUX_WINDOW}",
            shell=True,
            check=False
        )
        print("[-] ROS2 tmux window terminated")
    except Exception as e:
        print(f"[!] Error stopping ROS2 tmux window: {e}")


def manage_process():
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    print("[i] Process manager started. Press Ctrl+C to stop.")
    print("[i] Or use: kill -TERM <pid> to stop gracefully")

    try:
        while True:
            should_start = condition_to_start()
            should_stop = condition_to_stop()
            is_running = is_ros2_process_running()

            if should_start and not is_running:
                start_ros2_terminal()
                time.sleep(3)
            elif should_stop and is_running:
                stop_ros2_processes()
                time.sleep(2)

            time.sleep(2)

    except KeyboardInterrupt:
        if is_ros2_process_running():
            print("[-] Cleaning up ROS2 processes...")
            stop_ros2_processes()
        print("Yönetim döngüsü sonlandı.")


if __name__ == "__main__":
    manage_process()

