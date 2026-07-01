import subprocess
import time
import os
import signal
import sys

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
    """ROS2 ov_msckf process'inin çalışıp çalışmadığını kontrol et"""
    try:
        result = subprocess.run(
            ["pgrep", "-f", "run_subscribe_msckf"], 
            capture_output=True, 
            text=True, 
            check=False
        )
        return len(result.stdout.strip()) > 0
    except:
        return False

def start_ros2_terminal():
    """ROS2 terminal'ini başlat"""
    cmd = '''gnome-terminal \\
        --tab --title="OV_MSCKF" -- bash -c '
        echo "▶ Launching OV_MSCKF…"
        ros2 launch ov_msckf subscribe.launch.py config:=my_config max_cameras:=1
        ' '''
    
    subprocess.Popen(cmd, shell=True)
    print("[+] Gnome-terminal with ROS2 process started")

def stop_ros2_processes():
    """ROS2 proceslerini durdur"""
    print("[-] Stopping ROS2 processes...")
    try:
        subprocess.run(["pkill", "-f", "run_subscribe_msckf"], check=False)
        #subprocess.run(["pkill", "-f", "ros2 launch"], check=False)
        print("[-] ROS2 processes stopped")
    except Exception as e:
        print(f"[!] Error stopping processes: {e}")

def manage_process():
    # Signal handler'ları kaydet
    signal.signal(signal.SIGINT, signal_handler)   # Ctrl+C
    signal.signal(signal.SIGTERM, signal_handler)  # kill komutu
    
    print("[i] Process manager started. Press Ctrl+C to stop.")
    print("[i] Or use: kill -TERM <pid> to stop gracefully")
    
    try:
        while True:
            should_start = condition_to_start()
            should_stop = condition_to_stop()
            is_running = is_ros2_process_running()
            
            if should_start and not is_running:
                start_ros2_terminal()
                time.sleep(3)  # Terminal'in açılması için bekle
                
            elif should_stop and is_running:
                stop_ros2_processes()
                time.sleep(2)  # Process'lerin kapanması için bekle
            
            time.sleep(2)  # döngü periyodu
            
    except KeyboardInterrupt:
        if is_ros2_process_running():
            print("[-] Cleaning up ROS2 processes...")
            stop_ros2_processes()
        print("Yönetim döngüsü sonlandı.")

if __name__ == "__main__":
    manage_process()
