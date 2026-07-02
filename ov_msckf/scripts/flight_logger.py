"""
Flight Logger Utility
=====================
Provides a tee-logger that mirrors ROS 2 node log output to a file under:
    <scripts_dir>/logs/flight_<N>/<log_name>.log

Auto-increments flight_N.  If the latest flight_N directory was created
within the last 5 minutes, both nodes will reuse it (shared session).
"""

import os
import time
import logging
import re


class TeeLogger:
    """Wraps a ROS 2 logger and mirrors all output to a Python file logger."""

    # rclpy kwargs that we handle ourselves (stripped before passing to rclpy)
    _THROTTLE_KEYS = {"throttle_duration_sec", "throttle_time_source_type",
                      "skip_first", "once"}

    def __init__(self, ros_logger, file_path):
        self._ros = ros_logger
        self._py = logging.getLogger(f"tee.{ros_logger.name}")
        self._py.setLevel(logging.DEBUG)
        self._py.propagate = False
        # Avoid adding duplicate handlers on re-init
        if not self._py.handlers:
            fh = logging.FileHandler(file_path)
            fh.setFormatter(
                logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
            )
            self._py.addHandler(fh)
        # Unified throttle state: {caller_key: last_log_time}
        self._throttle = {}

    def _should_log(self, kwargs):
        """Check throttle; return True if the message should be emitted."""
        throttle = kwargs.get("throttle_duration_sec", 0)
        if throttle > 0:
            import sys
            frame = sys._getframe(3)  # 0=_should, 1=_log, 2=info/etc, 3=caller
            key = f"{frame.f_code.co_filename}:{frame.f_lineno}"
            now = time.monotonic()
            last = self._throttle.get(key, 0.0)
            if (now - last) < throttle:
                return False
            self._throttle[key] = now
        return True

    def _log(self, ros_level, py_level, msg, *args, **kwargs):
        """Unified log: throttle ourselves, then emit to both ROS and file."""
        if not self._should_log(kwargs):
            return
        # Strip rclpy-specific kwargs to avoid ValueError collisions
        clean = {k: v for k, v in kwargs.items() if k not in self._THROTTLE_KEYS}
        getattr(self._ros, ros_level)(msg, *args, **clean)
        getattr(self._py, py_level)(msg)

    def info(self, msg, *args, **kwargs):
        self._log("info", "info", msg, *args, **kwargs)

    def warn(self, msg, *args, **kwargs):
        self._log("warn", "warning", msg, *args, **kwargs)

    def warning(self, msg, *args, **kwargs):
        self._log("warning", "warning", msg, *args, **kwargs)

    def error(self, msg, *args, **kwargs):
        self._log("error", "error", msg, *args, **kwargs)

    def debug(self, msg, *args, **kwargs):
        self._log("debug", "debug", msg, *args, **kwargs)

    def fatal(self, msg, *args, **kwargs):
        self._log("fatal", "critical", msg, *args, **kwargs)

    # Delegate everything else (e.g. .name, .get_child) to the ROS logger
    def __getattr__(self, name):
        return getattr(self._ros, name)


def _uptime():
    """Seconds since boot (immune to wall-clock changes/NTP)."""
    try:
        with open("/proc/uptime") as f:
            return float(f.read().split()[0])
    except Exception:
        return time.monotonic()


def get_flight_dir(scripts_dir=None, reuse_window_sec=30):
    """
    Return the path to the current flight log directory.

    Uses a persistent ``counter.txt`` file to track the flight number.
    A ``.session`` marker (with system uptime) allows nodes launched
    together to share the same ``flight_N`` directory without relying
    on the wall clock (works offline / without NTP).

    Parameters
    ----------
    scripts_dir : str or None
        Base scripts directory.  Defaults to the directory containing this file.
    reuse_window_sec : float
        Maximum uptime delta (seconds) to reuse the current session.
    """
    if scripts_dir is None:
        scripts_dir = os.path.dirname(os.path.abspath(__file__))

    logs_base = os.path.join(scripts_dir, "logs")
    os.makedirs(logs_base, exist_ok=True)

    counter_path = os.path.join(logs_base, "counter.txt")
    session_path = os.path.join(logs_base, ".session")

    import fcntl
    fd = os.open(counter_path, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)

        # Read current counter
        content = os.read(fd, 100).decode().strip()
        current_n = int(content) if content else 0

        # Check if we should reuse the current flight directory
        reuse = False
        now_up = _uptime()
        if os.path.isfile(session_path):
            try:
                parts = open(session_path).read().strip().split()
                sess_n = int(parts[0])
                sess_up = float(parts[1])
                # Same flight number AND within reuse window (by uptime)
                # Uptime resets on reboot → old sessions are never reused
                if sess_n == current_n and 0 <= (now_up - sess_up) < reuse_window_sec:
                    reuse = True
            except (ValueError, IndexError):
                pass

        if not reuse:
            current_n += 1
            # Write updated counter
            os.lseek(fd, 0, os.SEEK_SET)
            os.ftruncate(fd, 0)
            os.write(fd, str(current_n).encode())

        flight_dir = os.path.join(logs_base, f"flight_{current_n}")
        os.makedirs(flight_dir, exist_ok=True)

        # Write/update session marker
        with open(session_path, "w") as sf:
            sf.write(f"{current_n} {now_up:.3f}\n")

    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)

    return flight_dir


def setup_flight_logger(node, log_filename):
    """
    Set up file logging for a ROS 2 node.

    Creates a TeeLogger that mirrors all ``node.get_logger()`` output to
    ``<flight_dir>/<log_filename>``, and monkey-patches ``node.get_logger``
    to return it.

    Parameters
    ----------
    node : rclpy.node.Node
        The ROS 2 node instance.
    log_filename : str
        Name of the log file (e.g. ``"cbf.log"`` or ``"traj.log"``).

    Returns
    -------
    str
        Absolute path to the flight directory (e.g. ``scripts/logs/flight_13/``).
    """
    scripts_dir = os.path.dirname(os.path.abspath(__file__))
    flight_dir = get_flight_dir(scripts_dir)
    log_path = os.path.join(flight_dir, log_filename)

    tee = TeeLogger(node.get_logger(), log_path)

    # Monkey-patch so all future get_logger() calls return the tee wrapper
    node.get_logger = lambda: tee

    # Store flight dir on the node for other uses (e.g. CSV logging)
    node._flight_dir = flight_dir

    tee.info(f"Flight log: {log_path}")
    return flight_dir
