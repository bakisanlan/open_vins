#!/usr/bin/env bash
#
# install_services.sh — Install and enable the VIO macro systemd services
# Usage: sudo bash install_services.sh
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== Installing VIO Macro systemd services ==="

# 1) Make wrapper scripts executable
echo "[1/4] Making wrapper scripts executable..."
chmod +x /home/ituarc/Desktop/vio_macro/run/setup_macro_nomavros_ssh_service.sh
chmod +x /home/ituarc/Desktop/vio_macro/bag/record_bag_ov_msckf_service.sh

# 2) Copy service files to systemd directory
echo "[2/4] Copying service files to /etc/systemd/system/..."
cp "$SCRIPT_DIR/vio-macro-stack.service" /etc/systemd/system/
cp "$SCRIPT_DIR/vio-macro-bag-recorder.service" /etc/systemd/system/

# 3) Reload systemd daemon
echo "[3/4] Reloading systemd daemon..."
systemctl daemon-reload

# 4) Enable services for auto-start on boot
echo "[4/4] Enabling services..."
systemctl enable vio-macro-stack.service
systemctl enable vio-macro-bag-recorder.service

echo ""
echo "=== Installation complete ==="
echo ""
echo "Services are now enabled and will auto-start on reboot."
echo ""
echo "Useful commands:"
echo "  sudo systemctl start vio-macro-stack          # Start the ROS stack now"
echo "  sudo systemctl start vio-macro-bag-recorder   # Start the bag recorder now"
echo "  sudo systemctl status vio-macro-stack         # Check stack status"
echo "  sudo systemctl status vio-macro-bag-recorder  # Check recorder status"
echo "  journalctl -u vio-macro-stack -f              # Follow stack logs"
echo "  journalctl -u vio-macro-bag-recorder -f       # Follow recorder logs"
echo "  sudo systemctl stop vio-macro-stack           # Stop both (recorder depends on stack)"
echo "  sudo systemctl disable vio-macro-stack        # Disable auto-start"
echo "  sudo systemctl disable vio-macro-bag-recorder # Disable auto-start"
