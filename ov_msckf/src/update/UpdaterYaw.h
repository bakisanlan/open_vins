/*
 * OpenVINS: An Open Platform for Visual-Inertial Research
 * Copyright (C) 2018-2023 Patrick Geneva
 * Copyright (C) 2018-2023 Guoquan Huang
 * Copyright (C) 2018-2023 OpenVINS Contributors
 * Copyright (C) 2018-2019 Kevin Eckenhoff
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#ifndef OV_MSCKF_UPDATER_YAW_H
#define OV_MSCKF_UPDATER_YAW_H

#include <memory>
#include <mutex>
#include <vector>

#include "update/UpdaterOptions.h"
#include "utils/sensor_data.h"

namespace ov_msckf {

class State;

/**
 * @brief Updater that fuses magnetometer-based yaw measurements into the EKF.
 *
 * This provides a global yaw reference from an external source (e.g. ArduPilot's EKF-fused heading
 * via MAVROS) to prevent yaw drift in GPS-denied VIO scenarios.
 *
 * The measurement model is:
 *   z = ψ_mag (yaw in ENU frame)
 *   h(x) = atan2(R_ItoG(1,0), R_ItoG(0,0))
 *
 * The Jacobian uses the full (non-approximate) derivation:
 *   H = 1/d² × [0, R_ItoG(2,1), R_ItoG(2,2)]
 *   d² = R_ItoG(0,0)² + R_ItoG(1,0)²
 */
class UpdaterYaw {

public:
  /**
   * @brief Default constructor
   * @param options Chi2 multipler and other options
   * @param sigma_yaw Standard deviation of yaw measurement (rad)
   * @param update_rate Maximum update rate in Hz (0 = unlimited)
   */
  UpdaterYaw(UpdaterOptions &options, double sigma_yaw, double update_rate);

  /**
   * @brief Feed a yaw measurement into the buffer
   * @param message Yaw measurement (timestamp, yaw in ENU rad, sigma)
   */
  void feed_yaw(const ov_core::YawData &message);

  /**
   * @brief Try to perform a yaw update on the state
   * @param state Current EKF state
   * @param timestamp Desired update time
   * @return True if the update was performed
   */
  bool try_update(std::shared_ptr<State> state, double timestamp);

  /**
   * @brief Check if we have any yaw measurements available
   * @return True if the yaw buffer is non-empty
   */
  bool has_data() const {
    std::lock_guard<std::mutex> lck(_mtx);
    return !_yaw_data.empty();
  }

  /**
   * @brief Get the most recent yaw measurement (for initialization)
   * @param data Output yaw data
   * @return True if a measurement was available
   */
  bool get_latest(ov_core::YawData &data) const {
    std::lock_guard<std::mutex> lck(_mtx);
    if (_yaw_data.empty())
      return false;
    data = _yaw_data.back();
    return true;
  }

private:
  /// Options (chi2 multiplier etc.)
  UpdaterOptions _options;

  /// Measurement noise standard deviation (rad)
  double _sigma_yaw;

  /// Maximum update rate (Hz), 0 = unlimited
  double _update_rate;

  /// Buffer of yaw measurements
  std::vector<ov_core::YawData> _yaw_data;

  /// Mutex for yaw data buffer
  mutable std::mutex _mtx;

  /// Last time we performed a yaw update
  double _last_update_time = -1;

  /// Chi squared lookup table (indexed by degrees of freedom)
  std::map<int, double> _chi_squared_table;
};

} // namespace ov_msckf

#endif // OV_MSCKF_UPDATER_YAW_H
