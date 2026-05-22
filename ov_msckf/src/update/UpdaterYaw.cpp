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

#include "UpdaterYaw.h"

#include "state/State.h"
#include "state/StateHelper.h"
#include "utils/colors.h"
#include "utils/print.h"
#include "utils/quat_ops.h"

#include <boost/math/distributions/chi_squared.hpp>

using namespace ov_core;
using namespace ov_type;
using namespace ov_msckf;

UpdaterYaw::UpdaterYaw(UpdaterOptions &options, double sigma_yaw, double update_rate)
    : _options(options), _sigma_yaw(sigma_yaw), _update_rate(update_rate) {

  // Pre-compute chi squared table (we only need 1 DOF for yaw)
  for (int i = 1; i < 10; i++) {
    boost::math::chi_squared chi_squared_dist(i);
    _chi_squared_table[i] = boost::math::quantile(chi_squared_dist, 0.95);
  }

  PRINT_DEBUG("[YAW]: Yaw updater created with sigma=%.4f rad (%.2f deg), rate=%.1f Hz, chi2_mult=%.1f\n", _sigma_yaw,
              _sigma_yaw * 180.0 / M_PI, _update_rate, _options.chi2_multipler);
}

void UpdaterYaw::feed_yaw(const ov_core::YawData &message) {
  std::lock_guard<std::mutex> lck(_mtx);
  _yaw_data.push_back(message);

  // Keep buffer bounded (at most 100 measurements)
  if (_yaw_data.size() > 100) {
    _yaw_data.erase(_yaw_data.begin());
  }
}

bool UpdaterYaw::try_update(std::shared_ptr<State> state, double timestamp) {

  // Get the closest yaw measurement to the desired timestamp
  ov_core::YawData yaw_meas;
  {
    std::lock_guard<std::mutex> lck(_mtx);

    if (_yaw_data.empty()) {
      return false;
    }

    // Find the measurement closest to the requested timestamp
    double best_dt = std::numeric_limits<double>::max();
    int best_idx = -1;
    for (int i = 0; i < (int)_yaw_data.size(); i++) {
      double dt = std::abs(_yaw_data[i].timestamp - timestamp);
      if (dt < best_dt) {
        best_dt = dt;
        best_idx = i;
      }
    }

    // Reject if the closest measurement is too far from the state time (>0.5s)
    if (best_dt > 0.5) {
      return false;
    }

    yaw_meas = _yaw_data[best_idx];

    // Remove all measurements older than or equal to this one
    _yaw_data.erase(_yaw_data.begin(), _yaw_data.begin() + best_idx + 1);
  }

  // Check rate throttling
  if (_update_rate > 0 && _last_update_time > 0) {
    double dt_since_last = timestamp - _last_update_time;
    if (dt_since_last < (1.0 / _update_rate)) {
      return false;
    }
  }

  //==========================================================================
  // Compute the predicted yaw from the CURRENT state (for residual)
  //==========================================================================

  // Residual always uses current estimate (NOT FEJ)
  Eigen::Matrix3d R_GtoI_cur = state->_imu->Rot();
  Eigen::Matrix3d R_ItoG_cur = R_GtoI_cur.transpose();

  // Extract predicted yaw: ψ = atan2(R_ItoG(1,0), R_ItoG(0,0))
  double yaw_est = std::atan2(R_ItoG_cur(1, 0), R_ItoG_cur(0, 0));

  //==========================================================================
  // Compute residual (with angle wrapping)
  //==========================================================================

  double res_yaw = yaw_meas.yaw - yaw_est;

  // Wrap residual to [-π, π]
  while (res_yaw > M_PI)
    res_yaw -= 2.0 * M_PI;
  while (res_yaw < -M_PI)
    res_yaw += 2.0 * M_PI;

  Eigen::VectorXd res(1);
  res(0) = res_yaw;

  //==========================================================================
  // Compute Jacobian (FULL, no small-angle approximation)
  //==========================================================================
  //
  // Jacobian uses FEJ rotation if enabled (for consistent linearization)
  // Residual above uses current rotation (for correct innovation)
  //
  // Measurement: ψ = atan2(R_ItoG(1,0), R_ItoG(0,0))
  //
  // Using left-multiplicative error model:
  //   R_GtoI = (I - [δθ×]) R̂_GtoI
  //   R_ItoG = R̂_ItoG (I + [δθ×])
  //
  // Full chain rule through atan2:
  //   ∂ψ/∂δθ = 1/d² × (R̂_ItoG(0,0) × ∂R_ItoG(1,0)/∂δθ − R̂_ItoG(1,0) × ∂R_ItoG(0,0)/∂δθ)
  //
  // Using rotation matrix cofactor property:
  //   H = 1/d² × [0, R̂_ItoG(2,1), R̂_ItoG(2,2)]
  //
  // This is exact for any attitude.
  //==========================================================================

  // Use FEJ rotation for Jacobian linearization point (if enabled)
  Eigen::Matrix3d R_GtoI_lin = (state->_options.do_fej) ? state->_imu->Rot_fej() : state->_imu->Rot();
  Eigen::Matrix3d R_ItoG_lin = R_GtoI_lin.transpose();

  double d_sq = R_ItoG_lin(0, 0) * R_ItoG_lin(0, 0) + R_ItoG_lin(1, 0) * R_ItoG_lin(1, 0);

  // Guard against gimbal lock (pitch = ±90°)
  if (d_sq < 1e-6) {
    PRINT_WARNING(YELLOW "[YAW]: near gimbal lock (d²=%.6f), skipping yaw update\n" RESET, d_sq);
    return false;
  }

  // Jacobian of yaw w.r.t. orientation error state δθ (1×3)
  Eigen::MatrixXd H = Eigen::MatrixXd::Zero(1, 3);
  H(0, 0) = 0.0;
  H(0, 1) = R_ItoG_lin(2, 1) / d_sq;
  H(0, 2) = R_ItoG_lin(2, 2) / d_sq;

  // Order of state variables for the update
  std::vector<std::shared_ptr<Type>> Hx_order;
  Hx_order.push_back(state->_imu->q());

  //==========================================================================
  // Measurement noise
  //==========================================================================

  // Use the per-measurement sigma if available, otherwise default
  double sigma = (yaw_meas.sigma > 0) ? yaw_meas.sigma : _sigma_yaw;
  Eigen::MatrixXd R = Eigen::MatrixXd::Identity(1, 1) * sigma * sigma;

  //==========================================================================
  // Chi2 distance check
  //==========================================================================

  Eigen::MatrixXd P_marg = StateHelper::get_marginal_covariance(state, Hx_order);
  Eigen::MatrixXd S = H * P_marg * H.transpose() + R;
  double chi2 = res.dot(S.llt().solve(res));
  double chi2_check = _chi_squared_table[1]; // 1 DOF

  if (chi2 > _options.chi2_multipler * chi2_check) {
    PRINT_WARNING(YELLOW "[YAW]: chi2 rejected (%.3f > %.3f), res=%.3f deg\n" RESET, chi2, _options.chi2_multipler * chi2_check,
                  res_yaw * 180.0 / M_PI);
    return false;
  }



  //==========================================================================
  // Perform EKF update
  //==========================================================================

  // Print diagnostic: yaw std dev from covariance
  double yaw_std_deg = std::sqrt(S(0, 0)) * 180.0 / M_PI;
  double P_yaw_std_deg = std::sqrt((H * P_marg * H.transpose())(0, 0)) * 180.0 / M_PI;

  StateHelper::EKFUpdate(state, Hx_order, H, res, R);
  _last_update_time = timestamp;

  // Post-update yaw
  Eigen::Matrix3d R_ItoG_post = state->_imu->Rot().transpose();
  double yaw_post = std::atan2(R_ItoG_post(1, 0), R_ItoG_post(0, 0));

  PRINT_INFO(CYAN "[YAW]: res=%.1f deg, chi2=%.1f, yaw_est=%.1f→%.1f deg, yaw_mag=%.1f deg, P_yaw=%.2f deg, S=%.2f deg\n" RESET,
             res_yaw * 180.0 / M_PI, chi2, yaw_est * 180.0 / M_PI, yaw_post * 180.0 / M_PI, yaw_meas.yaw * 180.0 / M_PI, P_yaw_std_deg,
             yaw_std_deg);

  return true;
}
