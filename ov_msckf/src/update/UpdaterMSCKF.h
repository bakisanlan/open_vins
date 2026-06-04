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

#ifndef OV_MSCKF_UPDATER_MSCKF_H
#define OV_MSCKF_UPDATER_MSCKF_H

#include <Eigen/Eigen>
#include <memory>

#include "feat/FeatureInitializerOptions.h"

#include "UpdaterOptions.h"

namespace ov_core {
class Feature;
class FeatureInitializer;
} // namespace ov_core

namespace ov_msckf {

class State;

/**
 * @brief Will compute the system for our sparse features and update the filter.
 *
 * This class is responsible for computing the entire linear system for all features that are going to be used in an update.
 * This follows the original MSCKF, where we first triangulate features, we then nullspace project the feature Jacobian.
 * After this we compress all the measurements to have an efficient update and update the state.
 */
class UpdaterMSCKF {

public:
  /**
   * @brief CBF observability metrics computed during each MSCKF update.
   *
   * These values are set by update() and can be read by VioManager/ROS2Visualizer
   * for publishing as ROS topics.
   */
  struct CbfOutput {
    double mean_logdet = 0.0;            ///< Average log-det metric: ℓ̄ = (1/s) Σ log(det(M_i))
    Eigen::Vector3d g_vec = Eigen::Vector3d::Zero(); ///< CBF control gradient g(x) in body (IMU) frame
    double drift = 0.0;                  ///< CBF drift f(x) from past poses
    int num_features = 0;                ///< Number of features used in the CBF computation
    bool valid = false;                  ///< True if CBF metrics were successfully computed
  };

  /**
   * @brief Default constructor for our MSCKF updater
   *
   * Our updater has a feature initializer which we use to initialize features as needed.
   * Also the options allow for one to tune the different parameters for update.
   *
   * @param options Updater options (include measurement noise value)
   * @param feat_init_options Feature initializer options
   */
  UpdaterMSCKF(UpdaterOptions &options, ov_core::FeatureInitializerOptions &feat_init_options);

  /**
   * @brief Given tracked features, this will try to use them to update the state.
   *
   * @param state State of the filter
   * @param feature_vec Features that can be used for update
   */
  void update(std::shared_ptr<State> state, std::vector<std::shared_ptr<ov_core::Feature>> &feature_vec);

  /// Accessor for the latest CBF output (computed during the last update() call)
  const CbfOutput &get_cbf_output() const { return _cbf_output; }

  /// Set the EMA smoothing factor for CBF metrics (0=full smooth, 1=no smooth)
  void set_ema_alpha(double alpha) { _ema_alpha = alpha; }

protected:
  /// Options used during update
  UpdaterOptions _options;

  /// Feature initializer class object
  std::shared_ptr<ov_core::FeatureInitializer> initializer_feat;

  /// Chi squared 95th percentile table (lookup would be size of residual)
  std::map<int, double> chi_squared_table;

  /// Latest CBF output from the most recent update() call
  CbfOutput _cbf_output;

  /// EMA state for smoothing CBF metrics
  bool _ema_initialized = false;
  double _ema_logdet = 0.0;
  double _ema_drift = 0.0;
  Eigen::Vector3d _ema_g = Eigen::Vector3d::Zero();
  static constexpr double _ema_alpha_default = 0.1;
  double _ema_alpha = _ema_alpha_default; ///< EMA smoothing factor (0=full smooth, 1=no smooth)
};

} // namespace ov_msckf

#endif // OV_MSCKF_UPDATER_MSCKF_H
