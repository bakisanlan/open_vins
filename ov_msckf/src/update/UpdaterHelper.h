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

#ifndef OV_MSCKF_UPDATER_HELPER_H
#define OV_MSCKF_UPDATER_HELPER_H

#include <Eigen/Eigen>
#include <memory>
#include <unordered_map>
#include <vector>

#include "feat/FeatureInitializer.h"
#include "types/LandmarkRepresentation.h"

namespace ov_type {
class Type;
} // namespace ov_type
namespace ov_core {
class Feature;
} // namespace ov_core

namespace ov_msckf {

class State;

/**
 * @brief Class that has helper functions for our updaters.
 *
 * Can compute the Jacobian for a single feature representation.
 * This will create the Jacobian based on what representation our state is in.
 * If we are using the anchor representation then we also have additional Jacobians in respect to the anchor state.
 * Also has functions such as nullspace projection and full jacobian construction.
 * For derivations look at @ref update-feat page which has detailed equations.
 *
 */
class UpdaterHelper {
public:
  /**
   * @brief Feature object that our UpdaterHelper leverages, has all measurements and means
   */
  struct UpdaterHelperFeature {

    /// Unique ID of this feature
    size_t featid;

    /// UV coordinates that this feature has been seen from (mapped by camera ID)
    std::unordered_map<size_t, std::vector<Eigen::VectorXf>> uvs;

    // UV normalized coordinates that this feature has been seen from (mapped by camera ID)
    std::unordered_map<size_t, std::vector<Eigen::VectorXf>> uvs_norm;

    /// Timestamps of each UV measurement (mapped by camera ID)
    std::unordered_map<size_t, std::vector<double>> timestamps;

    /// What representation our feature is in
    ov_type::LandmarkRepresentation::Representation feat_representation;

    /// What camera ID our pose is anchored in!! By default the first measurement is the anchor.
    int anchor_cam_id = -1;

    /// Timestamp of anchor clone
    double anchor_clone_timestamp = -1;

    /// Triangulated position of this feature, in the anchor frame
    Eigen::Vector3d p_FinA;

    /// Triangulated position of this feature, in the anchor frame first estimate
    Eigen::Vector3d p_FinA_fej;

    /// Triangulated position of this feature, in the global frame
    Eigen::Vector3d p_FinG;

    /// Triangulated position of this feature, in the global frame first estimate
    Eigen::Vector3d p_FinG_fej;
  };

  /**
   * @brief This gets the feature and state Jacobian in respect to the feature representation
   *
   * @param[in] state State of the filter system
   * @param[in] feature Feature we want to get Jacobians of (must have feature means)
   * @param[out] H_f Jacobians in respect to the feature error state (will be either 3x3 or 3x1 for single depth)
   * @param[out] H_x Extra Jacobians in respect to the state (for example anchored pose)
   * @param[out] x_order Extra variables our extra Jacobian has (for example anchored pose)
   */
  static void get_feature_jacobian_representation(std::shared_ptr<State> state, UpdaterHelperFeature &feature, Eigen::MatrixXd &H_f,
                                                  std::vector<Eigen::MatrixXd> &H_x, std::vector<std::shared_ptr<ov_type::Type>> &x_order);

  /**
   * @brief Will construct the "stacked" Jacobians for a single feature from all its measurements
   *
   * @param[in] state State of the filter system
   * @param[in] feature Feature we want to get Jacobians of (must have feature means)
   * @param[out] H_f Jacobians in respect to the feature error state
   * @param[out] H_x Extra Jacobians in respect to the state (for example anchored pose)
   * @param[out] res Measurement residual for this feature
   * @param[out] x_order Extra variables our extra Jacobian has (for example anchored pose)
   */
  static void get_feature_jacobian_full(std::shared_ptr<State> state, UpdaterHelperFeature &feature, Eigen::MatrixXd &H_f,
                                        Eigen::MatrixXd &H_x, Eigen::VectorXd &res, std::vector<std::shared_ptr<ov_type::Type>> &x_order);

  /**
   * @brief This will project the left nullspace of H_f onto the linear system.
   *
   * Please see the @ref update-null for details on how this works.
   * This is the MSCKF nullspace projection which removes the dependency on the feature state.
   * Note that this is done **in place** so all matrices will be different after a function call.
   *
   * @param H_f Jacobian with nullspace we want to project onto the system [res = Hx*(x-xhat)+Hf(f-fhat)+n]
   * @param H_x State jacobian
   * @param res Measurement residual
   */
  static void nullspace_project_inplace(Eigen::MatrixXd &H_f, Eigen::MatrixXd &H_x, Eigen::VectorXd &res);

  /**
   * @brief This will perform measurement compression
   *
   * Please see the @ref update-compress for details on how this works.
   * Note that this is done **in place** so all matrices will be different after a function call.
   *
   * @param H_x State jacobian
   * @param res Measurement residual
   */
  static void measurement_compress_inplace(Eigen::MatrixXd &H_x, Eigen::VectorXd &res);

  // ============================================================================
  // Triangulation-Aware CBF (TANGO-VIO) observability metrics
  // ============================================================================

  /**
   * @brief Triangulation-observability metrics used by the triangulation-aware CBF.
   *
   * These are the quantities consumed by the navigation-level safety filter:
   * the average log-determinant of the per-feature triangulation information
   * matrices, the control gradient g(x), and the drift f(x). The metrics can be
   * computed from either the MSCKF feature set or the persistent SLAM feature set
   * (selected via UpdaterOptions::cbf_use_slam_features).
   */
  struct CbfOutput {
    double mean_logdet = 0.0;                        ///< Average log-det metric: ℓ̄ = (1/s) Σ log(det(M_i))
    Eigen::Vector3d g_vec = Eigen::Vector3d::Zero(); ///< CBF control gradient g(x) in body (IMU) frame
    double drift = 0.0;                              ///< CBF drift f(x) from past poses
    int num_features = 0;                            ///< Number of features used in the CBF computation
    bool valid = false;                              ///< True if metrics were successfully computed
  };

  /**
   * @brief Persistent exponential-moving-average state for smoothing the CBF metrics.
   *
   * Each updater owns one of these so the metric stream is temporally smoothed
   * across consecutive update calls.
   */
  struct CbfEmaState {
    bool initialized = false;
    double logdet = 0.0;
    double drift = 0.0;
    Eigen::Vector3d g = Eigen::Vector3d::Zero();
    double alpha = 0.1; ///< EMA smoothing factor (0=full smooth, 1=no smooth)
  };

  /**
   * @brief Computes the raw (unsmoothed) triangulation-aware CBF metrics for a feature set.
   *
   * For each feature i the per-pose bearings are accumulated into the triangulation
   * information matrix M_i = Σ_j π_{i,j}, from which ℓ_i = log(det(M_i)) is taken.
   * The control-affine derivative ḣ = f(x) + g(x)ᵀ v_B is split so that the current
   * pose contributes the control gradient g(x) and the past poses contribute the
   * drift f(x). Features must carry valid anchor information (anchor_cam_id,
   * anchor_clone_timestamp) and an anchor-frame position (p_FinA).
   *
   * @param state State of the filter (provides clone poses, current velocity, calibration)
   * @param feature_vec Features to evaluate (MSCKF or SLAM features)
   * @param clones_cam Camera clone poses indexed by [cam_id][timestamp]
   * @return Raw CBF metrics (valid==false if no usable feature was found)
   */
  static CbfOutput compute_cbf_metrics(
      std::shared_ptr<State> state, const std::vector<std::shared_ptr<ov_core::Feature>> &feature_vec,
      const std::unordered_map<size_t, std::unordered_map<double, ov_core::FeatureInitializer::ClonePose>> &clones_cam);

  /**
   * @brief Applies EMA smoothing to raw CBF metrics, updating @p ema in place.
   *
   * @param raw Raw metrics from compute_cbf_metrics()
   * @param ema Persistent EMA state (updated in place)
   * @return Smoothed CBF metrics (valid mirrors @p raw.valid)
   */
  static CbfOutput smooth_cbf_metrics(const CbfOutput &raw, CbfEmaState &ema);
};

} // namespace ov_msckf

#endif // OV_MSCKF_UPDATER_HELPER_H