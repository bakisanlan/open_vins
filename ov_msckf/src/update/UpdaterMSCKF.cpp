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

#include "UpdaterMSCKF.h"

#include "UpdaterHelper.h"

#include "feat/Feature.h"
#include "feat/FeatureInitializer.h"
#include "state/State.h"
#include "state/StateHelper.h"
#include "types/LandmarkRepresentation.h"
#include "utils/colors.h"
#include "utils/print.h"
#include "utils/quat_ops.h"

#include <boost/date_time/posix_time/posix_time.hpp>
#include <boost/math/distributions/chi_squared.hpp>
#include <cmath>
#include <limits>

using namespace ov_core;
using namespace ov_type;
using namespace ov_msckf;

UpdaterMSCKF::UpdaterMSCKF(UpdaterOptions &options, ov_core::FeatureInitializerOptions &feat_init_options) : _options(options) {

  // Save our raw pixel noise squared
  _options.sigma_pix_sq = std::pow(_options.sigma_pix, 2);

  // Save our feature initializer
  initializer_feat = std::shared_ptr<ov_core::FeatureInitializer>(new ov_core::FeatureInitializer(feat_init_options));

  // Initialize the chi squared test table with confidence level 0.95
  // https://github.com/KumarRobotics/msckf_vio/blob/050c50defa5a7fd9a04c1eed5687b405f02919b5/src/msckf_vio.cpp#L215-L221
  for (int i = 1; i < 500; i++) {
    boost::math::chi_squared chi_squared_dist(i);
    chi_squared_table[i] = boost::math::quantile(chi_squared_dist, 0.95);
  }
}

void UpdaterMSCKF::update(std::shared_ptr<State> state, std::vector<std::shared_ptr<Feature>> &feature_vec) {

  // Return if no features
  if (feature_vec.empty())
    return;

  // Start timing
  boost::posix_time::ptime rT0, rT1, rT2, rT3, rT4, rT5;
  rT0 = boost::posix_time::microsec_clock::local_time();

  // 0. Get all timestamps our clones are at (and thus valid measurement times)
  std::vector<double> clonetimes;
  for (const auto &clone_imu : state->_clones_IMU) {
    clonetimes.emplace_back(clone_imu.first);
  }

  // 1. Clean all feature measurements and make sure they all have valid clone times
  auto it0 = feature_vec.begin();
  while (it0 != feature_vec.end()) {

    // Clean the feature
    (*it0)->clean_old_measurements(clonetimes);

    // Count how many measurements
    int ct_meas = 0;
    for (const auto &pair : (*it0)->timestamps) {
      ct_meas += (*it0)->timestamps[pair.first].size();
    }

    // Remove if we don't have enough
    if (ct_meas < 2) {
      (*it0)->to_delete = true;
      it0 = feature_vec.erase(it0);
    } else {
      it0++;
    }
  }
  rT1 = boost::posix_time::microsec_clock::local_time();

  // 2. Create vector of cloned *CAMERA* poses at each of our clone timesteps
  std::unordered_map<size_t, std::unordered_map<double, FeatureInitializer::ClonePose>> clones_cam;
  for (const auto &clone_calib : state->_calib_IMUtoCAM) {

    // For this camera, create the vector of camera poses
    std::unordered_map<double, FeatureInitializer::ClonePose> clones_cami;
    for (const auto &clone_imu : state->_clones_IMU) {

      // Get current camera pose
      Eigen::Matrix<double, 3, 3> R_GtoCi = clone_calib.second->Rot() * clone_imu.second->Rot();
      Eigen::Matrix<double, 3, 1> p_CioinG = clone_imu.second->pos() - R_GtoCi.transpose() * clone_calib.second->pos();

      // Append to our map
      clones_cami.insert({clone_imu.first, FeatureInitializer::ClonePose(R_GtoCi, p_CioinG)});
    }

    // Append to our map
    clones_cam.insert({clone_calib.first, clones_cami});
  }

  // 3. Try to triangulate all MSCKF or new SLAM features that have measurements
  // Aggregate statistics for triangulation diagnostics
  int tri_total_tried = 0;
  int tri_total_success = 0;
  double tri_max_cond = 0.0, tri_sum_cond = 0.0;
  int tri_count_cond = 0;
  double tri_max_baseline_ratio = 0.0, tri_sum_baseline_ratio = 0.0;
  int tri_count_baseline_ratio = 0;
  // CBF logdet observability metric: ell_i = log(det(M_i)) per feature
  double tri_sum_logdet = 0.0, tri_min_logdet = std::numeric_limits<double>::infinity();
  double tri_max_logdet = -std::numeric_limits<double>::infinity();
  int tri_count_logdet = 0;

  auto it1 = feature_vec.begin();
  while (it1 != feature_vec.end()) {
    tri_total_tried++;

    // Triangulate the feature and remove if it fails
    bool success_tri = true;
    double cond_out = 0.0;
    double logdet_out = -std::numeric_limits<double>::infinity();
    if (initializer_feat->config().triangulate_1d) {
      success_tri = initializer_feat->single_triangulation_1d(*it1, clones_cam);
    } else {
      success_tri = initializer_feat->single_triangulation(*it1, clones_cam, &cond_out, &logdet_out);
      // Accumulate condition number stats (even for failures)
      if (cond_out > 0.0) {
        tri_sum_cond += cond_out;
        tri_count_cond++;
        if (cond_out > tri_max_cond)
          tri_max_cond = cond_out;
      }
      // Accumulate logdet stats (even for failures, if the matrix was valid)
      if (std::isfinite(logdet_out)) {
        tri_sum_logdet += logdet_out;
        tri_count_logdet++;
        if (logdet_out < tri_min_logdet)
          tri_min_logdet = logdet_out;
        if (logdet_out > tri_max_logdet)
          tri_max_logdet = logdet_out;
      }
    }

    // Gauss-newton refine the feature
    bool success_refine = true;
    double baseline_ratio_out = 0.0;
    if (initializer_feat->config().refine_features) {
      success_refine = initializer_feat->single_gaussnewton(*it1, clones_cam, &baseline_ratio_out);
      // Accumulate baseline ratio stats (even for failures)
      if (baseline_ratio_out > 0.0) {
        tri_sum_baseline_ratio += baseline_ratio_out;
        tri_count_baseline_ratio++;
        if (baseline_ratio_out > tri_max_baseline_ratio)
          tri_max_baseline_ratio = baseline_ratio_out;
      }
    }

    // Remove the feature if not a success
    if (!success_tri || !success_refine) {
      (*it1)->to_delete = true;
      it1 = feature_vec.erase(it1);
      continue;
    }
    tri_total_success++;
    it1++;
  }
  rT2 = boost::posix_time::microsec_clock::local_time();

  // Print aggregate triangulation statistics
  {
    double mean_cond = (tri_count_cond > 0) ? (tri_sum_cond / tri_count_cond) : 0.0;
    double mean_baseline = (tri_count_baseline_ratio > 0) ? (tri_sum_baseline_ratio / tri_count_baseline_ratio) : 0.0;
    double mean_logdet = (tri_count_logdet > 0) ? (tri_sum_logdet / tri_count_logdet) : 0.0;
    PRINT_INFO(MAGENTA "[TRI]: tried=%d | ok=%d | cond: max=%.1f/%.0f mean=%.1f | baseline_ratio: max=%.1f/%.0f mean=%.1f\n" RESET,
               tri_total_tried, tri_total_success, tri_max_cond, initializer_feat->config().max_cond_number, mean_cond,
               tri_max_baseline_ratio, initializer_feat->config().max_baseline, mean_baseline);
    PRINT_INFO(CYAN "[CBF-LOGDET]: feats=%d | mean_logdet=%.4f | min_logdet=%.4f | max_logdet=%.4f\n" RESET,
               tri_count_logdet, mean_logdet,
               (tri_count_logdet > 0) ? tri_min_logdet : 0.0,
               (tri_count_logdet > 0) ? tri_max_logdet : 0.0);
  }

  // ============================================================================
  // CBF: Compute drift f(x) and control gradient g(x) in body (IMU) frame
  //
  //   ḣ = f(x) + g(x)ᵀ v_B
  //
  //   f(x) = (2/s) Σ_i Σ_{j=1..n-1}  1/(ρ_{i,j} η²) bᵀ M⁻¹ π R_{Cj}^A R_B^C v_{B,j}
  //   g(x) = (2/s) Σ_i  1/(ρ_{i,n} η²) (R_{Cn}^A R_B^C)ᵀ π_{i,n} M⁻¹ b_{i,n}
  //
  // Since OpenVINS unit-normalises bearings, η = 1 and π = I - b bᵀ.
  // R_{C_j}^A = R_AtoCi^T;  R_B^C = R_ItoC (IMU-to-camera calibration).
  // So (R_{C_j}^A R_B^C)^T = R_ItoC^T · R_AtoCi  (anchor → body).
  // ============================================================================
  _cbf_output = CbfOutput(); // reset
  _cbf_output.tri_tried = tri_total_tried;
  _cbf_output.tri_success = tri_total_success;
  if (!feature_vec.empty() && tri_count_logdet > 0) {
    // IMU-to-camera rotation for the first camera (cam 0)
    const Eigen::Matrix3d R_ItoC = state->_calib_IMUtoCAM.at(0)->Rot();  // R_B^C
    const Eigen::Matrix3d R_CtoI = R_ItoC.transpose();                   // (R_B^C)^T

    // ── Estimate past velocities from clone positions ──
    // Clone times are sorted in ascending order (std::map)
    // v_{G,j} ≈ (p_{j+1} - p_j) / Δt, then v_{B,j} = R_GtoI_j · v_{G,j}
    struct CloneVel {
      double time;
      Eigen::Vector3d v_body;   // velocity in body (IMU) frame
      bool is_current;          // true for the latest (current) pose
    };
    std::vector<CloneVel> clone_vels;
    {
      std::vector<std::pair<double, std::shared_ptr<ov_type::PoseJPL>>> sorted_clones(
          state->_clones_IMU.begin(), state->_clones_IMU.end());

      for (size_t idx = 0; idx < sorted_clones.size(); idx++) {
        double t_j = sorted_clones[idx].first;
        bool is_last = (idx == sorted_clones.size() - 1);

        Eigen::Vector3d v_body;
        if (is_last) {
          // Current pose: use the EKF's estimated velocity directly
          // state->_imu->vel() is v_IinG (global frame)
          Eigen::Matrix3d R_GtoI_j = sorted_clones[idx].second->Rot();
          v_body = R_GtoI_j * state->_imu->vel();
        } else {
          // Past pose: numerical differentiation of global positions
          double t_next = sorted_clones[idx + 1].first;
          double dt = t_next - t_j;
          if (dt < 1e-9) dt = 1e-9; // guard against division by zero
          Eigen::Vector3d p_j = sorted_clones[idx].second->pos();
          Eigen::Vector3d p_next = sorted_clones[idx + 1].second->pos();
          Eigen::Vector3d v_global = (p_next - p_j) / dt;
          Eigen::Matrix3d R_GtoI_j = sorted_clones[idx].second->Rot();
          v_body = R_GtoI_j * v_global;
        }
        clone_vels.push_back({t_j, v_body, is_last});
      }
    }

    // Build a lookup: clone_time → index in clone_vels
    std::unordered_map<double, size_t> time_to_idx;
    for (size_t i = 0; i < clone_vels.size(); i++) {
      time_to_idx[clone_vels[i].time] = i;
    }

    Eigen::Vector3d g_sum = Eigen::Vector3d::Zero();  // control gradient accumulator
    double f_sum = 0.0;                                // drift accumulator
    int s = 0; // number of valid features

    for (const auto &feat : feature_vec) {
      // Get anchor pose
      auto anchorclone = clones_cam.at(feat->anchor_cam_id).at(feat->anchor_clone_timestamp);
      const Eigen::Matrix3d &R_GtoA = anchorclone.Rot();
      const Eigen::Vector3d &p_AinG = anchorclone.pos();

      // 1. Build information matrix M_i = Σ_j π_{i,j}
      Eigen::Matrix3d M_i = Eigen::Matrix3d::Zero();
      struct BearingEntry {
        Eigen::Vector3d b;
        Eigen::Matrix3d R_AtoCi;
        Eigen::Vector3d p_CiinA;
        double clone_time;       // timestamp of this clone
      };
      std::vector<BearingEntry> entries;

      for (const auto &pair : feat->timestamps) {
        for (size_t m = 0; m < pair.second.size(); m++) {
          const Eigen::Matrix3d &R_GtoCi = clones_cam.at(pair.first).at(pair.second.at(m)).Rot();
          const Eigen::Vector3d &p_CiinG = clones_cam.at(pair.first).at(pair.second.at(m)).pos();

          Eigen::Matrix3d R_AtoCi = R_GtoCi * R_GtoA.transpose();
          Eigen::Vector3d p_CiinA = R_GtoA * (p_CiinG - p_AinG);

          // Bearing in anchor frame (unit-normalized)
          Eigen::Vector3d b_i;
          b_i << feat->uvs_norm.at(pair.first).at(m)(0), feat->uvs_norm.at(pair.first).at(m)(1), 1;
          b_i = R_AtoCi.transpose() * b_i;
          b_i = b_i / b_i.norm(); // η = 1

          Eigen::Matrix3d pi_j = Eigen::Matrix3d::Identity() - b_i * b_i.transpose();
          M_i += pi_j;

          entries.push_back({b_i, R_AtoCi, p_CiinA, pair.second.at(m)});
        }
      }

      // 2. Check if M_i is invertible
      Eigen::JacobiSVD<Eigen::Matrix3d> svd_Mi(M_i, Eigen::ComputeFullU | Eigen::ComputeFullV);
      double lambda_min = svd_Mi.singularValues()(2);
      if (lambda_min < 1e-8)
        continue; // rank-deficient, skip this feature

      Eigen::Matrix3d M_inv = svd_Mi.solve(Eigen::Matrix3d::Identity());

      // 3. Split contributions: g(x) from current pose, f(x) from past poses
      Eigen::Vector3d g_feat = Eigen::Vector3d::Zero();
      double f_feat = 0.0;

      for (const auto &e : entries) {
        double rho = (feat->p_FinA - e.p_CiinA).norm(); // Euclidean depth
        if (rho < 1e-6)
          continue;

        Eigen::Matrix3d pi_j = Eigen::Matrix3d::Identity() - e.b * e.b.transpose();

        // Check if this entry belongs to the current (latest) pose
        auto it_vel = time_to_idx.find(e.clone_time);
        bool is_current_pose = false;
        if (it_vel != time_to_idx.end()) {
          is_current_pose = clone_vels[it_vel->second].is_current;
        }

        if (is_current_pose) {
          // ── g(x): control gradient (body frame) ──
          // g_i = (2/ρ) (R_{Cn}^A R_B^C)^T π M^{-1} b
          Eigen::Vector3d pi_Minv_b = pi_j * M_inv * e.b;
          Eigen::Matrix3d R_AtoBody = R_CtoI * e.R_AtoCi;
          g_feat += (1.0 / rho) * R_AtoBody * pi_Minv_b;
        } else if (it_vel != time_to_idx.end()) {
          // ── f(x): drift from past pose ──
          // f_i += (2/ρ) bᵀ M^{-1} π R_{Cj}^A R_B^C v_{B,j}
          const Eigen::Vector3d &v_body_j = clone_vels[it_vel->second].v_body;
          // R_{Cj}^A · R_B^C · v_{B,j} = R_AtoCi^T · R_ItoC · v_{B,j}
          Eigen::Vector3d vel_in_anchor = e.R_AtoCi.transpose() * R_ItoC * v_body_j;
          Eigen::Vector3d Minv_pi_vel = M_inv * pi_j * vel_in_anchor;
          double contrib = e.b.dot(Minv_pi_vel);
          f_feat += (1.0 / rho) * contrib;
        }
      }
      g_feat *= 2.0;
      f_feat *= 2.0;

      g_sum += g_feat;
      f_sum += f_feat;
      s++;
    }

    if (s > 0) {
      double raw_logdet = tri_sum_logdet / tri_count_logdet;
      Eigen::Vector3d raw_g = g_sum / s;
      double raw_drift = f_sum / s;

      // Apply EMA smoothing (weight alpha by feature count for stability)
      if (!_ema_initialized) {
        _ema_logdet = raw_logdet;
        _ema_g = raw_g;
        _ema_drift = raw_drift;
        _ema_initialized = true;
      } else {
        double alpha = std::min(1.0, _ema_alpha * (s / 10.0));
        _ema_logdet = alpha * raw_logdet + (1.0 - alpha) * _ema_logdet;
        _ema_g = alpha * raw_g + (1.0 - alpha) * _ema_g;
        _ema_drift = alpha * raw_drift + (1.0 - alpha) * _ema_drift;
      }

      _cbf_output.mean_logdet = _ema_logdet;
      _cbf_output.g_vec = _ema_g;
      _cbf_output.drift = _ema_drift;
      _cbf_output.num_features = s;
      _cbf_output.valid = true;

      PRINT_INFO(CYAN "[CBF]: feats=%d | logdet=%.4f | drift=%.6f | g=[%.6f, %.6f, %.6f]\n" RESET,
                 s, _ema_logdet, _ema_drift, _cbf_output.g_vec(0), _cbf_output.g_vec(1), _cbf_output.g_vec(2));
    }
  }

  // Calculate the max possible measurement size
  size_t max_meas_size = 0;
  for (size_t i = 0; i < feature_vec.size(); i++) {
    for (const auto &pair : feature_vec.at(i)->timestamps) {
      max_meas_size += 2 * feature_vec.at(i)->timestamps[pair.first].size();
    }
  }

  // Calculate max possible state size (i.e. the size of our covariance)
  // NOTE: that when we have the single inverse depth representations, those are only 1dof in size
  size_t max_hx_size = state->max_covariance_size();
  for (auto &landmark : state->_features_SLAM) {
    max_hx_size -= landmark.second->size();
  }

  // Large Jacobian and residual of *all* features for this update
  Eigen::VectorXd res_big = Eigen::VectorXd::Zero(max_meas_size);
  Eigen::MatrixXd Hx_big = Eigen::MatrixXd::Zero(max_meas_size, max_hx_size);
  std::unordered_map<std::shared_ptr<Type>, size_t> Hx_mapping;
  std::vector<std::shared_ptr<Type>> Hx_order_big;
  size_t ct_jacob = 0;
  size_t ct_meas = 0;

  // 4. Compute linear system for each feature, nullspace project, and reject
  auto it2 = feature_vec.begin();
  while (it2 != feature_vec.end()) {

    // Convert our feature into our current format
    UpdaterHelper::UpdaterHelperFeature feat;
    feat.featid = (*it2)->featid;
    feat.uvs = (*it2)->uvs;
    feat.uvs_norm = (*it2)->uvs_norm;
    feat.timestamps = (*it2)->timestamps;

    // If we are using single inverse depth, then it is equivalent to using the msckf inverse depth
    feat.feat_representation = state->_options.feat_rep_msckf;
    if (state->_options.feat_rep_msckf == LandmarkRepresentation::Representation::ANCHORED_INVERSE_DEPTH_SINGLE) {
      feat.feat_representation = LandmarkRepresentation::Representation::ANCHORED_MSCKF_INVERSE_DEPTH;
    }

    // Save the position and its fej value
    if (LandmarkRepresentation::is_relative_representation(feat.feat_representation)) {
      feat.anchor_cam_id = (*it2)->anchor_cam_id;
      feat.anchor_clone_timestamp = (*it2)->anchor_clone_timestamp;
      feat.p_FinA = (*it2)->p_FinA;
      feat.p_FinA_fej = (*it2)->p_FinA;
    } else {
      feat.p_FinG = (*it2)->p_FinG;
      feat.p_FinG_fej = (*it2)->p_FinG;
    }

    // Our return values (feature jacobian, state jacobian, residual, and order of state jacobian)
    Eigen::MatrixXd H_f;
    Eigen::MatrixXd H_x;
    Eigen::VectorXd res;
    std::vector<std::shared_ptr<Type>> Hx_order;

    // Get the Jacobian for this feature
    UpdaterHelper::get_feature_jacobian_full(state, feat, H_f, H_x, res, Hx_order);

    // Nullspace project
    UpdaterHelper::nullspace_project_inplace(H_f, H_x, res);

    /// Chi2 distance check
    Eigen::MatrixXd P_marg = StateHelper::get_marginal_covariance(state, Hx_order);
    Eigen::MatrixXd S = H_x * P_marg * H_x.transpose();
    S.diagonal() += _options.sigma_pix_sq * Eigen::VectorXd::Ones(S.rows());
    double chi2 = res.dot(S.llt().solve(res));

    // Get our threshold (we precompute up to 500 but handle the case that it is more)
    double chi2_check;
    if (res.rows() < 500) {
      chi2_check = chi_squared_table[res.rows()];
    } else {
      boost::math::chi_squared chi_squared_dist(res.rows());
      chi2_check = boost::math::quantile(chi_squared_dist, 0.95);
      PRINT_WARNING(YELLOW "chi2_check over the residual limit - %d\n" RESET, (int)res.rows());
    }

    // Check if we should delete or not
    if (chi2 > _options.chi2_multipler * chi2_check) {
      (*it2)->to_delete = true;
      it2 = feature_vec.erase(it2);
      // PRINT_DEBUG("featid = %d\n", feat.featid);
      // PRINT_DEBUG("chi2 = %f > %f\n", chi2, _options.chi2_multipler*chi2_check);
      // std::stringstream ss;
      // ss << "res = " << std::endl << res.transpose() << std::endl;
      // PRINT_DEBUG(ss.str().c_str());
      continue;
    }

    // We are good!!! Append to our large H vector
    size_t ct_hx = 0;
    for (const auto &var : Hx_order) {

      // Ensure that this variable is in our Jacobian
      if (Hx_mapping.find(var) == Hx_mapping.end()) {
        Hx_mapping.insert({var, ct_jacob});
        Hx_order_big.push_back(var);
        ct_jacob += var->size();
      }

      // Append to our large Jacobian
      Hx_big.block(ct_meas, Hx_mapping[var], H_x.rows(), var->size()) = H_x.block(0, ct_hx, H_x.rows(), var->size());
      ct_hx += var->size();
    }

    // Append our residual and move forward
    res_big.block(ct_meas, 0, res.rows(), 1) = res;
    ct_meas += res.rows();
    it2++;
  }
  rT3 = boost::posix_time::microsec_clock::local_time();

  // We have appended all features to our Hx_big, res_big
  // Delete it so we do not reuse information
  for (size_t f = 0; f < feature_vec.size(); f++) {
    feature_vec[f]->to_delete = true;
  }

  // Return if we don't have anything and resize our matrices
  if (ct_meas < 1) {
    return;
  }
  assert(ct_meas <= max_meas_size);
  assert(ct_jacob <= max_hx_size);
  res_big.conservativeResize(ct_meas, 1);
  Hx_big.conservativeResize(ct_meas, ct_jacob);

  // 5. Perform measurement compression
  UpdaterHelper::measurement_compress_inplace(Hx_big, res_big);
  if (Hx_big.rows() < 1) {
    return;
  }
  rT4 = boost::posix_time::microsec_clock::local_time();

  // Our noise is isotropic, so make it here after our compression
  Eigen::MatrixXd R_big = _options.sigma_pix_sq * Eigen::MatrixXd::Identity(res_big.rows(), res_big.rows());

  // 6. With all good features update the state
  StateHelper::EKFUpdate(state, Hx_order_big, Hx_big, res_big, R_big);
  rT5 = boost::posix_time::microsec_clock::local_time();

  // Debug print timing information
  PRINT_ALL("[MSCKF-UP]: %.4f seconds to clean\n", (rT1 - rT0).total_microseconds() * 1e-6);
  PRINT_ALL("[MSCKF-UP]: %.4f seconds to triangulate\n", (rT2 - rT1).total_microseconds() * 1e-6);
  PRINT_ALL("[MSCKF-UP]: %.4f seconds create system (%d features)\n", (rT3 - rT2).total_microseconds() * 1e-6, (int)feature_vec.size());
  PRINT_ALL("[MSCKF-UP]: %.4f seconds compress system\n", (rT4 - rT3).total_microseconds() * 1e-6);
  PRINT_ALL("[MSCKF-UP]: %.4f seconds update state (%d size)\n", (rT5 - rT4).total_microseconds() * 1e-6, (int)res_big.rows());
  PRINT_ALL("[MSCKF-UP]: %.4f seconds total\n", (rT5 - rT1).total_microseconds() * 1e-6);
}
