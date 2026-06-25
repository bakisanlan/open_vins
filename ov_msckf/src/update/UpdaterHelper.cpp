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

#include "UpdaterHelper.h"

#include "state/State.h"

#include "feat/Feature.h"
#include "utils/quat_ops.h"

#include <algorithm>
#include <cmath>
#include <limits>

using namespace ov_core;
using namespace ov_type;
using namespace ov_msckf;

void UpdaterHelper::get_feature_jacobian_representation(std::shared_ptr<State> state, UpdaterHelperFeature &feature, Eigen::MatrixXd &H_f,
                                                        std::vector<Eigen::MatrixXd> &H_x, std::vector<std::shared_ptr<Type>> &x_order) {

  // Global XYZ representation
  if (feature.feat_representation == LandmarkRepresentation::Representation::GLOBAL_3D) {
    H_f.resize(3, 3);
    H_f.setIdentity();
    return;
  }

  // Global inverse depth representation
  if (feature.feat_representation == LandmarkRepresentation::Representation::GLOBAL_FULL_INVERSE_DEPTH) {

    // Get the feature linearization point
    Eigen::Matrix<double, 3, 1> p_FinG = (state->_options.do_fej) ? feature.p_FinG_fej : feature.p_FinG;

    // Get inverse depth representation (should match what is in Landmark.cpp)
    double g_rho = 1 / p_FinG.norm();
    double g_phi = std::acos(g_rho * p_FinG(2));
    // double g_theta = std::asin(g_rho*p_FinG(1)/std::sin(g_phi));
    double g_theta = std::atan2(p_FinG(1), p_FinG(0));
    Eigen::Matrix<double, 3, 1> p_invFinG;
    p_invFinG(0) = g_theta;
    p_invFinG(1) = g_phi;
    p_invFinG(2) = g_rho;

    // Get inverse depth bearings
    double sin_th = std::sin(p_invFinG(0, 0));
    double cos_th = std::cos(p_invFinG(0, 0));
    double sin_phi = std::sin(p_invFinG(1, 0));
    double cos_phi = std::cos(p_invFinG(1, 0));
    double rho = p_invFinG(2, 0);

    // Construct the Jacobian
    H_f.resize(3, 3);
    H_f << -(1.0 / rho) * sin_th * sin_phi, (1.0 / rho) * cos_th * cos_phi, -(1.0 / (rho * rho)) * cos_th * sin_phi,
        (1.0 / rho) * cos_th * sin_phi, (1.0 / rho) * sin_th * cos_phi, -(1.0 / (rho * rho)) * sin_th * sin_phi, 0.0,
        -(1.0 / rho) * sin_phi, -(1.0 / (rho * rho)) * cos_phi;
    return;
  }

  //======================================================================
  //======================================================================
  //======================================================================

  // Assert that we have an anchor pose for this feature
  assert(feature.anchor_cam_id != -1);

  // Anchor pose orientation and position, and camera calibration for our anchor camera
  Eigen::Matrix3d R_ItoC = state->_calib_IMUtoCAM.at(feature.anchor_cam_id)->Rot();
  Eigen::Vector3d p_IinC = state->_calib_IMUtoCAM.at(feature.anchor_cam_id)->pos();
  Eigen::Matrix3d R_GtoI = state->_clones_IMU.at(feature.anchor_clone_timestamp)->Rot();
  Eigen::Vector3d p_IinG = state->_clones_IMU.at(feature.anchor_clone_timestamp)->pos();
  Eigen::Vector3d p_FinA = feature.p_FinA;

  // If I am doing FEJ, I should FEJ the anchor states (should we fej calibration???)
  // Also get the FEJ position of the feature if we are
  if (state->_options.do_fej) {
    // "Best" feature in the global frame
    Eigen::Vector3d p_FinG_best = R_GtoI.transpose() * R_ItoC.transpose() * (feature.p_FinA - p_IinC) + p_IinG;
    // Transform the best into our anchor frame using FEJ
    R_GtoI = state->_clones_IMU.at(feature.anchor_clone_timestamp)->Rot_fej();
    p_IinG = state->_clones_IMU.at(feature.anchor_clone_timestamp)->pos_fej();
    p_FinA = (R_GtoI.transpose() * R_ItoC.transpose()).transpose() * (p_FinG_best - p_IinG) + p_IinC;
  }
  Eigen::Matrix3d R_CtoG = R_GtoI.transpose() * R_ItoC.transpose();

  // Jacobian for our anchor pose
  Eigen::Matrix<double, 3, 6> H_anc;
  H_anc.block(0, 0, 3, 3).noalias() = -R_GtoI.transpose() * skew_x(R_ItoC.transpose() * (p_FinA - p_IinC));
  H_anc.block(0, 3, 3, 3).setIdentity();

  // Add anchor Jacobians to our return vector
  x_order.push_back(state->_clones_IMU.at(feature.anchor_clone_timestamp));
  H_x.push_back(H_anc);

  // Get calibration Jacobians (for anchor clone)
  if (state->_options.do_calib_camera_pose) {
    Eigen::Matrix<double, 3, 6> H_calib;
    H_calib.block(0, 0, 3, 3).noalias() = -R_CtoG * skew_x(p_FinA - p_IinC);
    H_calib.block(0, 3, 3, 3) = -R_CtoG;
    x_order.push_back(state->_calib_IMUtoCAM.at(feature.anchor_cam_id));
    H_x.push_back(H_calib);
  }

  // If we are doing anchored XYZ feature
  if (feature.feat_representation == LandmarkRepresentation::Representation::ANCHORED_3D) {
    H_f = R_CtoG;
    return;
  }

  // If we are doing full inverse depth
  if (feature.feat_representation == LandmarkRepresentation::Representation::ANCHORED_FULL_INVERSE_DEPTH) {

    // Get inverse depth representation (should match what is in Landmark.cpp)
    double a_rho = 1 / p_FinA.norm();
    double a_phi = std::acos(a_rho * p_FinA(2));
    double a_theta = std::atan2(p_FinA(1), p_FinA(0));
    Eigen::Matrix<double, 3, 1> p_invFinA;
    p_invFinA(0) = a_theta;
    p_invFinA(1) = a_phi;
    p_invFinA(2) = a_rho;

    // Using anchored inverse depth
    double sin_th = std::sin(p_invFinA(0, 0));
    double cos_th = std::cos(p_invFinA(0, 0));
    double sin_phi = std::sin(p_invFinA(1, 0));
    double cos_phi = std::cos(p_invFinA(1, 0));
    double rho = p_invFinA(2, 0);
    // assert(p_invFinA(2,0)>=0.0);

    // Jacobian of anchored 3D position wrt inverse depth parameters
    Eigen::Matrix<double, 3, 3> d_pfinA_dpinv;
    d_pfinA_dpinv << -(1.0 / rho) * sin_th * sin_phi, (1.0 / rho) * cos_th * cos_phi, -(1.0 / (rho * rho)) * cos_th * sin_phi,
        (1.0 / rho) * cos_th * sin_phi, (1.0 / rho) * sin_th * cos_phi, -(1.0 / (rho * rho)) * sin_th * sin_phi, 0.0,
        -(1.0 / rho) * sin_phi, -(1.0 / (rho * rho)) * cos_phi;
    H_f = R_CtoG * d_pfinA_dpinv;
    return;
  }

  // If we are doing the MSCKF version of inverse depth
  if (feature.feat_representation == LandmarkRepresentation::Representation::ANCHORED_MSCKF_INVERSE_DEPTH) {

    // Get inverse depth representation (should match what is in Landmark.cpp)
    Eigen::Matrix<double, 3, 1> p_invFinA_MSCKF;
    p_invFinA_MSCKF(0) = p_FinA(0) / p_FinA(2);
    p_invFinA_MSCKF(1) = p_FinA(1) / p_FinA(2);
    p_invFinA_MSCKF(2) = 1 / p_FinA(2);

    // Using the MSCKF version of inverse depth
    double alpha = p_invFinA_MSCKF(0, 0);
    double beta = p_invFinA_MSCKF(1, 0);
    double rho = p_invFinA_MSCKF(2, 0);

    // Jacobian of anchored 3D position wrt inverse depth parameters
    Eigen::Matrix<double, 3, 3> d_pfinA_dpinv;
    d_pfinA_dpinv << (1.0 / rho), 0.0, -(1.0 / (rho * rho)) * alpha, 0.0, (1.0 / rho), -(1.0 / (rho * rho)) * beta, 0.0, 0.0,
        -(1.0 / (rho * rho));
    H_f = R_CtoG * d_pfinA_dpinv;
    return;
  }

  /// CASE: Estimate single depth of the feature using the initial bearing
  if (feature.feat_representation == LandmarkRepresentation::Representation::ANCHORED_INVERSE_DEPTH_SINGLE) {

    // Get inverse depth representation (should match what is in Landmark.cpp)
    double rho = 1.0 / p_FinA(2);
    Eigen::Vector3d bearing = rho * p_FinA;

    // Jacobian of anchored 3D position wrt inverse depth parameters
    Eigen::Vector3d d_pfinA_drho;
    d_pfinA_drho << -(1.0 / (rho * rho)) * bearing;
    H_f = R_CtoG * d_pfinA_drho;
    return;
  }

  // Failure, invalid representation that is not programmed
  assert(false);
}

void UpdaterHelper::get_feature_jacobian_full(std::shared_ptr<State> state, UpdaterHelperFeature &feature, Eigen::MatrixXd &H_f,
                                              Eigen::MatrixXd &H_x, Eigen::VectorXd &res, std::vector<std::shared_ptr<Type>> &x_order) {

  // Total number of measurements for this feature
  int total_meas = 0;
  for (auto const &pair : feature.timestamps) {
    total_meas += (int)pair.second.size();
  }

  // Compute the size of the states involved with this feature
  int total_hx = 0;
  std::unordered_map<std::shared_ptr<Type>, size_t> map_hx;
  for (auto const &pair : feature.timestamps) {

    // Our extrinsics and intrinsics
    std::shared_ptr<PoseJPL> calibration = state->_calib_IMUtoCAM.at(pair.first);
    std::shared_ptr<Vec> distortion = state->_cam_intrinsics.at(pair.first);

    // If doing calibration extrinsics
    if (state->_options.do_calib_camera_pose) {
      map_hx.insert({calibration, total_hx});
      x_order.push_back(calibration);
      total_hx += calibration->size();
    }

    // If doing calibration intrinsics
    if (state->_options.do_calib_camera_intrinsics) {
      map_hx.insert({distortion, total_hx});
      x_order.push_back(distortion);
      total_hx += distortion->size();
    }

    // Loop through all measurements for this specific camera
    for (size_t m = 0; m < feature.timestamps[pair.first].size(); m++) {

      // Add this clone if it is not added already
      std::shared_ptr<PoseJPL> clone_Ci = state->_clones_IMU.at(feature.timestamps[pair.first].at(m));
      if (map_hx.find(clone_Ci) == map_hx.end()) {
        map_hx.insert({clone_Ci, total_hx});
        x_order.push_back(clone_Ci);
        total_hx += clone_Ci->size();
      }
    }
  }

  // If we are using an anchored representation, make sure that the anchor is also added
  if (LandmarkRepresentation::is_relative_representation(feature.feat_representation)) {

    // Assert we have a clone
    assert(feature.anchor_cam_id != -1);

    // Add this anchor if it is not added already
    std::shared_ptr<PoseJPL> clone_Ai = state->_clones_IMU.at(feature.anchor_clone_timestamp);
    if (map_hx.find(clone_Ai) == map_hx.end()) {
      map_hx.insert({clone_Ai, total_hx});
      x_order.push_back(clone_Ai);
      total_hx += clone_Ai->size();
    }

    // Also add its calibration if we are doing calibration
    if (state->_options.do_calib_camera_pose) {
      // Add this anchor if it is not added already
      std::shared_ptr<PoseJPL> clone_calib = state->_calib_IMUtoCAM.at(feature.anchor_cam_id);
      if (map_hx.find(clone_calib) == map_hx.end()) {
        map_hx.insert({clone_calib, total_hx});
        x_order.push_back(clone_calib);
        total_hx += clone_calib->size();
      }
    }
  }

  //=========================================================================
  //=========================================================================

  // Calculate the position of this feature in the global frame
  // If anchored, then we need to calculate the position of the feature in the global
  Eigen::Vector3d p_FinG = feature.p_FinG;
  if (LandmarkRepresentation::is_relative_representation(feature.feat_representation)) {
    // Assert that we have an anchor pose for this feature
    assert(feature.anchor_cam_id != -1);
    // Get calibration for our anchor camera
    Eigen::Matrix3d R_ItoC = state->_calib_IMUtoCAM.at(feature.anchor_cam_id)->Rot();
    Eigen::Vector3d p_IinC = state->_calib_IMUtoCAM.at(feature.anchor_cam_id)->pos();
    // Anchor pose orientation and position
    Eigen::Matrix3d R_GtoI = state->_clones_IMU.at(feature.anchor_clone_timestamp)->Rot();
    Eigen::Vector3d p_IinG = state->_clones_IMU.at(feature.anchor_clone_timestamp)->pos();
    // Feature in the global frame
    p_FinG = R_GtoI.transpose() * R_ItoC.transpose() * (feature.p_FinA - p_IinC) + p_IinG;
  }

  // Calculate the position of this feature in the global frame FEJ
  // If anchored, then we can use the "best" p_FinG since the value of p_FinA does not matter
  Eigen::Vector3d p_FinG_fej = feature.p_FinG_fej;
  if (LandmarkRepresentation::is_relative_representation(feature.feat_representation)) {
    p_FinG_fej = p_FinG;
  }

  //=========================================================================
  //=========================================================================

  // Allocate our residual and Jacobians
  int c = 0;
  int jacobsize = (feature.feat_representation != LandmarkRepresentation::Representation::ANCHORED_INVERSE_DEPTH_SINGLE) ? 3 : 1;
  res = Eigen::VectorXd::Zero(2 * total_meas);
  H_f = Eigen::MatrixXd::Zero(2 * total_meas, jacobsize);
  H_x = Eigen::MatrixXd::Zero(2 * total_meas, total_hx);

  // Derivative of p_FinG in respect to feature representation.
  // This only needs to be computed once and thus we pull it out of the loop
  Eigen::MatrixXd dpfg_dlambda;
  std::vector<Eigen::MatrixXd> dpfg_dx;
  std::vector<std::shared_ptr<Type>> dpfg_dx_order;
  UpdaterHelper::get_feature_jacobian_representation(state, feature, dpfg_dlambda, dpfg_dx, dpfg_dx_order);

  // Assert that all the ones in our order are already in our local jacobian mapping
#ifndef NDEBUG
  for (auto &type : dpfg_dx_order) {
    assert(map_hx.find(type) != map_hx.end());
  }
#endif

  // Loop through each camera for this feature
  for (auto const &pair : feature.timestamps) {

    // Our calibration between the IMU and CAMi frames
    std::shared_ptr<Vec> distortion = state->_cam_intrinsics.at(pair.first);
    std::shared_ptr<PoseJPL> calibration = state->_calib_IMUtoCAM.at(pair.first);
    Eigen::Matrix3d R_ItoC = calibration->Rot();
    Eigen::Vector3d p_IinC = calibration->pos();

    // Loop through all measurements for this specific camera
    for (size_t m = 0; m < feature.timestamps[pair.first].size(); m++) {

      //=========================================================================
      //=========================================================================

      // Get current IMU clone state
      std::shared_ptr<PoseJPL> clone_Ii = state->_clones_IMU.at(feature.timestamps[pair.first].at(m));
      Eigen::Matrix3d R_GtoIi = clone_Ii->Rot();
      Eigen::Vector3d p_IiinG = clone_Ii->pos();

      // Get current feature in the IMU
      Eigen::Vector3d p_FinIi = R_GtoIi * (p_FinG - p_IiinG);

      // Project the current feature into the current frame of reference
      Eigen::Vector3d p_FinCi = R_ItoC * p_FinIi + p_IinC;
      Eigen::Vector2d uv_norm;
      uv_norm << p_FinCi(0) / p_FinCi(2), p_FinCi(1) / p_FinCi(2);

      // Distort the normalized coordinates (radtan or fisheye)
      Eigen::Vector2d uv_dist;
      uv_dist = state->_cam_intrinsics_cameras.at(pair.first)->distort_d(uv_norm);

      // Our residual
      Eigen::Vector2d uv_m;
      uv_m << (double)feature.uvs[pair.first].at(m)(0), (double)feature.uvs[pair.first].at(m)(1);
      res.block(2 * c, 0, 2, 1) = uv_m - uv_dist;

      //=========================================================================
      //=========================================================================

      // If we are doing first estimate Jacobians, then overwrite with the first estimates
      if (state->_options.do_fej) {
        R_GtoIi = clone_Ii->Rot_fej();
        p_IiinG = clone_Ii->pos_fej();
        // R_ItoC = calibration->Rot_fej();
        // p_IinC = calibration->pos_fej();
        p_FinIi = R_GtoIi * (p_FinG_fej - p_IiinG);
        p_FinCi = R_ItoC * p_FinIi + p_IinC;
        // uv_norm << p_FinCi(0)/p_FinCi(2),p_FinCi(1)/p_FinCi(2);
        // cam_d = state->get_intrinsics_CAM(pair.first)->fej();
      }

      // Compute Jacobians in respect to normalized image coordinates and possibly the camera intrinsics
      Eigen::MatrixXd dz_dzn, dz_dzeta;
      state->_cam_intrinsics_cameras.at(pair.first)->compute_distort_jacobian(uv_norm, dz_dzn, dz_dzeta);

      // Normalized coordinates in respect to projection function
      Eigen::MatrixXd dzn_dpfc = Eigen::MatrixXd::Zero(2, 3);
      dzn_dpfc << 1 / p_FinCi(2), 0, -p_FinCi(0) / (p_FinCi(2) * p_FinCi(2)), 0, 1 / p_FinCi(2), -p_FinCi(1) / (p_FinCi(2) * p_FinCi(2));

      // Derivative of p_FinCi in respect to p_FinIi
      Eigen::MatrixXd dpfc_dpfg = R_ItoC * R_GtoIi;

      // Derivative of p_FinCi in respect to camera clone state
      Eigen::MatrixXd dpfc_dclone = Eigen::MatrixXd::Zero(3, 6);
      dpfc_dclone.block(0, 0, 3, 3).noalias() = R_ItoC * skew_x(p_FinIi);
      dpfc_dclone.block(0, 3, 3, 3) = -dpfc_dpfg;

      //=========================================================================
      //=========================================================================

      // Precompute some matrices
      Eigen::MatrixXd dz_dpfc = dz_dzn * dzn_dpfc;
      Eigen::MatrixXd dz_dpfg = dz_dpfc * dpfc_dpfg;

      // CHAINRULE: get the total feature Jacobian
      H_f.block(2 * c, 0, 2, H_f.cols()).noalias() = dz_dpfg * dpfg_dlambda;

      // CHAINRULE: get state clone Jacobian
      H_x.block(2 * c, map_hx[clone_Ii], 2, clone_Ii->size()).noalias() = dz_dpfc * dpfc_dclone;

      // CHAINRULE: loop through all extra states and add their
      // NOTE: we add the Jacobian here as we might be in the anchoring pose for this measurement
      for (size_t i = 0; i < dpfg_dx_order.size(); i++) {
        H_x.block(2 * c, map_hx[dpfg_dx_order.at(i)], 2, dpfg_dx_order.at(i)->size()).noalias() += dz_dpfg * dpfg_dx.at(i);
      }

      //=========================================================================
      //=========================================================================

      // Derivative of p_FinCi in respect to camera calibration (R_ItoC, p_IinC)
      if (state->_options.do_calib_camera_pose) {

        // Calculate the Jacobian
        Eigen::MatrixXd dpfc_dcalib = Eigen::MatrixXd::Zero(3, 6);
        dpfc_dcalib.block(0, 0, 3, 3) = skew_x(p_FinCi - p_IinC);
        dpfc_dcalib.block(0, 3, 3, 3) = Eigen::Matrix<double, 3, 3>::Identity();

        // Chainrule it and add it to the big jacobian
        H_x.block(2 * c, map_hx[calibration], 2, calibration->size()).noalias() += dz_dpfc * dpfc_dcalib;
      }

      // Derivative of measurement in respect to distortion parameters
      if (state->_options.do_calib_camera_intrinsics) {
        H_x.block(2 * c, map_hx[distortion], 2, distortion->size()) = dz_dzeta;
      }

      // Move the Jacobian and residual index forward
      c++;
    }
  }
}

void UpdaterHelper::nullspace_project_inplace(Eigen::MatrixXd &H_f, Eigen::MatrixXd &H_x, Eigen::VectorXd &res) {

  // Apply the left nullspace of H_f to all variables
  // Based on "Matrix Computations 4th Edition by Golub and Van Loan"
  // See page 252, Algorithm 5.2.4 for how these two loops work
  // They use "matlab" index notation, thus we need to subtract 1 from all index
  Eigen::JacobiRotation<double> tempHo_GR;
  for (int n = 0; n < H_f.cols(); ++n) {
    for (int m = (int)H_f.rows() - 1; m > n; m--) {
      // Givens matrix G
      tempHo_GR.makeGivens(H_f(m - 1, n), H_f(m, n));
      // Multiply G to the corresponding lines (m-1,m) in each matrix
      // Note: we only apply G to the nonzero cols [n:Ho.cols()-n-1], while
      //       it is equivalent to applying G to the entire cols [0:Ho.cols()-1].
      (H_f.block(m - 1, n, 2, H_f.cols() - n)).applyOnTheLeft(0, 1, tempHo_GR.adjoint());
      (H_x.block(m - 1, 0, 2, H_x.cols())).applyOnTheLeft(0, 1, tempHo_GR.adjoint());
      (res.block(m - 1, 0, 2, 1)).applyOnTheLeft(0, 1, tempHo_GR.adjoint());
    }
  }

  // The H_f jacobian max rank is 3 if it is a 3d position, thus size of the left nullspace is Hf.rows()-3
  // NOTE: need to eigen3 eval here since this experiences aliasing!
  // H_f = H_f.block(H_f.cols(),0,H_f.rows()-H_f.cols(),H_f.cols()).eval();
  H_x = H_x.block(H_f.cols(), 0, H_x.rows() - H_f.cols(), H_x.cols()).eval();
  res = res.block(H_f.cols(), 0, res.rows() - H_f.cols(), res.cols()).eval();

  // Sanity check
  assert(H_x.rows() == res.rows());
}

void UpdaterHelper::measurement_compress_inplace(Eigen::MatrixXd &H_x, Eigen::VectorXd &res) {

  // Return if H_x is a fat matrix (there is no need to compress in this case)
  if (H_x.rows() <= H_x.cols())
    return;

  // Do measurement compression through givens rotations
  // Based on "Matrix Computations 4th Edition by Golub and Van Loan"
  // See page 252, Algorithm 5.2.4 for how these two loops work
  // They use "matlab" index notation, thus we need to subtract 1 from all index
  Eigen::JacobiRotation<double> tempHo_GR;
  for (int n = 0; n < H_x.cols(); n++) {
    for (int m = (int)H_x.rows() - 1; m > n; m--) {
      // Givens matrix G
      tempHo_GR.makeGivens(H_x(m - 1, n), H_x(m, n));
      // Multiply G to the corresponding lines (m-1,m) in each matrix
      // Note: we only apply G to the nonzero cols [n:Ho.cols()-n-1], while
      //       it is equivalent to applying G to the entire cols [0:Ho.cols()-1].
      (H_x.block(m - 1, n, 2, H_x.cols() - n)).applyOnTheLeft(0, 1, tempHo_GR.adjoint());
      (res.block(m - 1, 0, 2, 1)).applyOnTheLeft(0, 1, tempHo_GR.adjoint());
    }
  }

  // If H is a fat matrix, then use the rows
  // Else it should be same size as our state
  int r = std::min(H_x.rows(), H_x.cols());

  // Construct the smaller jacobian and residual after measurement compression
  assert(r <= H_x.rows());
  H_x.conservativeResize(r, H_x.cols());
  res.conservativeResize(r, res.cols());
}

UpdaterHelper::CbfOutput
UpdaterHelper::compute_cbf_metrics(std::shared_ptr<State> state, const std::vector<std::shared_ptr<ov_core::Feature>> &feature_vec,
                                   const std::unordered_map<size_t, std::unordered_map<double, FeatureInitializer::ClonePose>> &clones_cam) {

  // ============================================================================
  // CBF: Compute the average log-det observability metric ℓ̄ = (1/s) Σ log(det(M_i)),
  // the drift f(x), and the control gradient g(x) in the body (IMU) frame.
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
  CbfOutput out;
  if (feature_vec.empty() || clones_cam.empty())
    return out;

  // IMU-to-camera rotation for the first camera (cam 0)
  const Eigen::Matrix3d R_ItoC = state->_calib_IMUtoCAM.at(0)->Rot(); // R_B^C
  const Eigen::Matrix3d R_CtoI = R_ItoC.transpose();                  // (R_B^C)^T

  // ── Estimate past velocities from clone positions ──
  // Clone times are sorted in ascending order (std::map)
  // v_{G,j} ≈ (p_{j+1} - p_j) / Δt, then v_{B,j} = R_GtoI_j · v_{G,j}
  struct CloneVel {
    double time;
    Eigen::Vector3d v_body; // velocity in body (IMU) frame
    bool is_current;        // true for the latest (current) pose
  };
  std::vector<CloneVel> clone_vels;
  {
    std::vector<std::pair<double, std::shared_ptr<ov_type::PoseJPL>>> sorted_clones(state->_clones_IMU.begin(),
                                                                                    state->_clones_IMU.end());
    for (size_t idx = 0; idx < sorted_clones.size(); idx++) {
      double t_j = sorted_clones[idx].first;
      bool is_last = (idx == sorted_clones.size() - 1);
      Eigen::Vector3d v_body;
      if (is_last) {
        // Current pose: use the EKF's estimated velocity directly (state->_imu->vel() is v_IinG)
        Eigen::Matrix3d R_GtoI_j = sorted_clones[idx].second->Rot();
        v_body = R_GtoI_j * state->_imu->vel();
      } else {
        // Past pose: numerical differentiation of global positions
        double t_next = sorted_clones[idx + 1].first;
        double dt = t_next - t_j;
        if (dt < 1e-9)
          dt = 1e-9; // guard against division by zero
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

  Eigen::Vector3d g_sum = Eigen::Vector3d::Zero(); // control gradient accumulator
  double f_sum = 0.0;                              // drift accumulator
  double logdet_sum = 0.0;                         // average log-det accumulator
  int s = 0;                                       // number of valid features

  for (const auto &feat : feature_vec) {

    // Get anchor pose (skip if the anchor frame is not in our current window)
    auto cam_it = clones_cam.find(feat->anchor_cam_id);
    if (cam_it == clones_cam.end())
      continue;
    auto anchor_it = cam_it->second.find(feat->anchor_clone_timestamp);
    if (anchor_it == cam_it->second.end())
      continue;
    const Eigen::Matrix3d &R_GtoA = anchor_it->second.Rot();
    const Eigen::Vector3d &p_AinG = anchor_it->second.pos();

    // 1. Build information matrix M_i = Σ_j π_{i,j}
    Eigen::Matrix3d M_i = Eigen::Matrix3d::Zero();
    struct BearingEntry {
      Eigen::Vector3d b;
      Eigen::Matrix3d R_AtoCi;
      Eigen::Vector3d p_CiinA;
      double clone_time; // timestamp of this clone
    };
    std::vector<BearingEntry> entries;

    for (const auto &pair : feat->timestamps) {
      auto cam_clones_it = clones_cam.find(pair.first);
      if (cam_clones_it == clones_cam.end())
        continue;
      for (size_t m = 0; m < pair.second.size(); m++) {
        auto clone_it = cam_clones_it->second.find(pair.second.at(m));
        if (clone_it == cam_clones_it->second.end())
          continue; // measurement not at a clone time in the active window
        const Eigen::Matrix3d &R_GtoCi = clone_it->second.Rot();
        const Eigen::Vector3d &p_CiinG = clone_it->second.pos();

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

    // 2. Check if M_i is invertible (sufficient parallax)
    Eigen::JacobiSVD<Eigen::Matrix3d> svd_Mi(M_i, Eigen::ComputeFullU | Eigen::ComputeFullV);
    double lambda_min = svd_Mi.singularValues()(2);
    if (lambda_min < 1e-8)
      continue; // rank-deficient, skip this feature

    Eigen::Matrix3d M_inv = svd_Mi.solve(Eigen::Matrix3d::Identity());

    // log(det(M_i)) = Σ log(singular values); M_i is symmetric PSD so singular = eigen values
    double logdet_i = 0.0;
    for (int k = 0; k < 3; k++)
      logdet_i += std::log(svd_Mi.singularValues()(k));

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
    logdet_sum += logdet_i;
    s++;
  }

  if (s > 0) {
    out.mean_logdet = logdet_sum / s;
    out.g_vec = g_sum / s;
    out.drift = f_sum / s;
    out.num_features = s;
    out.valid = true;
  }
  return out;
}

UpdaterHelper::CbfOutput UpdaterHelper::smooth_cbf_metrics(const CbfOutput &raw, CbfEmaState &ema) {

  CbfOutput smoothed;
  if (!raw.valid)
    return smoothed; // valid == false

  // Apply EMA smoothing (weight alpha by feature count for stability)
  if (!ema.initialized) {
    ema.logdet = raw.mean_logdet;
    ema.g = raw.g_vec;
    ema.drift = raw.drift;
    ema.initialized = true;
  } else {
    double alpha = std::min(1.0, ema.alpha * (raw.num_features / 10.0));
    ema.logdet = alpha * raw.mean_logdet + (1.0 - alpha) * ema.logdet;
    ema.g = alpha * raw.g_vec + (1.0 - alpha) * ema.g;
    ema.drift = alpha * raw.drift + (1.0 - alpha) * ema.drift;
  }

  smoothed.mean_logdet = ema.logdet;
  smoothed.g_vec = ema.g;
  smoothed.drift = ema.drift;
  smoothed.num_features = raw.num_features;
  smoothed.valid = true;
  return smoothed;
}
