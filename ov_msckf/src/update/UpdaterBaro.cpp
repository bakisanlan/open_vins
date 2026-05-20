/*
 * OpenVINS: An Open Platform for Visual-Inertial Research
 * UpdaterBaro implementation for Barometric Altimeter Update
 */

#include "UpdaterBaro.h"
#include "UpdaterHelper.h"
#include "state/State.h"
#include "state/StateHelper.h"
#include "utils/colors.h"
#include "utils/print.h"

#include <boost/math/distributions/chi_squared.hpp>

using namespace ov_core;
using namespace ov_type;
using namespace ov_msckf;

UpdaterBaro::UpdaterBaro(UpdaterOptions &options) : _options(options) {
  // Initialize the chi squared test table with confidence level 0.95 for 1 DOF
  boost::math::chi_squared chi_squared_dist(1);
  chi_squared_value = boost::math::quantile(chi_squared_dist, 0.95);
}

bool UpdaterBaro::try_update(std::shared_ptr<State> state, double timestamp, double measured_altitude) {

  // For the baro update we need to update the active IMU position in the global frame.
  // We don't interpolate right now; we assume the measurement arrives close enough to the IMU state.
  // Note: if doing advanced interpolation we would use a clone, but for a simple 1D z update 
  // on a slow-changing variable we update the current state `p_IinG`.

  // The state order for our Jacobian: [p_IinG]
  std::vector<std::shared_ptr<Type>> Hx_order;
  Hx_order.push_back(state->_imu->p());

  // Creating the exact measurement Jacobian (1x3)
  // Our measurement equation: h(x) = p_IinG.z()
  // Therefore dh/dp = [0, 0, 1]
  Eigen::MatrixXd H = Eigen::MatrixXd::Zero(1, 3);
  H(0, 2) = 1.0;

  // The expected altitude is simply the z-coordinate of the current IMU position
  double expected_altitude = state->_imu->pos()(2);

  // Measurement residual: z_meas - h(x)
  Eigen::VectorXd res = Eigen::VectorXd::Zero(1);
  res(0) = measured_altitude - expected_altitude;

  // Measurement noise covariance (R)
  Eigen::MatrixXd R = Eigen::MatrixXd::Identity(1, 1);
  R(0, 0) = _options.sigma_pix_sq; // User configurable variance from yaml

  // Chi2 distance check
  Eigen::MatrixXd P_marg = StateHelper::get_marginal_covariance(state, Hx_order);
  Eigen::MatrixXd S = H * P_marg * H.transpose() + R;
  double chi2 = res.dot(S.llt().solve(res));

  // Check if we pass the chi-square test
  if (chi2 > _options.chi2_multipler * chi_squared_value) {
    PRINT_DEBUG(YELLOW "[BARO]: rejected altitude update (chi2 %.3f > %.3f)\n" RESET, chi2, _options.chi2_multipler * chi_squared_value);
    return false;
  }

  // PRINT_INFO(CYAN "[BARO]: accepted altitude update | res: %.3f m | (chi2 %.3f < %.3f)\n" RESET, res(0), chi2, _options.chi2_multipler * chi_squared_value);

  // Finally perform the EKF update on the current state
  StateHelper::EKFUpdate(state, Hx_order, H, res, R);

  return true;
}
