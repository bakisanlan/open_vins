/*
 * OpenVINS: An Open Platform for Visual-Inertial Research
 * UpdaterBaro header for Barometric Altimeter Update
 */

#ifndef OV_MSCKF_UPDATER_BARO_H
#define OV_MSCKF_UPDATER_BARO_H

#include <memory>
#include <vector>

#include "UpdaterOptions.h"

namespace ov_msckf {

class State;

/**
 * @brief Will try to update the filter using a barometric altimeter measurement.
 *
 * This update fuses a 1D vertical measurement (altitude) to correct the z-channel.
 * It assumes a standard pressure-to-altitude barometric formula.
 */
class UpdaterBaro {

public:
  /**
   * @brief Default constructor for our barometer updater.
   * @param options Updater options (contains baro noise sigma and chi2 multiplier)
   */
  UpdaterBaro(UpdaterOptions &options);

  /**
   * @brief Will attempt to do a state update using the barometric altitude.
   * @param state State of the filter
   * @param timestamp Time of the baro measurement
   * @param measured_altitude The calculated relative altitude from the barometer
   * @return True if the update was successful
   */
  bool try_update(std::shared_ptr<State> state, double timestamp, double measured_altitude);

protected:
  /// Options used during update (noise and chi2 multiplier)
  UpdaterOptions _options;

  /// Chi squared 95th percentile table for a 1D measurement
  double chi_squared_value;
};

} // namespace ov_msckf

#endif // OV_MSCKF_UPDATER_BARO_H
