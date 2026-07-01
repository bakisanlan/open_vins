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

#ifndef OV_MSCKF_ROS2VISUALIZER_H
#define OV_MSCKF_ROS2VISUALIZER_H

#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <geometry_msgs/msg/vector3_stamped.hpp>
#include <image_transport/image_transport.h>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/time_synchronizer.h>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/fluid_pressure.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/nav_sat_fix.hpp>
#include <sensor_msgs/msg/point_cloud.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <std_msgs/msg/float64.hpp>
#include <geographic_msgs/msg/geo_point_stamped.hpp>
#include <mavros_msgs/srv/command_home.hpp>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/transform_datatypes.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <tf2_ros/transform_broadcaster.h>

#include <atomic>
#include <fstream>
#include <memory>
#include <mutex>

#include <Eigen/Eigen>
#include <boost/date_time/posix_time/posix_time.hpp>
#include <boost/filesystem.hpp>
#include <cv_bridge/cv_bridge.h>

namespace ov_core {
class YamlParser;
struct CameraData;
} // namespace ov_core

namespace ov_msckf {

class VioManager;
class Simulator;

/**
 * @brief Helper class that will publish results onto the ROS framework.
 *
 * Also save to file the current total state and covariance along with the groundtruth if we are simulating.
 * We visualize the following things:
 * - State of the system on TF, pose message, and path
 * - Image of our tracker
 * - Our different features (SLAM, MSCKF, ARUCO)
 * - Groundtruth trajectory if we have it
 */
class ROS2Visualizer {

public:
  /**
   * @brief Default constructor
   * @param node ROS node pointer
   * @param app Core estimator manager
   * @param sim Simulator if we are simulating
   */
  ROS2Visualizer(std::shared_ptr<rclcpp::Node> node, std::shared_ptr<VioManager> app, std::shared_ptr<Simulator> sim = nullptr);

  /**
   * @brief Will setup ROS subscribers and callbacks
   * @param parser Configuration file parser
   */
  void setup_subscribers(std::shared_ptr<ov_core::YamlParser> parser);

  /**
   * @brief Will visualize the system if we have new things
   */
  void visualize();

  /**
   * @brief Will publish our odometry message for the current timestep.
   * This will take the current state estimate and get the propagated pose to the desired time.
   * This can be used to get pose estimates on systems which require high frequency pose estimates.
   */
  void visualize_odometry(double timestamp);

  /**
   * @brief After the run has ended, print results
   */
  void visualize_final();

  /// Callback for inertial information
  void callback_inertial(const sensor_msgs::msg::Imu::SharedPtr msg);

  /// Callback for barometric pressure information
  void callback_baro(const sensor_msgs::msg::FluidPressure::SharedPtr msg);

  /// Callback for MAVROS relative altitude (used to correct home altitude offset)
  void callback_rel_alt(const std_msgs::msg::Float64::SharedPtr msg);

  /// Callback for monocular cameras information
  void callback_monocular(const sensor_msgs::msg::Image::SharedPtr msg0, int cam_id0);

  /// Callback for synchronized stereo camera information
  void callback_stereo(const sensor_msgs::msg::Image::ConstSharedPtr msg0, const sensor_msgs::msg::Image::ConstSharedPtr msg1, int cam_id0,
                       int cam_id1);

  /// Callback for magnetometer yaw from MAVROS
  void callback_mag_yaw(const sensor_msgs::msg::Imu::SharedPtr msg);

  /**
   * @brief Callback for MAVROS local position (optional ground truth source)
   * @param msg PoseStamped from /mavros/local_position/pose in ENU frame
   */
  void callback_mavros_gt(const geometry_msgs::msg::PoseStamped::SharedPtr msg);

protected:
  /// Publish the current state
  void publish_state();

  /// Publish the active tracking image
  void publish_images();

  /// Publish current features
  void publish_features();

  /// Publish groundtruth (if we have it)
  void publish_groundtruth();

  /// Publish loop-closure information of current pose and active track information
  void publish_loopclosure_information();

  /// Global node handler
  std::shared_ptr<rclcpp::Node> _node;

  /// Core application of the filter system
  std::shared_ptr<VioManager> _app;

  /// Simulator (is nullptr if we are not sim'ing)
  std::shared_ptr<Simulator> _sim;

  // Our publishers
  image_transport::Publisher it_pub_tracks, it_pub_loop_img_depth, it_pub_loop_img_depth_color;
  rclcpp::Publisher<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr pub_poseimu;
  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr pub_fakegps_vision;
  rclcpp::Publisher<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr pub_fakegps_vision_cov;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pub_odomimu;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr pub_pathimu;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_points_msckf, pub_points_slam, pub_points_aruco, pub_points_sim;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pub_loop_pose, pub_loop_extrinsic;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud>::SharedPtr pub_loop_point;
  rclcpp::Publisher<sensor_msgs::msg::CameraInfo>::SharedPtr pub_loop_intrinsics;
  std::shared_ptr<tf2_ros::TransformBroadcaster> mTfBr;

  // CBF observability publishers
  rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr pub_cbf_mean_logdet;
  rclcpp::Publisher<geometry_msgs::msg::Vector3Stamped>::SharedPtr pub_cbf_g;
  rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr pub_cbf_drift;
  rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr pub_cbf_num_features;
  rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr pub_cbf_tri_tried;
  rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr pub_cbf_tri_success;

  // Our subscribers and camera synchronizers
  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr sub_imu;
  rclcpp::Subscription<sensor_msgs::msg::FluidPressure>::SharedPtr sub_baro;
  std::vector<rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr> subs_cam;
  typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::msg::Image, sensor_msgs::msg::Image> sync_pol;
  std::vector<std::shared_ptr<message_filters::Synchronizer<sync_pol>>> sync_cam;
  std::vector<std::shared_ptr<message_filters::Subscriber<sensor_msgs::msg::Image>>> sync_subs_cam;

  // Magnetometer yaw subscriber
  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr sub_mag_yaw;
  bool mag_yaw_received = false;

  // Output-mode yaw correction state (used when mag_yaw_mode == "output")
  bool yaw_output_initialized = false;
  Eigen::Vector3d yaw_corrected_position = Eigen::Vector3d::Zero();
  Eigen::Vector3d prev_filter_position = Eigen::Vector3d::Zero();
  double latest_mag_yaw = 0.0;
  bool have_mag_yaw = false;
  Eigen::Vector4d latest_imu_orientation = Eigen::Vector4d(0.0, 0.0, 0.0, 1.0); // [x, y, z, w] from AHRS
  std::mutex mag_yaw_mtx;

  // Corrected output state for GT error comparison (populated in publish_state)
  Eigen::Vector3d corrected_p_ENU = Eigen::Vector3d::Zero();
  Eigen::Vector4d corrected_q_GtoI_ENU = Eigen::Vector4d(0, 0, 0, 1);

  // Fixed frame offset between OpenVINS and MAVROS reference frames
  // Computed once on the first GT comparison call
  bool gt_frame_offset_computed = false;
  Eigen::Matrix3d R_offset = Eigen::Matrix3d::Identity();  // R_GtoI_gt * R_GtoI_est^{-1}
  Eigen::Matrix3d R_AtoB = Eigen::Matrix3d::Identity();    // maps position coords from est frame to gt frame
  Eigen::Vector3d p_est_init = Eigen::Vector3d::Zero();
  Eigen::Vector3d p_gt_init = Eigen::Vector3d::Zero();

  // MAVROS local position as optional ground truth source
  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr sub_mavros_gt;
  geometry_msgs::msg::PoseStamped latest_mavros_gt;
  bool have_mavros_gt = false;
  std::mutex mavros_gt_mtx;

  // For path viz
  std::vector<geometry_msgs::msg::PoseStamped> poses_imu;

  // Groundtruth infomation
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr pub_pathgt;
  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr pub_posegt;
  double summed_mse_ori = 0.0;
  double summed_mse_pos = 0.0;
  double summed_nees_ori = 0.0;
  double summed_nees_pos = 0.0;
  size_t summed_number = 0;

  // Start and end timestamps
  bool start_time_set = false;
  boost::posix_time::ptime rT1, rT2;

  // Thread atomics
  std::atomic<bool> thread_update_running;

  /// Queue up camera measurements sorted by time and trigger once we have
  /// exactly one IMU measurement with timestamp newer than the camera measurement
  /// This also handles out-of-order camera measurements, which is rare, but
  /// a nice feature to have for general robustness to bad camera drivers.
  std::deque<ov_core::CameraData> camera_queue;
  std::mutex camera_queue_mtx;

  // Last camera message timestamps we have received (mapped by cam id)
  std::map<int, double> camera_last_timestamp;

  // Last timestamp we visualized at
  double last_visualization_timestamp = 0;
  double last_visualization_timestamp_image = 0;

  // Our groundtruth states
  std::map<double, Eigen::Matrix<double, 17, 1>> gt_states;

  // For path viz
  std::vector<geometry_msgs::msg::PoseStamped> poses_gt;
  bool publish_global2imu_tf = true;
  bool publish_calibration_tf = true;

  // Files and if we should save total state
  bool save_total_state = false;
  std::ofstream of_state_est, of_state_std, of_state_gt;

  // Barometer reference pressure (first reading) for relative altitude
  double baro_ref_pressure = -1.0;

  // GPS origin and home setup (triggered once on first baro reading)
  bool set_home_origin = false;  // if true, publish GPS origin and call set_home on first baro reading
  bool origin_home_set = false;
  double home_latitude = 41.1006384902323;
  double home_longitude = 29.02551275632911;
  rclcpp::Publisher<geographic_msgs::msg::GeoPointStamped>::SharedPtr pub_gp_origin;
  rclcpp::Client<mavros_msgs::srv::CommandHome>::SharedPtr srv_set_home;
  rclcpp::CallbackGroup::SharedPtr srv_callback_group;  // separate thread for service calls

  // Rel-alt subscriber and home correction state
  rclcpp::Subscription<std_msgs::msg::Float64>::SharedPtr sub_rel_alt;
  bool first_home_set = false;       // true after the first set_home service call succeeds
  bool home_corrected = false;       // true after corrections are done (converged or max attempts)
  double home_altitude_used = 0.0;   // the altitude value used in the last set_home call
  int rel_alt_wait_count = 0;        // messages received since home set / last correction
  int home_correction_attempt = 0;   // number of correction attempts made
};

} // namespace ov_msckf

#endif // OV_MSCKF_ROS2VISUALIZER_H
