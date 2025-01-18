// Augmented Cartesian Impedance Controller
// This controller extends the functionality of the original Cartesian Impedance Controller
// by allowing a desired wrench to be specified in addition to the target
// end-effector pose. The controller adjusts the desired pose of the end-effector to achieve
// the specified wrench equilibrium, enhancing the ability to interact with the
// environment through controlled forces and torques.
// 
// This implementation is designed for the Franka Emika Panda robot, incorporating both
// Cartesian impedance control and direct wrench commands for improved manipulation tasks.
// 
// Author: Ekansh Sharma
// Copyright (c) 2023 Franka Robotics GmbH
// Licensed under the Apache-2.0 License. See the LICENSE file for details.

#pragma once

#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <controller_interface/multi_interface_controller.h>
#include <dynamic_reconfigure/server.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/WrenchStamped.h>
#include <hardware_interface/joint_command_interface.h>
#include <hardware_interface/robot_hw.h>
#include <ros/node_handle.h>
#include <ros/time.h>
#include <tf2_ros/transform_listener.h>
#include <geometry_msgs/TransformStamped.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <Eigen/Dense>

#include <franka_example_controllers/compliance_paramConfig.h>
#include <franka_hw/franka_model_interface.h>
#include <franka_hw/franka_state_interface.h>

#include <oae_msgs/DirectionalCompliance.h>

namespace franka_example_controllers {

class AugmentedCartesianImpedanceExampleController : public controller_interface::MultiInterfaceController<
                                                franka_hw::FrankaModelInterface,
                                                hardware_interface::EffortJointInterface,
                                                franka_hw::FrankaStateInterface> {
 public:
  bool init(hardware_interface::RobotHW* robot_hw, ros::NodeHandle& node_handle) override;
  void starting(const ros::Time&) override;
  void update(const ros::Time&, const ros::Duration& period) override;

 private:
  // Saturation
  Eigen::Matrix<double, 7, 1> saturateTorqueRate(
      const Eigen::Matrix<double, 7, 1>& tau_d_calculated,
      const Eigen::Matrix<double, 7, 1>& tau_J_d);  // NOLINT (readability-identifier-naming)

  std::unique_ptr<franka_hw::FrankaStateHandle> state_handle_;
  std::unique_ptr<franka_hw::FrankaModelHandle> model_handle_;
  std::vector<hardware_interface::JointHandle> joint_handles_;

  double filter_params_{0.005};
  double nullspace_stiffness_{20.0};
  double nullspace_stiffness_target_{20.0};
  const double delta_tau_max_{1.0};
  Eigen::Matrix<double, 6, 6> cartesian_stiffness_;
  Eigen::Matrix<double, 6, 6> cartesian_stiffness_target_;
  Eigen::Matrix<double, 6, 6> cartesian_damping_;
  Eigen::Matrix<double, 6, 6> cartesian_damping_target_;
  Eigen::Matrix<double, 7, 1> q_d_nullspace_;
  Eigen::Vector3d position_d_;
  Eigen::Quaterniond orientation_d_;
  std::mutex position_and_orientation_d_target_mutex_;
  Eigen::Vector3d position_d_target_;
  Eigen::Quaterniond orientation_d_target_;

  // Dynamic reconfigure
  std::unique_ptr<dynamic_reconfigure::Server<franka_example_controllers::compliance_paramConfig>>
      dynamic_server_compliance_param_;
  ros::NodeHandle dynamic_reconfigure_compliance_param_node_;
  void complianceParamCallback(franka_example_controllers::compliance_paramConfig& config,
                               uint32_t level);

  // Equilibrium pose subscriber
  ros::Subscriber sub_equilibrium_pose_;
  void equilibriumPoseCallback(const geometry_msgs::PoseStampedConstPtr& msg);

  // For augmented wrench control
  // Reference: (Matthias Mayr) https://joss.theoj.org/papers/10.21105/joss.05194
  double wrench_filter_params_{0.001};
  Eigen::Matrix<double, 6, 1> cartesian_wrench_target_, cartesian_wrench_, transformed_cartesian_wrench_target_root_, transformed_cartesian_wrench_target_ee_;
  std::mutex cartesian_wrench_target_mutex_;
  ros::Subscriber sub_cartesian_wrench_;
  ros::Publisher pub_cartesian_wrench_;
  std::string wrench_root_frame_id_;
  std::string wrench_ee_frame_id_;
  geometry_msgs::TransformStamped transform_ee; 
  std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
  std::unique_ptr<tf2_ros::TransformListener> tf_listener_;
  Eigen::Matrix<double, 6, 1> transformWrench(const geometry_msgs::WrenchStamped& wrench_msg, const geometry_msgs::TransformStamped& transform);
  std::shared_ptr<geometry_msgs::WrenchStamped> createWrenchMsg(const Eigen::Matrix<double, 6, 1>& wrench);
   // Add these new members
  ros::Timer tf_update_timer_;
  geometry_msgs::TransformStamped cached_transform_;
  std::mutex transform_mutex_;
  bool transform_valid_{false};

  // New method declarations
  void tfUpdateCallback(const ros::TimerEvent&);
  void updateTransformedWrench();

  void pubWrench(const Eigen::Matrix<double, 6, 1>& wrench, ros::Publisher& pub);
  void wrenchCommandCb(const geometry_msgs::WrenchStampedConstPtr& msg);
  
  // Weight matrices for selective control
  Eigen::Matrix<double, 6, 6> weight_matrix_;  // Current active weight matrix
  Eigen::Matrix<double, 6, 6> default_weight_matrix_;
  bool using_selective_weight_{false};  // Flag to track if selective weight is active
  ros::Subscriber sub_directional_compliance_;
  ros::Publisher pub_directional_compliance_;
  void directionalComplianceCallback(const oae_msgs::DirectionalComplianceConstPtr& msg);
};

}  // namespace franka_example_controllers
