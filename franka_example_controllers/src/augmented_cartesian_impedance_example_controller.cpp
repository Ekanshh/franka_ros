// Augmented Cartesian Impedance Controller
// This controller extends the functionality of the original Cartesian Impedance Controller
// by allowing a desired wrench to be specified in addition to the target
// end-effector pose. The controller adjusts the desired pose of the end-effector to achieve
// the specified wrench, enhancing the ability to interact with the environment through 
// controlled forces and torques.
// 
// This implementation is designed for the Franka Emika Panda robot, incorporating both
// Cartesian impedance control and direct wrench commands for improved manipulation tasks.
// 
// Author: Ekansh Sharma
// Copyright (c) 2023 Franka Robotics GmbH
// Licensed under the Apache-2.0 License. See the LICENSE file for details.

#include <franka_example_controllers/augmented_cartesian_impedance_example_controller.h>

#include <cmath>
#include <memory>

#include <controller_interface/controller_base.h>
#include <franka/robot_state.h>
#include <pluginlib/class_list_macros.h>
#include <ros/ros.h>


#include <franka_example_controllers/pseudo_inversion.h>


namespace franka_example_controllers {

bool AugmentedCartesianImpedanceExampleController::init(hardware_interface::RobotHW* robot_hw,
                                               ros::NodeHandle& node_handle) {
  std::vector<double> cartesian_stiffness_vector;
  std::vector<double> cartesian_damping_vector;

  sub_equilibrium_pose_ = node_handle.subscribe(
      "equilibrium_pose", 20, &AugmentedCartesianImpedanceExampleController::equilibriumPoseCallback, this,
      ros::TransportHints().reliable().tcpNoDelay());

  sub_cartesian_wrench_ = node_handle.subscribe(
      "equilibrium_wrench", 20, &AugmentedCartesianImpedanceExampleController::wrenchCommandCb, this,
      ros::TransportHints().reliable().tcpNoDelay());

  pub_cartesian_wrench_ = node_handle.advertise<geometry_msgs::WrenchStamped>("current_commanded_wrench", 10);

  // Directional compliance control
  sub_directional_compliance_ = node_handle.subscribe(
      "equilibrium_pose/selective", 20, &AugmentedCartesianImpedanceExampleController::directionalComplianceCallback, this,
      ros::TransportHints().reliable().tcpNoDelay());

  pub_directional_compliance_ = node_handle.advertise<geometry_msgs::WrenchStamped>("current_commanded_wrench/selective", 10);
  default_weight_matrix_.setIdentity();
  weight_matrix_ = default_weight_matrix_;

  // Initialize tf2 buffer with a dedicated thread
  tf_buffer_ = std::make_unique<tf2_ros::Buffer>();
  tf_buffer_->setUsingDedicatedThread(true);
  tf_listener_ = std::make_unique<tf2_ros::TransformListener>(*tf_buffer_);
  
  // Load frame IDs from parameters with defaults
  std::string ee_frame_id;
  if (!node_handle.getParam("ee_frame_id", ee_frame_id)) {
    ee_frame_id = "panda_K";  // Default end-effector frame for Franka
    ROS_WARN_STREAM("No ee_frame_id specified, using default: " << ee_frame_id);
  }
  wrench_ee_frame_id_ = ee_frame_id;
  
  std::string root_frame_id;
  if (!node_handle.getParam("root_frame_id", root_frame_id)) {
    root_frame_id = "panda_link0";  // Default base frame for Franka
    ROS_WARN_STREAM("No root_frame_id specified, using default: " << root_frame_id);
  }
  wrench_root_frame_id_ = root_frame_id;
  
  // Set up timer for tf updates at a lower frequency (e.g., 10Hz)
  tf_update_timer_ = node_handle.createTimer(
      ros::Duration(0.5), &AugmentedCartesianImpedanceExampleController::tfUpdateCallback, this);


  std::string arm_id;
  if (!node_handle.getParam("arm_id", arm_id)) {
    ROS_ERROR_STREAM("AugmentedCartesianImpedanceExampleController: Could not read parameter arm_id");
    return false;
  }
  std::vector<std::string> joint_names;
  if (!node_handle.getParam("joint_names", joint_names) || joint_names.size() != 7) {
    ROS_ERROR(
        "AugmentedCartesianImpedanceExampleController: Invalid or no joint_names parameters provided, "
        "aborting controller init!");
    return false;
  }

  auto* model_interface = robot_hw->get<franka_hw::FrankaModelInterface>();
  if (model_interface == nullptr) {
    ROS_ERROR_STREAM(
        "AugmentedCartesianImpedanceExampleController: Error getting model interface from hardware");
    return false;
  }
  try {
    model_handle_ = std::make_unique<franka_hw::FrankaModelHandle>(
        model_interface->getHandle(arm_id + "_model"));
  } catch (hardware_interface::HardwareInterfaceException& ex) {
    ROS_ERROR_STREAM(
        "AugmentedCartesianImpedanceExampleController: Exception getting model handle from interface: "
        << ex.what());
    return false;
  }

  auto* state_interface = robot_hw->get<franka_hw::FrankaStateInterface>();
  if (state_interface == nullptr) {
    ROS_ERROR_STREAM(
        "AugmentedCartesianImpedanceExampleController: Error getting state interface from hardware");
    return false;
  }
  try {
    state_handle_ = std::make_unique<franka_hw::FrankaStateHandle>(
        state_interface->getHandle(arm_id + "_robot"));
  } catch (hardware_interface::HardwareInterfaceException& ex) {
    ROS_ERROR_STREAM(
        "AugmentedCartesianImpedanceExampleController: Exception getting state handle from interface: "
        << ex.what());
    return false;
  }

  auto* effort_joint_interface = robot_hw->get<hardware_interface::EffortJointInterface>();
  if (effort_joint_interface == nullptr) {
    ROS_ERROR_STREAM(
        "AugmentedCartesianImpedanceExampleController: Error getting effort joint interface from hardware");
    return false;
  }
  for (size_t i = 0; i < 7; ++i) {
    try {
      joint_handles_.push_back(effort_joint_interface->getHandle(joint_names[i]));
    } catch (const hardware_interface::HardwareInterfaceException& ex) {
      ROS_ERROR_STREAM(
          "AugmentedCartesianImpedanceExampleController: Exception getting joint handles: " << ex.what());
      return false;
    }
  }

  dynamic_reconfigure_compliance_param_node_ =
      ros::NodeHandle(node_handle.getNamespace() + "/dynamic_reconfigure_compliance_param_node");

  dynamic_server_compliance_param_ = std::make_unique<
      dynamic_reconfigure::Server<franka_example_controllers::compliance_paramConfig>>(

      dynamic_reconfigure_compliance_param_node_);
  dynamic_server_compliance_param_->setCallback(
      boost::bind(&AugmentedCartesianImpedanceExampleController::complianceParamCallback, this, _1, _2));

  position_d_.setZero();
  orientation_d_.coeffs() << 0.0, 0.0, 0.0, 1.0;
  position_d_target_.setZero();
  orientation_d_target_.coeffs() << 0.0, 0.0, 0.0, 1.0;

  cartesian_stiffness_.setZero();
  cartesian_damping_.setZero();

  cartesian_wrench_.setZero();
  transformed_cartesian_wrench_target_root_.setZero();
  transformed_cartesian_wrench_target_ee_.setZero();

  return true;
}

void AugmentedCartesianImpedanceExampleController::starting(const ros::Time& /*time*/) {
  // compute initial velocity with jacobian and set x_attractor and q_d_nullspace
  // to initial configuration
  franka::RobotState initial_state = state_handle_->getRobotState();
  // get jacobian
  std::array<double, 42> jacobian_array =
      model_handle_->getZeroJacobian(franka::Frame::kEndEffector);
  // convert to eigen
  Eigen::Map<Eigen::Matrix<double, 7, 1>> q_initial(initial_state.q.data());
  Eigen::Affine3d initial_transform(Eigen::Matrix4d::Map(initial_state.O_T_EE.data()));
  Eigen::Map<Eigen::Matrix<double, 6, 1>> inital_wrench(initial_state.O_F_ext_hat_K.data());

  // set equilibrium point to current state
  position_d_ = initial_transform.translation();
  orientation_d_ = Eigen::Quaterniond(initial_transform.rotation());
  position_d_target_ = initial_transform.translation();
  orientation_d_target_ = Eigen::Quaterniond(initial_transform.rotation());

  // set nullspace equilibrium configuration to initial q
  q_d_nullspace_ = q_initial;
}


void AugmentedCartesianImpedanceExampleController::update(const ros::Time& /*time*/,
                                                 const ros::Duration& /*period*/) {
  // get state variables
  franka::RobotState robot_state = state_handle_->getRobotState();
  std::array<double, 7> coriolis_array = model_handle_->getCoriolis();
  std::array<double, 42> jacobian_array =
      model_handle_->getZeroJacobian(franka::Frame::kEndEffector);

  // convert to Eigen
  Eigen::Map<Eigen::Matrix<double, 7, 1>> coriolis(coriolis_array.data());
  Eigen::Map<Eigen::Matrix<double, 6, 7>> jacobian(jacobian_array.data());
  Eigen::Map<Eigen::Matrix<double, 7, 1>> q(robot_state.q.data());
  Eigen::Map<Eigen::Matrix<double, 7, 1>> dq(robot_state.dq.data());
  Eigen::Map<Eigen::Matrix<double, 7, 1>> tau_J_d(  // NOLINT (readability-identifier-naming)
      robot_state.tau_J_d.data());
  Eigen::Affine3d transform(Eigen::Matrix4d::Map(robot_state.O_T_EE.data()));
  Eigen::Vector3d position(transform.translation());
  Eigen::Quaterniond orientation(transform.rotation());

  // publish the cartesian wrench which is transformed to the end-effector frame
  AugmentedCartesianImpedanceExampleController::pubWrench(transformed_cartesian_wrench_target_ee_, pub_cartesian_wrench_);

  // compute error to desired pose
  // position error
  Eigen::Vector3d position_error = position - position_d_;
  
  Eigen::Matrix<double, 6, 1> error;
  // error.head(3) << position_error.cwiseProduct(weight_matrix_.topLeftCorner(3,
  // 3).diagonal());
  error.head(3) << position_error;

  // orientation error
  if (orientation_d_.coeffs().dot(orientation.coeffs()) < 0.0) {
    orientation.coeffs() << -orientation.coeffs();
  }
  // "difference" quaternion
  Eigen::Quaterniond error_quaternion(orientation.inverse() * orientation_d_);
  error.tail(3) << error_quaternion.x(), error_quaternion.y(), error_quaternion.z();
  // Transform to base frame
  error.tail(3) << -transform.rotation() * error.tail(3);

   // Convert orientation error to rotation matrix (R = q2R(e[3:]))
  Eigen::Matrix3d R = error_quaternion.toRotationMatrix();
  
  // Create augmented rotation matrix (R_augm = [R, zeros(); zeros(), R])
  Eigen::Matrix<double, 6, 6> R_augmented;
  R_augmented.setZero();
  R_augmented.topLeftCorner(3, 3) = R;
  R_augmented.bottomRightCorner(3, 3) = R;

  Eigen::Matrix<double, 6, 6> Wr = R_augmented * weight_matrix_ * R_augmented.transpose();

  // Apply weight to error (e_w = Wr * e)
  Eigen::Matrix<double, 6, 1> error_weighted = Wr * error;

  // compute control
  // allocate variables
  Eigen::VectorXd tau_task(7), tau_nullspace(7), tau_ext(7), tau_d(7);

  // pseudoinverse for nullspace handling
  // kinematic pseuoinverse
  Eigen::MatrixXd jacobian_transpose_pinv;
  pseudoInverse(jacobian.transpose(), jacobian_transpose_pinv);

  // Compute base forces
  Eigen::Matrix<double, 6, 1> F;
  F << (-cartesian_stiffness_ * error_weighted - cartesian_damping_ * (jacobian * dq));
  
  // Compute task torques
  tau_task << jacobian.transpose() * F;
                    
  AugmentedCartesianImpedanceExampleController::pubWrench(tau_task, pub_directional_compliance_);

  // nullspace PD control with damping ratio = 1
  tau_nullspace << (Eigen::MatrixXd::Identity(7, 7) -
                    jacobian.transpose() * jacobian_transpose_pinv) *
                       (nullspace_stiffness_ * (q_d_nullspace_ - q) -
                        (2.0 * sqrt(nullspace_stiffness_)) * dq);
  // Desired torque
  tau_d << tau_task + tau_nullspace + coriolis;
  
  // Saturate torque rate to avoid discontinuities
  tau_d << saturateTorqueRate(tau_d, tau_J_d);
  for (size_t i = 0; i < 7; ++i) {
    joint_handles_[i].setCommand(tau_d(i));
  }

  // update parameters changed online either through dynamic reconfigure or through the interactive
  // target by filtering
  cartesian_stiffness_ =
      filter_params_ * cartesian_stiffness_target_ + (1.0 - filter_params_) * cartesian_stiffness_;
  cartesian_damping_ =
      filter_params_ * cartesian_damping_target_ + (1.0 - filter_params_) * cartesian_damping_;
  nullspace_stiffness_ =
      filter_params_ * nullspace_stiffness_target_ + (1.0 - filter_params_) * nullspace_stiffness_;
  std::lock_guard<std::mutex> position_d_target_mutex_lock(
      position_and_orientation_d_target_mutex_);
  position_d_ = filter_params_ * position_d_target_ + (1.0 - filter_params_) * position_d_;
  orientation_d_ = orientation_d_.slerp(filter_params_, orientation_d_target_);

  std::lock_guard<std::mutex> cartesian_wrench_target_mutex_lock(cartesian_wrench_target_mutex_);
  cartesian_wrench_ = wrench_filter_params_ * transformed_cartesian_wrench_target_root_ + (1.0 - wrench_filter_params_) * cartesian_wrench_;
  
  // Update transformed wrench using cached transform
  AugmentedCartesianImpedanceExampleController::updateTransformedWrench();

}

Eigen::Matrix<double, 7, 1> AugmentedCartesianImpedanceExampleController::saturateTorqueRate(
    const Eigen::Matrix<double, 7, 1>& tau_d_calculated,
    const Eigen::Matrix<double, 7, 1>& tau_J_d) {  // NOLINT (readability-identifier-naming)
  Eigen::Matrix<double, 7, 1> tau_d_saturated{};
  for (size_t i = 0; i < 7; i++) {
    double difference = tau_d_calculated[i] - tau_J_d[i];
    tau_d_saturated[i] =
        tau_J_d[i] + std::max(std::min(difference, delta_tau_max_), -delta_tau_max_);
  }
  return tau_d_saturated;
}

void AugmentedCartesianImpedanceExampleController::complianceParamCallback(
    franka_example_controllers::compliance_paramConfig& config,
    uint32_t /*level*/) {
  cartesian_stiffness_target_.setIdentity();
  cartesian_stiffness_target_.topLeftCorner(3, 3)
      << config.translational_stiffness * Eigen::Matrix3d::Identity();
  cartesian_stiffness_target_.bottomRightCorner(3, 3)
      << config.rotational_stiffness * Eigen::Matrix3d::Identity();
  cartesian_damping_target_.setIdentity();
  // Damping ratio = 1
  cartesian_damping_target_.topLeftCorner(3, 3)
      << 2.0 * sqrt(config.translational_stiffness) * Eigen::Matrix3d::Identity();
  cartesian_damping_target_.bottomRightCorner(3, 3)
      << 2.0 * sqrt(config.rotational_stiffness) * Eigen::Matrix3d::Identity();
  nullspace_stiffness_target_ = config.nullspace_stiffness;
}

void AugmentedCartesianImpedanceExampleController::equilibriumPoseCallback(
    const geometry_msgs::PoseStampedConstPtr& msg) {

  ROS_INFO("Received new equilibrium pose command:= [%f %f %f %f %f %f %f]", msg->pose.position.x,
           msg->pose.position.y, msg->pose.position.z, msg->pose.orientation.x, msg->pose.orientation.y,
           msg->pose.orientation.z, msg->pose.orientation.w);

  // When receiving a regular pose command, revert to default weights
  if (using_selective_weight_) {
    weight_matrix_ = default_weight_matrix_;
    using_selective_weight_ = false;
    ROS_DEBUG("Reverting to default weights");
  }

  std::lock_guard<std::mutex> position_d_target_mutex_lock(
      position_and_orientation_d_target_mutex_);
  position_d_target_ << msg->pose.position.x, msg->pose.position.y, msg->pose.position.z;
  Eigen::Quaterniond last_orientation_d_target(orientation_d_target_);
  orientation_d_target_.coeffs() << msg->pose.orientation.x, msg->pose.orientation.y,
      msg->pose.orientation.z, msg->pose.orientation.w;
  if (last_orientation_d_target.coeffs().dot(orientation_d_target_.coeffs()) < 0.0) {
    orientation_d_target_.coeffs() << -orientation_d_target_.coeffs();
  }

}

// Helper function to transform wrench between frames
Eigen::Matrix<double, 6, 1> AugmentedCartesianImpedanceExampleController::transformWrench(const geometry_msgs::WrenchStamped& wrench_msg, const geometry_msgs::TransformStamped& transform) 
{   

    // Convert the input wrench to the target frame
    geometry_msgs::WrenchStamped transformed_wrench;
    tf2::doTransform(wrench_msg, transformed_wrench, transform);
    
    // Return as Eigen vector
    Eigen::Matrix<double, 6, 1> wrench_eigen;
    wrench_eigen << transformed_wrench.wrench.force.x,
                    transformed_wrench.wrench.force.y,
                    transformed_wrench.wrench.force.z,
                    transformed_wrench.wrench.torque.x,
                    transformed_wrench.wrench.torque.y,
                    transformed_wrench.wrench.torque.z;
    return wrench_eigen;
  }


// Modified wrench callback
void AugmentedCartesianImpedanceExampleController::wrenchCommandCb(
    const geometry_msgs::WrenchStampedConstPtr& msg) {
  ROS_INFO("Received new wrench command:= [%f %f %f %f %f %f]", msg->wrench.force.x,
           msg->wrench.force.y, msg->wrench.force.z, msg->wrench.torque.x,
           msg->wrench.torque.y, msg->wrench.torque.z);

  try {
    // Get transform from the wrench frame to Base frame : see reference 
    geometry_msgs::TransformStamped transform_root = tf_buffer_->lookupTransform(
        wrench_root_frame_id_,  // target frame
        msg->header.frame_id,  // source frame
        msg->header.stamp,
        ros::Duration(0.1)  // timeout
    );
    
    // Transform the wrench
    std::lock_guard<std::mutex> cartesian_wrench_target_mutex_lock(cartesian_wrench_target_mutex_);
    
    transformed_cartesian_wrench_target_root_ = transformWrench(*msg, transform_root);

  } catch (tf2::TransformException& ex) {
    ROS_WARN_STREAM("Failed to transform wrench to Base frame: " << ex.what());
    // Keep the previous target wrench
    return;
  }
}
// create wrench msg
std::shared_ptr<geometry_msgs::WrenchStamped> AugmentedCartesianImpedanceExampleController::createWrenchMsg(const Eigen::Matrix<double, 6, 1>& wrench) {
  auto msg = std::make_shared<geometry_msgs::WrenchStamped>();
  msg->header.stamp = ros::Time::now();
  msg->header.frame_id = wrench_ee_frame_id_;
  msg->wrench.force.x = wrench(0);
  msg->wrench.force.y = wrench(1);
  msg->wrench.force.z = wrench(2);
  msg->wrench.torque.x = wrench(3);
  msg->wrench.torque.y = wrench(4);
  msg->wrench.torque.z = wrench(5);
  return msg;
}

void AugmentedCartesianImpedanceExampleController::tfUpdateCallback(const ros::TimerEvent&) {
  // First, check if the frames exist
  if (!tf_buffer_->_frameExists(wrench_ee_frame_id_) || 
      !tf_buffer_->_frameExists(wrench_root_frame_id_)) {
    ROS_WARN_THROTTLE(5.0, "Frames not found in TF tree. Looking for transform from '%s' to '%s'",
                      wrench_root_frame_id_.c_str(), wrench_ee_frame_id_.c_str());
    std::lock_guard<std::mutex> lock(transform_mutex_);
    transform_valid_ = false;
    return;
  }

  try {
    // Perform tf lookup in the callback
    auto transform = tf_buffer_->lookupTransform(
        wrench_ee_frame_id_,      // Target frame
        wrench_root_frame_id_,    // Source frame
        ros::Time(0),            // Latest available transform
        ros::Duration(0.1)       // Small timeout
    );
    
    // Update cached transform
    std::lock_guard<std::mutex> lock(transform_mutex_);
    cached_transform_ = transform;
    transform_valid_ = true;
  } catch (tf2::TransformException &ex) {
    ROS_WARN_THROTTLE(5.0, "Could not get transform: %s", ex.what());
    std::lock_guard<std::mutex> lock(transform_mutex_);
    transform_valid_ = false;
  }
}

void AugmentedCartesianImpedanceExampleController::updateTransformedWrench() {
  std::lock_guard<std::mutex> lock(transform_mutex_);
  if (transform_valid_) {
    auto msg = AugmentedCartesianImpedanceExampleController::createWrenchMsg(cartesian_wrench_);
    transformed_cartesian_wrench_target_ee_ = AugmentedCartesianImpedanceExampleController::transformWrench(*msg, cached_transform_);
  }
}


void AugmentedCartesianImpedanceExampleController::pubWrench(const Eigen::Matrix<double, 6, 1>& wrench, ros::Publisher& pub) {
  geometry_msgs::WrenchStamped msg;
  msg.header.stamp = ros::Time::now();
  msg.header.frame_id = wrench_ee_frame_id_;
  msg.wrench.force.x = wrench(0);
  msg.wrench.force.y = wrench(1);
  msg.wrench.force.z = wrench(2);
  msg.wrench.torque.x = wrench(3);
  msg.wrench.torque.y = wrench(4);
  msg.wrench.torque.z = wrench(5);
  pub.publish(msg);
}

void AugmentedCartesianImpedanceExampleController::directionalComplianceCallback(
    const oae_msgs::DirectionalComplianceConstPtr& msg) {
    
    // Update target pose
    std::lock_guard<std::mutex> position_d_target_mutex_lock(
        position_and_orientation_d_target_mutex_);
    position_d_target_ << msg->target_pose.pose.position.x, 
                         msg->target_pose.pose.position.y, 
                         msg->target_pose.pose.position.z;
    
    orientation_d_target_.coeffs() << msg->target_pose.pose.orientation.x,
                                    msg->target_pose.pose.orientation.y,
                                    msg->target_pose.pose.orientation.z,
                                    msg->target_pose.pose.orientation.w;

    // Create diagonal weight matrices for position and orientation
    Eigen::Matrix3d pos_weights = Eigen::Matrix3d::Zero();
    Eigen::Matrix3d rot_weights = Eigen::Matrix3d::Zero();

    // Set weights based on control flags (1.0 for controlled, small value for compliant)
    const double COMPLIANT_WEIGHT = 1E-16;  // More compliant when false
    
    for (size_t i = 0; i < 3; ++i) {
        pos_weights(i,i) = msg->position_control[i] ? 1.0 : COMPLIANT_WEIGHT;
        rot_weights(i,i) = msg->orientation_control[i] ? 1.0 : COMPLIANT_WEIGHT;
    }

    // Update the weight matrix
    Eigen::Matrix<double, 6, 6> new_weight_matrix;
    new_weight_matrix.setZero();
    new_weight_matrix.topLeftCorner(3, 3) = pos_weights;
    new_weight_matrix.bottomRightCorner(3, 3) = rot_weights;
    
    weight_matrix_ = new_weight_matrix;
    using_selective_weight_ = true;
    
    ROS_DEBUG("Updated selective weight matrix");
}

}  // namespace franka_example_controllers

PLUGINLIB_EXPORT_CLASS(franka_example_controllers::AugmentedCartesianImpedanceExampleController,
                       controller_interface::ControllerBase)
