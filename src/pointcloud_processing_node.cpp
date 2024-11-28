// Include necessary headers
#include <rclcpp/rclcpp.hpp>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <tf2_sensor_msgs/tf2_sensor_msgs.hpp>
//#include "obstacle_detection/pointcloud_processing_node.hpp"


class PointCloudProcessingNode : public rclcpp::Node
{
public:
  PointCloudProcessingNode(const std::string & node_name)
  : Node(node_name)
  {
    tfbuffer = std::make_shared<tf2_ros::Buffer>(this->get_clock());
    tflistener = std::make_shared<tf2_ros::TransformListener>(*tfbuffer);

    this->declare_parameter<std::string>("target_frame", "base_link");
    targetframe = this->get_parameter("target_frame").as_string();
  }
protected:
  bool transformPointCloud(
    const sensor_msgs::msg::PointCloud2::SharedPtr & input_cloud,
    sensor_msgs::msg::PointCloud2 & output_cloud,
    const std::string & target_frame)
  {
    try
    {
      tfbuffer->transform(
        *input_cloud,
        output_cloud,
        target_frame,
        tf2::durationFromSec(0.1));
      return true;
    }
    catch (tf2::TransformException &ex)
    {
      RCLCPP_WARN(this->get_logger(), "Could not transform point cloud: %s", ex.what());
      return false;
    }
  }
  // TF2 buffer and listener
  std::shared_ptr<tf2_ros::Buffer> tfbuffer;
  std::shared_ptr<tf2_ros::TransformListener> tflistener;
  std::string targetframe;
};