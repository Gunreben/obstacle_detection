#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/common/common.h> 
#include <pcl_ros/transforms.hpp>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/create_timer_ros.h>
#include <tf2_sensor_msgs/tf2_sensor_msgs.hpp>
#include <random>

// Include CUDA cluster extraction header
#include "cuda_runtime.h"
#include "obstacle_detection/cudaCluster.h"

class ObstacleDetectionCudaNode : public rclcpp::Node
{
public:
  ObstacleDetectionCudaNode()
  : Node("obstacle_detection_cuda_node")
  {
    // Declare and get parameters
    this->declare_parameter<std::string>("input_topic", "/filtered_fov_points");
    this->declare_parameter<std::string>("cluster_topic", "/detected_obstacles");
    this->declare_parameter<std::string>("marker_topic", "/obstacle_markers");
    this->declare_parameter<std::string>("target_frame", "base_link");
    this->declare_parameter<double>("cluster_tolerance", 0.5);
    this->declare_parameter<int>("min_cluster_size", 20);
    this->declare_parameter<int>("max_cluster_size", 100000);
    this->declare_parameter<double>("voxel_leaf_size_x", 0.7);
    this->declare_parameter<double>("voxel_leaf_size_y", 0.7);
    this->declare_parameter<double>("voxel_leaf_size_z", 0.7);
    this->declare_parameter<int>("count_threshold", 10);
    //not sure if this is actually makes sense, false on default:
    this->declare_parameter<bool>("use_cpu_pre_downsampling", false); 

    input_topic_ = this->get_parameter("input_topic").as_string();
    cluster_topic_ = this->get_parameter("cluster_topic").as_string();
    marker_topic_ = this->get_parameter("marker_topic").as_string();
    target_frame_ = this->get_parameter("target_frame").as_string();
    cluster_tolerance_ = this->get_parameter("cluster_tolerance").as_double();
    min_cluster_size_ = this->get_parameter("min_cluster_size").as_int();
    max_cluster_size_ = this->get_parameter("max_cluster_size").as_int();
    voxel_leaf_size_x_ = this->get_parameter("voxel_leaf_size_x").as_double();
    voxel_leaf_size_y_ = this->get_parameter("voxel_leaf_size_y").as_double();
    voxel_leaf_size_z_ = this->get_parameter("voxel_leaf_size_z").as_double();
    count_threshold_ = this->get_parameter("count_threshold").as_int();
    use_cpu_pre_downsampling_ = this->get_parameter("use_cpu_pre_downsampling").as_bool();

    // Initialize TF buffer and listener
    tf_buffer_ = std::make_unique<tf2_ros::Buffer>(this->get_clock());
    tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

    // Subscriber and Publishers
    sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
      input_topic_, 10,
      std::bind(&ObstacleDetectionCudaNode::pointCloudCallback, this, std::placeholders::_1));
    cluster_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(cluster_topic_, 10);
    marker_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(marker_topic_, 10);
  }

private:
  void pointCloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr cloud_msg)
  {
    try {
      // Transform point cloud to target frame
      sensor_msgs::msg::PointCloud2 cloud_transformed;
      geometry_msgs::msg::TransformStamped transform_stamped;
      
      transform_stamped = tf_buffer_->lookupTransform(
        target_frame_, cloud_msg->header.frame_id,
        tf2::TimePointZero);
      
      tf2::doTransform(*cloud_msg, cloud_transformed, transform_stamped);

      // Convert transformed ROS2 message to PCL PointCloud
      pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());
      pcl::fromROSMsg(cloud_transformed, *cloud);

      // Optional downsampling, check if necessary anyways. 
      pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZ>());
      if (use_cpu_pre_downsampling_)
      {
        pcl::VoxelGrid<pcl::PointXYZ> vg;
        vg.setInputCloud(cloud);
        vg.setLeafSize(voxel_leaf_size_x_, voxel_leaf_size_y_, voxel_leaf_size_z_);
        vg.filter(*cloud_filtered);
      }
      else
      {
        cloud_filtered = cloud;
      }

      if (cloud_filtered->empty()) {
        RCLCPP_WARN(this->get_logger(), "Filtered cloud is empty. Skipping clustering.");
        return;
      }

      // ----------------------
      // CUDA-based clustering
      // ----------------------
      cudaStream_t stream = NULL;
      cudaStreamCreate(&stream);

      unsigned int sizeEC = static_cast<unsigned int>(cloud_filtered->size());
      float *inputEC = NULL;
      float *outputEC = NULL;
      unsigned int *indexEC = NULL;

      // Allocate unified memory
      cudaMallocManaged(&inputEC, sizeof(float) * 4 * sizeEC, cudaMemAttachHost);
      cudaStreamAttachMemAsync(stream, inputEC);

      cudaMallocManaged(&outputEC, sizeof(float) * 4 * sizeEC, cudaMemAttachHost);
      cudaStreamAttachMemAsync(stream, outputEC);

      cudaMallocManaged(&indexEC, sizeof(unsigned int) * (sizeEC+1), cudaMemAttachHost);
      cudaStreamAttachMemAsync(stream, indexEC);

      // Copy point data to inputEC
      // Each point: x,y,z data in PointXYZ is float[4] with padding
      for (unsigned int i = 0; i < sizeEC; ++i) {
        inputEC[i*4+0] = cloud_filtered->points[i].x;
        inputEC[i*4+1] = cloud_filtered->points[i].y;
        inputEC[i*4+2] = cloud_filtered->points[i].z;
        inputEC[i*4+3] = 0.0f; // padding
      }

      cudaMemcpyAsync(outputEC, inputEC, sizeof(float)*4*sizeEC, cudaMemcpyHostToDevice, stream);
      cudaMemsetAsync(indexEC, 0, sizeof(unsigned int)*(sizeEC+1), stream);
      cudaStreamSynchronize(stream);

      // Set clustering parameters
      extractClusterParam_t ecp;
      ecp.minClusterSize = static_cast<unsigned int>(min_cluster_size_);
      ecp.maxClusterSize = static_cast<unsigned int>(max_cluster_size_);
      ecp.voxelX = static_cast<float>(voxel_leaf_size_x_);
      ecp.voxelY = static_cast<float>(voxel_leaf_size_y_);
      ecp.voxelZ = static_cast<float>(voxel_leaf_size_z_);
      ecp.countThreshold = count_threshold_;


      cudaExtractCluster cudaec(stream);
      cudaec.set(ecp);

      // Perform CUDA extraction
      cudaec.extract(inputEC, sizeEC, outputEC, indexEC);
      cudaStreamSynchronize(stream);

      // indexEC[0] = number of clusters
      unsigned int num_clusters = indexEC[0];
      //RCLCPP_INFO(this->get_logger(), "Found %u clusters using CUDA clustering.", num_clusters);

      // Construct a combined colored cluster cloud
      pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_clusters(new pcl::PointCloud<pcl::PointXYZRGB>());

      visualization_msgs::msg::MarkerArray marker_array;
      int cluster_id = 0;
      unsigned int offset = 0;

      for (unsigned int c = 1; c <= num_clusters; ++c) {
        unsigned int cluster_size = indexEC[c];
        if (cluster_size == 0) {
          RCLCPP_WARN(this->get_logger(), "Empty cluster encountered.");
          continue;
        }

        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cluster_cloud(new pcl::PointCloud<pcl::PointXYZRGB>());
        cluster_cloud->resize(cluster_size);

        // Generate a random but consistent color
        std::mt19937 gen(cluster_id);
        std::uniform_int_distribution<> dis(0, 255);
        uint8_t r = static_cast<uint8_t>(dis(gen));
        uint8_t g = static_cast<uint8_t>(dis(gen));
        uint8_t b = static_cast<uint8_t>(dis(gen));

        for (unsigned int k = 0; k < cluster_size; ++k) {
          pcl::PointXYZRGB p;
          p.x = outputEC[(offset+k)*4 + 0];
          p.y = outputEC[(offset+k)*4 + 1];
          p.z = outputEC[(offset+k)*4 + 2];
          p.r = r; p.g = g; p.b = b;
          cluster_cloud->points[k] = p;
        }
        offset += cluster_size;

        *cloud_clusters += *cluster_cloud;

        // Compute bounding box
        pcl::PointXYZRGB min_pt, max_pt;
        pcl::getMinMax3D(*cluster_cloud, min_pt, max_pt);

        visualization_msgs::msg::Marker marker;
        marker.header.frame_id = target_frame_;
        marker.header.stamp = cloud_msg->header.stamp;
        marker.ns = "obstacles";
        marker.id = cluster_id;
        marker.type = visualization_msgs::msg::Marker::CUBE;
        marker.action = visualization_msgs::msg::Marker::ADD;
        marker.pose.position.x = (min_pt.x + max_pt.x) / 2.0;
        marker.pose.position.y = (min_pt.y + max_pt.y) / 2.0;
        marker.pose.position.z = (min_pt.z + max_pt.z) / 2.0;
        marker.pose.orientation.x = 0.0;  
        marker.pose.orientation.y = 0.0;
        marker.pose.orientation.z = 0.0;
        marker.pose.orientation.w = 1.0;
        marker.scale.x = std::max((max_pt.x - min_pt.x), 0.1f);
        marker.scale.y = std::max((max_pt.y - min_pt.y), 0.1f);
        marker.scale.z = std::max((max_pt.z - min_pt.z), 0.1f);
        marker.color.r = static_cast<float>(r) / 255.0f;
        marker.color.g = static_cast<float>(g) / 255.0f;
        marker.color.b = static_cast<float>(b) / 255.0f;
        marker.color.a = 0.5f;
        marker.lifetime = rclcpp::Duration::from_seconds(0.1);

        marker_array.markers.push_back(marker);
        cluster_id++;
      }

      // Publish clusters
      sensor_msgs::msg::PointCloud2 output_clusters;
      pcl::toROSMsg(*cloud_clusters, output_clusters);
      output_clusters.header.frame_id = target_frame_;
      output_clusters.header.stamp = cloud_msg->header.stamp;
      cluster_pub_->publish(output_clusters);

      // Publish markers
      marker_pub_->publish(marker_array);

      // Cleanup CUDA memory
      cudaFree(inputEC);
      cudaFree(outputEC);
      cudaFree(indexEC);
      cudaStreamDestroy(stream);

    } catch (tf2::TransformException &ex) {
      RCLCPP_WARN(this->get_logger(), "Could not transform point cloud: %s", ex.what());
    }
  }

  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr cluster_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_pub_;

  std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;

  std::string input_topic_;
  std::string cluster_topic_;
  std::string marker_topic_;
  std::string target_frame_;
  double cluster_tolerance_;
  int min_cluster_size_;
  int max_cluster_size_;
  double voxel_leaf_size_x_;
  double voxel_leaf_size_y_;
  double voxel_leaf_size_z_;
  int count_threshold_;
  bool use_cpu_pre_downsampling_;
};

int main(int argc, char *argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<ObstacleDetectionCudaNode>());
  rclcpp::shutdown();
  return 0;
}
