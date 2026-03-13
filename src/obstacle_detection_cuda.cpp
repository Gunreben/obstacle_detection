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

#include "cuda_runtime.h"
#include "obstacle_detection/cudaCluster.h"

// ---------------------------------------------------------------------------
// CUDA error checking macro
// ---------------------------------------------------------------------------
#define CUDA_CHECK(call)                                                         \
  do {                                                                           \
    cudaError_t _err = (call);                                                   \
    if (_err != cudaSuccess) {                                                   \
      RCLCPP_ERROR(this->get_logger(),                                           \
        "CUDA error at %s:%d — %s", __FILE__, __LINE__,                          \
        cudaGetErrorString(_err));                                                \
      return;                                                                    \
    }                                                                            \
  } while (0)

class ObstacleDetectionCudaNode : public rclcpp::Node
{
public:
  ObstacleDetectionCudaNode()
  : Node("obstacle_detection_cuda_node")
  {
    // ── Parameters ────────────────────────────────────────────────────────────
    this->declare_parameter<std::string>("input_topic",    "/filtered_fov_points");
    this->declare_parameter<std::string>("cluster_topic",  "/detected_obstacles");
    this->declare_parameter<std::string>("marker_topic",   "/obstacle_markers");
    this->declare_parameter<std::string>("target_frame",   "base_link");
    this->declare_parameter<int>        ("min_cluster_size",  20);
    this->declare_parameter<int>        ("max_cluster_size",  100000);
    // Voxel sizes act as the spatial tolerance for the CUDA cluster.
    // Larger values → fewer, coarser clusters (faster); smaller → more precise.
    this->declare_parameter<double>("voxel_leaf_size_x",  0.5);
    this->declare_parameter<double>("voxel_leaf_size_y",  0.5);
    this->declare_parameter<double>("voxel_leaf_size_z",  0.5);
    // Minimum points in a voxel for it to be considered non-noise
    this->declare_parameter<int>   ("count_threshold",    2);
    // Optional CPU pre-downsampling before GPU cluster (reduces GPU transfer size)
    this->declare_parameter<bool>  ("use_cpu_pre_downsampling", false);

    input_topic_    = this->get_parameter("input_topic").as_string();
    cluster_topic_  = this->get_parameter("cluster_topic").as_string();
    marker_topic_   = this->get_parameter("marker_topic").as_string();
    target_frame_   = this->get_parameter("target_frame").as_string();
    min_cluster_size_ = this->get_parameter("min_cluster_size").as_int();
    max_cluster_size_ = this->get_parameter("max_cluster_size").as_int();
    voxel_leaf_size_x_ = this->get_parameter("voxel_leaf_size_x").as_double();
    voxel_leaf_size_y_ = this->get_parameter("voxel_leaf_size_y").as_double();
    voxel_leaf_size_z_ = this->get_parameter("voxel_leaf_size_z").as_double();
    count_threshold_   = this->get_parameter("count_threshold").as_int();
    use_cpu_pre_downsampling_ = this->get_parameter("use_cpu_pre_downsampling").as_bool();

    // ── CUDA initialisation ───────────────────────────────────────────────────
    cudaError_t err = cudaStreamCreate(&stream_);
    if (err != cudaSuccess) {
      RCLCPP_FATAL(this->get_logger(), "cudaStreamCreate failed: %s",
        cudaGetErrorString(err));
      throw std::runtime_error("CUDA stream creation failed");
    }

    // Pre-allocate buffers for a typical LiDAR scan (128k points).
    // Buffers grow automatically when larger clouds arrive.
    const unsigned int initial_capacity = 131072;
    allocateBuffers(initial_capacity);

    // ── TF ───────────────────────────────────────────────────────────────────
    tf_buffer_   = std::make_unique<tf2_ros::Buffer>(this->get_clock());
    tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

    // ── ROS I/O ──────────────────────────────────────────────────────────────
    sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
      input_topic_, 10,
      std::bind(&ObstacleDetectionCudaNode::pointCloudCallback, this,
                std::placeholders::_1));
    cluster_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
      cluster_topic_, 10);
    marker_pub_  = this->create_publisher<visualization_msgs::msg::MarkerArray>(
      marker_topic_, 10);

    RCLCPP_INFO(this->get_logger(),
      "ObstacleDetectionCudaNode started (voxel=%.2fx%.2fx%.2f, "
      "cluster_size=%d..%d, count_threshold=%d)",
      voxel_leaf_size_x_, voxel_leaf_size_y_, voxel_leaf_size_z_,
      min_cluster_size_, max_cluster_size_, count_threshold_);
  }

  ~ObstacleDetectionCudaNode()
  {
    freeBuffers();
    if (stream_) cudaStreamDestroy(stream_);
  }

private:
  // ── Buffer helpers ──────────────────────────────────────────────────────────

  void allocateBuffers(unsigned int capacity)
  {
    cudaMallocManaged(&input_buf_,  sizeof(float)        * 4 * capacity,
                      cudaMemAttachHost);
    cudaMallocManaged(&output_buf_, sizeof(float)        * 4 * capacity,
                      cudaMemAttachHost);
    cudaMallocManaged(&index_buf_,  sizeof(unsigned int) * (capacity + 1),
                      cudaMemAttachHost);
    buf_capacity_ = capacity;
  }

  void freeBuffers()
  {
    if (input_buf_)  { cudaFree(input_buf_);  input_buf_  = nullptr; }
    if (output_buf_) { cudaFree(output_buf_); output_buf_ = nullptr; }
    if (index_buf_)  { cudaFree(index_buf_);  index_buf_  = nullptr; }
    buf_capacity_ = 0;
  }

  // Grow buffers if the incoming cloud is larger than current capacity
  void ensureCapacity(unsigned int n)
  {
    if (n <= buf_capacity_) return;
    RCLCPP_INFO(this->get_logger(),
      "Growing CUDA buffers: %u → %u points", buf_capacity_, n);
    freeBuffers();
    allocateBuffers(n);
  }

  // ── Main callback ───────────────────────────────────────────────────────────

  void pointCloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr cloud_msg)
  {
    try {
      // Transform to target frame
      geometry_msgs::msg::TransformStamped tf_stamped =
        tf_buffer_->lookupTransform(
          target_frame_, cloud_msg->header.frame_id,
          tf2::TimePointZero);

      sensor_msgs::msg::PointCloud2 cloud_transformed;
      tf2::doTransform(*cloud_msg, cloud_transformed, tf_stamped);

      // Convert to PCL
      pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());
      pcl::fromROSMsg(cloud_transformed, *cloud);

      // Optional CPU voxel downsampling (reduces GPU transfer size)
      if (use_cpu_pre_downsampling_) {
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_ds(new pcl::PointCloud<pcl::PointXYZ>());
        pcl::VoxelGrid<pcl::PointXYZ> vg;
        vg.setInputCloud(cloud);
        vg.setLeafSize(
          static_cast<float>(voxel_leaf_size_x_),
          static_cast<float>(voxel_leaf_size_y_),
          static_cast<float>(voxel_leaf_size_z_));
        vg.filter(*cloud_ds);
        cloud = cloud_ds;
      }

      if (cloud->empty()) {
        RCLCPP_WARN(this->get_logger(), "Input cloud is empty after filtering.");
        return;
      }

      // ── CUDA clustering ───────────────────────────────────────────────────
      const unsigned int n = static_cast<unsigned int>(cloud->size());
      ensureCapacity(n);

      // Attach buffers to stream for coherent access
      CUDA_CHECK(cudaStreamAttachMemAsync(stream_, input_buf_));
      CUDA_CHECK(cudaStreamAttachMemAsync(stream_, output_buf_));
      CUDA_CHECK(cudaStreamAttachMemAsync(stream_, index_buf_));

      // Fill input buffer
      for (unsigned int i = 0; i < n; ++i) {
        input_buf_[i * 4 + 0] = cloud->points[i].x;
        input_buf_[i * 4 + 1] = cloud->points[i].y;
        input_buf_[i * 4 + 2] = cloud->points[i].z;
        input_buf_[i * 4 + 3] = 0.0f;
      }

      CUDA_CHECK(cudaMemcpyAsync(output_buf_, input_buf_,
        sizeof(float) * 4 * n, cudaMemcpyHostToDevice, stream_));
      CUDA_CHECK(cudaMemsetAsync(index_buf_, 0,
        sizeof(unsigned int) * (n + 1), stream_));
      CUDA_CHECK(cudaStreamSynchronize(stream_));

      // Set clustering parameters
      extractClusterParam_t ecp;
      ecp.minClusterSize = static_cast<unsigned int>(min_cluster_size_);
      ecp.maxClusterSize = static_cast<unsigned int>(max_cluster_size_);
      ecp.voxelX         = static_cast<float>(voxel_leaf_size_x_);
      ecp.voxelY         = static_cast<float>(voxel_leaf_size_y_);
      ecp.voxelZ         = static_cast<float>(voxel_leaf_size_z_);
      ecp.countThreshold = count_threshold_;

      cudaExtractCluster cuda_ec(stream_);
      cuda_ec.set(ecp);
      cuda_ec.extract(input_buf_, static_cast<int>(n), output_buf_, index_buf_);
      CUDA_CHECK(cudaStreamSynchronize(stream_));

      const unsigned int num_clusters = index_buf_[0];

      // ── Build colored cluster cloud + bounding-box markers ────────────────
      pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_clusters(
        new pcl::PointCloud<pcl::PointXYZRGB>());
      visualization_msgs::msg::MarkerArray marker_array;

      unsigned int offset = 0;
      for (unsigned int c = 1; c <= num_clusters; ++c) {
        const unsigned int cluster_size = index_buf_[c];
        if (cluster_size == 0) continue;

        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cluster_cloud(
          new pcl::PointCloud<pcl::PointXYZRGB>());
        cluster_cloud->resize(cluster_size);

        // Deterministic colour per cluster id
        const int cluster_id = static_cast<int>(c - 1);
        std::mt19937 gen(cluster_id);
        std::uniform_int_distribution<> dis(0, 255);
        const uint8_t r = static_cast<uint8_t>(dis(gen));
        const uint8_t g = static_cast<uint8_t>(dis(gen));
        const uint8_t b = static_cast<uint8_t>(dis(gen));

        for (unsigned int k = 0; k < cluster_size; ++k) {
          pcl::PointXYZRGB& p = cluster_cloud->points[k];
          p.x = output_buf_[(offset + k) * 4 + 0];
          p.y = output_buf_[(offset + k) * 4 + 1];
          p.z = output_buf_[(offset + k) * 4 + 2];
          p.r = r; p.g = g; p.b = b;
        }
        offset += cluster_size;
        *cloud_clusters += *cluster_cloud;

        // Bounding box marker
        pcl::PointXYZRGB min_pt, max_pt;
        pcl::getMinMax3D(*cluster_cloud, min_pt, max_pt);

        visualization_msgs::msg::Marker marker;
        marker.header.frame_id = target_frame_;
        marker.header.stamp    = cloud_msg->header.stamp;
        marker.ns              = "obstacles_cuda";
        marker.id              = cluster_id;
        marker.type            = visualization_msgs::msg::Marker::CUBE;
        marker.action          = visualization_msgs::msg::Marker::ADD;
        marker.pose.position.x = static_cast<double>(min_pt.x + max_pt.x) / 2.0;
        marker.pose.position.y = static_cast<double>(min_pt.y + max_pt.y) / 2.0;
        marker.pose.position.z = static_cast<double>(min_pt.z + max_pt.z) / 2.0;
        marker.pose.orientation.w = 1.0;
        marker.scale.x = std::max(max_pt.x - min_pt.x, 0.1f);
        marker.scale.y = std::max(max_pt.y - min_pt.y, 0.1f);
        marker.scale.z = std::max(max_pt.z - min_pt.z, 0.1f);
        marker.color.r = static_cast<float>(r) / 255.0f;
        marker.color.g = static_cast<float>(g) / 255.0f;
        marker.color.b = static_cast<float>(b) / 255.0f;
        marker.color.a = 0.5f;
        marker.lifetime = rclcpp::Duration::from_seconds(0.1);

        marker_array.markers.push_back(marker);
      }

      // ── Publish ───────────────────────────────────────────────────────────
      sensor_msgs::msg::PointCloud2 out_cloud;
      pcl::toROSMsg(*cloud_clusters, out_cloud);
      out_cloud.header.frame_id = target_frame_;
      out_cloud.header.stamp    = cloud_msg->header.stamp;
      cluster_pub_->publish(out_cloud);
      marker_pub_->publish(marker_array);

    } catch (const tf2::TransformException& ex) {
      RCLCPP_WARN(this->get_logger(), "TF transform failed: %s", ex.what());
    }
  }

  // ── ROS members ─────────────────────────────────────────────────────────────
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr    cluster_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_pub_;
  std::unique_ptr<tf2_ros::Buffer>              tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener>   tf_listener_;

  // ── Parameters ─────────────────────────────────────────────────────────────
  std::string input_topic_, cluster_topic_, marker_topic_, target_frame_;
  int    min_cluster_size_, max_cluster_size_, count_threshold_;
  double voxel_leaf_size_x_, voxel_leaf_size_y_, voxel_leaf_size_z_;
  bool   use_cpu_pre_downsampling_;

  // ── CUDA persistent resources ───────────────────────────────────────────────
  cudaStream_t  stream_       {nullptr};
  float*        input_buf_    {nullptr};
  float*        output_buf_   {nullptr};
  unsigned int* index_buf_    {nullptr};
  unsigned int  buf_capacity_ {0};
};

int main(int argc, char* argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<ObstacleDetectionCudaNode>());
  rclcpp::shutdown();
  return 0;
}
