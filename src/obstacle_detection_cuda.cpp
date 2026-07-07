#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

#include <pcl/common/common.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl_conversions/pcl_conversions.h>

#include <algorithm>
#include <cmath>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <pcl_ros/transforms.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/create_timer_ros.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_sensor_msgs/tf2_sensor_msgs.hpp>

#include "cuda_runtime.h"
#include "obstacle_detection/cluster_tracker.hpp"
#include "obstacle_detection/cudaCluster.h"
#include "obstacle_detection/msg/tracked_object.hpp"
#include "obstacle_detection/msg/tracked_object_array.hpp"

class ObstacleDetectionCudaNode : public rclcpp::Node
{
public:
  ObstacleDetectionCudaNode()
  : Node("obstacle_detection_cuda_node")
  {
    declareParameters();
    getParameters();

    const auto error = cudaStreamCreate(&stream_);
    if (error != cudaSuccess) {
      RCLCPP_FATAL(this->get_logger(), "cudaStreamCreate failed: %s", cudaGetErrorString(error));
      throw std::runtime_error("CUDA stream creation failed");
    }

    allocateBuffers(131072U);

    tf_buffer_ = std::make_unique<tf2_ros::Buffer>(this->get_clock());
    tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

    sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
      input_topic_,
      rclcpp::SensorDataQoS(),
      std::bind(&ObstacleDetectionCudaNode::pointCloudCallback, this, std::placeholders::_1));
    cluster_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(cluster_topic_, 10);
    marker_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(marker_topic_, 10);
    objects_pub_ = this->create_publisher<obstacle_detection::msg::TrackedObjectArray>(
      detections_topic_, 10);
  }

  ~ObstacleDetectionCudaNode()
  {
    freeBuffers();
    if (stream_ != nullptr) {
      cudaStreamDestroy(stream_);
    }
  }

private:
  struct ClusterParameters
  {
    double voxel_x {0.0};
    double voxel_y {0.0};
    double voxel_z {0.0};
    int count_threshold {0};
    int min_cluster_size {0};
    int max_cluster_size {0};
  };

  struct ClusterCandidate
  {
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud;
    obstacle_detection::DetectionGeometry geometry;
  };

  void declareParameters()
  {
    this->declare_parameter<std::string>("input_topic", "/filtered_fov_points");
    this->declare_parameter<std::string>("cluster_topic", "/detected_obstacles");
    this->declare_parameter<std::string>("marker_topic", "/obstacle_markers");
    this->declare_parameter<std::string>("detections_topic", "/lidar/objects");
    this->declare_parameter<bool>("publish_markers", true);
    this->declare_parameter<bool>("publish_cluster_cloud", true);
    this->declare_parameter<std::string>("target_frame", "base_link");
    this->declare_parameter<int>("min_cluster_size", 20);
    this->declare_parameter<int>("max_cluster_size", 100000);
    this->declare_parameter<double>("voxel_leaf_size_x", 0.5);
    this->declare_parameter<double>("voxel_leaf_size_y", 0.5);
    this->declare_parameter<double>("voxel_leaf_size_z", 0.5);
    this->declare_parameter<int>("count_threshold", 2);
    this->declare_parameter<bool>("use_cpu_pre_downsampling", false);
    this->declare_parameter<bool>("use_adaptive_clustering", false);
    this->declare_parameter<double>("near_range_max", 15.0);
    this->declare_parameter<double>("mid_range_max", 30.0);
    this->declare_parameter<double>("near_voxel_leaf_size_x", 0.35);
    this->declare_parameter<double>("near_voxel_leaf_size_y", 0.35);
    this->declare_parameter<double>("near_voxel_leaf_size_z", 0.35);
    this->declare_parameter<double>("mid_voxel_leaf_size_x", 0.5);
    this->declare_parameter<double>("mid_voxel_leaf_size_y", 0.5);
    this->declare_parameter<double>("mid_voxel_leaf_size_z", 0.5);
    this->declare_parameter<double>("far_voxel_leaf_size_x", 0.8);
    this->declare_parameter<double>("far_voxel_leaf_size_y", 0.8);
    this->declare_parameter<double>("far_voxel_leaf_size_z", 0.8);
    this->declare_parameter<int>("near_count_threshold", 2);
    this->declare_parameter<int>("mid_count_threshold", 1);
    this->declare_parameter<int>("far_count_threshold", 1);
    this->declare_parameter<int>("near_min_cluster_size", 20);
    this->declare_parameter<int>("mid_min_cluster_size", 12);
    this->declare_parameter<int>("far_min_cluster_size", 6);
    this->declare_parameter<bool>("enable_tracking", true);
    this->declare_parameter<double>("tracking_max_association_distance", 1.5);
    this->declare_parameter<double>("tracking_range_gate_scale", 0.03);
    this->declare_parameter<int>("tracking_max_missed_frames", 3);
    this->declare_parameter<double>("tracking_smoothing_alpha", 0.6);
    this->declare_parameter<double>("tf_lookup_timeout_sec", 0.05);
  }

  void getParameters()
  {
    input_topic_ = this->get_parameter("input_topic").as_string();
    cluster_topic_ = this->get_parameter("cluster_topic").as_string();
    marker_topic_ = this->get_parameter("marker_topic").as_string();
    detections_topic_ = this->get_parameter("detections_topic").as_string();
    publish_markers_ = this->get_parameter("publish_markers").as_bool();
    publish_cluster_cloud_ = this->get_parameter("publish_cluster_cloud").as_bool();
    target_frame_ = this->get_parameter("target_frame").as_string();
    min_cluster_size_ = this->get_parameter("min_cluster_size").as_int();
    max_cluster_size_ = this->get_parameter("max_cluster_size").as_int();
    voxel_leaf_size_x_ = this->get_parameter("voxel_leaf_size_x").as_double();
    voxel_leaf_size_y_ = this->get_parameter("voxel_leaf_size_y").as_double();
    voxel_leaf_size_z_ = this->get_parameter("voxel_leaf_size_z").as_double();
    count_threshold_ = this->get_parameter("count_threshold").as_int();
    use_cpu_pre_downsampling_ = this->get_parameter("use_cpu_pre_downsampling").as_bool();
    use_adaptive_clustering_ = this->get_parameter("use_adaptive_clustering").as_bool();
    near_range_max_ = this->get_parameter("near_range_max").as_double();
    mid_range_max_ = this->get_parameter("mid_range_max").as_double();
    near_voxel_leaf_size_x_ = this->get_parameter("near_voxel_leaf_size_x").as_double();
    near_voxel_leaf_size_y_ = this->get_parameter("near_voxel_leaf_size_y").as_double();
    near_voxel_leaf_size_z_ = this->get_parameter("near_voxel_leaf_size_z").as_double();
    mid_voxel_leaf_size_x_ = this->get_parameter("mid_voxel_leaf_size_x").as_double();
    mid_voxel_leaf_size_y_ = this->get_parameter("mid_voxel_leaf_size_y").as_double();
    mid_voxel_leaf_size_z_ = this->get_parameter("mid_voxel_leaf_size_z").as_double();
    far_voxel_leaf_size_x_ = this->get_parameter("far_voxel_leaf_size_x").as_double();
    far_voxel_leaf_size_y_ = this->get_parameter("far_voxel_leaf_size_y").as_double();
    far_voxel_leaf_size_z_ = this->get_parameter("far_voxel_leaf_size_z").as_double();
    near_count_threshold_ = this->get_parameter("near_count_threshold").as_int();
    mid_count_threshold_ = this->get_parameter("mid_count_threshold").as_int();
    far_count_threshold_ = this->get_parameter("far_count_threshold").as_int();
    near_min_cluster_size_ = this->get_parameter("near_min_cluster_size").as_int();
    mid_min_cluster_size_ = this->get_parameter("mid_min_cluster_size").as_int();
    far_min_cluster_size_ = this->get_parameter("far_min_cluster_size").as_int();
    enable_tracking_ = this->get_parameter("enable_tracking").as_bool();
    tracking_max_association_distance_ =
      this->get_parameter("tracking_max_association_distance").as_double();
    tracking_range_gate_scale_ = this->get_parameter("tracking_range_gate_scale").as_double();
    tracking_max_missed_frames_ = this->get_parameter("tracking_max_missed_frames").as_int();
    tracking_smoothing_alpha_ = this->get_parameter("tracking_smoothing_alpha").as_double();
    tf_lookup_timeout_sec_ = this->get_parameter("tf_lookup_timeout_sec").as_double();
  }

  void pointCloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr cloud_msg)
  {
    try {
      const auto tf_stamped = tf_buffer_->lookupTransform(
        target_frame_,
        cloud_msg->header.frame_id,
        rclcpp::Time(cloud_msg->header.stamp),
        rclcpp::Duration::from_seconds(tf_lookup_timeout_sec_));

      sensor_msgs::msg::PointCloud2 cloud_transformed;
      tf2::doTransform(*cloud_msg, cloud_transformed, tf_stamped);

      pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());
      pcl::fromROSMsg(cloud_transformed, *cloud);

      if (use_cpu_pre_downsampling_) {
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_ds(new pcl::PointCloud<pcl::PointXYZ>());
        pcl::VoxelGrid<pcl::PointXYZ> voxel_grid;
        voxel_grid.setInputCloud(cloud);
        voxel_grid.setLeafSize(
          static_cast<float>(voxel_leaf_size_x_),
          static_cast<float>(voxel_leaf_size_y_),
          static_cast<float>(voxel_leaf_size_z_));
        voxel_grid.filter(*cloud_ds);
        cloud = cloud_ds;
      }

      if (!cloud || cloud->empty()) {
        RCLCPP_WARN(this->get_logger(), "Input cloud is empty after filtering.");
        return;
      }

      const auto candidates = extractClusterCandidates(cloud);
      const auto tracked_detections = assignTrackIds(candidates);

      publishObjects(tracked_detections, cloud_msg->header.stamp);

      if (publish_markers_) {
        visualization_msgs::msg::MarkerArray marker_array;
        std::set<int> active_marker_ids;
        for (const auto & tracked_detection : tracked_detections) {
          const auto [r, g, b] = colorForTrack(tracked_detection.track_id);
          active_marker_ids.insert(tracked_detection.track_id);
          marker_array.markers.push_back(
            makeMarker(
              tracked_detection.track_id,
              tracked_detection.geometry,
              r,
              g,
              b,
              cloud_msg->header.stamp));
        }
        appendDeleteMarkers(marker_array, active_marker_ids, cloud_msg->header.stamp);
        last_published_marker_ids_ = active_marker_ids;
        marker_pub_->publish(marker_array);
      }

      if (publish_cluster_cloud_) {
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_clusters(new pcl::PointCloud<pcl::PointXYZRGB>());
        for (const auto & tracked_detection : tracked_detections) {
          const auto & candidate = candidates[tracked_detection.detection_index];
          const auto [r, g, b] = colorForTrack(tracked_detection.track_id);
          for (auto & point_rgb : candidate.cloud->points) {
            point_rgb.r = r;
            point_rgb.g = g;
            point_rgb.b = b;
          }
          *cloud_clusters += *candidate.cloud;
        }
        sensor_msgs::msg::PointCloud2 out_cloud;
        pcl::toROSMsg(*cloud_clusters, out_cloud);
        out_cloud.header.frame_id = target_frame_;
        out_cloud.header.stamp = cloud_msg->header.stamp;
        cluster_pub_->publish(out_cloud);
      }
    } catch (const tf2::TransformException & ex) {
      RCLCPP_WARN(this->get_logger(), "Could not transform point cloud: %s", ex.what());
    } catch (const std::exception & ex) {
      RCLCPP_ERROR(this->get_logger(), "CUDA clustering failed: %s", ex.what());
    }
  }

  std::vector<ClusterCandidate> extractClusterCandidates(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr & cloud)
  {
    if (!use_adaptive_clustering_) {
      return runCudaClustering(
        cloud,
        {voxel_leaf_size_x_, voxel_leaf_size_y_, voxel_leaf_size_z_,
          count_threshold_, min_cluster_size_, max_cluster_size_});
    }

    pcl::PointCloud<pcl::PointXYZ>::Ptr near_cloud(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::PointCloud<pcl::PointXYZ>::Ptr mid_cloud(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::PointCloud<pcl::PointXYZ>::Ptr far_cloud(new pcl::PointCloud<pcl::PointXYZ>());

    for (const auto & point : cloud->points) {
      const auto range = computeRange(point);
      if (range <= near_range_max_) {
        near_cloud->points.push_back(point);
      } else if (range <= mid_range_max_) {
        mid_cloud->points.push_back(point);
      } else {
        far_cloud->points.push_back(point);
      }
    }

    setCloudMetadata(near_cloud, cloud->is_dense);
    setCloudMetadata(mid_cloud, cloud->is_dense);
    setCloudMetadata(far_cloud, cloud->is_dense);

    std::vector<ClusterCandidate> candidates;
    appendClusters(
      candidates,
      runCudaClustering(
        near_cloud,
        {near_voxel_leaf_size_x_, near_voxel_leaf_size_y_, near_voxel_leaf_size_z_,
          near_count_threshold_, near_min_cluster_size_, max_cluster_size_}));
    appendClusters(
      candidates,
      runCudaClustering(
        mid_cloud,
        {mid_voxel_leaf_size_x_, mid_voxel_leaf_size_y_, mid_voxel_leaf_size_z_,
          mid_count_threshold_, mid_min_cluster_size_, max_cluster_size_}));
    appendClusters(
      candidates,
      runCudaClustering(
        far_cloud,
        {far_voxel_leaf_size_x_, far_voxel_leaf_size_y_, far_voxel_leaf_size_z_,
          far_count_threshold_, far_min_cluster_size_, max_cluster_size_}));

    std::sort(
      candidates.begin(),
      candidates.end(),
      [](const ClusterCandidate & lhs, const ClusterCandidate & rhs) {
        return std::tie(lhs.geometry.center[0], lhs.geometry.center[1], lhs.geometry.center[2]) <
               std::tie(rhs.geometry.center[0], rhs.geometry.center[1], rhs.geometry.center[2]);
      });

    return candidates;
  }

  std::vector<ClusterCandidate> runCudaClustering(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr & cloud,
    const ClusterParameters & parameters)
  {
    std::vector<ClusterCandidate> candidates;
    if (!cloud || cloud->empty()) {
      return candidates;
    }

    const auto point_count = static_cast<unsigned int>(cloud->size());
    ensureCapacity(point_count);

    throwIfCudaError(cudaStreamAttachMemAsync(stream_, input_buf_), "attach input buffer");
    throwIfCudaError(cudaStreamAttachMemAsync(stream_, output_buf_), "attach output buffer");
    throwIfCudaError(cudaStreamAttachMemAsync(stream_, index_buf_), "attach index buffer");

    for (unsigned int i = 0; i < point_count; ++i) {
      input_buf_[i * 4 + 0] = cloud->points[i].x;
      input_buf_[i * 4 + 1] = cloud->points[i].y;
      input_buf_[i * 4 + 2] = cloud->points[i].z;
      input_buf_[i * 4 + 3] = 0.0F;
    }

    throwIfCudaError(
      cudaMemsetAsync(index_buf_, 0, sizeof(unsigned int) * (point_count + 1), stream_),
      "reset index buffer");

    extractClusterParam_t ecp;
    ecp.minClusterSize = static_cast<unsigned int>(parameters.min_cluster_size);
    ecp.maxClusterSize = static_cast<unsigned int>(parameters.max_cluster_size);
    ecp.voxelX = static_cast<float>(parameters.voxel_x);
    ecp.voxelY = static_cast<float>(parameters.voxel_y);
    ecp.voxelZ = static_cast<float>(parameters.voxel_z);
    ecp.countThreshold = parameters.count_threshold;

    cudaExtractCluster extractor(stream_);
    extractor.set(ecp);
    extractor.extract(input_buf_, static_cast<int>(point_count), output_buf_, index_buf_);
    throwIfCudaError(cudaStreamSynchronize(stream_), "synchronize clustering stream");

    const auto num_clusters = index_buf_[0];
    unsigned int offset = 0;
    for (unsigned int cluster_index = 1; cluster_index <= num_clusters; ++cluster_index) {
      const auto cluster_size = index_buf_[cluster_index];
      if (cluster_size == 0) {
        continue;
      }

      pcl::PointCloud<pcl::PointXYZRGB>::Ptr cluster_cloud(new pcl::PointCloud<pcl::PointXYZRGB>());
      cluster_cloud->resize(cluster_size);
      for (unsigned int point_index = 0; point_index < cluster_size; ++point_index) {
        auto & point = cluster_cloud->points[point_index];
        point.x = output_buf_[(offset + point_index) * 4 + 0];
        point.y = output_buf_[(offset + point_index) * 4 + 1];
        point.z = output_buf_[(offset + point_index) * 4 + 2];
      }
      offset += cluster_size;

      cluster_cloud->width = cluster_size;
      cluster_cloud->height = 1;
      cluster_cloud->is_dense = cloud->is_dense;

      pcl::PointXYZRGB min_point;
      pcl::PointXYZRGB max_point;
      pcl::getMinMax3D(*cluster_cloud, min_point, max_point);

      obstacle_detection::DetectionGeometry geometry;
      geometry.center = {
        (min_point.x + max_point.x) * 0.5F,
        (min_point.y + max_point.y) * 0.5F,
        (min_point.z + max_point.z) * 0.5F};
      geometry.size = {
        std::max(max_point.x - min_point.x, 0.1F),
        std::max(max_point.y - min_point.y, 0.1F),
        std::max(max_point.z - min_point.z, 0.1F)};
      geometry.range_xy = std::hypot(geometry.center[0], geometry.center[1]);
      candidates.push_back({cluster_cloud, geometry});
    }

    return candidates;
  }

  std::vector<obstacle_detection::TrackedDetection> assignTrackIds(
    const std::vector<ClusterCandidate> & candidates)
  {
    std::vector<obstacle_detection::DetectionGeometry> detections;
    detections.reserve(candidates.size());
    for (const auto & candidate : candidates) {
      detections.push_back(candidate.geometry);
    }

    if (enable_tracking_) {
      return tracker_.update(
        detections,
        tracking_max_association_distance_,
        tracking_range_gate_scale_,
        tracking_max_missed_frames_,
        tracking_smoothing_alpha_);
    }

    std::vector<obstacle_detection::TrackedDetection> tracked_detections;
    tracked_detections.reserve(detections.size());
    for (std::size_t i = 0; i < detections.size(); ++i) {
      tracked_detections.push_back({static_cast<int>(i), i, detections[i], true});
    }
    return tracked_detections;
  }

  void publishObjects(
    const std::vector<obstacle_detection::TrackedDetection> & tracked_detections,
    const builtin_interfaces::msg::Time & stamp)
  {
    obstacle_detection::msg::TrackedObjectArray objects;
    objects.header.stamp = stamp;
    objects.header.frame_id = target_frame_;
    objects.objects.reserve(tracked_detections.size());
    for (const auto & tracked_detection : tracked_detections) {
      obstacle_detection::msg::TrackedObject object;
      object.track_id = tracked_detection.track_id;
      object.class_name = "unknown";
      object.confidence = 0.0;
      object.position.x = tracked_detection.geometry.center[0];
      object.position.y = tracked_detection.geometry.center[1];
      object.position.z = tracked_detection.geometry.center[2];
      object.size.x = tracked_detection.geometry.size[0];
      object.size.y = tracked_detection.geometry.size[1];
      object.size.z = tracked_detection.geometry.size[2];
      object.source = "lidar";
      objects.objects.push_back(object);
    }
    objects_pub_->publish(objects);
  }

  visualization_msgs::msg::Marker makeMarker(
    int marker_id,
    const obstacle_detection::DetectionGeometry & geometry,
    const uint8_t r,
    const uint8_t g,
    const uint8_t b,
    const builtin_interfaces::msg::Time & stamp) const
  {
    visualization_msgs::msg::Marker marker;
    marker.header.frame_id = target_frame_;
    marker.header.stamp = stamp;
    marker.ns = "obstacles_cuda";
    marker.id = marker_id;
    marker.type = visualization_msgs::msg::Marker::CUBE;
    marker.action = visualization_msgs::msg::Marker::ADD;
    marker.pose.position.x = geometry.center[0];
    marker.pose.position.y = geometry.center[1];
    marker.pose.position.z = geometry.center[2];
    marker.pose.orientation.w = 1.0;
    marker.scale.x = std::max(geometry.size[0], 0.1F);
    marker.scale.y = std::max(geometry.size[1], 0.1F);
    marker.scale.z = std::max(geometry.size[2], 0.1F);
    marker.color.r = static_cast<float>(r) / 255.0F;
    marker.color.g = static_cast<float>(g) / 255.0F;
    marker.color.b = static_cast<float>(b) / 255.0F;
    marker.color.a = 0.5F;
    marker.lifetime = rclcpp::Duration::from_seconds(0.2);
    return marker;
  }

  void appendDeleteMarkers(
    visualization_msgs::msg::MarkerArray & marker_array,
    const std::set<int> & active_marker_ids,
    const builtin_interfaces::msg::Time & stamp) const
  {
    for (const auto marker_id : last_published_marker_ids_) {
      if (active_marker_ids.count(marker_id) > 0) {
        continue;
      }

      visualization_msgs::msg::Marker marker;
      marker.header.frame_id = target_frame_;
      marker.header.stamp = stamp;
      marker.ns = "obstacles_cuda";
      marker.id = marker_id;
      marker.action = visualization_msgs::msg::Marker::DELETE;
      marker_array.markers.push_back(marker);
    }
  }

  void allocateBuffers(unsigned int capacity)
  {
    throwIfCudaError(
      cudaMallocManaged(&input_buf_, sizeof(float) * 4 * capacity, cudaMemAttachHost),
      "allocate input buffer");
    throwIfCudaError(
      cudaMallocManaged(&output_buf_, sizeof(float) * 4 * capacity, cudaMemAttachHost),
      "allocate output buffer");
    throwIfCudaError(
      cudaMallocManaged(&index_buf_, sizeof(unsigned int) * (capacity + 1), cudaMemAttachHost),
      "allocate index buffer");
    buf_capacity_ = capacity;
  }

  void freeBuffers()
  {
    if (input_buf_ != nullptr) {
      cudaFree(input_buf_);
      input_buf_ = nullptr;
    }
    if (output_buf_ != nullptr) {
      cudaFree(output_buf_);
      output_buf_ = nullptr;
    }
    if (index_buf_ != nullptr) {
      cudaFree(index_buf_);
      index_buf_ = nullptr;
    }
    buf_capacity_ = 0;
  }

  void ensureCapacity(unsigned int point_count)
  {
    if (point_count <= buf_capacity_) {
      return;
    }

    RCLCPP_INFO(
      this->get_logger(),
      "Growing CUDA buffers: %u -> %u points",
      buf_capacity_,
      point_count);
    freeBuffers();
    allocateBuffers(point_count);
  }

  void throwIfCudaError(cudaError_t error, const char * context) const
  {
    if (error != cudaSuccess) {
      throw std::runtime_error(std::string(context) + ": " + cudaGetErrorString(error));
    }
  }

  static void setCloudMetadata(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr & cloud,
    bool is_dense)
  {
    cloud->width = cloud->points.size();
    cloud->height = 1;
    cloud->is_dense = is_dense;
  }

  static void appendClusters(
    std::vector<ClusterCandidate> & destination,
    std::vector<ClusterCandidate> && source)
  {
    destination.insert(
      destination.end(),
      std::make_move_iterator(source.begin()),
      std::make_move_iterator(source.end()));
  }

  static double computeRange(const pcl::PointXYZ & point)
  {
    return std::hypot(static_cast<double>(point.x), static_cast<double>(point.y));
  }

  static std::tuple<uint8_t, uint8_t, uint8_t> colorForTrack(int track_id)
  {
    std::mt19937 generator(static_cast<std::mt19937::result_type>(track_id * 2654435761U));
    std::uniform_int_distribution<> distribution(0, 255);
    return {
      static_cast<uint8_t>(distribution(generator)),
      static_cast<uint8_t>(distribution(generator)),
      static_cast<uint8_t>(distribution(generator))};
  }

  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr cluster_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_pub_;
  rclcpp::Publisher<obstacle_detection::msg::TrackedObjectArray>::SharedPtr objects_pub_;

  std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;

  obstacle_detection::ClusterTracker tracker_;
  std::set<int> last_published_marker_ids_;

  std::string input_topic_;
  std::string cluster_topic_;
  std::string marker_topic_;
  std::string detections_topic_;
  bool publish_markers_ {true};
  bool publish_cluster_cloud_ {true};
  std::string target_frame_;

  int min_cluster_size_ {20};
  int max_cluster_size_ {100000};
  int count_threshold_ {2};
  double voxel_leaf_size_x_ {0.5};
  double voxel_leaf_size_y_ {0.5};
  double voxel_leaf_size_z_ {0.5};
  bool use_cpu_pre_downsampling_ {false};

  bool use_adaptive_clustering_ {false};
  double near_range_max_ {15.0};
  double mid_range_max_ {30.0};
  double near_voxel_leaf_size_x_ {0.35};
  double near_voxel_leaf_size_y_ {0.35};
  double near_voxel_leaf_size_z_ {0.35};
  double mid_voxel_leaf_size_x_ {0.5};
  double mid_voxel_leaf_size_y_ {0.5};
  double mid_voxel_leaf_size_z_ {0.5};
  double far_voxel_leaf_size_x_ {0.8};
  double far_voxel_leaf_size_y_ {0.8};
  double far_voxel_leaf_size_z_ {0.8};
  int near_count_threshold_ {2};
  int mid_count_threshold_ {1};
  int far_count_threshold_ {1};
  int near_min_cluster_size_ {20};
  int mid_min_cluster_size_ {12};
  int far_min_cluster_size_ {6};

  bool enable_tracking_ {true};
  double tracking_max_association_distance_ {1.5};
  double tracking_range_gate_scale_ {0.03};
  int tracking_max_missed_frames_ {3};
  double tracking_smoothing_alpha_ {0.6};
  double tf_lookup_timeout_sec_ {0.05};

  cudaStream_t stream_ {nullptr};
  float * input_buf_ {nullptr};
  float * output_buf_ {nullptr};
  unsigned int * index_buf_ {nullptr};
  unsigned int buf_capacity_ {0};
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<ObstacleDetectionCudaNode>());
  rclcpp::shutdown();
  return 0;
}
