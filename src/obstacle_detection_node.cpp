#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

#include <pcl/common/common.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/search/kdtree.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl_conversions/pcl_conversions.h>

#include <algorithm>
#include <cmath>
#include <random>
#include <set>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <pcl_ros/transforms.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/create_timer_ros.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_sensor_msgs/tf2_sensor_msgs.hpp>

#include "obstacle_detection/cluster_tracker.hpp"

class ObstacleDetectionNode : public rclcpp::Node
{
public:
  ObstacleDetectionNode()
  : Node("obstacle_detection_node")
  {
    declareParameters();
    getParameters();

    tf_buffer_ = std::make_unique<tf2_ros::Buffer>(this->get_clock());
    tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

    sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
      input_topic_,
      rclcpp::SensorDataQoS(),
      std::bind(&ObstacleDetectionNode::pointCloudCallback, this, std::placeholders::_1));
    cluster_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(cluster_topic_, 10);
    marker_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(marker_topic_, 10);
  }

private:
  struct RangeBinParameters
  {
    double max_range {0.0};
    double cluster_tolerance {0.0};
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
    this->declare_parameter<std::string>("input_topic", "/ouster/points/filtered");
    this->declare_parameter<std::string>("cluster_topic", "/detected_obstacles");
    this->declare_parameter<std::string>("marker_topic", "/obstacle_markers");
    this->declare_parameter<std::string>("target_frame", "base_link");
    this->declare_parameter<double>("cluster_tolerance", 0.5);
    this->declare_parameter<int>("min_cluster_size", 50);
    this->declare_parameter<int>("max_cluster_size", 10000);
    this->declare_parameter<double>("voxel_leaf_size", 0.1);
    this->declare_parameter<bool>("use_downsampling", true);
    this->declare_parameter<bool>("use_adaptive_clustering", false);
    this->declare_parameter<double>("near_range_max", 15.0);
    this->declare_parameter<double>("mid_range_max", 30.0);
    this->declare_parameter<double>("near_cluster_tolerance", 0.45);
    this->declare_parameter<double>("mid_cluster_tolerance", 0.7);
    this->declare_parameter<double>("far_cluster_tolerance", 1.0);
    this->declare_parameter<int>("near_min_cluster_size", 50);
    this->declare_parameter<int>("mid_min_cluster_size", 25);
    this->declare_parameter<int>("far_min_cluster_size", 10);
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
    target_frame_ = this->get_parameter("target_frame").as_string();
    cluster_tolerance_ = this->get_parameter("cluster_tolerance").as_double();
    min_cluster_size_ = this->get_parameter("min_cluster_size").as_int();
    max_cluster_size_ = this->get_parameter("max_cluster_size").as_int();
    voxel_leaf_size_ = this->get_parameter("voxel_leaf_size").as_double();
    use_downsampling_ = this->get_parameter("use_downsampling").as_bool();
    use_adaptive_clustering_ = this->get_parameter("use_adaptive_clustering").as_bool();
    near_range_max_ = this->get_parameter("near_range_max").as_double();
    mid_range_max_ = this->get_parameter("mid_range_max").as_double();
    near_cluster_tolerance_ = this->get_parameter("near_cluster_tolerance").as_double();
    mid_cluster_tolerance_ = this->get_parameter("mid_cluster_tolerance").as_double();
    far_cluster_tolerance_ = this->get_parameter("far_cluster_tolerance").as_double();
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
      const auto transform_stamped = tf_buffer_->lookupTransform(
        target_frame_,
        cloud_msg->header.frame_id,
        rclcpp::Time(cloud_msg->header.stamp),
        rclcpp::Duration::from_seconds(tf_lookup_timeout_sec_));

      sensor_msgs::msg::PointCloud2 cloud_transformed;
      tf2::doTransform(*cloud_msg, cloud_transformed, transform_stamped);

      pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());
      pcl::fromROSMsg(cloud_transformed, *cloud);

      pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZ>());
      if (use_downsampling_) {
        pcl::VoxelGrid<pcl::PointXYZ> voxel_grid;
        voxel_grid.setInputCloud(cloud);
        voxel_grid.setLeafSize(voxel_leaf_size_, voxel_leaf_size_, voxel_leaf_size_);
        voxel_grid.filter(*cloud_filtered);
      } else {
        cloud_filtered = cloud;
      }

      const auto candidates = extractClusterCandidates(cloud_filtered);
      const auto tracked_detections = assignTrackIds(candidates);

      visualization_msgs::msg::MarkerArray marker_array;
      pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_clusters(new pcl::PointCloud<pcl::PointXYZRGB>());
      std::set<int> active_marker_ids;

      for (const auto & tracked_detection : tracked_detections) {
        const auto & candidate = candidates[tracked_detection.detection_index];
        const auto [r, g, b] = colorForTrack(tracked_detection.track_id);
        for (auto & point_rgb : candidate.cloud->points) {
          point_rgb.r = r;
          point_rgb.g = g;
          point_rgb.b = b;
        }
        *cloud_clusters += *candidate.cloud;
        active_marker_ids.insert(tracked_detection.track_id);
        marker_array.markers.push_back(
          makeMarker(tracked_detection.track_id, tracked_detection.geometry, r, g, b, cloud_msg->header.stamp));
      }

      appendDeleteMarkers(marker_array, active_marker_ids, cloud_msg->header.stamp);
      last_published_marker_ids_ = active_marker_ids;

      sensor_msgs::msg::PointCloud2 output_clusters;
      pcl::toROSMsg(*cloud_clusters, output_clusters);
      output_clusters.header.frame_id = target_frame_;
      output_clusters.header.stamp = cloud_msg->header.stamp;
      cluster_pub_->publish(output_clusters);
      marker_pub_->publish(marker_array);
    } catch (const tf2::TransformException & ex) {
      RCLCPP_WARN(this->get_logger(), "Could not transform point cloud: %s", ex.what());
    }
  }

  std::vector<ClusterCandidate> extractClusterCandidates(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr & cloud) const
  {
    if (!use_adaptive_clustering_) {
      return extractClusters(cloud, cluster_tolerance_, min_cluster_size_, max_cluster_size_);
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

    near_cloud->width = near_cloud->points.size();
    near_cloud->height = 1;
    near_cloud->is_dense = cloud->is_dense;
    mid_cloud->width = mid_cloud->points.size();
    mid_cloud->height = 1;
    mid_cloud->is_dense = cloud->is_dense;
    far_cloud->width = far_cloud->points.size();
    far_cloud->height = 1;
    far_cloud->is_dense = cloud->is_dense;

    std::vector<ClusterCandidate> candidates;
    appendClusters(
      candidates,
      extractClusters(near_cloud, near_cluster_tolerance_, near_min_cluster_size_, max_cluster_size_));
    appendClusters(
      candidates,
      extractClusters(mid_cloud, mid_cluster_tolerance_, mid_min_cluster_size_, max_cluster_size_));
    appendClusters(
      candidates,
      extractClusters(far_cloud, far_cluster_tolerance_, far_min_cluster_size_, max_cluster_size_));

    std::sort(
      candidates.begin(),
      candidates.end(),
      [](const ClusterCandidate & lhs, const ClusterCandidate & rhs) {
        return std::tie(lhs.geometry.center[0], lhs.geometry.center[1], lhs.geometry.center[2]) <
               std::tie(rhs.geometry.center[0], rhs.geometry.center[1], rhs.geometry.center[2]);
      });

    return candidates;
  }

  std::vector<ClusterCandidate> extractClusters(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr & cloud,
    double cluster_tolerance,
    int min_cluster_size,
    int max_cluster_size) const
  {
    std::vector<ClusterCandidate> candidates;
    if (!cloud || cloud->empty()) {
      return candidates;
    }

    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());
    tree->setInputCloud(cloud);

    std::vector<pcl::PointIndices> cluster_indices;
    pcl::EuclideanClusterExtraction<pcl::PointXYZ> extraction;
    extraction.setClusterTolerance(cluster_tolerance);
    extraction.setMinClusterSize(min_cluster_size);
    extraction.setMaxClusterSize(max_cluster_size);
    extraction.setSearchMethod(tree);
    extraction.setInputCloud(cloud);
    extraction.extract(cluster_indices);

    candidates.reserve(cluster_indices.size());
    for (const auto & indices : cluster_indices) {
      pcl::PointCloud<pcl::PointXYZRGB>::Ptr cluster_cloud(new pcl::PointCloud<pcl::PointXYZRGB>());
      cluster_cloud->reserve(indices.indices.size());
      for (const auto point_index : indices.indices) {
        const auto & point = cloud->points[point_index];
        pcl::PointXYZRGB point_rgb;
        point_rgb.x = point.x;
        point_rgb.y = point.y;
        point_rgb.z = point.z;
        cluster_cloud->points.push_back(point_rgb);
      }

      if (cluster_cloud->empty()) {
        continue;
      }

      cluster_cloud->width = cluster_cloud->points.size();
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
    marker.ns = "obstacles";
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
      marker.ns = "obstacles";
      marker.id = marker_id;
      marker.action = visualization_msgs::msg::Marker::DELETE;
      marker_array.markers.push_back(marker);
    }
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

  std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;

  obstacle_detection::ClusterTracker tracker_;
  std::set<int> last_published_marker_ids_;

  std::string input_topic_;
  std::string cluster_topic_;
  std::string marker_topic_;
  std::string target_frame_;

  double cluster_tolerance_ {0.5};
  int min_cluster_size_ {50};
  int max_cluster_size_ {10000};
  double voxel_leaf_size_ {0.1};
  bool use_downsampling_ {true};

  bool use_adaptive_clustering_ {false};
  double near_range_max_ {15.0};
  double mid_range_max_ {30.0};
  double near_cluster_tolerance_ {0.45};
  double mid_cluster_tolerance_ {0.7};
  double far_cluster_tolerance_ {1.0};
  int near_min_cluster_size_ {50};
  int mid_min_cluster_size_ {25};
  int far_min_cluster_size_ {10};

  bool enable_tracking_ {true};
  double tracking_max_association_distance_ {1.5};
  double tracking_range_gate_scale_ {0.03};
  int tracking_max_missed_frames_ {3};
  double tracking_smoothing_alpha_ {0.6};
  double tf_lookup_timeout_sec_ {0.05};
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<ObstacleDetectionNode>());
  rclcpp::shutdown();
  return 0;
}
