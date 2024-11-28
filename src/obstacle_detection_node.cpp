#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/search/kdtree.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/common/common.h> 
#include <random>
#include "pointcloud_processing_node.cpp"

class ObstacleDetectionNode : public PointCloudProcessingNode
{
public:
  ObstacleDetectionNode()
  : PointCloudProcessingNode("obstacle_detection_node")
  {

    // Declare and get parameters
    this->declare_parameter<std::string>("input_topic", "/filtered_fov_points");
    this->declare_parameter<std::string>("cluster_topic", "/detected_obstacles");
    this->declare_parameter<std::string>("marker_topic", "/obstacle_markers");
    this->declare_parameter<double>("cluster_tolerance", 0.5); // meters
    this->declare_parameter<int>("min_cluster_size", 50);
    this->declare_parameter<int>("max_cluster_size", 10000);
    this->declare_parameter<double>("voxel_leaf_size", 0.1); // meters
    this->declare_parameter<bool>("use_downsampling", true);

    input_topic_ = this->get_parameter("input_topic").as_string();
    cluster_topic_ = this->get_parameter("cluster_topic").as_string();
    marker_topic_ = this->get_parameter("marker_topic").as_string();
    cluster_tolerance_ = this->get_parameter("cluster_tolerance").as_double();
    min_cluster_size_ = this->get_parameter("min_cluster_size").as_int();
    max_cluster_size_ = this->get_parameter("max_cluster_size").as_int();
    voxel_leaf_size_ = this->get_parameter("voxel_leaf_size").as_double();
    use_downsampling_ = this->get_parameter("use_downsampling").as_bool();

    // Subscriber and Publishers
    sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
      input_topic_, 10,
      std::bind(&ObstacleDetectionNode::pointCloudCallback, this, std::placeholders::_1));
    cluster_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(cluster_topic_, 10);
    marker_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(marker_topic_, 10);
  }

private:
  void pointCloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr cloud_msg)
  {
    // Transform the point cloud to 'base_link' frame
    sensor_msgs::msg::PointCloud2 cloud_in_base_link;
    if (!transformPointCloud(cloud_msg, cloud_in_base_link, "base_link"))
    {
      RCLCPP_WARN(this->get_logger(), "Skipping point cloud processing due to transform failure.");
      return;
    }

    // Convert to PCL and proceed with processing
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::fromROSMsg(cloud_in_base_link, *cloud);
    // Downsample the point cloud (optional)
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZ>());
    if (use_downsampling_)
    {
      pcl::VoxelGrid<pcl::PointXYZ> vg;
      vg.setInputCloud(cloud);
      vg.setLeafSize(voxel_leaf_size_, voxel_leaf_size_, voxel_leaf_size_);
      vg.filter(*cloud_filtered);
    }
    else
    {
      cloud_filtered = cloud;
    }

    // Create the KdTree object for the search method of the extraction
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());
    tree->setInputCloud(cloud_filtered);

    // Perform clustering
    std::vector<pcl::PointIndices> cluster_indices;
    pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
    ec.setClusterTolerance(cluster_tolerance_);
    ec.setMinClusterSize(min_cluster_size_);
    ec.setMaxClusterSize(max_cluster_size_);
    ec.setSearchMethod(tree);
    ec.setInputCloud(cloud_filtered);
    ec.extract(cluster_indices);

    // Prepare to publish clusters and markers
    visualization_msgs::msg::MarkerArray marker_array;
    int cluster_id = 0;

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_clusters(new pcl::PointCloud<pcl::PointXYZRGB>());

    for (const auto& indices : cluster_indices)
    {
      // Assign a random color to each cluster
      //uint8_t r = static_cast<uint8_t>(rand() % 256);
      //uint8_t g = static_cast<uint8_t>(rand() % 256);
      //uint8_t b = static_cast<uint8_t>(rand() % 256);

      // Create a new point cloud for each cluster
      pcl::PointCloud<pcl::PointXYZRGB>::Ptr cluster_cloud(new pcl::PointCloud<pcl::PointXYZRGB>());
      for (const auto& idx : indices.indices)
      {
        pcl::PointXYZ point = cloud_filtered->points[idx];
        pcl::PointXYZRGB point_rgb;
        point_rgb.x = point.x;
        point_rgb.y = point.y;
        point_rgb.z = point.z;
        //point_rgb.r = r;
        //point_rgb.g = g;
        //point_rgb.b = b;
        cluster_cloud->points.push_back(point_rgb);
      }

      //check if cloud is empty
      if (cluster_cloud->points.empty())
      {
        RCLCPP_WARN(this->get_logger(), "Encountered an empty cluster. Skipping...");
        continue; // Skip processing this cluster
      }

      // Seed the random number generator with the cluster ID for consistent coloring
      std::mt19937 gen(cluster_id); // Standard mersenne_twister_engine seeded with cluster_id
      std::uniform_int_distribution<> dis(0, 255);
      uint8_t r = static_cast<uint8_t>(dis(gen));
      uint8_t g = static_cast<uint8_t>(dis(gen));
      uint8_t b = static_cast<uint8_t>(dis(gen));

      // Assign the color to each point in the cluster
      for (auto& point_rgb : cluster_cloud->points)
      {
        point_rgb.r = r;
        point_rgb.g = g;
        point_rgb.b = b;
      }


      // Add cluster cloud to the aggregated cloud
      *cloud_clusters += *cluster_cloud;

      // Compute the bounding box of the cluster
      pcl::PointXYZRGB min_pt, max_pt;  // Changed to PointXYZRGB
      pcl::getMinMax3D<pcl::PointXYZRGB>(*cluster_cloud, min_pt, max_pt);  // Specify template parameter

      // Create a bounding box marker
      visualization_msgs::msg::Marker marker;
      marker.header = cloud_msg->header;
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
      marker.scale.x = std::max((max_pt.x - min_pt.x), 0.1f);  // Added minimum size
      marker.scale.y = std::max((max_pt.y - min_pt.y), 0.1f);
      marker.scale.z = std::max((max_pt.z - min_pt.z), 0.1f);
      marker.color.r = static_cast<float>(r) / 255.0;
      marker.color.g = static_cast<float>(g) / 255.0;
      marker.color.b = static_cast<float>(b) / 255.0;
      marker.color.a = 0.5; // Semi-transparent
      marker.lifetime = rclcpp::Duration::from_seconds(0);

      marker_array.markers.push_back(marker);

      cluster_id++;
    }

    // Publish the clustered point cloud
    sensor_msgs::msg::PointCloud2 output_clusters;
    pcl::toROSMsg(*cloud_clusters, output_clusters);
    output_clusters.header = cloud_msg->header;
    cluster_pub_->publish(output_clusters);

    // Publish the markers
    marker_pub_->publish(marker_array);
  }

  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr cluster_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_pub_;

  std::string input_topic_;
  std::string cluster_topic_;
  std::string marker_topic_;
  double cluster_tolerance_;
  int min_cluster_size_;
  int max_cluster_size_;
  double voxel_leaf_size_;
  bool use_downsampling_;
};

int main(int argc, char *argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<ObstacleDetectionNode>());
  rclcpp::shutdown();
  return 0;
}
