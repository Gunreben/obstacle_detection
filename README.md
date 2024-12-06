# Obstacle Detection ROS 2 Package

This repository contains a ROS 2 package for obstacle detection based on point cloud clustering. It provides two nodes: a CPU-based implementation using standard PCL Euclidean clustering, and a CUDA-accelerated version for faster performance on systems with Jetson Platform with Jetpack 6.0 (General NVIDIA GPU Compatibility not tested, used branch jp_6.x of cuPCL: https://github.com/NVIDIA-AI-IOT/cuPCL)

## Features:

    CPU Node: Leverages the PCL library's EuclideanClusterExtraction for identifying clusters in 3D point clouds.
    CUDA Node: Utilizes a CUDA-based clustering approach to accelerate segmentation, potentially offering significant speed-ups on supported hardware.
    Transforms and Visualization: Automatically transforms incoming point clouds into a target frame and publishes visualization markers (bounding boxes) around detected clusters.

## Requirements

    ROS 2 (tested on Humble)
    PCL (Point Cloud Library)
    NVIDIA Jetson with CUDA (for the CUDA node)
    tf2 and tf2_ros for handling transforms
    rclcpp, sensor_msgs, visualization_msgs for ROS integration

## Installation

    Clone the repository into your ROS 2 workspace:

cd ~/ros2_ws/src
git clone https://github.com/yourusername/obstacle_detection.git

Install dependencies: Make sure all necessary dependencies are installed. For example on Ubuntu:

sudo apt-get update
sudo apt-get install -y ros-${ROS_DISTRO}-pcl-ros ros-${ROS_DISTRO}-tf2-ros ros-${ROS_DISTRO}-visualization-msgs
Install CUDA if not already installed. On Jetson devices, CUDA is generally pre-installed.

Build the package:

    cd ~/ros2_ws
    colcon build --packages-select obstacle_detection
    source install/setup.bash

## Running the Nodes

### CPU Node:

ros2 run obstacle_detection obstacle_detection_node

### CUDA Node:

ros2 run obstacle_detection obstacle_detection_cuda

Before running, ensure that a point cloud source is available (e.g., from a lidar sensor, RGB-D camera, or a recorded rosbag). Adjust parameters as needed in the node’s YAML configuration file or via command-line parameters.
Parameters

    input_topic (string): The input point cloud topic. Default: /filtered_fov_points
    cluster_topic (string): The output point cloud topic for clusters. Default: /detected_obstacles
    marker_topic (string): The topic for visualization markers. Default: /obstacle_markers
    target_frame (string): The frame to which the input clouds are transformed. Default: base_link
    cluster_tolerance (double): Spatial tolerance for clustering (CPU node) / a heuristic used to set thresholds for CUDA node.
    min_cluster_size (int): The minimum number of points in a cluster.
    max_cluster_size (int): The maximum number of points in a cluster.
    voxel_leaf_size (double): The leaf size for optional voxel grid downsampling.
    use_downsampling (bool): Whether to apply voxel grid downsampling before clustering.

For the CUDA node, parameters like cluster_tolerance affect countThreshold and voxelization parameters internally. Adjust them if no clusters appear.
Troubleshooting

    If no clusters are detected in the CUDA node, experiment with reducing countThreshold or decreasing the voxel size parameters.
    If you encounter linking errors, ensure that libcudacluster.so is properly installed and located in a directory known to the linker at runtime.
    If tf warnings appear, ensure the appropriate transform frames are being broadcasted.

