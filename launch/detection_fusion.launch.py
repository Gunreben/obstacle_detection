import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    pkg_dir = get_package_share_directory("obstacle_detection")
    params_file = os.path.join(pkg_dir, "config", "fusion_params.yaml")

    return LaunchDescription([
        Node(
            package="obstacle_detection",
            executable="detection_fusion_node.py",
            name="detection_fusion_node",
            parameters=[params_file],
            output="screen",
        ),
    ])
