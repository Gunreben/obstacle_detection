"""Launch the CPU-based obstacle detection node with a parameter file."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    pkg_share = FindPackageShare("obstacle_detection")

    params_file_arg = DeclareLaunchArgument(
        "params_file",
        default_value=PathJoinSubstitution([pkg_share, "config", "params.yaml"]),
        description="Path to the parameter YAML file",
    )

    node = Node(
        package="obstacle_detection",
        executable="obstacle_detection_node",
        name="obstacle_detection_node",
        parameters=[LaunchConfiguration("params_file")],
        output="screen",
        emulate_tty=True,
    )

    return LaunchDescription([params_file_arg, node])
