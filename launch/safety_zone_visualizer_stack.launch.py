"""Launch full SafetyZone visualizer runtime stack."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    input_topic_arg = DeclareLaunchArgument(
        "input_topic",
        default_value="/ouster/points/filtered",
        description="PointCloud2 input topic for obstacle_detection_cuda.",
    )

    enable_box_filter_arg = DeclareLaunchArgument(
        "enable_box_filter",
        default_value="true",
        description="Enable lidar box filter.",
    )
    enable_aperture_filter_arg = DeclareLaunchArgument(
        "enable_aperture_filter",
        default_value="false",
        description="Enable lidar aperture filter.",
    )
    enable_ground_filter_arg = DeclareLaunchArgument(
        "enable_ground_filter",
        default_value="true",
        description="Enable lidar ground filter.",
    )

    robot_tf_calibrated_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [
                    FindPackageShare("vario700_sensorrig"),
                    "launch",
                    "robot_tf_calibrated.launch.py",
                ]
            )
        ),
        launch_arguments={"calibration_source": "msa_new"}.items(),
    )

    # params_cuda.yaml first, so the input_topic launch arg overrides it.
    obstacle_detection_cuda_node = Node(
        package="obstacle_detection",
        executable="obstacle_detection_cuda",
        name="obstacle_detection_cuda_node",
        output="screen",
        emulate_tty=True,
        parameters=[
            PathJoinSubstitution(
                [FindPackageShare("obstacle_detection"), "config", "params_cuda.yaml"]
            ),
            {"input_topic": LaunchConfiguration("input_topic")},
        ],
    )

    combined_lidar_filter_node = Node(
        package="lidar_filter",
        executable="combined_lidar_filter_node",
        name="combined_lidar_filter_node",
        output="screen",
        emulate_tty=True,
        parameters=[
            {
                "enable_box_filter": LaunchConfiguration("enable_box_filter"),
                "enable_aperture_filter": LaunchConfiguration(
                    "enable_aperture_filter"
                ),
                "enable_ground_filter": LaunchConfiguration("enable_ground_filter"),
            }
        ],
    )

    yolo_multi_cam_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [
                    FindPackageShare("yolo_multi_cam"),
                    "launch",
                    "yolo_multi_cam.launch.py",
                ]
            )
        )
    )

    detection_fusion_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [
                    FindPackageShare("obstacle_detection"),
                    "launch",
                    "detection_fusion.launch.py",
                ]
            )
        )
    )

    return LaunchDescription(
        [
            input_topic_arg,
            enable_box_filter_arg,
            enable_aperture_filter_arg,
            enable_ground_filter_arg,
            robot_tf_calibrated_launch,
            obstacle_detection_cuda_node,
            combined_lidar_filter_node,
            yolo_multi_cam_launch,
            detection_fusion_launch,
        ]
    )
