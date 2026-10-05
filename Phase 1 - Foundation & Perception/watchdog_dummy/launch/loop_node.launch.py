import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    params_file = os.path.join(
        get_package_share_directory("watchdog_dummy"),
        "config",
        "loop_node_params.yaml",
    )

    return LaunchDescription([
        Node(
            package="watchdog_dummy",
            executable="loop_node",
            name="watchdog_dummy_loop",
            output="screen",
            parameters=[params_file],
        ),
    ])
