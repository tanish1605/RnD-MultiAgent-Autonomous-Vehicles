import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    params_file = os.path.join(
        get_package_share_directory('lidar_obstacle_detector'),
        'config',
        'obstacle_detector_params.yaml',
    )

    return LaunchDescription([
        Node(
            package='lidar_obstacle_detector',
            executable='obstacle_detector',
            name='obstacle_detector',
            output='screen',
            parameters=[params_file],
        ),
    ])
