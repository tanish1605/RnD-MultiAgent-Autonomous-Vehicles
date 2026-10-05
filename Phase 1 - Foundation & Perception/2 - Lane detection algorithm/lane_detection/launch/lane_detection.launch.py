import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    config_dir = os.path.join(get_package_share_directory('lane_detection'), 'config')

    return LaunchDescription([
        Node(
            package='lane_detection',
            executable='lane_detection',
            name='lane_detection',
            output='screen',
            parameters=[
                os.path.join(config_dir, 'lane_detection_params.yaml'),
                os.path.join(config_dir, 'camera_calibration.yaml'),
            ],
        ),
    ])
