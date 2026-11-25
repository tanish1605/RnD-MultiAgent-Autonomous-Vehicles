import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    params_file = os.path.join(
        get_package_share_directory('semantic_segmentation'),
        'config',
        'segmentation_params.yaml',
    )

    return LaunchDescription([
        Node(
            package='semantic_segmentation',
            executable='segmentation_node',
            name='segmentation_node',
            output='screen',
            parameters=[params_file],
        ),
    ])
