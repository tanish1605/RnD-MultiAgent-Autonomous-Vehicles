from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription

from launch.launch_description_sources import PythonLaunchDescriptionSource

from launch_ros.actions import Node

from ament_index_python.packages import get_package_share_directory

import os


def generate_launch_description():

    depthai_launch = os.path.join(

        get_package_share_directory("depthai_ros_driver_v3"),

        "launch",

        "driver.launch.py"

    )


    return LaunchDescription([

        IncludeLaunchDescription(

            PythonLaunchDescriptionSource(depthai_launch)

        ),

        Node(

            package="object_detection",

            executable="inference_node",

            name="inference_node",

            output="screen"

        )

    ])
