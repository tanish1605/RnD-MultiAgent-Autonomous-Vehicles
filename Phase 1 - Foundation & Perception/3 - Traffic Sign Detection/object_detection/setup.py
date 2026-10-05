from setuptools import setup
import os
from glob import glob

package_name = 'object_detection'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),

        ('share/' + package_name,
            ['package.xml']),

        ('share/' + package_name + '/launch',
            glob('launch/*.py')),

        ('share/' + package_name + '/config',
            glob('config/*')),

        ('share/' + package_name + '/models',
            glob('models/*')),

        ("share/" + package_name + "/launch",
            glob("launch/*.launch.py")),
    ],

    install_requires=['setuptools'],

    zip_safe=True,

    maintainer='car-02',

    maintainer_email='car-02@todo.todo',

    description='YOLOv8 Object Detection',

    license='Apache License 2.0',

    tests_require=['pytest'],

    entry_points={
        'console_scripts': [
            'inference_node = object_detection.inference_node:main',
        ],
    },
)
