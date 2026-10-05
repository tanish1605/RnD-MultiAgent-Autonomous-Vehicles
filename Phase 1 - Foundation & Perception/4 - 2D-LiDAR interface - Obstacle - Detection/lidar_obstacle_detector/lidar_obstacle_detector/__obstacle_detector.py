#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Point

import tf2_ros
import tf2_geometry_msgs

import math
import matplotlib.pyplot as plt

from obstacle_msgs.msg import Obstacle
from visualization_msgs.msg import Marker

from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy



class ObstacleDetector(Node):

    def __init__(self):

        super().__init__("obstacle_detector")


        # =============================
        # TF SETUP
        # =============================

        self.tf_buffer = tf2_ros.Buffer()

        self.tf_listener = tf2_ros.TransformListener(
            self.tf_buffer,
            self
        )


        # =============================
        # SUBSCRIBER
        # =============================

        scan_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT
        )


        self.create_subscription(
            LaserScan,
            "/scan",
            self.scan_callback,
            scan_qos
        )



        # =============================
        # PUBLISHERS
        # =============================

        self.obstacle_pub = self.create_publisher(
            Obstacle,
            "/front_obstacle",
            10
        )


        self.marker_pub = self.create_publisher(
            Marker,
            "/obstacle_marker",
            10
        )



        # =============================
        # PARAMETERS
        # =============================

        self.front_fov = math.radians(20)

        self.fov_offset = 0.0

        self.cluster_threshold = 0.20



        # =============================
        # MATPLOTLIB LIDAR VIEW
        # =============================

        plt.ion()

        self.fig, self.ax = plt.subplots()


        self.ax.set_xlim(-1,5)

        self.ax.set_ylim(-3,3)

        self.ax.set_aspect("equal")


        self.ax.set_xlabel(
            "X (m)"
        )

        self.ax.set_ylabel(
            "Y (m)"
        )


        self.ax.grid()



        # laser rays

        self.ray_lines = []



        # scan points

        self.scan_plot, = self.ax.plot(
            [],
            [],
            '.',
            markersize=3
        )



        # obstacle width line

        self.obstacle_edge, = self.ax.plot(
            [],
            [],
            linewidth=3
        )



        # obstacle center

        self.obstacle_center, = self.ax.plot(
            [],
            [],
            'o',
            markersize=8
        )



        self.info_text = self.ax.text(
            -0.8,
            2.5,
            ""
        )



        self.get_logger().info(
            "Obstacle detector running"
        )




    # =====================================================
    # SCAN CALLBACK
    # =====================================================

    def scan_callback(self, scan):


        points=[]



        # =============================
        # LASER -> XY + FOV
        # =============================

        for i,r in enumerate(scan.ranges):


            angle = (
                scan.angle_min +
                i*scan.angle_increment
            )


            angle += self.fov_offset



            if abs(angle) > self.front_fov:

                continue



            if scan.range_min < r < scan.range_max:


                x = r*math.cos(angle)

                y = r*math.sin(angle)


                points.append(
                    (x,y)
                )



        if len(points)<5:

            return



        # =============================
        # DRAW DENSE LASER RAYS
        # =============================


        for line in self.ray_lines:

            line.remove()


        self.ray_lines.clear()



        for p in points:


            line, = self.ax.plot(
                [0,p[0]],
                [0,p[1]],
                linewidth=0.5
            )


            self.ray_lines.append(line)



        # scan dots

        self.scan_plot.set_data(

            [p[0] for p in points],

            [p[1] for p in points]

        )





        # =============================
        # CLUSTERING
        # =============================

        clusters=[]

        current=[]



        for p in points:


            if not current:

                current.append(p)

                continue



            last=current[-1]


            d=math.hypot(
                p[0]-last[0],
                p[1]-last[1]
            )



            if d < self.cluster_threshold:

                current.append(p)


            else:


                if len(current)>5:

                    clusters.append(current)


                current=[p]



        if len(current)>5:

            clusters.append(current)



        if not clusters:

            return




        # =============================
        # CLOSEST OBSTACLE
        # =============================


        obstacle=min(
            clusters,
            key=lambda c:
            math.hypot(
                c[len(c)//2][0],
                c[len(c)//2][1]
            )
        )



        left=obstacle[0]

        right=obstacle[-1]



        cx=(left[0]+right[0])/2

        cy=(left[1]+right[1])/2



        width=math.hypot(
            right[0]-left[0],
            right[1]-left[1]
        )


        distance=math.hypot(
            cx,
            cy
        )



        # =============================
        # MATPLOTLIB OBSTACLE DISPLAY
        # =============================


        self.obstacle_edge.set_data(

            [left[0],right[0]],

            [left[1],right[1]]

        )


        self.obstacle_center.set_data(

            [cx],

            [cy]

        )



        self.info_text.set_text(

            f"Distance : {distance:.2f} m\n"
            f"Obstacle Width : {width:.2f} m"

        )


        self.fig.canvas.draw()

        self.fig.canvas.flush_events()




        # =============================
        # TF
        # =============================


        center_odom=self.transform_point(
            cx,
            cy
        )


        left_odom=self.transform_point(
            left[0],
            left[1]
        )


        right_odom=self.transform_point(
            right[0],
            right[1]
        )



        self.publish_marker()



        if center_odom is None:

            return




        # =============================
        # MESSAGE
        # =============================

        msg=Obstacle()


        msg.center_x=center_odom[0]

        msg.center_y=center_odom[1]


        msg.distance=distance

        msg.width=width



        msg.left_x=left_odom[0] if left_odom else 0.0

        msg.left_y=left_odom[1] if left_odom else 0.0



        msg.right_x=right_odom[0] if right_odom else 0.0

        msg.right_y=right_odom[1] if right_odom else 0.0



        msg.angle=math.atan2(
            cy,
            cx
        )



        self.obstacle_pub.publish(msg)




    # =====================================================
    # TF TRANSFORM
    # =====================================================

    def transform_point(self,x,y):


        p=tf2_geometry_msgs.PointStamped()


        p.header.frame_id="lidar_link"


        p.header.stamp=(
            self.get_clock()
            .now()
            .to_msg()
        )


        p.point.x=x

        p.point.y=y

        p.point.z=0.0



        try:


            tf=self.tf_buffer.lookup_transform(

                "odom",

                "lidar_link",

                rclpy.time.Time()

            )


            out=tf2_geometry_msgs.do_transform_point(
                p,
                tf
            )


            return (
                out.point.x,
                out.point.y
            )


        except Exception:

            return None




    # =====================================================
    # RVIZ FOV MARKER
    # =====================================================

    def publish_marker(self):


        m=Marker()


        m.header.frame_id="lidar_link"


        m.header.stamp=(

            self.get_clock()
            .now()
            .to_msg()

        )


        m.ns="fov"

        m.id=1

        m.type=Marker.LINE_LIST

        m.action=Marker.ADD


        m.scale.x=0.02


        m.color.g=1.0

        m.color.a=1.0



        length=5.0



        c=Point()

        c.x=0.0

        c.y=0.0



        left=Point()

        left.x=length*math.cos(
            self.front_fov
        )

        left.y=length*math.sin(
            self.front_fov
        )



        right=Point()

        right.x=length*math.cos(
            -self.front_fov
        )

        right.y=length*math.sin(
            -self.front_fov
        )



        m.points=[

            c,left,

            c,right

        ]


        self.marker_pub.publish(m)




# =====================================================
# MAIN
# =====================================================

def main():

    rclpy.init()


    node=ObstacleDetector()


    try:

        rclpy.spin(node)


    except KeyboardInterrupt:

        pass



    node.destroy_node()

    rclpy.shutdown()



if __name__=="__main__":

    main()