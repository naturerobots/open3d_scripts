import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
import sensor_msgs_py.point_cloud2 as pc2
from rclpy.qos import qos_profile_sensor_data


class PointCloudToPly(Node):
    def __init__(self):
        super().__init__("pointcloud_to_ply")
        self.subscription = self.create_subscription(
            PointCloud2,
            "/ouster/points",
            self.pointcloud_callback,
            qos_profile=qos_profile_sensor_data,
        )
        self.subscription  # prevent unused variable warning
        self.received_first_scan = False

    def pointcloud_callback(self, msg):
        if not self.received_first_scan:
            self.received_first_scan = True
            self.save_ply(msg)

    def save_ply(self, msg):
        points = []
        for point in pc2.read_points(msg, skip_nans=True):
            points.append(point)

        ply_header = """ply
format ascii 1.0
element vertex {0}
property float x
property float y
property float z
end_header
""".format(len(points))

        with open("output.ply", "w") as f:
            f.write(ply_header)
            for point in points:
                f.write("{0} {1} {2}\n".format(point[0], point[1], point[2]))

        self.get_logger().info("Saved point cloud to output.ply")


def main(args=None):
    rclpy.init(args=args)
    node = PointCloudToPly()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
