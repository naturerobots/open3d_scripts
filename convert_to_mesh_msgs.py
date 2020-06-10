#!python3
import open3d as o3d
import argparse
import rospy
import numpy as np
from mesh_msgs.msg import MeshGeometryStamped, TriangleIndices
from geometry_msgs.msg import Point


def open3d2mesh_msgs(mesh):
    msg = MeshGeometryStamped()
    vertex_array = np.asarray(mesh.vertices)
    msg.mesh_geometry.vertices = [Point(x=v[0], y=v[1], z=v[2]) for v in np.asarray(mesh.vertices)]
    msg.mesh_geometry.faces = [TriangleIndices(v) for v in np.asarray(mesh.triangles)]
    return msg


def main():
    rospy.init_node("convert_to_mesh_msgs")
    parser = argparse.ArgumentParser(
        description='Reconstruction of point cloud data (PCD) to a reconstructed mesh using Open3D',
    )

    parser.add_argument('-i', '--input', dest='input', required=True, action='store',
                        help='triangle mesh input file', type=str)

    parser.add_argument('--frame', dest='frame', required=False, action='store',
                        help='ros frame_id', type=str, default='mesh')

    args = parser.parse_args()
    mesh = o3d.io.read_triangle_mesh(args.input)
    pub = rospy.Publisher('mesh', MeshGeometryStamped, queue_size=1, latch=True)
    msg = open3d2mesh_msgs(mesh)
    msg.header.frame_id = args.frame
    msg.header.stamp = rospy.Time.now()
    pub.publish(msg)
    print("published mesh...")
    rospy.spin()


if __name__ == "__main__":
    main()
