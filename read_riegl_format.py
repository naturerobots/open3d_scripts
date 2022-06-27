#!python3

import pandas as pd
import open3d as o3d
import argparse
import os
import glob

from numpy import array


def convert_pc_data(args):
    os.chdir(args.dataset)
    files = glob.glob("*.ascii")
    files.sort()
    for file in files:
        print("read", file)
        df = pd.read_csv(file, names=['x', 'y', 'z', 'r', 'g', 'b'])
        points = df[['x', 'y', 'z']]
        colors = df[['r', 'g', 'b']] / 255.0

        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(points.to_numpy())
        cloud.colors = o3d.utility.Vector3dVector(colors.to_numpy())
        print("voxel grid down sample...")
        cloud = cloud.voxel_down_sample(voxel_size=0.03)
        print("remove radius outlier...")
        cloud, ind = cloud.remove_radius_outlier(nb_points=16, radius=0.15)
        print("estimate point normals...")
        cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        print("orient normals consistently...")
        #cloud.orient_normals_consistent_tangent_plane(8)
        cloud.orient_normals_to_align_with_direction(orientation_reference=array([0., 0., 1.]))
        cloud.orient_normals_towards_camera_location(camera_location=array([0., 0., 5.]))
        print("write point cloud to file...")
        o3d.io.write_point_cloud(os.path.splitext(file)[0]+".ply", cloud)
        print("----")


def main():
    parser = argparse.ArgumentParser(description='Reads [x, y, z, r, g, b,] ascii files and writes ply files',)

    # IO
    parser.add_argument('-i', '--input', dest='dataset', required=True, action='store',
                        help='Dataset directory, point cloud input directory', type=str)

    args = parser.parse_args()
    convert_pc_data(args)


if __name__ == "__main__":
    main()
