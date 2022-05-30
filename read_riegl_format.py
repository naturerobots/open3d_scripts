#!python3

import pandas as pd
import open3d as o3d
import argparse
import os
import glob


def convert_pc_data(args):
    os.chdir(args.dataset)
    files = glob.glob("*.ascii")
    files.sort()
    for file in files:
        print(file)
        df = pd.read_csv(file, names=['x', 'y', 'z', 'r', 'g', 'b'])
        points = df[['x', 'y', 'z']]
        colors = df[['r', 'g', 'b']] / 255.0

        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(points.to_numpy())
        cloud.colors = o3d.utility.Vector3dVector(colors.to_numpy())
        o3d.io.write_point_cloud(os.path.splitext(file)[0]+".ply", cloud)


def main():
    parser = argparse.ArgumentParser(description='Reads [x, y, z, r, g, b,] ascii files and writes ply files',)

    # IO
    parser.add_argument('-i', '--input', dest='dataset', required=True, action='store',
                        help='Dataset directory, point cloud input directory', type=str)

    args = parser.parse_args()
    convert_pc_data(args)


if __name__ == "__main__":
    main()
